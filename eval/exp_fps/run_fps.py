"""E1.01 / E1.04: IndEgo pixel-feature frame-selection experiments.

sweep (E1.01): decode each window ONCE at native 10 fps, subsample by the
frame-index rule idx=arange(0,T,step) with step in {10:1,5:2,3.33:3,2:5,1:10},
compute the 26 cpu_window_features with the TRUE effective fps (10/step).

select (E1.04): fixed 20-frame budget. Arms: uniform20 (step 5), motion20
(top-20 inter-frame-diff frames + frame 0), random20 (seed 0), keyframe20
(I-frames from ffprobe). Selection arms are not uniformly spaced, so the
fps-dependent features are not used for E1.04 (merge_e1_04.py keeps only
SPATIAL_FEATURES). The keyframe arm is INVALID unless every window has at
least `budget` distinct I-frame positions on the 10 fps grid; otherwise its
features are NaN and the shard records keyframe_fallback=true with the count.

Regen (CPU, 2 processes; then merge):
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python \\
    eval/exp_fps/run_fps.py --mode sweep --shard 0/2   # and --shard 1/2
  PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/exp_fps/merge_e1_01.py
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fps_lib import (FPS_STEPS, NATIVE_FPS, W, H, WIN_S, decode_window,
                     gpu_features, load_manifest, subsample_idx)

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "exp_fps")
NAN_ROW_LEN = 26


def compute_for_window(gf, rgb, luma, idx, fps_eff):
    t0 = time.perf_counter()
    d = gf.cpu_window_features(luma[idx].copy(), rgb[idx].copy(),
                               fps=float(fps_eff))
    dt = (time.perf_counter() - t0) * 1000.0
    return d, dt


def motion_select(luma, budget=20):
    """Top-(budget-1) inter-frame-diff frames plus frame 0. Sorted, unique."""
    T = len(luma)
    e = np.abs(np.diff(luma.astype(np.float64), axis=0)).mean(axis=(1, 2))
    r = np.argsort(-e, kind="stable")
    keep = np.concatenate([[0], (r[:budget - 1] + 1).clip(0, T - 1)])
    return np.unique(keep).astype(int)


def random_select(T, budget=20, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(T, budget, replace=False)).astype(int)


def parse_ffprobe_iframes(stdout, start_s, win_s=WIN_S):
    """Parse `ffprobe -show_entries frame=pts_time,pict_type -of csv=p=0`
    output and return I-frame timestamps inside [start_s, start_s+win_s).
    Raises ValueError on a malformed timestamp (no silent skipping)."""
    ts = []
    for line in stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2 or parts[1] != "I":
            continue
        t = float(parts[0])  # ValueError on N/A or junk: fail loudly
        if start_s <= t < start_s + win_s:
            ts.append(t)
    return ts


def iframe_times(path, start_s):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "frame=pts_time,pict_type",
           "-of", "csv=p=0", path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {r.stderr[:300]}")
    return parse_ffprobe_iframes(r.stdout, start_s)


def keyframe_select(i_times, start_s, T, budget=20):
    """Map I-frame times to decoded-frame indices (10 fps grid).

    Returns (idx, n_iframes, fallback). fallback=True when fewer than
    `budget` distinct I-frame positions exist; idx is then None and the
    keyframe arm must be reported as INVALID for that window.
    """
    pos = sorted(set(int(round((t - start_s) * NATIVE_FPS)) for t in i_times))
    pos = [i for i in pos if 0 <= i < T]
    n = len(pos)
    if n < budget:
        return None, n, True
    sel = [pos[int(k)] for k in np.linspace(0, n - 1, budget)]
    return np.array(sel, dtype=int), n, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sweep", "select"], default="sweep")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    si, sn = [int(x) for x in a.shard.split("/")]
    items = load_manifest()
    mine = [it for k, it in enumerate(items) if k % sn == si]
    gf = gpu_features()
    names = list(gf.FEATURE_NAMES)
    assert len(names) == NAN_ROW_LEN, names

    if a.mode == "sweep":
        cfgs = [(f, s, NATIVE_FPS / s) for f, s in FPS_STEPS.items()]
    else:  # E1.04: fixed 20-frame budget, uniform reference at step 5
        cfgs = [("uniform20", 5, 2.0)]

    feats = {c[0]: [] for c in cfgs}
    feat_ms = {c[0]: [] for c in cfgs}
    decode_ms, nframes, ids, clips, y = [], [], [], [], []
    kf_iframes, kf_fallback = [], []
    for it in mine:
        rgb, luma, dms = decode_window(it["path"], it["start_s"])
        T = len(luma)
        decode_ms.append(dms)
        nframes.append(T)
        ids.append(it["id"])
        clips.append(it["clip"])
        y.append(it["label"])
        for tag, step, feff in cfgs:
            idx = subsample_idx(T, step)
            d, fms = compute_for_window(gf, rgb, luma, idx, feff)
            feats[tag].append([d[n] for n in names])
            feat_ms[tag].append(fms)
        if a.mode == "select":
            # motion-adaptive selection (dt-dependent features discarded later)
            keep = motion_select(luma, 20)
            d, fms = compute_for_window(gf, rgb, luma, keep, 2.0)
            feats.setdefault("motion20", []).append([d[n] for n in names])
            feat_ms.setdefault("motion20", []).append(fms)
            # random control, seed fixed
            keep = random_select(T, 20, a.seed)
            d, fms = compute_for_window(gf, rgb, luma, keep, 2.0)
            feats.setdefault("random20", []).append([d[n] for n in names])
            feat_ms.setdefault("random20", []).append(fms)
            # keyframe-only: I-frame positions from ffprobe
            iT = iframe_times(it["path"], it["start_s"])
            keep, n_if, fb = keyframe_select(iT, it["start_s"], T, 20)
            kf_iframes.append(n_if)
            kf_fallback.append(bool(fb))
            if fb:
                row, kms = [float("nan")] * NAN_ROW_LEN, 0.0
            else:
                d, kms = compute_for_window(gf, rgb, luma, keep, 2.0)
                row = [d[n] for n in names]
            feats.setdefault("keyframe", []).append(row)
            feat_ms.setdefault("keyframe", []).append(kms)
    out = {"ids": ids, "clips": clips, "y": y, "names": names,
           "feats": {k: np.array(v).tolist() for k, v in feats.items()},
           "decode_ms": decode_ms, "feat_ms": feat_ms, "nframes": nframes,
           "mode": a.mode, "decode_size": [W, H]}
    if a.mode == "select":
        out["keyframe_n_iframes"] = kf_iframes
        out["keyframe_fallback"] = kf_fallback
        n_fb = sum(kf_fallback)
        print(f"keyframe_fallback windows: {n_fb}/{len(ids)}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"shard_{a.mode}_{si}of{sn}.json")
    json.dump(out, open(p, "w"))
    print(f"wrote {p} n={len(ids)}")


if __name__ == "__main__":
    main()
