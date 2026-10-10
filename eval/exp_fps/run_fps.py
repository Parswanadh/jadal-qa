"""E1.01 (sweep) + E1.04 (select): IndEgo pixel-feature fps / frame-selection eval.

--mode sweep  : decode each window ONCE at native ~10fps, deterministically
                subsample by frame-index rule idx=arange(0,T,step), compute the
                26 cpu_window_features with the TRUE effective fps, LOCO
                logistic (C=0.3) + clip-bootstrap CI. (E1.01, already done)
--mode select : E1.04 equal 20-frame budget (~2fps):
                uniform20 / motion20 / random20 / keyframe.

Data paths come from $JADAL_BUNDLE; results go to eval/results/exp_fps (or
$JADAL_FPS_RESULTS). Shard with --shard i/n.

Regen:
  OMP_NUM_THREADS=1 python3 eval/exp_fps/run_fps.py --mode sweep  --shard 0/2
  OMP_NUM_THREADS=1 python3 eval/exp_fps/run_fps.py --mode select --shard 0/2
then merge_e1_01.py / merge_e1_04.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fps_lib import (FPS_STEPS, NATIVE_FPS, RESULTS, W, H, decode_window,
                     git_sha, gpu_features, load_manifest, loco_proba,
                     clip_bootstrap_ci, bal_acc, subsample_idx)

STRATEGIES = ["uniform20", "motion20", "random20", "keyframe"]
BUDGET = 20


def compute_for_window(gf, rgb, luma, idx, fps_eff):
    t0 = time.perf_counter()
    d = gf.cpu_window_features(luma[idx].copy(), rgb[idx].copy(),
                               fps=float(fps_eff))
    dt = (time.perf_counter() - t0) * 1000.0
    return d, dt


def motion_rank(luma):
    """Descending mean |inter-frame diff| frame order (stable)."""
    e = np.abs(np.diff(luma.astype(np.float64), axis=0)).mean(axis=(1, 2))
    return np.argsort(-e, kind="stable")


def motion_keep(luma, budget=BUDGET):
    """top-(budget-1) inter-frame-diff frames + frame 0 as anchor."""
    r = motion_rank(luma)
    return np.unique(np.concatenate([[0], (r[:budget - 1] + 1)]))


def random_keep(T, seed, budget=BUDGET):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(T, min(budget, T), replace=False))


def keyframe_positions(path, t0=None, t1=None, timeout=180):
    """I-frame presentation timestamps (s) of a video via ffprobe.

    ffprobe 6.x prints `pts_time,pict_type,` -> lines like '4.600000,I,'.
    (The old code asked for `pkt_pts_time`, which is empty in ffprobe 6.1 and
    silently returned no I-frames -> the keyframe arm was identical to uniform.
    Fixed here.)

    Uses `-skip_frame nokey -read_intervals` so only the window is scanned:
    a full `frame=pict_type` scan of an 11-minute IndEgo clip takes ~43 s,
    while a 10 s interval costs ~0.6 s.
    """
    import subprocess
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey"]
    if t0 is not None and t1 is not None:
        cmd += ["-read_intervals", f"{max(t0, 0.0):.3f}%{t1:.3f}"]
    cmd += ["-show_entries", "frame=pts_time,pict_type", "-of", "csv=p=0", path]
    ts = []
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        for line in r.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2 and parts[1] == "I" and parts[0]:
                try:
                    ts.append(float(parts[0]))
                except ValueError:
                    continue
    except Exception:
        return []
    return ts


def keyframe_idx(path, start_s, T, budget=BUDGET):
    """I-frames inside [start_s, start_s+10s] mapped to the native 10fps grid.

    If fewer than `budget` I-frames exist the arm is padded with uniform frames
    (that padding is reported: the arm degenerates to uniform for windows with
    <budget I-frames, which is the honest result at 10fps IndEgo).
    Returns (idx, n_iframes_in_window, n_iframes_used).
    """
    end_s = start_s + 10.0
    ts = [t for t in keyframe_positions(path, start_s - 1.0, end_s)
          if start_s - 1.0 <= t <= end_s]
    n_if = len(ts)
    idx = sorted(set(int(round((t - start_s) * NATIVE_FPS)) for t in ts))
    idx = [i for i in idx if 0 <= i < T]
    if len(idx) >= budget:
        pick = [idx[int(k)] for k in np.linspace(0, len(idx) - 1, budget)]
        return np.array(sorted(set(pick)), dtype=int), n_if, len(set(pick))
    uni = list(subsample_idx(T, 5))
    merged = sorted(set(idx) | set(uni))[:budget]
    return np.array(merged, dtype=int), n_if, len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sweep", "select"], default="sweep")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    si, sn = [int(x) for x in a.shard.split("/")]
    items = load_manifest()
    mine = [it for k, it in enumerate(items) if k % sn == si]
    gf = gpu_features()
    names = list(gf.FEATURE_NAMES)
    assert len(names) == 26, names

    if a.mode == "sweep":
        cfgs = [(str(f), s, NATIVE_FPS / s) for f, s in FPS_STEPS.items()]
    else:
        cfgs = [("uniform20", 5, 2.0)]

    feats = {c[0]: [] for c in cfgs}
    feat_ms = {c[0]: [] for c in cfgs}
    keep_idx = {c[0]: [] for c in cfgs}
    meta = {c[0]: [] for c in cfgs}
    decode_ms, nframes, ids, clips, y = [], [], [], [], []
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
            keep_idx[tag].append([int(i) for i in idx])
            meta[tag].append({"n_frames": int(len(idx)), "fps_eff": float(feff)})
        if a.mode == "select":
            arms = [("motion20", motion_keep(luma), {"n_iframes": None}),
                    ("random20", random_keep(T, a.seed), {"seed": a.seed})]
            kidx, nif, nif_used = keyframe_idx(it["path"], it["start_s"], T)
            arms.append(("keyframe", kidx,
                         {"n_iframes": nif, "n_iframes_used": nif_used}))
            for tag, idx, extra in arms:
                d, fms = compute_for_window(gf, rgb, luma, idx, 2.0)
                feats.setdefault(tag, []).append([d[n] for n in names])
                feat_ms.setdefault(tag, []).append(fms)
                keep_idx.setdefault(tag, []).append([int(i) for i in idx])
                meta.setdefault(tag, []).append(
                    {"n_frames": int(len(idx)), "fps_eff": 2.0, **extra})
    out = {"ids": ids, "clips": clips, "y": y, "names": names,
           "feats": {k: np.array(v).tolist() for k, v in feats.items()},
           "keep_idx": keep_idx, "meta": meta,
           "decode_ms": decode_ms, "feat_ms": feat_ms, "nframes": nframes,
           "mode": a.mode, "decode_size": [W, H], "seed": a.seed,
           "git_sha": git_sha()}
    os.makedirs(RESULTS, exist_ok=True)
    p = a.out or os.path.join(RESULTS, f"shard_{a.mode}_{si}of{sn}.json")
    json.dump(out, open(p, "w"))
    print(f"wrote {p} n={len(ids)}")


if __name__ == "__main__":
    main()
