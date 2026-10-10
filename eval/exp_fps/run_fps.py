"""E1.01: IndEgo pixel-feature fps sweep {10,5,3.33,2,1}.

Decode each window ONCE at native 10fps, deterministically subsample by
frame-index rule idx=arange(0,T,step), compute the 26 cpu_window_features
with the TRUE effective fps, LOCO logistic (C=0.3) + clip-bootstrap CI.

Also supports --mode select for E1.04 (20-frame budget selection variants).
Shard with --shard i/n to run 2 processes.

Regen: PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python \
  eval/exp_fps/run_fps.py --mode sweep --shard 0/1 ... (then merge_shards)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fps_lib import (FPS_STEPS, NATIVE_FPS, W, H, decode_window, gpu_features,
                     load_manifest, loco_proba, clip_bootstrap_ci, bal_acc,
                     subsample_idx)

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "exp_fps")


def compute_for_window(gf, rgb, luma, idx, fps_eff):
    t0 = time.perf_counter()
    d = gf.cpu_window_features(luma[idx].copy(), rgb[idx].copy(),
                               fps=float(fps_eff))
    dt = (time.perf_counter() - t0) * 1000.0
    return d, dt


def motion_rank(luma):
    e = np.abs(np.diff(luma.astype(np.float64), axis=0)).mean(axis=(1, 2))
    return np.argsort(-e, kind="stable")


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
    assert len(names) == 26, names

    if a.mode == "sweep":
        cfgs = [(f, s, NATIVE_FPS / s) for f, s in FPS_STEPS.items()]
    else:  # E1.04: fixed 20-frame budget
        cfgs = [("uniform20", 5, 2.0)]

    feats = {c[0]: [] for c in cfgs}
    feat_ms = {c[0]: [] for c in cfgs}
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
        if a.mode == "select":
            # motion-adaptive: top-20 inter-frame-diff frames (+first frame anchor)
            r = motion_rank(luma)
            keep = np.sort(np.concatenate([[0], (r[:19] + 1).clip(0, T - 1)]))
            keep = np.unique(keep)
            d, fms = compute_for_window(gf, rgb, luma, keep, 2.0)
            feats.setdefault("motion20", []).append([d[n] for n in names])
            feat_ms.setdefault("motion20", []).append(fms)
            # random control seed fixed
            rng = np.random.default_rng(a.seed)
            keep = np.sort(rng.choice(T, 20, replace=False))
            d, fms = compute_for_window(gf, rgb, luma, keep, 2.0)
            feats.setdefault("random20", []).append([d[n] for n in names])
            feat_ms.setdefault("random20", []).append(fms)
            # keyframe-only: I-frame positions from ffprobe, nearest decoded idx
            keep = keyframe_idx(it["path"], it["start_s"], T)
            d, fms = compute_for_window(gf, rgb, luma, keep, 2.0)
            feats.setdefault("keyframe", []).append([d[n] for n in names])
            feat_ms.setdefault("keyframe", []).append(fms)
    out = {"ids": ids, "clips": clips, "y": y, "names": names,
           "feats": {k: np.array(v).tolist() for k, v in feats.items()},
           "decode_ms": decode_ms, "feat_ms": feat_ms, "nframes": nframes,
           "mode": a.mode, "decode_size": [W, H]}
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"shard_{a.mode}_{si}of{sn}.json")
    json.dump(out, open(p, "w"))
    print(f"wrote {p} n={len(ids)}")


def keyframe_idx(path, start_s, T, budget=20):
    import subprocess
    end_s = start_s + 10.0
    # NOTE: -skip_frame nokey + pkt_pts_time prints nothing (ffprobe quirk);
    # list pict_type and filter I.
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "frame=pkt_pts_time,pict_type",
           "-of", "csv=p=0", path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        ts = []
        for line in r.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and parts[1] == "I":
                try:
                    t = float(parts[0])
                except ValueError:
                    continue
                if start_s - 1.0 <= t <= end_s:
                    ts.append(t)
    except Exception:
        ts = []
    # map I-frame timestamps to nearest decoded frame idx (10fps grid)
    idx = sorted(set(int(round((t - start_s) * NATIVE_FPS)) for t in ts))
    idx = [i for i in idx if 0 <= i < T]
    if len(idx) >= budget:
        # uniform-pick budget among I-frames
        idx = [idx[int(k)] for k in np.linspace(0, len(idx) - 1, budget)]
    else:  # pad deterministically with uniform frames
        uni = list(subsample_idx(T, 5))
        idx = sorted(set(idx) | set(uni))[:budget]
    return np.array(idx, dtype=int)


if __name__ == "__main__":
    main()
