"""E1.03: OOD 1080p30-source fps sweep {30,15,10,5,3,2,1}.

Decode ORIGINAL source_path at source_start_s (1080p30 HEVC), 300 frames,
downscaled to 160x90 (fixed analysis res; fps effect only), subsample by
index rule, cpu_window_features with TRUE fps. Eval vs human OOD labels:
 (a) frozen IndEgo-trained logistic (all-IndEgo 10fps features) AUROC per fps
 (b) per-feature direct AUROC vs fps (no transfer needed)
 (c) label-free score rank-correlation across fps (stability)
CI = window bootstrap (note: ~1 window/factory so ≈ factory bootstrap).
n_bad=9 -> wide CI, reported.

PARTIAL (E1.03): shard outputs are NOT committed (local only, see .gitignore).
Regen: PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python \
  eval/exp_fps/run_ood.py --shard {0,1}/2; then eval/exp_fps/merge_ood.py
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fps_lib import WT_ROOT, decode_window, gpu_features, subsample_idx

D = os.path.join(os.path.dirname(__file__), "..", "results", "exp_fps")
Q = os.path.join(WT_ROOT, "scale-data", "eval", "results", "scale", "label_queue_random.json")
LAB = os.path.join(WT_ROOT, "ood_labels", "labels.json")
OOD_STEPS = {30: 1, 15: 2, 10: 3, 5: 6, 3: 10, 2: 15, 1: 30}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    si, sn = [int(x) for x in a.shard.split("/")]
    segs = json.load(open(Q))["segments"]
    lab = json.load(open(LAB))["latest"]
    mine = [s for k, s in enumerate(segs) if k % sn == si]
    gf = gpu_features()
    names = list(gf.FEATURE_NAMES)
    out = {"names": names, "windows": []}
    for s in mine:
        sid = s["segment_id"]
        v = lab.get(sid)
        if v is None:
            continue
        rgb, luma, dms = decode_window(s["source_path"], s["source_start_s"],
                                       n_frames=300)
        T = len(luma)
        rec = {"segment_id": sid, "factory": s.get("factory"),
               "y": 1 if v["verdict"] == "good" else 0,
               "reason": v.get("reason"), "n_decoded": T,
               "decode_ms": round(dms, 1), "fps": {}}
        import time
        for fps, step in OOD_STEPS.items():
            idx = subsample_idx(T, step)
            feff = 30.0 / step
            t0 = time.perf_counter()
            d = gf.cpu_window_features(luma[idx].copy(), rgb[idx].copy(),
                                       fps=float(feff))
            fms = (time.perf_counter() - t0) * 1000.0
            rec["fps"][str(fps)] = {"feats": [d[n] for n in names],
                                    "n_frames": len(idx),
                                    "feat_ms": round(fms, 1)}
        out["windows"].append(rec)
        print(f"{sid} y={rec['y']} T={T} dec={dms:.0f}ms", flush=True)
    os.makedirs(D, exist_ok=True)
    p = os.path.join(D, f"ood_shard_{si}of{sn}.json")
    json.dump(out, open(p, "w"))
    print(f"wrote {p} n={len(out['windows'])}")


if __name__ == "__main__":
    main()
