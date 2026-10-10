#!/usr/bin/env python3
"""F30 cost: ms/window of the 4 added invariants vs the 26 E5 pixel features.

Decodes a fixed sample of OOD windows at 160x90, subsamples to an effective
10 fps (matching the cached feature view), then times (a) decode, (b)
``cpu_window_features`` (26), (c) ``window_invariants`` (4) and (d)
``cpu_window_features30`` (30). One warm-up window is excluded. CPU only,
single process, fixed order.

Run: OMP_NUM_THREADS=2 python3 eval/exp_feat30/measure_cost.py [n_windows]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
OUT = os.path.join(REPO, "eval", "results", "exp_feat30")
BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")

from jadal.ensemble.scorer import decode_pixel_window  # noqa: E402
from jadal.gpu.features import cpu_window_features  # noqa: E402
from jadal.gpu.features30 import window_invariants, cpu_window_features30  # noqa: E402


def _t(fn, *a, **k):
    t0 = time.perf_counter()
    fn(*a, **k)
    return (time.perf_counter() - t0) * 1000.0


def main(n=16):
    os.makedirs(OUT, exist_ok=True)
    w = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))["windows"][:n + 1]
    dec_ms, f26_ms, f4_ms, f30_ms, nfr = [], [], [], [], []
    for i, x in enumerate(w):
        path = os.path.join(BUNDLE, x["video"])
        t0 = time.perf_counter()
        luma, rgb, fps = decode_pixel_window(path, 0.0, 10.0, 160, 90)
        dt = (time.perf_counter() - t0) * 1000.0
        # subsample to effective 10 fps (matches the cached feature view)
        step = max(1, int(round(fps / 10.0)))
        luma, rgb = luma[::step], rgb[::step]
        if i == 0:
            cpu_window_features30(luma, rgb, fps=min(fps, 30.0))  # warm-up
            continue
        dec_ms.append(dt)
        nfr.append(int(rgb.shape[0]))
        f26_ms.append(_t(cpu_window_features, luma, rgb, fps=min(fps, 30.0)))
        f4_ms.append(_t(window_invariants, rgb))
        f30_ms.append(_t(cpu_window_features30, luma, rgb, fps=min(fps, 30.0)))

    def s(a):
        a = np.asarray(a, float)
        return {"mean": round(float(a.mean()), 2), "median": round(float(np.median(a)), 2),
                "p90": round(float(np.percentile(a, 90)), 2)}
    res = {
        "experiment": "F30.02", "n_windows": n,
        "frames_per_window_mean": round(float(np.mean(nfr)), 1),
        "view": "ffmpeg cpu rgb24 160x90, 10 s window, subsampled to ~10 fps",
        "device": "CPU (single process, OMP_NUM_THREADS=2)",
        "ms_per_window": {
            "decode": s(dec_ms),
            "features_26": s(f26_ms),
            "added_4_invariants": s(f4_ms),
            "features_30_total": s(f30_ms),
        },
        "marginal_4_median_ms": round(float(np.median(f4_ms)), 2),
        "marginal_4_mean_ms": round(float(np.mean(f4_ms)), 2),
        "marginal_4_frac_of_26": round(float(np.median(f4_ms) / np.median(f26_ms)), 4),
        "command": "OMP_NUM_THREADS=2 python3 eval/exp_feat30/measure_cost.py",
        "note": ("timing only (no GPU); per-window = one 10-s 160x90 ~10 fps window. "
                 "The 4 invariants are O(HW) per frame plus one 64x64 DCT."),
    }
    json.dump(res, open(os.path.join(OUT, "F30.02_cost.json"), "w"), indent=1)
    print(json.dumps(res["ms_per_window"], indent=1))
    print("marginal 4 median:", res["marginal_4_median_ms"], "ms/window",
          f"({res['marginal_4_frac_of_26']*100:.1f}% of the 26-feature cost)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
