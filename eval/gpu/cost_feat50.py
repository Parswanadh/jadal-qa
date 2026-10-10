#!/usr/bin/env python3
"""F50 — per-window cost of the added 24 features (ms/window).

Measures, on one real 10-s window (160x90, 100 frames) and on synthetic frames:
  decode (ffmpeg) | frozen-26 kernels | new-24 kernels | ffprobe (ct_fsize_std)

Run: OMP_NUM_THREADS=2 python3 eval/gpu/cost_feat50.py
Writes eval/results/exp_feat50/cost.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from eval.gpu.decode import decode_window  # noqa: E402
from jadal.gpu.features import cpu_window_features  # noqa: E402
from jadal.gpu.features50 import cpu_new_features  # noqa: E402

OUT = os.path.join(REPO, "eval", "results", "exp_feat50")
CLIP = os.environ.get("JADAL_COST_CLIP",
                      "/workspace/jadal_pod_bundle/data/indego/1_Assembly/mp4_480/"
                      "User_15_410_0411_2_480.mp4")
N = 15


def _time(fn, n=N):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return 1000.0 * (time.perf_counter() - t) / n


def main():
    rng = np.random.default_rng(0)
    luma = rng.random((100, 90, 160))
    rgb = rng.random((100, 90, 160, 3))
    synth = {
        "frozen26_kernel_ms": _time(lambda: cpu_window_features(luma, rgb, fps=10.0)),
        "new24_kernel_ms": _time(lambda: cpu_new_features(luma, rgb, fps=10.0, fsize_std=0.1)),
    }
    real = {}
    try:
        real["decode_ms"] = _time(
            lambda: decode_window(CLIP, 0.0, 10.0, 160, 90, "rgb24"))
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "packet=size", "-of", "csv=p=0", CLIP],
                           capture_output=True, text=True, timeout=120)
        real["ffprobe_ms"] = _time(lambda: subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=size", "-of", "csv=p=0", CLIP],
            capture_output=True, text=True, timeout=120))
    except Exception as e:  # noqa: BLE001
        real["error"] = str(e)[:200]
    res = {
        "experiment": "F50.05",
        "title": "Per-window cost of the added 24 per-pixel features",
        "device": "CPU (single core; ffmpeg decode on CPU)",
        "n_repeats": N, "frames": 100, "resolution": [160, 90],
        "synthetic_frames": synth, "real_window": real,
        "total_new24_end_to_end_ms": round(synth["new24_kernel_ms"] +
                                           real.get("ffprobe_ms", 0.0), 2),
        "frozen26_kernel_ms": synth["frozen26_kernel_ms"],
        "note": ("decode is shared with the frozen 26 pipeline; the added cost is the "
                 "new-24 kernels + one ffprobe for ct_fsize_std. Measured on CPU, not GPU."),
        "command": "OMP_NUM_THREADS=2 python3 eval/gpu/cost_feat50.py",
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "cost.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
