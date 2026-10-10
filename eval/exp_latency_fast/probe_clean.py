#!/usr/bin/env python3
"""L3.01 clean probe: uncontended per-window per-stage timing (1 worker).

The full-resources run uses 7 workers on one A40; the NVDEC/EG2 GPU service
serialises, so per-window decode wall grows with the worker count while the
aggregate throughput rises. This probe runs the *same optimized pipeline* on a
spread of windows with **one** worker (no GPU queue) to recover the uncontended
per-window per-stage cost, and separately times 4 s vs 10 s decode at 1 worker.

Run (GPU slot held):
  source /workspace/env.sh
  export OMP_NUM_THREADS=1 JADAL_GPU_SLOTS=6
  bash tools/pod/gpu_run.sh python3 eval/exp_latency_fast/probe_clean.py \
       --pixel-fps 3.33 --eg2-fps 1 --early-exit-s 4 --skip-mosaic --n 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval", "exp_latency_fast"))
sys.path.insert(0, os.path.join(REPO, "eval", "exp_rescore"))

import run_fast  # noqa: E402
import run_e8 as e8  # noqa: E402

STAGES = ["io_probe", "decode", "pixel_features", "descriptors", "eg2_embed",
          "mosaic", "decision", "write"]


def pctl(a):
    a = np.asarray(a, float)
    return {"p50": round(float(np.percentile(a, 50)), 1),
            "p95": round(float(np.percentile(a, 95)), 1),
            "mean": round(float(a.mean()), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--pixel-fps", type=float, default=3.33)
    ap.add_argument("--eg2-fps", type=float, default=1.0)
    ap.add_argument("--early-exit-s", type=float, default=4.0)
    ap.add_argument("--native-fps", type=float, default=0.8)
    ap.add_argument("--skip-mosaic", action="store_true")
    ap.add_argument("--out", default=os.path.join(
        REPO, "eval", "results", "exp_latency_fast", "probe_clean.json"))
    args = ap.parse_args()

    os.environ["JADAL_PIXEL_FPS"] = str(args.pixel_fps)
    os.environ["JADAL_EG2_FPS"] = str(args.eg2_fps)
    os.environ["JADAL_EARLY_EXIT_S"] = str(args.early_exit_s)
    os.environ["JADAL_NATIVE_FPS"] = str(args.native_fps)
    if args.skip_mosaic:
        os.environ["JADAL_SKIP_MOSAIC"] = "1"

    run_fast.init_worker()
    _, items = e8.load_manifest()
    step = max(1, len(items) // args.n)
    sample = items[::step][:args.n]
    pmap = e8.profiles()

    # warm the EG2 model once
    rows = []
    t0 = time.perf_counter()
    for i, it in enumerate(sample, 1):
        try:
            rows.append(run_fast.score_window_timed(it, pmap, do_compare=False))
        except Exception as ex:  # noqa: BLE001
            print("ERR", it["window_id"], ex, flush=True)
        if i % 10 == 0:
            print(f"  {i}/{len(sample)} {time.perf_counter()-t0:.0f}s", flush=True)

    # decode 4 s vs 10 s at 1 worker (no queue)
    dec = []
    for it in sample[:8]:
        path, s0 = it["path"], float(it["start_s"])
        t = time.perf_counter()
        run_fast.decode_three_branch_cfg(path, s0, 4.0, args.pixel_fps,
                                         args.eg2_fps, args.native_fps)
        d4 = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        run_fast.decode_three_branch_cfg(path, s0, 10.0, args.pixel_fps,
                                         args.eg2_fps, args.native_fps)
        d10 = (time.perf_counter() - t) * 1000
        dec.append({"window_id": it["window_id"], "decode4_ms": round(d4, 1),
                    "decode10_ms": round(d10, 1)})

    stage = {s: pctl([r["latency_ms"].get(s, 0.0) for r in rows]) for s in STAGES}
    tot = [sum(r["latency_ms"].get(s, 0.0) for s in STAGES) for r in rows]
    res = {
        "what": "uncontended (1 worker) per-window per-stage timing of the "
                "optimized pipeline on a spread of windows",
        "n": len(rows),
        "workers": 1,
        "optimizations": run_fast.read_opts(),
        "per_stage_ms": stage,
        "total_ms": pctl(tot),
        "frames_per_window": {
            "pixel": pctl([r["n_frames"]["pixel"] for r in rows]),
            "eg2": pctl([r["n_frames"]["eg2"] for r in rows]),
            "native": pctl([r["n_frames"]["native"] for r in rows]),
        },
        "decode_4s_vs_10s_1worker": {
            "decode4_p50_ms": round(float(np.median([d["decode4_ms"] for d in dec])), 1),
            "decode10_p50_ms": round(float(np.median([d["decode10_ms"] for d in dec])), 1),
            "rows": dec,
        },
    }
    json.dump(res, open(args.out, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("decode_4s_vs_10s_1worker",)}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
