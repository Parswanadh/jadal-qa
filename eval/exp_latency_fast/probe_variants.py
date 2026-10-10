#!/usr/bin/env python3
"""L3.01 diagnostic: which adopted optimization shifts the frozen-head score?

The frozen heads (configs/ensemble_v1.json) were fit on full-sampling features.
Applying sparser sampling to the *frozen* heads can shift the score. This probe
runs the optimized pipeline on a spread of windows under one optimization at a
time and records the score/verdict shift vs the frozen-sampling path.

Run (GPU slot held):
  bash tools/pod/gpu_run.sh python3 eval/exp_latency_fast/probe_variants.py --n 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval", "exp_latency_fast"))
sys.path.insert(0, os.path.join(REPO, "eval", "exp_rescore"))

import run_fast  # noqa: E402
import run_e8 as e8  # noqa: E402

# mosaic is evidence-only (E2: dropped as a model input), so it never changes
# the score; skip it in every variant to isolate the sampling effect.
BASE = {"pixel_fps": 0.0, "eg2_fps": 2.0, "native_fps": 0.8,
        "early_exit_s": 0.0, "skip_mosaic": True}
VARIANTS = {
    "baseline": dict(BASE),
    "pixel_3.33fps": {**BASE, "pixel_fps": 3.33},
    "eg2_1fps": {**BASE, "eg2_fps": 1.0},
    "early_exit_4s": {**BASE, "early_exit_s": 4.0},
    "all_adopted": {"pixel_fps": 3.33, "eg2_fps": 1.0, "native_fps": 0.8,
                    "early_exit_s": 4.0, "skip_mosaic": True},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(
        REPO, "eval", "results", "exp_latency_fast", "probe_variants.json"))
    args = ap.parse_args()

    os.environ.setdefault("JADAL_PIXEL_FPS", "0")
    os.environ.setdefault("JADAL_EG2_FPS", "2")
    run_fast.init_worker()
    _, items = e8.load_manifest()
    step = max(1, len(items) // args.n)
    sample = items[::step][:args.n]
    pmap = e8.profiles()

    out = {"n": len(sample), "variants": {}}

    def dump():
        json.dump(out, open(args.out, "w"), indent=1, sort_keys=True)

    for name, opts in VARIANTS.items():
        run_fast._OPT = opts
        scores, verdicts, probs = [], [], []
        for it in sample:
            try:
                r = run_fast.score_window_timed(it, pmap, do_compare=False)
                scores.append(r["score"])
                verdicts.append(r["verdict"])
                probs.append((r["contract"]["features"]["p_eg2"],
                              r["contract"]["features"]["p_pixel"]))
            except Exception as ex:  # noqa: BLE001
                print("ERR", name, it["window_id"], str(ex)[:80], flush=True)
        if not scores:
            print("SKIP", name, "(all failed)", flush=True)
            continue
        scores = np.array(scores)
        out["variants"][name] = {
            "opts": opts,
            "n": len(scores),
            "score_mean": round(float(scores.mean()), 4),
            "score_p50": round(float(np.percentile(scores, 50)), 4),
            "verdict_distribution": dict(sorted(Counter(verdicts).items())),
            "mean_p_eg2": round(float(np.mean([p[0] for p in probs])), 4),
            "mean_p_pixel": round(float(np.mean([p[1] for p in probs])), 4),
        }
        dump()  # write incrementally so a later OOM does not lose the result
        print(name, out["variants"][name]["score_mean"],
              out["variants"][name]["verdict_distribution"], flush=True)

    base = out["variants"].get("baseline")
    if base:
        for name, v in out["variants"].items():
            v["score_mean_delta_vs_baseline"] = round(
                v["score_mean"] - base["score_mean"], 4)
            v["p_eg2_delta"] = round(v["mean_p_eg2"] - base["mean_p_eg2"], 4)
            v["p_pixel_delta"] = round(v["mean_p_pixel"] - base["mean_p_pixel"], 4)
    dump()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
