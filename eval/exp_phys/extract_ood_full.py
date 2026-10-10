#!/usr/bin/env python3
"""Extract full.* + phys.* features for the 160 OOD windows (CPU, cached).

Uses the SAME kernels as the packaged model (tools/phys/extract_features.py),
i.e. the kernels that produced Q1's gold features. Output is git-ignored.

Run: OMP_NUM_THREADS=2 python3 eval/exp_phys/extract_ood_full.py --jobs 2
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
BUNDLE = os.environ.get("JADAL_BUNDLE")
if not BUNDLE:
    raise SystemExit("JADAL_BUNDLE is required (path to the pod data bundle)")
OUT = os.path.join(ROOT, "eval", "results", "exp_phys", "inputs")


def _one(job):
    wid, path, y, factory, batch = job
    from tools.phys.extract_features import extract_window_features
    try:
        feats = extract_window_features(path, 0.0, 10.0)
        return {"id": wid, "y": int(y), "factory": factory, "batch": batch,
                "feats": feats, "error": ""}
    except Exception as e:  # noqa: BLE001
        return {"id": wid, "y": int(y), "factory": factory, "batch": batch,
                "feats": {}, "error": f"{e}"[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))
    w = d["windows"]
    jobs = [(x["segment_id"], os.path.join(BUNDLE, x["video"]),
             1 if x["verdict"] == "good" else 0, x["factory"], x["batch"]) for x in w]
    t0 = time.time()
    with mp.Pool(args.jobs) as pool:
        out = pool.map(_one, jobs, chunksize=2)
    names = sorted({k for r in out for k in r["feats"]})
    ids = [r["id"] for r in out]
    X = np.array([[r["feats"].get(n, np.nan) for n in names] for r in out], float)
    np.savez_compressed(
        os.path.join(OUT, "ood_full.npz"), ids=np.array(ids),
        y=np.array([r["y"] for r in out]),
        factory=np.array([r["factory"] for r in out]),
        batch=np.array([r["batch"] for r in out]),
        names=np.array(names), X=X,
        errors=np.array([r["error"] for r in out]))
    nerr = sum(1 for r in out if r["error"])
    print(f"n={len(out)} features={len(names)} errors={nerr} in {time.time()-t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
