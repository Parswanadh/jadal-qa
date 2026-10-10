#!/usr/bin/env python3
"""DSH-PHYS: build the CONFIRMATORY hold-out (round 3, 108 windows) feature matrix.

Scored exactly ONCE with the frozen final table. No tuning here.
Source: labels/round3/ood_labels3_108_with_factory.json, clips under
data/ego10k_scale/{clips,clips_more}/<source_clip>, window = [source_start_s, +10s).

Output: eval/results/exp_phys_dsh/inputs/holdout108_full.npz
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
OUT = os.path.join(REPO, "eval", "results", "exp_phys_dsh", "inputs")


def _one(job):
    wid, path, y, start, dur = job
    from tools.phys.extract_features import extract_window_features
    r = {"id": wid, "y": int(y), "error": None, "feats": None}
    try:
        r["feats"] = extract_window_features(path, start, dur)
    except Exception as e:
        r["error"] = f"{type(e).__name__}:{e}"[:200]
    return r


def main() -> int:
    d = json.load(open(os.path.join(
        BUNDLE, "labels", "round3", "ood_labels3_108_with_factory.json")))
    wins = d["windows"]
    roots = [os.path.join(BUNDLE, "data", "ego10k_scale", "clips"),
             os.path.join(BUNDLE, "data", "ego10k_scale", "clips_more")]
    ids, y, extra, work = [], [], [], []
    missing = []
    for w in wins:
        rel = w["source_clip"]
        path = next((os.path.join(r, rel) for r in roots
                     if os.path.exists(os.path.join(r, rel))), None)
        if path is None:
            missing.append(rel)
            continue
        ids.append(w["segment_id"])
        y.append(1 if w["verdict"] == "good" else 0)
        extra.append({"factory": w.get("factory", ""), "reason": w.get("reason", ""),
                      "source_clip": rel, "source_start_s": w["source_start_s"]})
        work.append((w["segment_id"], path, y[-1], float(w["source_start_s"]), 10.0))
    print(f"holdout: found {len(work)}/{len(wins)} videos, missing {len(set(missing))}",
          flush=True)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=5) as pool:
        out = list(pool.map(_one, work))
    by = {r["id"]: r for r in out}
    names = None
    for r in out:
        if r["feats"]:
            names = sorted(r["feats"])
            break
    X = np.full((len(ids), len(names)), np.nan, float)
    for i, wid in enumerate(ids):
        f = by[wid]["feats"]
        if f:
            X[i] = [f[k] for k in names]
    nfail = int(np.isnan(X).all(axis=1).sum())
    np.savez_compressed(os.path.join(OUT, "holdout108_full.npz"),
                        ids=np.array(ids), y=np.asarray(y, int),
                        names=np.array(names), X=X,
                        meta=np.array([json.dumps(e, sort_keys=True) for e in extra]),
                        errors=np.array([by[i]["error"] or "" for i in ids]))
    print(f"holdout108: n={len(ids)} feats={len(names)} failed={nfail} "
          f"in {time.time()-t0:.1f}s -> {OUT}/holdout108_full.npz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
