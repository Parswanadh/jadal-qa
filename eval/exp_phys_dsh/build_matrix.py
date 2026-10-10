#!/usr/bin/env python3
"""DSH-PHYS: build the unified full.*+phys.* feature matrix for a labelled window set.

Uses the FROZEN vendored kernels in tools/phys/vendor (the exact code that produced
the Q1/Q4 rule-table thresholds), so every matrix built here is comparable to the
frozen rule table.

  --set ood160   : the 160 human-labelled Egocentric-10K windows (label_windows*/x.mp4)
  --set indego   : the 150 IndEgo gold windows (manifest_gold.json)

Output: eval/results/exp_phys_dsh/inputs/<set>_full.npz
  keys: ids, y (1=good), names, X, meta(json strings: clip/factory/batch)
"""
from __future__ import annotations

import argparse
import json
import concurrent.futures as cf
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
OUT = os.path.join(REPO, "eval", "results", "exp_phys_dsh", "inputs")


def load_ood160():
    d = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))
    ids, y, clips, extra, videos = [], [], [], [], []
    for x in d["windows"]:
        ids.append(x["segment_id"])
        y.append(1 if x["verdict"] == "good" else 0)
        clips.append(x["video"])
        extra.append({"factory": x["factory"], "batch": x["batch"],
                      "reason": x.get("reason", "")})
        videos.append((os.path.join(BUNDLE, x["video"]), 0.0, 10.0))
    return ids, np.asarray(y, int), clips, extra, videos


def _one(job):
    wid, path, y, start, dur = job
    from tools.phys.extract_features import extract_window_features
    res = {"id": wid, "y": int(y), "error": None, "feats": None}
    try:
        res["feats"] = extract_window_features(path, start, dur)
    except Exception as e:  # keep going, record the failure
        res["error"] = f"{type(e).__name__}:{e}"[:200]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["ood160", "indego"])
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.set == "ood160":
        ids, y, clips, extra, jobs = load_ood160()
    else:
        raise SystemExit("indego set: build via build_indego.py")

    if args.limit:
        ids, y, clips, extra, jobs = (ids[:args.limit], y[:args.limit],
                                     clips[:args.limit], extra[:args.limit],
                                     jobs[:args.limit])
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    work = [(ids[i], jobs[i][0], y[i], jobs[i][1], jobs[i][2]) for i in range(len(ids))]
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        out = list(pool.map(_one, work))
    by = {r["id"]: r for r in out}
    from tools.phys.extract_features import extract_window_features  # noqa: F401
    names = None
    for r in out:
        if r["feats"]:
            names = sorted(r["feats"])
            break
    if names is None:
        raise SystemExit("every extraction failed")
    X = np.full((len(ids), len(names)), np.nan, float)
    for i, wid in enumerate(ids):
        f = by[wid]["feats"]
        if f:
            X[i] = [f[k] for k in names]
    nerr = int(np.isnan(X).all(axis=1).sum())
    np.savez_compressed(
        os.path.join(OUT, f"{args.set}_full.npz"),
        ids=np.array(ids), y=y, names=np.array(names), X=X,
        clips=np.array(clips),
        meta=np.array([json.dumps(e, sort_keys=True) for e in extra]),
        errors=np.array([by[i]["error"] or "" for i in ids]))
    print(f"{args.set}: n={len(ids)} feats={len(names)} failed={nerr} "
          f"in {time.time()-t0:.1f}s -> {OUT}/{args.set}_full.npz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
