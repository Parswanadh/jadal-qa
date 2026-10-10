#!/usr/bin/env python3
"""DSH-PHYS: build the IndEgo gold-150 matrix in the SAME feature name space as
the OOD matrix (full.* + phys.*), using the frozen vendored kernels.

Reads /workspace/jadal_pod_bundle/labels/manifest_gold.json and the IndEgo mp4s.
Output: eval/results/exp_phys_dsh/inputs/indego_full.npz
"""
from __future__ import annotations

import concurrent.futures as cf
import glob
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


def clip_index():
    """basename -> path for every IndEgo mp4 we actually have."""
    idx = {}
    for p in glob.glob(os.path.join(BUNDLE, "data", "indego", "*", "mp4_480", "*.mp4")):
        idx[os.path.basename(p)] = p
    for p in glob.glob("/tmp/eg2videos/*.mp4"):
        idx.setdefault(os.path.basename(p), p)
    return idx


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
    man = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))
    idx = clip_index()
    print(f"indego mp4s available: {len(idx)}", flush=True)
    ids, y, clips, extra, work = [], [], [], [], []
    for it in man["items"]:
        base = os.path.basename(it["path"])
        p = idx.get(base)
        if p is None:
            continue
        ids.append(it["id"])
        y.append(int(it["label"]))
        clips.append(it["clip"])
        extra.append({"clip": it["clip"], "reason": it.get("reason", ""),
                      "start_s": it["start_s"]})
        work.append((it["id"], p, it["label"], float(it["start_s"]),
                     float(it["end_s"]) - float(it["start_s"])))
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
    if names is None:
        raise SystemExit("every IndEgo extraction failed")
    X = np.full((len(ids), len(names)), np.nan, float)
    for i, wid in enumerate(ids):
        f = by[wid]["feats"]
        if f:
            X[i] = [f[k] for k in names]
    nfail = int(np.isnan(X).all(axis=1).sum())
    np.savez_compressed(os.path.join(OUT, "indego_full.npz"),
                        ids=np.array(ids), y=np.asarray(y, int),
                        names=np.array(names), X=X, clips=np.array(clips),
                        meta=np.array([json.dumps(e, sort_keys=True) for e in extra]),
                        errors=np.array([by[i]["error"] or "" for i in ids]))
    print(f"indego: n={len(ids)}/{len(man['items'])} feats={len(names)} failed={nfail} "
          f"in {time.time()-t0:.1f}s -> {OUT}/indego_full.npz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
