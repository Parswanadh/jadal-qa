#!/usr/bin/env python3
"""O4: Z3 candidate features on the 160 OOD windows (q1 reuse + q2 compute).

Reuses the Z3 extractor verbatim (`z3extract.py`, copied from
`m/new-features:eval/exp_features/extract.py`). Batch q1 (60 windows) is read
from the frozen Z3 `candidates.npz`; batch q2 (100 windows) is decoded with the
SAME recipe (ffprobe frame table + 160x90 decode + per-pair phase-correlation /
Lucas-Kanade + per-frame invariants).

CPU only, 2 worker processes, OMP_NUM_THREADS=2.

Output (git-ignored, *.npz):
  eval/results/exp_adapt160/inputs/z3_160.npz
    {ids (160,), y (160,), batch (160,), factory (160,), names (20,),
     X (160,20), n_q1_reused, n_q2_computed, X_indego (150,20), y_indego (150,),
     clips_indego (150,)}

Run: OMP_NUM_THREADS=2 python3 eval/exp_adapt160/extract160.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import z3extract as E  # noqa: E402

# Pod paths come from the environment only (no tracked defaults, review finding 6):
# JADAL_BUNDLE = pod bundle; JADAL_Z3_CAND = frozen Z3 batch-q1 candidates.npz.
BUNDLE = os.environ.get("JADAL_BUNDLE")
Z3_CAND = os.environ.get("JADAL_Z3_CAND")
OUT = os.path.join(REPO, "eval", "results", "exp_adapt160", "inputs")
CAND = E.CAND


def _require_env(name, value):
    if not value:
        raise SystemExit(
            f"{name} is not set and no pod path is hard-coded; export it, e.g. "
            f"export {name}=/path/on/pod")
    return value


def load_labels160():
    d = json.load(open(os.path.join(_require_env("JADAL_BUNDLE", BUNDLE),
                                    "labels", "ood160_labels.json")))
    w = d["windows"]
    return {
        "ids": [x["segment_id"] for x in w],
        "y": np.asarray([1 if x["verdict"] == "good" else 0 for x in w], int),
        "batch": np.asarray([x["batch"] for x in w]),
        "factory": np.asarray([x["factory"] for x in w]),
        "video": [x["video"] for x in w],
    }


def decode_job(job):
    kind, wid, path, start_s, end_s, label = job
    res = {"kind": kind, "id": wid, "label": int(label), "error": None}
    try:
        d = E.decode_features(path, start_s, end_s)
        if d is None:
            raise RuntimeError("too few frames")
        res.update(d)
    except Exception as e:  # keep going
        res["error"] = f"decode:{e}"[:200]
    for k in CAND:
        res.setdefault(k, np.nan)
    return res


def main():
    lab = load_labels160()
    order = {sid: i for i, sid in enumerate(lab["ids"])}

    # ---- batch q1: reuse frozen Z3 candidates ----
    z = np.load(_require_env("JADAL_Z3_CAND", Z3_CAND), allow_pickle=True)
    q1names = [str(x) for x in z["names"]]
    assert q1names == CAND, (q1names, CAND)
    q1ids = {str(i): k for k, i in enumerate(z["ood_ids"])}
    X = np.full((len(lab["ids"]), len(CAND)), np.nan)
    n1 = 0
    for sid in lab["ids"]:
        if lab["batch"][order[sid]] == "q1" and sid in q1ids:
            X[order[sid]] = z["ood_X"][q1ids[sid]]
            n1 += 1
    print(f"batch-q1 reused from {Z3_CAND}: {n1}/60", flush=True)
    assert n1 == 60, f"expected 60 q1 windows, got {n1}"

    # ---- batch q2: compute with the same recipe ----
    bundle = _require_env("JADAL_BUNDLE", BUNDLE)
    q2 = json.load(open(os.path.join(bundle, "queues", "label_queue_random2.json")))["segments"]
    segs = [s for s in q2 if s["segment_id"] in order]
    print(f"batch-q2 decoding {len(segs)} windows (probe + 160x90 decode + optical flow)", flush=True)

    # compression telemetry: probe unique files once
    uniq = {s["path"]: (s["start_s"], s["end_s"]) for s in segs}
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        tables = dict(zip(uniq.keys(), ex.map(E.probe_full, uniq.keys())))
    print(f"probe done {time.time()-t0:.1f}s", flush=True)

    seg_by_id = {s["segment_id"]: s for s in segs}
    jobs = [("ood", s["segment_id"], s["path"], s["start_s"], s["end_s"], 0) for s in segs]
    with mp.Pool(2) as pool:
        rows = pool.map(decode_job, jobs, chunksize=2)
    n2, errs = 0, []
    for r in rows:
        sid = r["id"]
        s = seg_by_id[sid]
        path, s0, s1 = s["path"], s["start_s"], s["end_s"]
        ct = E.compression_from_table(tables.get(path), s0, s1)
        if ct is None:
            for k in E.COMPRESSION:
                r[k] = np.nan
        else:
            r.update(ct)
        if r.get("error"):
            errs.append((sid, r["error"]))
        else:
            n2 += 1
        X[order[sid]] = [float(r.get(n, np.nan)) for n in CAND]
    print(f"batch-q2 decoded ok: {n2}/{len(segs)}; errors={errs[:5]}", flush=True)

    # ---- IndEgo 150 (from the same frozen candidates) ----
    Xi = z["indego_X"].astype(float)
    yi = z["indego_y"].astype(int)
    gold = json.load(open(os.path.join(bundle, "labels", "manifest_gold.json")))["items"]
    gorder = {it["id"]: it for it in gold}
    clips = [gorder[str(i)]["clip"] for i in z["indego_ids"]]
    assert Xi.shape[0] == 150, Xi.shape

    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "z3_160.npz")
    np.savez_compressed(
        out, ids=np.asarray(lab["ids"]), y=lab["y"], batch=lab["batch"],
        factory=lab["factory"], names=np.asarray(CAND), X=X,
        X_indego=Xi, y_indego=yi, clips_indego=np.asarray(clips))
    ok = ~np.isnan(X).any(axis=1)
    print(f"wrote {out}: {int(ok.sum())}/160 complete, good={int(lab['y'][ok].sum())} "
          f"bad={int((1-lab['y'][ok]).sum())}", flush=True)
    if not ok.all():
        print("incomplete ids:", list(np.asarray(lab["ids"])[~ok]), flush=True)


if __name__ == "__main__":
    main()
