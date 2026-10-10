#!/usr/bin/env python3
"""O4 review point: Z3 candidate features on a sample of the 1,546 E8 windows.

The 1,546 Egocentric-10K windows (E8 rescored corpus) have NO human labels and
their cached 26 pixel features but NOT the Z3 candidate features. Decode a
deterministic random sample with the SAME Z3 extractor recipe.

Output (git-ignored): eval/results/exp_adapt160/inputs/e8_sample.npz
  {ids, factory, clip, start_s, end_s, names (20,), X (N,20), X26 (N,26)}

Run: OMP_NUM_THREADS=2 python3 eval/exp_adapt160/extract_e8_sample.py [--n 100]
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
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import z3extract as E  # noqa: E402
from run_adapt160 import FEATURE_NAMES26  # noqa: E402

# Pod paths from the environment only (no tracked defaults, review finding 6).
BUNDLE = os.environ.get("JADAL_BUNDLE")
E8_JSONL = os.environ.get("JADAL_E8_JSONL")
OUT = os.path.join(REPO, "eval", "results", "exp_adapt160", "inputs")


def _require_env(name, value):
    if not value:
        raise SystemExit(
            f"{name} is not set and no pod path is hard-coded; export it, e.g. "
            f"export {name}=/path/on/pod")
    return value


def _clips_dir():
    return os.path.join(_require_env("JADAL_BUNDLE", BUNDLE), "data", "ego10k_scale", "clips")


def load_e8():
    recs = []
    for line in open(_require_env("JADAL_E8_JSONL", E8_JSONL)):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "header":
            continue
        recs.append(d)
    return recs


def decode_job(job):
    wid, path, s0, s1 = job
    res = {"id": wid, "error": None}
    try:
        d = E.decode_features(path, s0, s1)
        if d is None:
            raise RuntimeError("too few frames")
        res.update(d)
    except Exception as e:
        res["error"] = f"decode:{e}"[:200]
    for k in E.CAND:
        res.setdefault(k, np.nan)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()

    recs = load_e8()
    recs = sorted(recs, key=lambda r: r["window_id"])
    rng = np.random.RandomState(args.seed)
    idx = np.sort(rng.choice(len(recs), size=min(args.n, len(recs)), replace=False))
    sample = [recs[i] for i in idx]
    print(f"E8 total {len(recs)}; sampling n={len(sample)} (seed={args.seed})", flush=True)

    X26names = list(FEATURE_NAMES26)
    X26 = np.full((len(sample), len(X26names)), np.nan)
    X = np.full((len(sample), len(E.CAND)), np.nan)
    ids = [r["window_id"] for r in sample]
    fac = [r["factory"] for r in sample]

    # cached 26 pixel features (already available, no decode)
    for k, r in enumerate(sample):
        f = r.get("features", {})
        for j, n in enumerate(X26names):
            if n in f:
                X26[k, j] = float(f[n])

    # decode Z3 candidates
    clips_dir = _clips_dir()
    uniq = {os.path.join(clips_dir, r["clip"]): (r["start_s"], r["end_s"]) for r in sample}
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        tables = dict(zip(uniq.keys(), ex.map(E.probe_full, uniq.keys())))
    print(f"probe {time.time()-t0:.1f}s", flush=True)

    jobs = [(r["window_id"], os.path.join(clips_dir, r["clip"]), r["start_s"], r["end_s"])
            for r in sample]
    with mp.Pool(args.jobs) as pool:
        rows = pool.map(decode_job, jobs, chunksize=2)
    sid2k = {wid: k for k, wid in enumerate(ids)}
    errs = 0
    for r in rows:
        k = sid2k[r["id"]]
        path = os.path.join(clips_dir, sample[k]["clip"])
        ct = E.compression_from_table(tables.get(path), sample[k]["start_s"], sample[k]["end_s"])
        if ct is not None:
            r.update(ct)
        if r.get("error"):
            errs += 1
        X[k] = [float(r.get(n, np.nan)) for n in E.CAND]
    print(f"decoded ok {len(sample)-errs}/{len(sample)}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "e8_sample.npz")
    np.savez_compressed(p, ids=np.asarray(ids), factory=np.asarray(fac),
                        clip=np.asarray([r["clip"] for r in sample]),
                        names=np.asarray(E.CAND), X=X,
                        x26_names=np.asarray(X26names), X26=X26,
                        sample_seed=args.seed, n_total=len(recs))
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
