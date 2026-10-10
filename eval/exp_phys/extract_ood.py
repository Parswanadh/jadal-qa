#!/usr/bin/env python3
"""Q2: per-window physics + theory features on the 160 OOD human-labelled windows.

Reuses the VERIFIED extractors `jadal/perpixel/{series,physics,theory}.py`
(vendored read-only from b/verify-rules / b/verify-lb2-main, the same code that
produced the frozen IndEgo rule table). The 160x90 decode recipe is identical to
`jadal.perpixel.series.decode_160x90`; features are NOT refit or tuned.

Evaluation only: no fitting on OOD. Fixed thresholds come from IndEgo training.

Output (git-ignored, *.npz):
  eval/results/exp_phys/inputs/ood_phys.npz
    {ids (160,), y (160,), factory (160,), batch (160,),
     phys_names (16,), Xphys (160,16),
     th_names (18,),  Xth (160,18),
     indego_phys_names (16,), Xphys_indego (150,16),
     indego_th_names (18,),  Xth_indego (150,18),
     clips_indego (150,)}

Run: OMP_NUM_THREADS=2 python3 eval/exp_phys/extract_ood.py [--jobs 2]
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

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
OUT = os.path.join(REPO, "eval", "results", "exp_phys", "inputs")
# frozen IndEgo per-window features (150 gold windows) produced by the same code
INDEGO_PHYS = os.environ.get(
    "JADAL_INDEGO_PHYS",
    "/workspace/wt/b/verify-lb2-main/eval/results/perpixel/physics_feats.npz")
INDEGO_TH = os.environ.get(
    "JADAL_INDEGO_TH",
    "/workspace/wt/b/verify-lb2-main/eval/results/perpixel/theory_feats.npz")
INDEGO_MANIFEST = os.environ.get(
    "JADAL_INDEGO_MANIFEST",
    "/workspace/jadal_pod_bundle/labels/manifest_gold.json")


def load_labels160():
    d = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))
    w = d["windows"]
    return {
        "ids": [x["segment_id"] for x in w],
        "y": np.asarray([1 if x["verdict"] == "good" else 0 for x in w], int),
        "batch": np.asarray([x["batch"] for x in w]),
        "factory": np.asarray([x["factory"] for x in w]),
        "video": [os.path.join(BUNDLE, x["video"]) for x in w],
    }


def _one(job):
    wid, path, y, start_s, end_s = job
    from jadal.perpixel import physics as P
    from jadal.perpixel import series as S
    from jadal.perpixel import theory as T
    res = {"id": wid, "y": int(y), "error": None}
    try:
        frames, fps = S.decode_160x90(path, start_s, end_s)
        ph = P.extract_physics(frames, fps, path, start_s, end_s)
        th = T.extract_theory(frames, fps)
        res["phys"] = [float(ph[k]) for k in P.PHYSICS_FEATURES]
        res["th"] = [float(th[k]) for k in T.THEORY_FEATURES]
        res["nframes"] = int(frames.shape[0])
        res["fps"] = float(fps)
    except Exception as e:  # keep going
        res["error"] = f"extract:{e}"[:200]
        res["phys"] = [np.nan] * 16
        res["th"] = [np.nan] * 18
    return res


def load_indego():
    from jadal.perpixel import physics as P
    from jadal.perpixel import theory as T
    zp = np.load(INDEGO_PHYS, allow_pickle=True)
    pids = [str(x) for x in zp["ids"]]
    pmap = {i: np.asarray(zp["X"][j], float) for j, i in enumerate(pids)}
    zt = np.load(INDEGO_TH, allow_pickle=True)
    tmeta = {json.loads(m)["id"]: m for m in zt["meta"]}
    gm = json.load(open(INDEGO_MANIFEST))
    items = {it["id"]: it for it in gm["items"]}
    order = sorted(set(pmap) & set(tmeta) & set(items))
    Xp = np.array([pmap[i] for i in order])
    Xt = np.array([[json.loads(tmeta[i])["float"][k]
                    for k in T.THEORY_FEATURES] for i in order])
    y = np.array([0 if items[i]["label"] == 1 else 1 for i in order], int)
    clips = np.array([items[i]["clip"] for i in order])
    return order, Xp, Xt, y, clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lab = load_labels160()
    dur = 10.0
    jobs = [(lab["ids"][i], lab["video"][i], lab["y"][i], 0.0, dur)
            for i in range(len(lab["ids"]))]
    t0 = time.time()
    with mp.Pool(args.jobs) as pool:
        out = pool.map(_one, jobs, chunksize=4)
    order = {r["id"]: r for r in out}
    ids = lab["ids"]
    Xphys = np.array([order[i]["phys"] for i in ids], float)
    Xth = np.array([order[i]["th"] for i in ids], float)
    nerr = sum(1 for r in out if r["error"])
    from jadal.perpixel import physics as P
    from jadal.perpixel import theory as T
    iorder, Xp_i, Xt_i, y_i, clips_i = load_indego()
    np.savez_compressed(
        os.path.join(OUT, "ood_phys.npz"),
        ids=np.array(ids), y=lab["y"], factory=lab["factory"],
        batch=lab["batch"], phys_names=np.array(P.PHYSICS_FEATURES),
        Xphys=Xphys, th_names=np.array(T.THEORY_FEATURES), Xth=Xth,
        indego_phys_names=np.array(P.PHYSICS_FEATURES), Xphys_indego=Xp_i,
        indego_th_names=np.array(T.THEORY_FEATURES), Xth_indego=Xt_i,
        y_indego=y_i, clips_indego=clips_i,
        nframes=np.array([order[i].get("nframes", 0) for i in ids]),
        fps=np.array([order[i].get("fps", np.nan) for i in ids]),
        errors=np.array([order[i]["error"] or "" for i in ids]))
    print(f"n={len(ids)} errors={nerr} in {time.time()-t0:.1f}s -> "
          f"{OUT}/ood_phys.npz", flush=True)
    print("indego n=", len(iorder), flush=True)


if __name__ == "__main__":
    main()
