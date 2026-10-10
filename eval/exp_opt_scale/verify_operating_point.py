#!/usr/bin/env python3
"""R1/T3 — verify the recalibrated operating point matches the frozen one.

Loads the N5 sparse-sampled feature dumps (pixel 3.33 fps, EG2 1 fps, 4 s
prefix; extracted by ``eval/exp_recal/extract_sparse.py`` from branch m/recal)
and re-applies BOTH head sets independently:

  * frozen  heads  -> ``configs/ensemble_v1.json``
  * sparse  heads  -> ``configs/ensemble_v1_sparse.json`` (N5 refit, PR #156)

on the 150 gold (in-domain) and 160 OOD (Egocentric-10K, third-party) windows.
Reports review share + confident balanced accuracy (BA) at the frozen band
[0.4, 0.6] and the recalibrated band [0.4193, 0.6112], plus AUROC@0.5.

Pure numpy: the logistic heads are ``jadal.ensemble.scorer.head_prob``.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from jadal.ensemble.scorer import ensemble_prob  # noqa: E402

RECAL = os.environ.get("JADAL_RECAL_DIR", "/workspace/wt/m-recal/eval/results/exp_recal")
OUT = os.path.join(REPO, "eval", "results", "exp_opt_scale", "operating_point.json")


def probs(npz, cfg):
    d = np.load(npz, allow_pickle=True)
    names = [str(x) for x in d["names"]]
    P = np.empty(len(d["ids"]), dtype=np.float64)
    for i in range(len(d["ids"])):
        feats = {k: float(d["X_pixel"][i, j]) for j, k in enumerate(names)}
        P[i], _, _ = ensemble_prob(d["X_eg2"][i], feats, cfg)
    return P, d["labels"].astype(int), d["clips"]


def auroc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = {u: (csum[u] - cnt[u] + 1 + csum[u]) / 2.0 for u in range(len(cnt))}
    ranks = np.array([avg[i] for i in inv])
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def band_stats(P, y, lo, hi):
    accept = P > hi
    reject = P < lo
    review = ~(accept | reject)
    conf = accept | reject
    pred = np.where(accept, 1, 0)  # accept -> good
    if conf.sum() == 0:
        ba = None
    else:
        yc, pc = y[conf], pred[conf]
        tpr = ((pc == 1) & (yc == 1)).sum() / max((yc == 1).sum(), 1)
        tnr = ((pc == 0) & (yc == 0)).sum() / max((yc == 0).sum(), 1)
        ba = float((tpr + tnr) / 2.0)
    return {
        "n": int(len(P)), "n_good": int((y == 1).sum()), "n_bad": int((y == 0).sum()),
        "accept": int(accept.sum()), "reject": int(reject.sum()),
        "review": int(review.sum()), "review_share": round(float(review.mean()), 4),
        "confident_n": int(conf.sum()),
        "confident_balacc": round(ba, 4) if ba is not None else None,
        "auroc": round(auroc(y, P), 4),
    }


def main():
    frozen = json.load(open(os.path.join(REPO, "configs", "ensemble_v1.json")))
    sparse = json.load(open(os.path.join(REPO, "configs", "ensemble_v1_sparse.json")))
    sets = {
        "gold150": os.path.join(RECAL, "sparse_gold.npz"),
        "ood160": os.path.join(RECAL, "sparse_ood160.npz"),
    }
    out = {
        "experiment": "R1.03-operating-point",
        "source_features": "N5 sparse-sampled features (pixel 3.33 fps, EG2 1 fps, 4 s prefix)",
        "feature_files": {k: os.path.relpath(v, "/workspace") for k, v in sets.items()},
        "frozen_config": "configs/ensemble_v1.json",
        "sparse_config": "configs/ensemble_v1_sparse.json",
        "frozen_band": frozen["abstain_band"],
        "sparse_band": sparse["abstain_band"],
        "sets": {},
    }
    for name, npz in sets.items():
        pf, y, clips = probs(npz, frozen)
        ps, _, _ = probs(npz, sparse)
        d = {
            "n": int(len(y)), "n_good": int((y == 1).sum()), "n_bad": int((y == 0).sum()),
            "n_clips": int(len(set(map(str, clips)))),
            "mean_prob_frozen_heads": round(float(pf.mean()), 4),
            "mean_prob_sparse_heads": round(float(ps.mean()), 4),
            "frozen_heads": {
                "frozen_band": band_stats(pf, y, *frozen["abstain_band"]),
                "sparse_band": band_stats(pf, y, *sparse["abstain_band"]),
            },
            "sparse_heads": {
                "frozen_band": band_stats(ps, y, *frozen["abstain_band"]),
                "sparse_band": band_stats(ps, y, *sparse["abstain_band"]),
            },
        }
        out["sets"][name] = d
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
