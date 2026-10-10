"""End-to-end: does the headline EG2 0.848 / two-view 0.8524 survive when the EG2
features are regenerated from the SOURCE VIDEO instead of read from the cache?

Uses the regenerated full-run frame embeddings (verify_full_reproduction.py --phase
encode) with the EXACT evaluation protocol of eval/exp_fps/run_eg2.py:
  window_mean(step=1): rows = E[k, :counts[k]] -> mean over L2-normalised frames ->
  L2-normalise; LOCO logistic C=0.01 (StandardScaler per fold, class_weight balanced);
  AUROC + clip-bootstrap CI B=1000 seed 0.
Two-view ensemble: mean of EG2 LOCO probability and the cached pixel-view LOCO
probability (26 features, C=0.3) at matching fps, exactly as run_eg2.py.

CPU only.
  OMP_NUM_THREADS=2 python3 eval/frames10k/verify_end_to_end_auroc.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp_fps"))
from fps_lib import bal_acc, clip_bootstrap_ci, loco_proba  # noqa: E402

from verify_cache_reproduction import CACHE, SCRATCH, WT  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

D = WT / "eval" / "results" / "exp_fps"
OUT = WT / "eval" / "results" / "pod_verify" / "eg2_end_to_end_auroc.json"


def pooled(E, F, cnt, ids):
    """window_mean with step=1, cache and regenerated; identical code path."""
    out = {}
    for tag, src in (("cache", F), ("regenerated", E)):
        Xs = []
        for k in range(len(ids)):
            n = max(1, int(cnt[k]))
            rows = src[k, :n].astype(np.float32)
            Xs.append(rows.mean(axis=0))
        X = np.stack(Xs)
        out[tag] = X / np.linalg.norm(X, axis=1, keepdims=True)
    return out


def main() -> None:
    z = np.load(CACHE, allow_pickle=True)
    ids = [str(i) for i in z["ids"]]
    clips = np.array([str(c) for c in z["clips"]])
    y = np.array(z["labels"]).astype(int)
    cnt = np.array(z["counts"]).astype(int)
    Er = np.load(SCRATCH / "full_frame_embeddings.fp16.npy")
    assert Er.shape[0] == int(cnt.sum()), (Er.shape, cnt.sum())
    # rebuild (150, T, 768) from the flat regenerated frame stream, cache layout
    T = z["F"].shape[1]
    E = np.full((len(ids), T, Er.shape[1]), np.nan, dtype=np.float16)
    o = 0
    for k in range(len(ids)):
        c = int(cnt[k])
        E[k, :c] = Er[o:o + c]
        o += c
    X = pooled(E, z["F"], cnt, ids)

    base = json.load(open(D / "e1_02.json"))
    sh = [json.load(open(D / f"shard_sweep_{i}of2.json")) for i in range(2)]
    order = np.argsort([x for s in sh for x in s["ids"]], kind="stable")
    sids = [x for s in sh for x in s["ids"]]
    sids = [sids[i] for i in order]
    assert sids == ids, "id order mismatch between EG2 cache and pixel shards"
    px = {t: np.nan_to_num(np.array([x for s in sh for x in s["feats"][t]])[order])
          for t in sh[0]["feats"]}

    res = {"experiment": "LB1-eg2-repro end-to-end",
           "protocol": "identical to eval/exp_fps/run_eg2.py: LOCO, L2-normalised mean "
                       "embedding + StandardScaler + logistic C=0.01; CI=clip-bootstrap "
                       "B=1000 seed0; two-view = mean of EG2 and pixel-view LOCO probs "
                       "(26 features, C=0.3)",
           "n": len(ids), "n_good": int(y.sum()), "n_bad": int((1 - y).sum()),
           "eg2_only": {}, "two_view": {}, "baseline_e1_02": {}}
    probs = {}
    for tag in ("cache", "regenerated"):
        p = loco_proba(X[tag], y, clips, C=0.01)
        probs[tag] = p
        lo, hi = clip_bootstrap_ci(y, p, clips)
        res["eg2_only"][tag] = {"auroc": round(float(roc_auc_score(y, p)), 6),
                                "ci95": [round(lo, 4), round(hi, 4)],
                                "bal_acc": round(bal_acc(y, p), 4)}
    pg = loco_proba(px["2"], y, clips, C=0.3)
    for tag in ("cache", "regenerated"):
        ensp = (probs[tag] + pg) / 2
        lo, hi = clip_bootstrap_ci(y, ensp, clips)
        res["two_view"][tag] = {"auroc": round(float(roc_auc_score(y, ensp)), 6),
                                "ci95": [round(lo, 4), round(hi, 4)],
                                "bal_acc": round(bal_acc(y, ensp), 4)}
    res["baseline_e1_02"] = {"eg2_only_2fps": base["eg2_rows"][0]["auroc"],
                             "two_view_2fps": base["two_view_prob_avg"][0]["auroc"]}
    res["deltas"] = {
        "eg2_only_regenerated_minus_published": round(
            res["eg2_only"]["regenerated"]["auroc"] - base["eg2_rows"][0]["auroc"], 6),
        "two_view_regenerated_minus_published": round(
            res["two_view"]["regenerated"]["auroc"] - base["two_view_prob_avg"][0]["auroc"], 6),
        "feature_cosine_cache_vs_regenerated": {
            "min": float((X["cache"] * X["regenerated"]).sum(1).min()),
            "median": float(np.median((X["cache"] * X["regenerated"]).sum(1))),
            "max": float((X["cache"] * X["regenerated"]).sum(1).max())},
        "probability_max_abs_diff_eg2": float(np.max(np.abs(probs["cache"] - probs["regenerated"]))),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
