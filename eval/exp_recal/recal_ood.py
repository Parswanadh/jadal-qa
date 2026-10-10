#!/usr/bin/env python3
"""N5.02 — OOD validation of the recalibrated sparse heads (160 windows).

Compares three score sources on the 160 human-labelled Egocentric-10K windows
(138 good / 22 bad, 47 factories):
  1. frozen heads on frozen features   (cached eval/results/ensemble/ood_scores{,2}.json)
  2. frozen heads on SPARSE features   (the L3 shift, out-of-domain)
  3. sparse-refit heads on SPARSE features (the shipped N5.01 head)
CIs are factory-grouped bootstrap (resample factories with replacement) — honest
for the factory/worker correlation that makes the plain window-bootstrap optimistic.

Writes eval/results/exp_recal/N5.02.json.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
BUNDLE = os.environ.get("JADAL_BUNDLE", "jadal_pod_bundle")
OUTDIR = os.path.join(REPO, "eval", "results", "exp_recal")
CONFIG = os.path.join(REPO, "configs", "ensemble_v1.json")
BOOT = 2000
SEED = 0


def head_prob(X, h):
    z = (np.asarray(X, float) - np.array(h["mean"])) / np.array(h["scale"])
    return 1.0 / (1.0 + np.exp(-(z @ np.array(h["coef"]) + h["intercept"])))


def group_boot(y, p, groups, B=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    names = sorted(set(groups))
    out = []
    for _ in range(B):
        gs = rng.choice(names, len(names))
        idx = np.concatenate([np.where(groups == g)[0] for g in gs])
        if len(set(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    out = np.asarray(out)
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def balacc(y, pred, thr):
    pr = (np.asarray(pred) > thr).astype(int)
    return round(float(((pr[y == 1] == 1).mean() + (pr[y == 0] == 0).mean()) / 2), 4)


def main():
    labels = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))
    lab = {}
    for w in labels["windows"]:
        stem = os.path.splitext(os.path.basename(w["video"]))[0]
        lab[stem] = (1 if w["verdict"] == "good" else 0, w.get("factory", ""))

    # frozen scores (60 q + 100 r)
    frozen = {}
    for f in ("ood_scores.json", "ood_scores2.json"):
        p = os.path.join(REPO, "eval", "results", "ensemble", f)
        if not os.path.exists(p):
            p = os.path.join(BUNDLE, "caches", "ensemble", f)
        for s in json.load(open(p))["scores"]:
            if "error" in s:
                continue
            stem = os.path.splitext(os.path.basename(s["id"]))[0]
            frozen[stem] = s

    sp = np.load(os.path.join(OUTDIR, "sparse_ood160.npz"), allow_pickle=True)
    ids = [str(i) for i in sp["ids"]]
    Xe, Xg = np.asarray(sp["X_eg2"], float), np.asarray(sp["X_pixel"], float)
    y = np.array([lab[i][0] for i in ids])
    groups = np.array([lab[i][1] for i in ids])
    cfg = json.load(open(CONFIG))
    refit = json.load(open(os.path.join(OUTDIR, "N5.01.json")))["refit_heads"]

    have_frozen = [i for i in ids if i in frozen]
    idx_f = [ids.index(i) for i in have_frozen]
    p_frozen = np.array([frozen[i]["usable_prob"] for i in have_frozen])
    y_f, g_f = y[idx_f], groups[idx_f]

    p_eg2_frozen = head_prob(Xe, cfg["views"]["eg2"])
    p_px_frozen = head_prob(Xg, cfg["views"]["pixel"])
    p_frozen_on_sparse = (p_eg2_frozen + p_px_frozen) / 2

    p_eg2_refit = head_prob(Xe, refit["eg2"])
    p_px_refit = head_prob(Xg, refit["pixel"])
    p_refit = (p_eg2_refit + p_px_refit) / 2

    def rec(p, tag, yy, gg):
        auc = float(roc_auc_score(yy, p))
        return {"tag": tag, "n": int(len(yy)), "n_good": int(yy.sum()),
                "n_bad": int((yy == 0).sum()), "n_groups": int(len(set(gg))),
                "auroc": round(auc, 4), "group_ci95": group_boot(yy, p, gg),
                "balacc@0.5": balacc(yy, p, 0.5),
                "mean_good": round(float(p[yy == 1].mean()), 4),
                "mean_bad": round(float(p[yy == 0].mean()), 4)}

    res = {
        "experiment": "N5.02",
        "label_source": "human (160 Egocentric-10K windows; third-party, not published)",
        "protocol": "frozen vs sparse-refit heads on the 160 OOD windows; factory-grouped bootstrap B=2000 seed0",
        "device": "A40",
        "frozen_scores_available": len(have_frozen),
        "frozen_features": rec(p_frozen, "frozen heads / frozen features", y_f, g_f),
        "sparse_frozen_heads": rec(p_frozen_on_sparse, "frozen heads / sparse features", y, groups),
        "sparse_refit": rec(p_refit, "sparse-refit heads / sparse features", y, groups),
    }
    d_shift = res["sparse_frozen_heads"]["auroc"] - res["frozen_features"]["auroc"]
    d_refit = res["sparse_refit"]["auroc"] - res["frozen_features"]["auroc"]
    res["deltas"] = {
        "sparse_frozen_heads_minus_frozen": round(d_shift, 4),
        "sparse_refit_minus_frozen": round(d_refit, 4),
        "interpretation": ("frozen heads on sparse features reproduce the L3 shift out-of-domain; "
                           "the sparse-refit head must keep OOD AUROC within the grouped CI of the frozen head"),
    }
    json.dump(res, open(os.path.join(OUTDIR, "N5.02.json"), "w"), indent=1)
    for k in ("frozen_features", "sparse_frozen_heads", "sparse_refit"):
        r = res[k]
        print(f"{r['tag']:36s} AUROC {r['auroc']:.4f} CI {r['group_ci95']} "
              f"bal@0.5 {r['balacc@0.5']:.3f} good/bad mean {r['mean_good']:.3f}/{r['mean_bad']:.3f}")
    print("deltas:", res["deltas"])
    return res


if __name__ == "__main__":
    main()
