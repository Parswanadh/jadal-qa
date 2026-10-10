#!/usr/bin/env python3
"""DSH-PHYS: how much accuracy is left on the table for this MODEL CLASS?

The v2 model is a deterministic function of the per-feature QUANTILE LEVELS a window
crosses -- i.e. of each feature's rank within the domain. Any deterministic rule model
built from per-feature order statistics lives in that class. So the honest ceiling is the
best LOCO classifier trained on the RANK-TRANSFORMED features (monotone-invariant), not on
the raw values.

Reported under the same LOCO folds as everything else:
  * v2 rule model                 (the model we ship)
  * Q1 rule model                 (the baseline)
  * LOCO logistic on raw 42 feats (not monotone-invariant; uses scale information the rule
                                   model cannot)
  * LOCO logistic on per-fold RANK-transformed 42 feats  <- the class ceiling
  * LOCO logistic on the 8 per-rule fire counts          <- best achievable aggregation of
                                                            exactly the rules we have

This is a diagnostic, not a new model: nothing here is proposed for the decision path.

Output: eval/results/exp_phys_dsh/LEAD_ceiling.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from eval.exp_phys.phys_rules import CANDIDATE_RULES  # noqa: E402
from eval.exp_phys_dsh.run_v2_lead import (  # noqa: E402
    auroc_bad, boot_ci, feat, load_npz, q_int, rule_matrix, thresholds_q)

OUTDIR = os.path.join(REPO, "eval", "results", "exp_phys_dsh")


def rank_transform(X):
    """Per-column rank -> uniform (0,1). Monotone-invariant representation."""
    Xr = np.array(X, float, copy=True)
    order = np.argsort(np.argsort(Xr, axis=0), axis=0).astype(float)
    fin = np.isfinite(Xr)
    cnt = fin.sum(axis=0)
    return np.where(fin, (order + 0.5) / np.maximum(cnt, 1), np.nan)


def loco_logistic(X, y, clips, C=0.3):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    p = np.zeros(len(y))
    for c in sorted(set(clips)):
        te = np.array([x == c for x in clips])
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        m.fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def main() -> int:
    t0 = time.time()
    d = load_npz("indego")
    X, y, clips, names = d["X"], d["y"], d["clips"], d["names"]
    out = {"experiment": "LEAD_ceiling", "device": "CPU (RunPod A40 host)",
           "protocol": "leave-one-clip-out over the 7 IndEgo clips; clip-bootstrap B=2000 seed 0",
           "label_source": "human (manifest_gold.json, 100 good / 50 bad)",
           "command": "python3 eval/exp_phys_dsh/run_ceiling.py",
           "caveat": ("a diagnostic upper bound for the model CLASS, not a proposed model; "
                      "logistic regression is not deterministic-integer and cannot ship"),
           "results": {}}

    # Q1 + v2 rule scores
    v1 = json.load(open(os.path.join(REPO, "eval", "results", "exp_phys",
                                     "rule_table.json")))["rules"]
    thr1 = {r["rule"] + "|" + r["feature"]: [q_int(t) for t in r["thresholds"]]
            for r in v1}
    M1 = rule_matrix(X, names, CANDIDATE_RULES, thr1)
    out["results"]["q1_rule_model"] = round(auroc_bad(y, M1.sum(axis=1)), 4)

    cfg = json.load(open(os.path.join(REPO, "configs", "phys_rules_v2.json")))
    thr2 = {r["rule"] + "|" + r["feature"]: [int(t) for t in r["thresholds_q"]]
            for r in cfg["rules"]}
    M2 = rule_matrix(X, names, cfg["rules"], thr2)
    w2 = np.array([r.get("weight", 1) for r in cfg["rules"]], int)
    out["results"]["v2_rule_model"] = round(auroc_bad(y, M2 @ w2), 4)

    # per-fold rank transform, then logistic (the class ceiling)
    Xr = np.zeros_like(X)
    for c in sorted(set(clips)):
        te = np.array([x == c for x in clips])
        Xr[te] = rank_transform(X[te])          # target fold's OWN unlabelled stats
    Xr = np.where(np.isfinite(Xr), Xr, 0.5)
    Xf = np.where(np.isfinite(X), X, np.nan)
    col_med = np.nanmedian(Xf, axis=0)
    Xf = np.where(np.isfinite(Xf), Xf, col_med)

    for tag, XX in (("logistic_raw42", Xf), ("logistic_rank42_CLASS_CEILING", Xr)):
        p = loco_logistic(XX, y, clips)
        out["results"][tag] = round(auroc_bad(y, 1.0 - p), 4)
        out["results"][tag + "_ci95"] = [round(v, 4) for v in boot_ci(y, 1.0 - p, clips)]
        print(f"{tag:34s} AUROC={out['results'][tag]:.4f}", flush=True)

    # ceiling of aggregating exactly the rules we have: logistic on the 16 fire counts
    Mf = np.where(np.isfinite(M2), M2, 0)
    p = loco_logistic(Mf.astype(float), y, clips)
    out["results"]["logistic_on_16_rule_firecounts"] = round(auroc_bad(y, 1.0 - p), 4)
    out["results"]["logistic_on_16_rule_firecounts_ci95"] = [
        round(v, 4) for v in boot_ci(y, 1.0 - p, clips)]
    print("logistic_on_16_rule_firecounts       AUROC="
          f"{out['results']['logistic_on_16_rule_firecounts']:.4f}", flush=True)

    out["headroom"] = {
        "v2_vs_class_ceiling": round(out["results"]["logistic_rank42_CLASS_CEILING"]
                                     - out["results"]["v2_rule_model"], 4),
        "v2_vs_best_aggregation_of_same_rules": round(
            out["results"]["logistic_on_16_rule_firecounts"]
            - out["results"]["v2_rule_model"], 4),
        "interpretation": ("the rank-42 logistic is the best any per-feature-order-statistic "
                           "model can do on these features under LOCO; the gap to v2 is the "
                           "headroom left inside the model class, and it is bought with a "
                           "non-integer, non-shippable model")}
    out["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUTDIR, "LEAD_ceiling.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps(out["headroom"], indent=1))
    print("wrote LEAD_ceiling.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
