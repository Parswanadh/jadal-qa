#!/usr/bin/env python3
"""GRID.06 — out-of-domain transfer of the mosaic head on the 160 factory windows.

Two questions, both on the SAME windows as the frozen shipped-head scores:

  1. FROZEN TRANSFER: train the mosaic head on the 150 IndEgo windows only, apply
     it unchanged to the 160 Egocentric-10K windows.  Paired, factory-clustered
     comparison against the frozen 26-number shipped head (`load_ood160()["p26"]`).
  2. FACTORY-GROUPED CV: is there a mosaic-native recipe that works on factory
     footage when it is TRAINED on factory footage?  Grouped 5-fold CV by factory
     (never leave-one-window-out), hyperparameters fixed.

Every OOD number is printed next to n=160, n_bad=22, 19/22 bad = NO_ACTIVE_WORK,
and the reason-stratified result is given for NO_ACTIVE_WORK only (the other two
reasons have 2 and 1 windows: not reportable).

Run: JADAL_BUNDLE=<bundle> python3 eval/exp_grid/run_grid06.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "dsh"))

import common as C  # noqa: E402
from eval.exp_grid.run_grid import matrix  # noqa: E402

RESULTS = os.path.join(ROOT, "eval", "results", "exp_grid")
SEED, N_BOOT = 0, 2000


def grouped_cv_proba(X, y, groups, C_reg=0.3, n_folds=5, seed=SEED):
    """Deterministic grouped CV: factories are assigned to folds by sorted name, so
    the split is reproducible and independent of the label vector."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    uniq = sorted(set(groups))
    fold_of = {g: i % n_folds for i, g in enumerate(uniq)}
    fid = np.asarray([fold_of[g] for g in groups])
    p = np.zeros(len(y), float)
    for k in range(n_folds):
        tr, te = fid != k, fid == k
        if tr.sum() < 10 or te.sum() == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=3000, C=C_reg, class_weight="balanced").fit(
            sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def main():
    ood = C.load_ood160()
    ids, y, fac, reason = ood["ids"], ood["y"], ood["factory"], ood["reason"]
    gold = C.load_gold()

    doc = json.load(open(os.path.join(RESULTS, "features_ood_m3x3_t224.json")))
    names = doc["feature_names"]
    Xo = matrix(doc, ids, names)
    missing = int(np.isnan(Xo).any(axis=1).sum())
    print(f"OOD features: {Xo.shape}, windows with any missing feature: {missing}")
    if missing:
        col = np.nanmean(Xo, axis=0)
        idx = np.where(np.isnan(Xo))
        Xo[idx] = np.take(col, idx[1])
    Xi = matrix(json.load(open(os.path.join(RESULTS, "features_m3x3_t224.json"))),
                gold["ids"], names)

    # ---- 1. frozen transfer (train on IndEgo only) ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xi)
    clf = LogisticRegression(max_iter=3000, C=0.3, class_weight="balanced").fit(
        sc.transform(Xi), gold["y"])
    p_frozen = clf.predict_proba(sc.transform(Xo))[:, 1]
    a_frozen = C.fast_auc(y, p_frozen)
    lo, hi, _ = C.group_boot_ci(y, p_frozen, fac, n_boot=N_BOOT, seed=SEED)
    wlo, whi, _ = C.window_boot_ci(y, p_frozen, n_boot=N_BOOT, seed=SEED)
    a_base = C.fast_auc(y, ood["p26"])
    blo, bhi, _ = C.group_boot_ci(y, ood["p26"], fac, n_boot=N_BOOT, seed=SEED)
    d = C.paired_group_boot_diff(y, p_frozen, ood["p26"], fac, n_boot=N_BOOT, seed=SEED)
    print(f"frozen 26-number shipped head   AUROC {a_base:.4f} [{blo:.3f},{bhi:.3f}] factory-grouped")
    print(f"FROZEN mosaic head on OOD-160   AUROC {a_frozen:.4f} [{lo:.3f},{hi:.3f}] factory-grouped")
    print(f"frozen 26-number shipped head   AUROC {a_base:.4f}  paired delta {d['delta']:+.4f} "
          f"[{d['lo']:+.3f},{d['hi']:+.3f}] excludes0={d['excludes_zero']}")

    # ---- 2. factory-grouped CV on the OOD set itself ----
    p_cv = grouped_cv_proba(Xo, y, fac)
    a_cv = C.fast_auc(y, p_cv)
    clo, chi, _ = C.group_boot_ci(y, p_cv, fac, n_boot=N_BOOT, seed=SEED)
    dcv = C.paired_group_boot_diff(y, p_cv, ood["p26"], fac, n_boot=N_BOOT, seed=SEED)
    print(f"mosaic head, factory-grouped CV AUROC {a_cv:.4f} [{clo:.3f},{chi:.3f}] "
          f"paired vs 26-number {dcv['delta']:+.4f} [{dcv['lo']:+.3f},{dcv['hi']:+.3f}]")

    # ---- reason-stratified (NO_ACTIVE_WORK only: n=19) ----
    naw = (reason == "NO_ACTIVE_WORK")
    strat = {}
    if naw.sum() >= 10:
        sel = naw | (y == 1)
        strat = {
            "reason": "NO_ACTIVE_WORK", "n_bad": int(naw.sum()),
            "frozen_mosaic_auroc": round(C.fast_auc(y[sel], p_frozen[sel]), 4),
            "frozen_26_auroc": round(C.fast_auc(y[sel], ood["p26"][sel]), 4),
            "grouped_cv_auroc": round(C.fast_auc(y[sel], p_cv[sel]), 4),
            "ci95_frozen": [round(v, 3) for v in C.group_boot_ci(y[sel], p_frozen[sel], fac[sel],
                                                                 n_boot=N_BOOT, seed=SEED)[:2]],
            "note": "HANDS_NOT_VISIBLE (2) and SHAKY (1) are not reportable at this n",
        }

    # ---- acceptance rates at the shipped band (descriptive, frozen head) ----
    def band(p):
        return {"accept": float((p >= 0.75).mean()), "reject": float((p <= 0.25).mean()),
                "review": float(((p > 0.25) & (p < 0.75)).mean())}
    payload = {
        "id": "GRID.06", "n": int(len(y)), "n_bad": int((y == 0).sum()),
        "n_good": int((y == 1).sum()), "n_factories": int(len(set(fac))),
        "label_source": ood["label_source"],
        "protocol": "mosaic features from the 480p Egocentric-10K window clips (same files the "
                    "frozen shipped-head scores were produced on); head trained on the 150 IndEgo "
                    "windows only and applied unchanged; CI = factory-grouped bootstrap (2000, seed 0)",
        "frozen_transfer": {"auroc": round(a_frozen, 4), "ci95": [round(lo, 3), round(hi, 3)],
                            "ci95_type": f"factory-grouped bootstrap ({N_BOOT}, seed {SEED})",
                            "ci95_window_optimistic": [round(wlo, 3), round(whi, 3)],
                            "auroc_vs_frozen_26": round(a_base, 4),
                            "ci95_vs_frozen_26": [round(blo, 3), round(bhi, 3)],
                            "paired_vs_26": d, "band_share": band(p_frozen)},
        "grouped_cv_on_ood": {"auroc": round(a_cv, 4), "ci95": [round(clo, 3), round(chi, 3)],
                              "paired_vs_26": dcv, "n_folds": 5, "C": 0.3,
                              "split": "factories assigned to folds by sorted name (deterministic)"},
        "reason_stratified": strat,
        "bad_reason_mix": {r: int(((reason == r) & (y == 0)).sum())
                           for r in sorted(set(reason)) if r},
        "context_required_with_every_ood_number": "n=160, n_bad=22, ONE labeller, 19 of 22 bad are "
                                                  "NO_ACTIVE_WORK; window bootstrap is optimistic and "
                                                  "is never used alone",
        "ci95": [round(lo, 3), round(hi, 3)],
        "ci95_type": f"factory-grouped bootstrap ({N_BOOT}, seed {SEED})",
        "device": C.DEVICE_CPU,
        "cost": f"feature extraction {doc.get('ms_per_window')} ms/window (CPU); 0 GPU",
        "command": "JADAL_BUNDLE=<bundle> python3 eval/exp_grid/run_grid06.py",
        "git_sha": C.git_sha(), "seed": SEED,
        "config_sha256": C.config_sha256({"arm": "m3x3_t224", "frozen": True, "cv_folds": 5,
                                          "C": 0.3, "seed": SEED}),
        "config": {"arm": "m3x3_t224", "frozen": True, "cv_folds": 5, "C": 0.3, "seed": SEED},
    }
    C.save_results(os.path.join(RESULTS, "GRID.06.json"), payload)
    print("wrote", os.path.join(RESULTS, "GRID.06.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
