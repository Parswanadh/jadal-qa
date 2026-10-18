#!/usr/bin/env python3
"""AUTO.04 - cost-sensitive thresholds at false-accept : false-reject = 1:1, 3:1, 1:3.

accept = p >= t.  The threshold is chosen on TRAINING FOLDS ONLY by minimising
the empirical cost  c_fa * (bad windows accepted) + c_fr * (good windows
rejected)  on the training clips' inner-LOCO probabilities of every outer fold
(in-domain), and on the IndEgo-150 out-of-fold probabilities for OOD-160 (OOD
has no training folds of its own).  The analytic Bayes threshold
c_fa/(c_fa+c_fr) is printed as arithmetic for contrast: it is only optimal for
a calibrated probability and the balanced-class-weight head is deliberately not
calibrated.

Run:  JADAL_BUNDLE=<bundle> OMP_NUM_THREADS=2 nice -n 10 \
      /home/parshu/envs/jadal/bin/python tools/dsh/auto04.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auto_util as A  # noqa: E402
import common as C  # noqa: E402

RATIOS = [("1:1", 1.0, 1.0), ("3:1", 3.0, 1.0), ("1:3", 1.0, 3.0)]
CONFIG = {
    "id": "AUTO.04",
    "cost_ratios_false_accept_false_reject": [r[0] for r in RATIOS],
    "selection": "empirical cost minimiser on the training clips' inner-LOCO probabilities per outer fold",
    "ood_selection": "cost minimiser on the IndEgo-150 out-of-fold probabilities (training data w.r.t. OOD)",
    "cost_definition": ("c_fa * (bad windows accepted) + c_fr * (good windows rejected), counted on the "
                        "evaluation set; cost per 1000 windows is arithmetic from the evaluation set's own class mix"),
    "band": "|p-0.5| < 0.25 (reported as a share only)",
    "n_boot": 2000, "seed": 0,
}
OUT = os.path.join(C.REPO, "eval", "results", "auto", "AUTO.04.json")


def _round(o, nd=4):
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return round(f, nd) if np.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def cost_of(y, pred, c_fa, c_fr):
    y = np.asarray(y)
    pred = np.asarray(pred)
    fa = int(((pred == 1) & (y == 0)).sum())
    fr = int(((pred == 0) & (y == 1)).sum())
    return {"fa_count": fa, "fr_count": fr, "cost": float(c_fa * fa + c_fr * fr),
            "cost_per_1000": float(1000.0 * (c_fa * fa + c_fr * fr) / len(y))}


def main():
    n_boot, seed = CONFIG["n_boot"], CONFIG["seed"]
    g = C.load_gold()
    X, y, clip = g["X"], g["y"], g["clip"]
    p = C.loco_proba(X, y, clip, C=0.3)

    indomain = {}
    for name, c_fa, c_fr in RATIOS:
        thr, folds = A.loco_fold_thresholds(X, y, clip, kind="cost", C=0.3, c_fa=c_fa, c_fr=c_fr)
        pred = (p >= thr).astype(int)
        indomain[name] = {
            "cost_ratio_fa_fr": name,
            "threshold_per_fold_min_median_max": [round(float(np.min(thr)), 4),
                                                  round(float(np.median(thr)), 4),
                                                  round(float(np.max(thr)), 4)],
            "thresholds_by_fold": {k: round(v["threshold"], 4) for k, v in folds.items()},
            "analytic_bayes_threshold_arithmetic": round(c_fa / (c_fa + c_fr), 4),
            "rates": _round(A.policy_rates(y, pred)),
            "rates_ci95_clip": A.rates_ci(y, clip, pred, n_boot, seed),
            "cost_on_eval": _round(cost_of(y, pred, c_fa, c_fr)),
            "fixed_0.5_contrast": dict(
                _round(A.policy_rates(y, (p >= 0.5).astype(int))),
                cost_on_eval=_round(cost_of(y, (p >= 0.5).astype(int), c_fa, c_fr)),
                note="the same cost function evaluated at the fixed 0.5 threshold; a cost-sensitive "
                     "threshold that does not beat this on the held-out clips has not paid for itself"),
        }

    o = C.load_ood160()
    oy, ofac, op = o["y"], o["factory"], o["p26"]
    ood = {}
    for name, c_fa, c_fr in RATIOS:
        t, val = A.cost_threshold(y, p, c_fa=c_fa, c_fr=c_fr)          # IndEgo = training data
        pred = (op >= t).astype(int)
        t_or, _ = A.cost_threshold(oy, op, c_fa=c_fa, c_fr=c_fr)       # in-sample, contrast only
        ood[name] = {
            "cost_ratio_fa_fr": name,
            "threshold_from_indego": round(t, 4),
            "threshold_train_cost": round(val, 4),
            "analytic_bayes_threshold_arithmetic": round(c_fa / (c_fa + c_fr), 4),
            "rates": _round(A.policy_rates(oy, pred)),
            "rates_ci95_factory": A.rates_ci(oy, ofac, pred, n_boot, seed),
            "cost_on_eval": _round(cost_of(oy, pred, c_fa, c_fr)),
            "fixed_0.5_contrast": dict(
                _round(A.policy_rates(oy, (op >= 0.5).astype(int))),
                cost_on_eval=_round(cost_of(oy, (op >= 0.5).astype(int), c_fa, c_fr)),
            ),
            "optimistic_in_sample_ood_contrast": {
                "threshold": round(t_or, 4),
                "rates": _round(A.policy_rates(oy, (op >= t_or).astype(int))),
                "note": "threshold chosen on the OOD-160 labels themselves -> in-sample, not a policy",
            },
        }

    payload = {
        "id": "AUTO.04", "status": "done",
        "n": int(len(y)), "n_bad": int((y == 0).sum()), "n_good": int((y == 1).sum()),
        "n_ood": int(len(oy)), "n_bad_ood": int((oy == 0).sum()),
        "n_groups": len(set(clip)), "groups": "7 IndEgo clips / 47 OOD factories",
        "label_source": g["label_source"] + " | " + o["label_source"],
        "protocol": ("per-fold cost minimiser on the 6 training clips' inner-LOCO probabilities (in-domain); "
                     "IndEgo-150 OOF cost minimiser transferred to the frozen OOD-160 scores; clip- and "
                     "factory-grouped bootstrap CIs"),
        "ci95": [indomain["1:1"]["rates_ci95_clip"]["bal_acc_ci95"][0],
                 indomain["1:1"]["rates_ci95_clip"]["bal_acc_ci95"][1]],
        "ci95_type": A.CI_TYPE_CLIP + " on balanced accuracy at the 1:1 ratio",
        "device": C.DEVICE_CPU,
        "cost": "0 GPU; cached features and frozen scores; ~12 s CPU (3 ratios x 7 outer + 42 inner fits)",
        "command": ("JADAL_BUNDLE=<bundle> OMP_NUM_THREADS=2 nice -n 10 "
                    "/home/parshu/envs/jadal/bin/python tools/dsh/auto04.py"),
        "git_sha": C.git_sha(), "seed": seed,
        "config_sha256": C.config_sha256(CONFIG), "config": CONFIG,
        "indomain": indomain,
        "ood160": {"caveat": A.decision_note_ood(), **ood},
        "notes": [
            "False accept = bad window accepted (cost c_fa); false reject = good window rejected (cost c_fr).",
            "3:1 makes the policy accept more and raises the false-accept rate; 1:3 does the reverse.",
            "The analytic Bayes threshold is arithmetic, not a measurement, and assumes calibrated probabilities.",
            "cost_per_1000 is expressed in that row's own cost units (c_fa, c_fr as named); rows with "
            "different ratios are therefore NOT comparable with each other.",
            "All policies here are forced decisions at a threshold; the shipped cascade accepts outside the "
            "band at p >= 0.75 instead, so these operating points are not the shipped ones.",
        ],
    }
    C.save_results(OUT, payload)
    for side in ("indomain", "ood160"):
        for name in indomain:
            r = payload[side][name]
            print(side, name, "thr", r.get("threshold_from_indego") or r["threshold_per_fold_min_median_max"],
                  "BA", r["rates"]["bal_acc"], "FAR", round(r["rates"]["false_accept_rate"], 3),
                  "FRR", round(r["rates"]["false_reject_rate"], 3),
                  "cost/1000", r["cost_on_eval"]["cost_per_1000"], "accept", r["rates"]["accept_share"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
