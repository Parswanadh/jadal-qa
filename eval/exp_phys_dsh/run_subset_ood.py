#!/usr/bin/env python3
"""DSH-PHYS: does a-priori rule SUBSET selection improve label-free OOD transfer?

Motivation (from the per-rule diagnostics): under per-domain re-quantiling the
rules whose physics is *scale-like* recover on OOD (BLUR +0.17 AUROC,
CODING_NOISE +0.10) while the semantic/motion rules stay at chance. The subset
{BLUR, CODING_NOISE, DUPLICATE} is chosen A PRIORI from the physics of the
features (sharpness, bitrate/compression telemetry, duplicate hash) -- NOT from the
OOD labels, which are used to report the result only.

Everything else is identical to run_v2_lead.py: same integer Q=4 thresholds, same
IndEgo sign/rule list, same integer weights = 1 (the uniform count that won
in-domain), same target-side rank normalisation.

Output: eval/results/exp_phys_dsh/LEAD_subset_ood.json
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
    LEVELS, Q, auroc_bad, boot_ci, feat, load_npz, pick_reject_train, q_int,
    rule_matrix, thresholds_q, ba_at)

OUTDIR = os.path.join(REPO, "eval", "results", "exp_phys_dsh")
APRIORI_SCALE_SUBSET = ("BLUR", "CODING_NOISE", "DUPLICATE")


def main() -> int:
    t0 = time.time()
    indego, ood = load_npz("indego"), load_npz("ood160")
    y, clips = indego["y"], indego["clips"]

    subsets = {
        "all_rules": None,
        "apriori_scale_subset": APRIORI_SCALE_SUBSET,
        "motion_only": ("NO_ACTIVE_WORK",),
        "quality_only": ("TOO_DARK", "OVEREXPOSED", "BLUR", "FROZEN_STATIC"),
    }
    out = {"experiment": "LEAD_subset_ood",
           "device": "CPU (RunPod A40 host; nvidia-smi unavailable)",
           "protocol": ("OOD = 160 human-labelled Egocentric-10K windows, "
                        "window-bootstrap 95% CI (optimistic); in-domain = IndEgo 150 "
                        "LOCO over 7 clips"),
           "label_sources": {"indego": "human 100 good/50 bad",
                             "ood160": "human 138 good/22 bad"},
           "subset_choice_note": ("the scale subset is chosen from feature physics "
                                  "(sharpness / bitrate / hash), a priori, NOT from OOD "
                                  "labels; OOD labels only report the outcome"),
           "apriori_scale_subset": list(APRIORI_SCALE_SUBSET),
           "quantile_levels": list(LEVELS), "quantisation": Q,
           "results": {}}

    # thresholds: IndEgo training values, fixed level ladder, integer Q=4
    thr_indego = {}
    for r in CANDIDATE_RULES:
        vals = np.array([feat(indego["names"], indego["X"][i], r["feature"])
                         for i in range(len(y))])
        thr_indego[r["rule"] + "|" + r["feature"]] = thresholds_q(vals, r["side"])

    # target-side rank normalisation (uniform in (0,1) per feature), target only
    Xr = np.array(ood["X"], float, copy=True)
    order = np.argsort(np.argsort(Xr, axis=0), axis=0).astype(float)
    finite = np.isfinite(Xr)
    cnt = finite.sum(axis=0)
    Xr = np.where(finite, (order + 0.5) / np.maximum(cnt, 1), np.nan)
    thr_rank = {r["rule"] + "|" + r["feature"]:
                [q_int(1.0 - l) if r["side"] == "high" else q_int(l) for l in LEVELS]
                for r in CANDIDATE_RULES}

    for name, sub in subsets.items():
        rules = [r for r in CANDIDATE_RULES if sub is None or r["rule"] in sub]
        M_ind = rule_matrix(indego["X"], indego["names"], rules, thr_indego)
        M_raw = rule_matrix(ood["X"], ood["names"], rules, thr_indego)
        M_rank = rule_matrix(Xr, ood["names"], rules, thr_rank)
        w = np.ones(M_ind.shape[1], int)          # uniform count (the in-domain winner)
        s_ind = M_ind @ w
        t = pick_reject_train(s_ind, y)           # operating point from IndEgo training only
        rec = {"n_rule_specs": int(M_ind.shape[1]),
               "rules": sorted({r["rule"] for r in rules}),
               "indego_auroc_in_sample": round(auroc_bad(y, s_ind), 4),
               "indego_auroc_ci95_clip_bootstrap": [round(v, 4) for v in boot_ci(y, s_ind, clips)],
               "indego_reject_threshold": int(t)}
        for tag, M in (("ood_raw", M_raw), ("ood_requantile", M_rank)):
            s = M @ w
            ba, tpr, tnr = ba_at(ood["y"], s, t)
            rec[tag] = {"auroc": round(auroc_bad(ood["y"], s), 4),
                        "auroc_ci95_window_bootstrap": [round(v, 4) for v in boot_ci(ood["y"], s, ood["clips"])],
                        "reject_rate_all": round(float((s >= t).mean()), 4),
                        "balanced_accuracy": round(ba, 4),
                        "reject_rate_bad": round(tpr, 4),
                        "false_reject_good": round(1 - tnr, 4)}
        rec["delta_auroc_requantile_minus_raw"] = round(
            rec["ood_requantile"]["auroc"] - rec["ood_raw"]["auroc"], 4)
        out["results"][name] = rec
        print(f"{name:24s} in={rec['indego_auroc_in_sample']:.3f} "
              f"raw={rec['ood_raw']['auroc']:.4f} rank={rec['ood_requantile']['auroc']:.4f} "
              f"d={rec['delta_auroc_requantile_minus_raw']:+.4f}", flush=True)

    out["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUTDIR, "LEAD_subset_ood.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("wrote", os.path.join(OUTDIR, "LEAD_subset_ood.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
