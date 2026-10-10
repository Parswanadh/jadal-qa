#!/usr/bin/env python3
"""DSH-PHYS: freeze the v2 rule table + integer weights, and LOCO it in-domain.

Candidates (all fully frozen -- no threshold is fitted inside a test fold, which
makes the LOCO numbers a strict evaluation of a zero-degree-of-freedom model):

  C1 v1_uniform      : the Q1 16-spec quantile table (PR #141), uniform weight 1,
                       t_reject = 14. My independent LOCO reproduction of Q1.
  C2 a1_fixed5       : the five rules of the A1 derivation (commit 2ad71fd) that
                       live in the packaged 42-feature space, integer Q=4 constants
                       derived as k x a training-median physical reference,
                       integer weights {2,3,3,3,3}, tau = 9.
  C3 merged          : C2 plus the Q1 soft specs as graded evidence (the union).

Selection is on IN-DOMAIN LOCO ONLY. The 108-window hold-out is scored once,
afterwards, by score_holdout.py -- never here.

Output: configs/phys_rules_v2.json  +  eval/results/exp_phys_dsh/LEAD_freeze.json
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
    BOOT, LEVELS, Q, SEED, auroc_bad, ba_at, boot_ci, feat, load_npz,
    pick_reject_train, q_int, rule_matrix, thresholds_q)

OUTDIR = os.path.join(REPO, "eval", "results", "exp_phys_dsh")
CONFIG = os.path.join(REPO, "configs", "phys_rules_v2.json")

# A1's deployable integer constants in the packaged 42-feature space (commit 2ad71fd)
A1_FIXED5 = [
    {"rule": "HANDS_NOT_VISIBLE", "feature": "full.skin_frac", "op": "<",
     "side": "low", "strength": "weak", "thresholds_q": [261], "weight": 2,
     "reference": "median_train skin fraction", "k": 0.75, "reference_value": 0.0348575,
     "derivation": "0.75 x median training-window skin fraction (hands out of frame)"},
    {"rule": "BLUR", "feature": "full.dct_energy", "op": "<",
     "side": "low", "strength": "quality", "thresholds_q": [65], "weight": 3,
     "reference": "median_train DCT AC energy", "k": 0.75, "reference_value": 0.008682,
     "derivation": "0.75 x median training DCT AC energy (acuity floor)"},
    {"rule": "SHAKY", "feature": "full.global_shift", "op": ">",
     "side": "high", "strength": "weak", "thresholds_q": [211595], "weight": 3,
     "reference": "median_train global shift", "k": 2.0, "reference_value": 14.106335,
     "derivation": "2.0 x median training global shift (camera shake)"},
    {"rule": "BLUR", "feature": "full.hf_ratio", "op": "<",
     "side": "low", "strength": "quality", "thresholds_q": [127], "weight": 3,
     "reference": "median_train HF share", "k": 0.75, "reference_value": 0.016952,
     "derivation": "0.75 x median training HF share (noise-normalised sharpness)"},
    {"rule": "NO_ACTIVE_WORK", "feature": "full.motion_energy", "op": ">",
     "side": "high", "strength": "weak", "thresholds_q": [577], "weight": 3,
     "reference": "median_train motion energy", "k": 2.0, "reference_value": 0.038485,
     "derivation": "2.0 x median training motion energy (motion after ego-compensation)"},
]


def v1_specs():
    d = json.load(open(os.path.join(REPO, "eval", "results", "exp_phys",
                                    "rule_table.json")))
    out = []
    for r in d["rules"]:
        out.append({"rule": r["rule"], "feature": r["feature"], "op": r["op"],
                    "side": r["side"], "strength": r["strength"],
                    "thresholds_q": [q_int(t) for t in r["thresholds"]],
                    "weight": 1,
                    "reference": r.get("source_quantile", ""),
                    "derivation": ("IndEgo training quantile ladder "
                                   f"{r.get('source_quantile','')} at Q=4 "
                                   "(Q1/PR #141 thresholds, frozen)")})
    return out


def evaluate(rules, indego, budget=0.05):
    """Frozen rules; the operating point tau is the ONLY per-fold quantity, chosen
    inside each fold on the training clips under the <=5% false-reject budget."""
    M = rule_matrix(indego["X"], indego["names"], rules,
                    {r["rule"] + "|" + r["feature"]: r["thresholds_q"] for r in rules})
    w = np.array([r.get("weight", 1) for r in rules], int)
    s = M @ w
    y, clips = indego["y"], indego["clips"]
    uc = sorted(set(clips))
    rej = np.zeros(len(y), bool)
    taus = {}
    for c in uc:
        te = np.array([x == c for x in clips])
        t = pick_reject_train(s[~te], y[~te], budget)
        taus[str(c)] = int(t)
        rej[te] = s[te] >= t
    good, bad = y == 1, y == 0
    tpr = float(rej[bad].mean())          # bad rejected
    tnr = float((~rej[good]).mean())      # good accepted
    tau_all = pick_reject_train(s, y, budget)
    return {"auroc": round(auroc_bad(y, s), 4),
            "auroc_ci95_clip_bootstrap": [round(v, 4) for v in boot_ci(y, s, clips)],
            "loco_balanced_accuracy": round(0.5 * (tpr + tnr), 4),
            "loco_reject_rate_bad": round(tpr, 4),
            "loco_false_reject_good": round(1 - tnr, 4),
            "loco_review_as_accept_ba": round(float((rej[bad].mean() + (~rej[good]).mean()) / 2), 4),
            "per_fold_tau": taus,
            "tau_frozen_on_all_training": int(tau_all),
            "score_stats": {"min": int(s.min()), "max": int(s.max()),
                            "mean": round(float(s.mean()), 3)},
            "n": int(len(y))}


def main() -> int:
    t0 = time.time()
    indego = load_npz("indego")
    v1 = v1_specs()
    # A4 (commit 250eb63) found FROZEN_STATIC harmful (drop = +0.050 BA) and the
    # QUALITY hard-reject override expensive in-domain; both are tested here.
    v1_no_frozen = [r for r in v1 if r["rule"] != "FROZEN_STATIC"]
    v1_no_frozen_noveto = [dict(r, strength="weak") for r in v1_no_frozen]
    candidates = {
        "C1_v1_uniform": (v1, 14),
        "C2_a1_fixed5": (A1_FIXED5, 9),
        "C3_merged_a1_plus_v1soft": (A1_FIXED5 + v1, 12),
        "C4_merged_drop_frozen_static": (A1_FIXED5 + v1_no_frozen, 12),
        "C5_merged_drop_frozen_no_veto": (A1_FIXED5 + v1_no_frozen_noveto, 12),
    }
    res = {}
    for name, (rules, tau) in candidates.items():
        ev = evaluate(rules, indego)
        res[name] = {"nominal_t_reject_ignored": tau, "n_rule_specs": len(rules),
                     "rules": [f'{r["rule"]}:{r["feature"]}' for r in rules], **ev}
        print(f"{name:26s} AUROC={ev['auroc']:.4f} "
              f"LOCO-BA={ev['loco_balanced_accuracy']:.4f} "
              f"LOCO-FRR={ev['loco_false_reject_good']:.3f} "
              f"tau*={ev['tau_frozen_on_all_training']}", flush=True)

    best = max(candidates, key=lambda k: res[k]["auroc"])
    best_rules = candidates[best][0]
    best_tau = res[best]["tau_frozen_on_all_training"]
    print("best in-domain candidate:", best, "tau =", best_tau, flush=True)

    cfg = {
        "name": "phys_rules_v2",
        "description": ("Deterministic physics rule model v2: fixed integer Q=4 "
                        "thresholds, fixed integer weights, no learned parameter in the "
                        "decision path, byte-identical receipts."),
        "version": 2,
        "selected_candidate": best,
        "score_rule": ("score = sum over rules of weight * (number of thresholds "
                       "crossed); REJECT if a QUALITY veto fires or score >= t_reject; "
                       "REVIEW if t_accept <= score < t_reject; else ACCEPT"),
        "quantisation": Q,
        "t_accept": 0,
        "t_reject": int(best_tau),
        "rules": best_rules,
        "domain_mode": {
            "absolute": ("thresholds_q are absolute values from IndEgo training; use "
                         "in-domain"),
            "requantile": ("LABEL-FREE domain adaptation: replace each feature value by "
                           "its within-target rank (uniform in (0,1), target's own "
                           "unlabelled values only), then the thresholds become the fixed "
                           "quantile levels [9900,9500,9000,7500] for high-is-bad and "
                           "[100,500,1000,2500] for low-is-bad. Rule list, directions, "
                           "weights and t_reject stay frozen from IndEgo."),
        },
        "derivation_summary": (
            "Rule list and directions: the a-priori product vocabulary (TOO_DARK, "
            "OVEREXPOSED, BLUR, FROZEN_STATIC, NO_ACTIVE_WORK, HANDS_NOT_VISIBLE, "
            "DUPLICATE, CODING_NOISE) plus SHAKY. Thresholds: k x a physical reference "
            "measured on TRAINING windows only (median skin fraction, DCT AC energy, "
            "global shift, HF share, motion energy), k in {0.75, 2.0}, rounded to an "
            "integer at Q=4. Weights: small integers derived by A1 on training clips. "
            "t_reject: chosen on training windows under a <=5% false-reject budget."),
        "evidence_provenance": {
            "C1": "eval/results/exp_phys/rule_table.json (Q1, PR #141); LOCO "
                  "reproduced independently at 0.8222 in LEAD_v2.json",
            "C2": "eval/results/exp_phys_dsh/A1_indomain_v2.json (commit 2ad71fd)",
            "merge": "eval/results/exp_phys_dsh/LEAD_freeze.json",
        },
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "rules"}
                    for k, v in res.items()},
        "limitations": (
            "IndEgo n=150 (100 good / 50 bad) over 7 clips; one clip can move AUROC by "
            ">0.05. Candidate families were shortlisted on the same pool (mild selection "
            "bias). OOD labels were never used for fitting, weighting, thresholding or "
            "selection."),
    }
    os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
    with open(CONFIG, "w") as f:
        json.dump(cfg, f, indent=1, sort_keys=True)
    out = {"experiment": "LEAD_freeze", "device": "CPU (RunPod A40 host)",
           "protocol": "in-domain frozen-model evaluation, IndEgo 150, 7 clips, clip-bootstrap B=2000 seed 0",
           "label_source": "human (manifest_gold.json, 100 good / 50 bad)",
           "command": "python3 eval/exp_phys_dsh/freeze_v2.py",
           "candidates": res, "selected": best,
           "frozen_config": os.path.relpath(CONFIG, REPO),
           "holdout_note": "the 108-window hold-out was NOT touched here",
           "runtime_s": round(time.time() - t0, 1)}
    with open(os.path.join(OUTDIR, "LEAD_freeze.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("wrote", CONFIG, "and LEAD_freeze.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
