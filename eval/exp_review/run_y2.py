#!/usr/bin/env python3
"""Y2 — reduce the PoC REVIEW share with decision-theoretic gating + stage-0.

Produces (deterministic; cached inputs only, no decode, no network):
  eval/results/exp_review/Y2.01.json   decision-theoretic gating sweep
  eval/results/exp_review/Y2.02.json   stage-0 triage + combined pipeline
  eval/results/exp_review/Y2.recommend.json  the recommended operating point

Data (read-only):
  $JADAL_BUNDLE/labels/manifest_gold.json          gold manifest (150 IndEgo)
  $JADAL_BUNDLE/caches/ensemble/oof_probs.npz      gold LOCO-OOF probs (p_ens)
  $JADAL_BUNDLE/labels/ood_labels.json             60 human OOD labels (n_bad=9)
  $JADAL_BUNDLE/caches/ensemble/ood_scores.json    frozen scores on the OOD-60
  $JADAL_BUNDLE/caches/gpu/features.json           210 windows: 150 gold + 60 synthetic
  /workspace/artifacts/windows_rescored.jsonl      1,546 Egocentric-10K (unlabelled)

Honesty: the OOD-60 is evaluation-only; nothing is fit on it.  The 1,546
Egocentric windows have NO human labels, so every number computed on them is
descriptive (review share / accept share), never accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.exp_review.gate import (  # noqa: E402
    QUALITY_RULES, SEED, auroc, clip_bootstrap_indices, cls_metrics,
    evaluate_gate, fit_temperature, gate, loco_temperature, quality_fires,
    wilson_ci, apply_temperature,
)

COVERAGES = [0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00]
COST_RATIOS = [1, 2, 5]
N_BOOT = 2000


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_inputs(bundle: Path, artifacts: Path):
    z = np.load(bundle / "caches/ensemble/oof_probs.npz", allow_pickle=True)
    gold = {
        "ids": np.array([str(x) for x in z["ids"]]),
        "y": z["y"].astype(int),
        "clip": np.array([str(x) for x in z["clip"]]),
        "p_ens": z["p_ens"].astype(float),
        "p_eg2": z["p_eg2"].astype(float),
        "p_pixel": z["p_pixel"].astype(float),
    }
    ood_lab = json.loads((bundle / "labels/ood_labels.json").read_text())["latest"]
    ood_sc = json.loads((bundle / "caches/ensemble/ood_scores.json").read_text())["scores"]
    ood = {
        "ids": np.array([s["id"] for s in ood_sc]),
        "y": np.array([1 if ood_lab[s["id"]]["verdict"] == "good" else 0
                       for s in ood_sc]),
        "p": np.array([float(s["usable_prob"]) for s in ood_sc]),
    }
    feats = json.loads((bundle / "caches/gpu/features.json").read_text())["windows"]
    rescored = []
    with open(artifacts / "windows_rescored.jsonl") as f:
        f.readline()  # header
        for line in f:
            rescored.append(json.loads(line))
    return gold, ood, feats, rescored


# --------------------------------------------------------------------------
def gating_sweep(gold, ood, rescored, calibrated_gold, calibrated_ood):
    """Full sweep on gold (LOCO) + OOD-60 (eval only) + descriptive 1546."""
    prov = {
        "gold_n": int(len(gold["y"])),
        "gold_label_source": "human (IndEgo gold, single labeller)",
        "gold_protocol": "cached clip-grouped LOCO-OOF p_ens from "
                         "caches/ensemble/oof_probs.npz (evaluation-only here)",
        "gold_auroc_measured": round(auroc(gold["p_ens"], gold["y"]), 4),
        "ood_n": int(len(ood["y"])),
        "ood_n_bad": int((ood["y"] == 0).sum()),
        "ood_label_source": "human (Egocentric-10K OOD-60, n_bad=9)",
        "ood_protocol": "frozen usable_prob from caches/ensemble/ood_scores.json; "
                        "EVALUATION ONLY, never used to fit anything",
        "ood_auroc_measured": round(auroc(ood["p"], ood["y"]), 4),
        "n_rescored": len(rescored),
        "rescored_label_source": "none (descriptive only)",
    }

    def sweep(p, y, pcal, tag):
        rows = []
        for r in COST_RATIOS:
            for c in COVERAGES:
                m, _, _, _ = evaluate_gate(p, y, c, r)
                mc, _, _, _ = evaluate_gate(pcal, y, c, r)
                m["variant"] = "raw"
                m2 = dict(mc)
                m2["variant"] = "loco_temperature"
                rows.append(m)
                rows.append(m2)
        return rows

    gold_rows = sweep(gold["p_ens"], gold["y"],
                      calibrated_gold, "gold")
    ood_rows = sweep(ood["p"], ood["y"], calibrated_ood, "ood")

    # descriptive assign on the 1,546 unlabelled windows
    s = np.array([float(r["score"]) for r in rescored])
    desc = []
    for r in COST_RATIOS:
        pstar = r / (r + 1.0)
        marg = np.abs(s - pstar)
        order = np.argsort(-marg, kind="mergesort")
        for c in COVERAGES:
            k = int(round(c * len(s)))
            dec = order[:k]
            rev = order[k:]
            acc = int((s[dec] >= pstar).sum())
            rej = int(len(dec) - acc)
            desc.append({
                "cost_ratio_fa_fr": r, "p_star": round(pstar, 4),
                "coverage": round(c, 4), "review": round(1 - c, 4),
                "n_accept": acc, "n_reject": rej, "n_review": len(rev),
                "review_share": round(len(rev) / len(s), 4),
                "review_share_wilson": wilson_ci(len(rev), len(s)),
            })

    # clip-bootstrap CI for the recommended gold config (cost 1:1, cov 0.45)
    rng = np.random.default_rng(SEED)
    rec_c, rec_r = 0.45, 1
    bas = []
    for _ in range(N_BOOT):
        idx = clip_bootstrap_indices(gold["clip"], rng)
        m, _, _, _ = evaluate_gate(gold["p_ens"][idx], gold["y"][idx],
                                   rec_c, rec_r)
        if m["TP"] + m["FN"] and m["TN"] + m["FP"]:
            bas.append(m["balanced_accuracy"])
    bas = np.array(bas)
    ba_ci = [round(float(np.percentile(bas, 2.5)), 4),
             round(float(np.percentile(bas, 97.5)), 4)]

    return {
        "id": "Y2.01",
        "title": "Decision-theoretic gating: review-vs-error sweep",
        "n": int(len(gold["y"])),
        "n_ood": int(len(ood["y"])),
        "label_source": "human (gold LOCO) + human (OOD-60, eval only) + none (1,546)",
        "protocol": ("Selective classification: decide the `coverage` fraction "
                     "with the largest cost-margin |p - p*|, review the rest; "
                     "p* = C_fa/(C_fa+C_fr). Gold uses cached LOCO-OOF p_ens; "
                     "OOD-60 evaluation-only; 1,546 descriptive."),
        "seed": SEED,
        "device": "CPU (cached signals only)",
        "provenance": prov,
        "gold_sweep": gold_rows,
        "ood_sweep": ood_rows,
        "rescored_descriptive": desc,
        "calibration": {
            "method": "temperature scaling on the logit (1 param, monotone)",
            "global_T_gold": round(fit_temperature(gold["p_ens"], gold["y"]), 3),
            "loco_note": "per-clip T fit on the other 6 clips only (honest); "
                         "T is unstable across clips (small n)",
            "ood_note": "the gold global T is applied to OOD probs as a "
                        "transfer test (NOT fit on OOD)",
        },
        "recommended_gold_ba_ci": ba_ci,
        "recommended_config": {"coverage": rec_c, "review": 1 - rec_c,
                               "cost_ratio_fa_fr": rec_r},
        "command": "python3 eval/exp_review/run_y2.py",
    }


# --------------------------------------------------------------------------
def stage0_and_pipeline(gold, ood, feats, rescored):
    # --- validate the rules on the synthetic degradation set ---
    synth = Counter()
    synth_tot = Counter()
    for k, v in feats.items():
        g = v.get("degradation")
        if g is None:
            continue
        synth_tot[g] += 1
        if quality_fires(v["full"]):
            synth[g] += 1
    synth_table = {g: {"fired": synth[g], "n": synth_tot[g]}
                   for g in sorted(synth_tot)}

    # --- validate on the 150 human gold windows ---
    gold_ids = set(gold["ids"].tolist())
    ylab = {str(i): int(v) for i, v in zip(gold["ids"], gold["y"])}
    fired_gold = []
    for k, v in feats.items():
        if v.get("degradation") is not None or k not in gold_ids:
            continue
        rr = quality_fires(v["full"])
        if rr:
            fired_gold.append({"id": k, "human_y": ylab[k], "codes": rr,
                               "features": {f: round(v["full"][f], 4)
                                            for f in ("luma_mean", "dark_frac",
                                                      "bright_frac", "frozen_frac",
                                                      "zncc_self", "motion_energy")}})
    n_fired = len(fired_gold)
    n_bad_fired = sum(1 for d in fired_gold if d["human_y"] == 0)
    gold_val = {
        "n_gold": int(len(gold["y"])),
        "n_fired": n_fired,
        "n_bad_fired": n_bad_fired,
        "precision_bad": round(n_bad_fired / n_fired, 4) if n_fired else None,
        "recall_bad": round(n_bad_fired / int((gold["y"] == 0).sum()), 4),
        "note": "n is tiny (rules fire on 2%% of gold): treat as indicative only; "
                "the funnel certification found NO safe quality-reject rule on gold",
        "items": fired_gold,
    }

    # --- apply to the 1,546 windows (cached features only) ---
    QR = Counter()
    n_fire = 0
    fire_idx = []
    for i, r in enumerate(rescored):
        rr = quality_fires(r["features"])
        if rr:
            n_fire += 1
            fire_idx.append(i)
        for c in rr:
            QR[c] += 1
    s = np.array([float(r["score"]) for r in rescored])
    n = len(s)
    fire = np.zeros(n, bool)
    fire[fire_idx] = True

    # pipeline review% after each lever, at the recommended coverage
    cov, r = 0.45, 1
    pstar = r / (r + 1.0)
    order = np.argsort(-np.abs(s - pstar), kind="mergesort")
    k = int(round(cov * n))
    review = order[k:]
    fire_in_review = int(fire[review].sum())
    review_after_gate = len(review)
    review_after_stage0 = len(review) - fire_in_review

    # lever (c): calibration — no effect on review share (monotone rank
    # transform); it only moves the accept/reject boundary. Report honestly.
    pipeline = {
        "base_frozen_policy": {
            "rule": "score>=0.75 accept / <=0.25 reject / else review (symmetric band)",
            "n_review": 1167, "review_share": 0.7549,
            "source": "eval/results/exp_rescore/E8.01.json",
        },
        "a_gating_alone": {
            "rule": "decide top-45% by cost-margin, review rest",
            "n_review": review_after_gate,
            "review_share": round(review_after_gate / n, 4),
        },
        "b_plus_stage0": {
            "rule": "a) + deterministic quality REJECT on review windows",
            "stage0_fired_in_review": fire_in_review,
            "n_review": review_after_stage0,
            "review_share": round(review_after_stage0 / n, 4),
        },
        "c_plus_calibration": {
            "rule": "b) + gold-LOCO temperature on the score",
            "n_review": review_after_stage0,
            "review_share": round(review_after_stage0 / n, 4),
            "note": "monotone transform: review share unchanged; it only shifts "
                    "the accept/reject split (measured on gold, not on the 1,546)",
        },
    }

    return {
        "id": "Y2.02",
        "title": "Deterministic stage-0 triage + combined review pipeline",
        "n": n,
        "label_source": ("none for the 1,546 (descriptive); synthetic degradation "
                         "labels for rule derivation; human gold for validation"),
        "protocol": ("Stage-0 rules over cached descriptor features, fixed a "
                     "priori, validated on 6-window-per-degradation synthetics "
                     "and on the 150 human gold windows; applied to the 1,546 "
                     "review windows. No decode."),
        "seed": SEED,
        "rules": [{"code": c, "feature": f, "op": o, "threshold": t}
                  for (c, f, o, t) in QUALITY_RULES],
        "synthetic_validation": synth_table,
        "gold_validation": gold_val,
        "rescored_application": {
            "n": n, "n_fired": n_fire,
            "fire_share": round(n_fire / n, 4),
            "code_counts": dict(QR),
        },
        "pipeline": pipeline,
        "honesty": (
            "Stage-0 fires on only ~2.5%% of the 1,546 windows and the fired "
            "windows have NO labels, so the review reduction it buys is a "
            "descriptive count, not a measured accuracy. On the 150 gold "
            "windows the rules fire 3 times (precision_bad 0.67, n=3): the "
            "funnel's own certification found no safe quality-reject rule."),
        "command": "python3 eval/exp_review/run_y2.py",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path,
                    default=Path(os.environ.get("JADAL_BUNDLE",
                                                "/workspace/jadal_pod_bundle")))
    ap.add_argument("--artifacts", type=Path,
                    default=Path("/workspace/artifacts"))
    ap.add_argument("--out", type=Path,
                    default=ROOT / "eval/results/exp_review")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    gold, ood, feats, rescored = load_inputs(args.bundle, args.artifacts)

    T = fit_temperature(gold["p_ens"], gold["y"])
    cal_gold = loco_temperature(gold["p_ens"], gold["y"], gold["clip"])
    cal_ood = apply_temperature(ood["p"], T)  # transfer gold T (not fit on OOD)
    print(f"gold n={len(gold['y'])} AUROC={auroc(gold['p_ens'], gold['y']):.4f} "
          f"Tglobal={T:.3f}; OOD n={len(ood['y'])} AUROC={auroc(ood['p'], ood['y']):.4f}")

    r1 = gating_sweep(gold, ood, rescored, cal_gold, cal_ood)
    r2 = stage0_and_pipeline(gold, ood, feats, rescored)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for obj in (r1, r2):
        obj["generated_utc"] = ts

    (args.out / "Y2.01.json").write_text(json.dumps(r1, indent=1) + "\n")
    (args.out / "Y2.02.json").write_text(json.dumps(r2, indent=1) + "\n")

    # compact recommendation
    rec = {
        "id": "Y2.recommend",
        "config": {"gating_coverage": 0.45, "review_share_target": 0.55,
                   "cost_ratio_fa_fr": 1, "stage0": True, "temperature": False},
        "gold_ba_raw": next(m["balanced_accuracy"] for m in r1["gold_sweep"]
                            if m["cost_ratio_fa_fr"] == 1 and m["coverage"] == 0.45
                            and m["variant"] == "raw"),
        "gold_ba_loco_cal": next(m["balanced_accuracy"] for m in r1["gold_sweep"]
                                 if m["cost_ratio_fa_fr"] == 1 and m["coverage"] == 0.45
                                 and m["variant"] == "loco_temperature"),
        "gold_ba_ci": r1["recommended_gold_ba_ci"],
        "ood_ba_eval_only": next(m["balanced_accuracy"] for m in r1["ood_sweep"]
                                 if m["cost_ratio_fa_fr"] == 1 and m["coverage"] == 0.45
                                 and m["variant"] == "raw"),
        "rescored_review_share_gating": 0.55,
        "rescored_review_share_stage0": r2["pipeline"]["b_plus_stage0"]["review_share"],
        "note": "residual OOD reliability ~chance; see Y2.01/02.",
        "generated_utc": ts,
        "command": "python3 eval/exp_review/run_y2.py",
    }
    (args.out / "Y2.recommend.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
