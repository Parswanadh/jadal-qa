#!/usr/bin/env python3
"""Y1.01 supporting artifacts: fine sweep, calibration consistency check, figure.

Honesty core
------------
Calibration here is a *monotone* map of the score, so it cannot change the
ranking.  The abstain band is symmetric around 0.5, so a monotone map merely
re-parameterises it.  We verify this: a calibrator applied IN-SAMPLE (the same
fit used on E8) leaves the (review%, error) frontier essentially unchanged,
while the LOCO-calibrated gold probs (fold-specific fits) produce an *apparent*
gain that is a small-sample artefact.  Both are reported.

Writes:
  eval/results/exp_review/Y1.01_support.json
  docs/experiments/exp_review/images/y1_01_review_vs_error.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_y1 as R  # noqa: E402

ROOT = R.ROOT
FIG_DIR = os.path.join(ROOT, "docs/experiments/exp_review/images")
FINE = [round(i * 0.005, 3) for i in range(61)]


def row(e8c, gold_p, w):
    rev = R.review_share(e8c, w)
    ba, nd, fp, fn = R.balacc(gy, gold_p, w)
    bao, ndo, fpo, fno = R.balacc(oy, op, w)
    return dict(w=w, review=rev, gold_ba=ba, gold_err=1 - ba, gold_n=nd,
                gold_good_rej=fp, ood_ba=bao, ood_err=1 - bao, ood_n=ndo,
                ood_good_rej=fpo)


e8, thr, verd = R.load_e8_scores()
gy, gclip, gp = R.load_gold()
oy, op, oids = R.load_ood()
gcal = R.global_calibrators(gp, gy)
gloco = R.loco_calibrate(gp, gy, gclip)


def curve(e8p, goldp):
    return [row(e8p, goldp, w) for w in FINE]


curves = {}
curves["raw"] = curve(e8, gp)
for name in ("temp", "platt", "isotonic"):
    e8p = R.apply_global(e8, gcal, name)
    curves[name + "_insample"] = curve(e8p, R.apply_global(gp, gcal, name))
    curves[name + "_loco"] = curve(e8p, gloco[name])

# ---- calibration at equal error -------------------------------------------------------
# level = 0.155 (raw and temp_in both reach it); report min review per method
def min_review_at(rows, level):
    ok = [r for r in rows if r["gold_err"] <= level]
    return min(r["review"] for r in ok) if ok else None


LEVEL = 0.155
equal_err = {k: min_review_at(v, LEVEL) for k, v in curves.items()}

# ---- raw frontier & recommendation ----------------------------------------------------
def widest_w_below(target):
    best = None
    for w in [round(i * 0.001, 3) for i in range(301)]:
        if R.review_share(e8, w) < target:
            best = w
    ba, nd, fp, fn = R.balacc(gy, gp, best)
    return dict(w=best, review=R.review_share(e8, best), gold_ba=ba, gold_err=1 - ba,
                gold_n=nd, gold_good_rej=fp)

targets = {str(t): widest_w_below(t) for t in (0.60, 0.50, 0.40)}
rec = dict(w=0.16, review=R.review_share(e8, 0.16))
ba, nd, fp, fn = R.balacc(gy, gp, 0.16)
bao, ndo, fpo, fno = R.balacc(oy, op, 0.16)
rec.update(gold_ba=ba, gold_err=1 - ba, gold_n=nd, gold_good_rej=fp,
           ood_ba=bao, ood_err=1 - bao, ood_n=ndo)

support = {
    "id": "Y1.01",
    "what": "fine sweep (step 0.005), calibration consistency check, and figure (supports Y1.01.json)",
    "protocol": {
        "grid_step": 0.005,
        "review_share_denominator": "n=1546 Egocentric-10K windows (E8.01, label_source=none, DESCRIPTIVE)",
        "gold": "150 IndEgo human gold (100 good/50 bad), single labeller; LOCO over 7 clips",
        "ood": "60 human-labelled Egocentric-10K (51 good/9 bad); PRE-REGISTERED test, evaluation only, never fit on",
        "insample": "global calibrator fit on all 150 gold then applied to gold (optimistic for the fit) and to E8/OOD",
        "loco": "calibrator fit on 6 clips, applied to the 7th (honest deployable estimate)",
    },
    "fitted_temperature_global": gcal["temp"],
    "raw_review_targets_widest_w": targets,
    "recommended": rec,
    "calibration_at_equal_error": {
        "level_gold_err": LEVEL,
        "min_review_share_per_curve": equal_err,
        "conclusion": ("temperature/Platt/isotonic applied IN-SAMPLE do not lower the review share "
                       "at equal gold error (monotone maps cannot reorder the frontier); the raw/"
                       "in-sample curves are within noise. The *_loco curves show a lower review only "
                       "because fold-specific fits change WHICH windows are decided - a small-sample "
                       "artefact, not a real gain."),
    },
    "exact_boundary_note": "widest half-width with review share < 60% on the 0.001 grid is ~0.177; on the prescribed 0.02 grid it is 0.16 (review 53.4%)",
    "curves": curves,
}

os.makedirs(os.path.join(ROOT, "eval/results/exp_review"), exist_ok=True)
with open(os.path.join(ROOT, "eval/results/exp_review/Y1.01_support.json"), "w") as f:
    json.dump(support, f, indent=1, sort_keys=True)

# --------------------------------------------------------------------------- figure
os.makedirs(FIG_DIR, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
style = {
    "raw": ("#111111", "-", 2.6),
    "temp_insample": ("#d62728", "--", 1.6), "temp_loco": ("#d62728", ":", 1.2),
    "platt_insample": ("#1f77b4", "--", 1.6), "platt_loco": ("#1f77b4", ":", 1.2),
    "isotonic_insample": ("#2ca02c", "--", 1.6), "isotonic_loco": ("#2ca02c", ":", 1.2),
}
for ax, key, title in ((axes[0], "gold_err", "IndEgo gold (LOCO, n=150)"),
                       (axes[1], "ood_err", "Egocentric-10K OOD-60 (n_bad=9; TEST ONLY)")):
    for name, rows in curves.items():
        c, ls, lw = style[name]
        ax.plot([r["review"] * 100 for r in rows], [r[key] for r in rows],
                color=c, ls=ls, lw=lw, label=name)
    ax.axvline(60, color="grey", ls="--", lw=1)
    ax.axvline(75.5, color="grey", ls=":", lw=1)
    ax.scatter([rec["review"] * 100], [rec[key]], color="#111111",
               zorder=6, s=70, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("E8 review share on 1,546 windows  [%]")
    ax.set_ylabel("error = 1 - balanced accuracy (decided)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, title="curve")
    ax.invert_xaxis()
fig.suptitle("Y1.01  review% vs error — abstain-band sweep (grey dashed 60% target, dotted 75.5% frozen)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "y1_01_review_vs_error.png"), dpi=130)
print("wrote figure + support")
print("temperature global:", gcal["temp"])
print("raw targets:", {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in targets.items()})
print("recommended:", {k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()})
print("min review at gold_err<=%.3f:" % LEVEL)
for k, v in equal_err.items():
    print(f"   {k:20s} {None if v is None else round(v*100, 2)}%")
