#!/usr/bin/env python3
"""F30 figure: per-pixel features 26 -> 30 (in-domain LOCO + OOD + hold-out).

Run: OMP_NUM_THREADS=2 python3 eval/exp_feat30/make_figure.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(REPO, "eval", "results", "exp_feat30", "F30.01.json")
OUTDIR = os.path.join(REPO, "docs", "reports", "images")


def main():
    r = json.load(open(RES))
    os.makedirs(OUTDIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    C26, C30, CACC = "#8a8a8a", "#2b6cb0", "#c05621"

    # --- Panel A: in-domain LOCO ---
    ax = axes[0]
    loco = r["indomain_loco"]
    vals = [loco["f26"]["auroc"], loco["f30"]["auroc"]]
    ci = [loco["f26"]["auroc_ci95_clip"], loco["f30"]["auroc_ci95_clip"]]
    err = [[vals[i] - ci[i][0] for i in range(2)], [ci[i][1] - vals[i] for i in range(2)]]
    ax.bar([0, 1], vals, color=[C26, C30], yerr=err, capsize=5, width=0.6)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["26 feat", "30 feat"])
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("AUROC")
    ax.set_title(f"In-domain LOCO (7 clips, n=150)\n26→30: {vals[1]-vals[0]:+.4f} (ns)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)

    # --- Panel B: OOD-160 adapted z-score ---
    ax = axes[1]
    z = r["ood_adapted_zscore"]
    labels = ["26 feat\n(full)", "30 feat\n(full)", "frozen\nens.", "all14\nbase",
              "new4\nonly"]
    vals = [z["f26"]["ood_auroc"], z["f30"]["ood_auroc"],
            r["frozen_ensemble_v1_ood160"]["ood_auroc"],
            r["baseline_reproduce_all14_zscore"]["ood_auroc"],
            r["adapted_zscore_new4_only"]["ood_auroc"]]
    cis = [z["f26"]["ood_ci95_grouped"], z["f30"]["ood_ci95_grouped"],
           r["frozen_ensemble_v1_ood160"]["ood_ci95_grouped"],
           r["baseline_reproduce_all14_zscore"]["ood_ci95_grouped"],
           r["adapted_zscore_new4_only"]["ood_ci95_grouped"]]
    cols = [C26, C30, "#999999", "#999999", CACC]
    err = [[max(0, vals[i] - cis[i][0]) for i in range(len(vals))],
           [max(0, cis[i][1] - vals[i]) for i in range(len(vals))]]
    ax.bar(range(len(vals)), vals, color=cols, yerr=err, capsize=4, width=0.65)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.1, 1.0); ax.set_ylabel("AUROC")
    ax.set_title("OOD-160 adapted per-domain z-score\n(factory-grouped 95% CI)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)

    # --- Panel C: hold-out 108 (scored once) ---
    ax = axes[2]
    vals = [z["f26"]["holdout_auroc"], z["f30"]["holdout_auroc"],
            r["baseline_reproduce_all14_zscore"]["holdout_auroc"],
            r["adapted_zscore_new4_only"]["holdout_auroc"]]
    cis = [z["f26"]["holdout_ci95_grouped"], z["f30"]["holdout_ci95_grouped"],
           r["baseline_reproduce_all14_zscore"]["holdout_ci95_grouped"],
           r["adapted_zscore_new4_only"]["holdout_ci95_grouped"]]
    labels = ["26 feat", "30 feat", "all14\nbase", "new4\nonly"]
    cols = [C26, C30, "#999999", CACC]
    err = [[max(0, vals[i] - cis[i][0]) for i in range(len(vals))],
           [max(0, cis[i][1] - vals[i]) for i in range(len(vals))]]
    ax.bar(range(len(vals)), vals, color=cols, yerr=err, capsize=4, width=0.65)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.1, 1.0); ax.set_ylabel("AUROC")
    ax.set_title("Hold-out 108 (92 good/16 bad, 44 factories)\nscored once, no tuning")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle("F30 — per-pixel feature vector 26 → 30 (4 Z3 transferable invariants)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "exp_feat30.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
