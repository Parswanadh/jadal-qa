#!/usr/bin/env python3
"""Render the few-shot adaptation curve figure from O2.02.json.

docs/experiments/exp_fewshot/images/adaptation_curve.png
Run: python3 eval/exp_fewshot/make_figure.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from eval.exp_fewshot import o2lib as L  # noqa: E402

IMG = os.path.join(L.REPO, "docs", "experiments", "exp_fewshot", "images")
COLORS = {"frozen_platt_calibrator": "#1f77b4", "two_view_logistic": "#ff7f0e",
          "pixel26_logistic": "#2ca02c"}
LABELS = {"frozen_platt_calibrator": "Platt on logit(frozen prob)",
          "two_view_logistic": "logistic on [p_eg2, p_pixel]",
          "pixel26_logistic": "logistic on 26 pixel feats"}


def main():
    d = json.load(open(os.path.join(L.OUT, "O2.02.json")))
    arms = d["arms"]
    base_err = arms["frozen_usable_prob_0labels"]["curve"][0]["error_mean"]
    base_auroc = arms["frozen_usable_prob_0labels"]["curve"][0]["auroc_mean"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for nm, col in COLORS.items():
        c = [x for x in arms[nm]["curve"] if x["k_target"] > 0]
        ks = [x["k_target"] for x in c]
        au = [x["auroc_mean"] for x in c]
        er = [x["error_mean"] for x in c]
        sd = [x["auroc_std"] or 0 for x in c]
        ax[0].errorbar(ks, au, yerr=sd, marker="o", color=col, label=LABELS[nm], capsize=3)
        ax[1].plot(ks, er, marker="s", color=col, label=LABELS[nm])
    ax[0].axhline(0.5, ls=":", c="k", lw=1)
    ax[0].axhline(base_auroc, ls="--", c="grey", lw=1,
                  label=f"frozen, 0 labels ({base_auroc:.2f})")
    ax[1].axhline(base_err, ls="--", c="grey", lw=1,
                  label=f"frozen, 0 labels ({base_err:.2f})")
    ax[0].set_xlabel("human labels k"); ax[0].set_ylabel("OOD AUROC")
    ax[0].set_title("AUROC vs labels (factory-grouped)")
    ax[1].set_xlabel("human labels k"); ax[1].set_ylabel("error = 1 - balanced accuracy @0.5")
    ax[1].set_title("Error vs labels")
    for a in ax:
        a.set_xticks([0, 10, 20, 40, 80, 160]); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("Few-shot adaptation on 160 human OOD labels "
                 "(k=160 = grouped-CV OOF; k<160 = grouped subsampling)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(IMG, exist_ok=True)
    p = os.path.join(IMG, "adaptation_curve.png")
    fig.savefig(p, dpi=140)
    print("wrote", p)


if __name__ == "__main__":
    main()
