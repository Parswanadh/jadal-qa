#!/usr/bin/env python3
"""Figure for C1.01: confirmatory one-shot on the 108 round-3 hold-out."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(REPO, "eval", "results", "exp_confirm")
IMG = os.path.join(REPO, "docs", "reports", "images")
os.makedirs(IMG, exist_ok=True)


def main():
    d = json.load(open(os.path.join(RES, "C1.01.json")))
    blocks = d["blocks"]
    labels = ["O4 (a)\nZ3-6", "O4 (b)\nE5-8+Z3-6", "sharpness-3\nhead", "frozen\nensemble"]
    short = ["(a)", "(b)", "(c)", "(d)"]
    aucs, los, his = [], [], []
    for b in blocks:
        ci = b["ood_auroc_ci95_factory_grouped_honest"]
        aucs.append(b["ood_auroc"])
        los.append(ci[0]); his.append(ci[1])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(labels))
    err = [np.array(aucs) - np.array(los), np.array(his) - np.array(aucs)]
    ax[0].bar(x, aucs, yerr=err, capsize=4,
              color=["#3b6fb0", "#1f4e8c", "#d98a2b", "#7a7a7a"])
    ax[0].axhline(0.5, ls="--", c="k", lw=1)
    ax[0].axhline(0.781, ls=":", c="r", lw=1.2)
    ax[0].text(0.02, 0.785, "O4 (b) pooled-160 AUROC 0.781", color="r", fontsize=8,
               transform=ax[0].get_yaxis_transform())
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel("AUROC (factory-grouped 95% CI)")
    ax[0].set_ylim(0.35, 0.95)
    ax[0].set_title("One-shot on 108 round-3 hold-out")

    # confusion for the headline recipe (b) and frozen
    names = []
    confs = []
    for b in blocks:
        for k, v in b.items():
            if k.startswith("confusion"):
                names.append(f"{short[blocks.index(b)]} {b['recipe'].split(')')[1].strip()[:22]}")
                confs.append(v)
    M = np.array([[c["tp"], c["fn"]], [c["fp"], c["tn"]] for c in confs], float)
    ax[1].imshow(np.ones((2, 2)), cmap="Greys", vmin=0, vmax=1, alpha=0)
    for i, (nm, c) in enumerate(zip(names, confs)):
        y0 = i * 1.15
        for (r, cc, txt) in ((0, 0, c["tp"]), (0, 1, c["fn"]), (1, 0, c["fp"]), (1, 1, c["tn"])):
            ax[1].add_patch(plt.Rectangle((cc, y0 + (1 - r)), 1, 1,
                                          fill=True, alpha=0.25,
                                          color=("#2e7d32" if (r == cc) else "#c62828")))
            ax[1].text(cc + 0.5, y0 + (1 - r) + 0.5, str(txt), ha="center", va="center",
                       fontsize=11)
        ax[1].text(2.15, y0 + 1.0, nm, fontsize=8, va="center")
    ax[1].set_xlim(0, 4.6); ax[1].set_ylim(0, 1.15 * len(names))
    ax[1].set_xticks([0.5, 1.5]); ax[1].set_xticklabels(["pred good", "pred bad"], fontsize=8)
    ax[1].set_yticks([])
    ax[1].set_title("Confusion (good/bad, thr)")
    fig.tight_layout()
    out = os.path.join(IMG, "exp_confirm.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
