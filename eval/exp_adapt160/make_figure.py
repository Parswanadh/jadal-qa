#!/usr/bin/env python3
"""O4 figure: OOD-160 AUROC with factory-grouped CI, q1 vs q2, gain vs frozen.

Run: python3 eval/exp_adapt160/make_figure.py
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
OUT = os.path.join(REPO, "eval", "results", "exp_adapt160")
FIGS = os.path.join(REPO, "docs", "experiments", "exp_adapt160", "images")


def main():
    d = json.load(open(os.path.join(OUT, "O4.01.json")))
    blocks = d["blocks"]
    labels = [b["recipe"].replace(" (", "\n(") for b in blocks]
    au = [b["ood_auroc"] for b in blocks]
    ci = [b["ood_auroc_ci95_factory_grouped_honest"] for b in blocks]
    lo = [a - c[0] for a, c in zip(au, ci)]
    hi = [c[1] - a for a, c in zip(au, ci)]
    colors = ["#2a6", "#2a6", "#a83", "#a83", "#777"]

    os.makedirs(FIGS, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    ax.barh(range(len(blocks)), au, color=colors, xerr=[lo, hi], capsize=3)
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("OOD-160 AUROC")
    ax.set_title("160 OOD labels · factory-grouped 95% CI (honest)")

    ax = axes[1]
    w = 0.38
    ys = np.arange(len(blocks))
    q1 = [b["ood_batch_split"]["q1"]["auroc"] for b in blocks]
    q2 = [b["ood_batch_split"]["q2"]["auroc"] for b in blocks]
    ax.barh(ys - w / 2, q1, height=w, color="#48c", label="q1 (n=60, bad=9)")
    ax.barh(ys + w / 2, q2, height=w, color="#c84", label="q2 (n=100, bad=13)")
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([b["recipe"].split(" ")[0] for b in blocks], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("OOD AUROC (per-batch z-score)")
    ax.set_title("batch q1 vs q2")
    ax.legend(fontsize=7)

    ax = axes[2]
    gains = d["gain_vs_frozen_paired_factory_grouped_bootstrap"]
    names = list(gains.keys())
    m = [gains[k]["mean"] for k in names]
    gci = [gains[k]["ci95"] for k in names]
    glo = [a - c[0] for a, c in zip(m, gci)]
    ghi = [c[1] - a for a, c in zip(m, gci)]
    ax.barh(range(len(names)), m, color="#36a", xerr=[glo, ghi], capsize=3)
    ax.axvline(0.0, color="k", ls="--", lw=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.split(")")[0] + ")" for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC(recipe) − AUROC(frozen), same resamples")
    ax.set_title("paired factory-grouped gain vs frozen (0.566)")

    fig.tight_layout()
    p = os.path.join(FIGS, "o4_adapt160.png")
    fig.savefig(p, dpi=140)
    print("figure ->", p)


if __name__ == "__main__":
    main()
