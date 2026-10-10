#!/usr/bin/env python3
"""A4 figure: decided balanced-accuracy (and accuracy) vs coverage."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(REPO, "eval", "results", "exp_acc", "A4.01.json")
OUT = os.path.join(REPO, "docs", "experiments", "exp_acc", "images",
                   "a4_coverage_accuracy.png")

d = json.load(open(RES))
regimes = d["regimes"]
panels = [("in_domain", "In-domain gold 150 (LOCO, clip-grouped)"),
          ("ood160", "OOD-160 (factory-grouped)"),
          ("holdout108", "Hold-out 108 (evaluate only)")]
colors = {"frozen": "#1f77b4", "adapted": "#d62728"}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, (name, title) in zip(axes, panels):
    for sc in ("frozen", "adapted"):
        r = regimes[name][sc]
        env = r["frontier_envelope"]
        if not env:
            continue
        cov = [e["coverage"] for e in env]
        ba = [e["ba"] for e in env]
        acc = [e["acc"] for e in env]
        ax.plot(cov, ba, "-", color=colors[sc], lw=2,
                label=f"{sc} (AUROC {r['auroc']:.3f})")
        ax.plot(cov, acc, "--", color=colors[sc], lw=1, alpha=0.55)
        b = r["max_coverage_at_ba95"]
        if b:
            ax.plot([b["coverage"]], [b["ba"]], "o", color=colors[sc], ms=7,
                    mec="k", mew=0.6)
    ax.axhline(0.95, color="k", ls=":", lw=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("coverage (fraction decided)")
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("decided balanced-accuracy (solid) / accuracy (dashed)")
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("A4 — decided balanced-accuracy vs coverage (marker = max coverage at BA>=0.95)", fontsize=11)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
