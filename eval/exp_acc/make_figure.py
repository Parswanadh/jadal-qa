#!/usr/bin/env python3
"""Figure for N1: error-vs-labels curve + model-shape AUROC with grouped CIs."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(REPO, "eval", "results", "exp_acc")
IMG = os.path.join(REPO, "docs", "experiments", "exp_acc", "images")
os.makedirs(IMG, exist_ok=True)

n101 = json.load(open(os.path.join(RES, "N1.01.json")))
n103 = json.load(open(os.path.join(RES, "N1.03.json")))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

# --- panel A: error-vs-labels curve (AUROC) ---
colors = {"equal-weight mean (label-free)": "tab:blue",
          "median signed-z (label-free)": "tab:green",
          "logistic C=0.1 (uses k labels)": "tab:red",
          "top-12 selection (uses k labels)": "tab:orange"}
for lab, rows in n103["error_vs_labels_curve"].items():
    ks = [r["k"] for r in rows]
    au = [r["auroc"] for r in rows]
    ax1.plot(ks, au, "o-", color=colors.get(lab, "gray"), label=lab)
ax1.axhline(0.781, ls="--", color="tab:blue", alpha=0.5)
ax1.axhline(0.5, ls=":", color="gray")
ax1.set_xlabel("k = human OOD labels used (grouped CV)")
ax1.set_ylabel("OOD AUROC")
ax1.set_title("A. Error vs labels (factory-grouped CV)")
ax1.set_ylim(0.5, 0.85)
ax1.legend(fontsize=8, loc="lower right")
ax1.grid(alpha=0.3)

# --- panel B: model shape AUROC + grouped CI ---
blocks = n101["model_shape"]
names = [b["recipe"] for b in blocks]
au = [b["ood_auroc"] for b in blocks]
lo = [b["ood_auroc_ci95_factory_grouped"][0] for b in blocks]
hi = [b["ood_auroc_ci95_factory_grouped"][1] for b in blocks]
yy = np.arange(len(names))[::-1]
ax2.errorbar(au, yy, xerr=[np.array(au) - np.array(lo), np.array(hi) - np.array(au)],
             fmt="o", color="tab:purple", capsize=3)
ax2.axvline(0.781, ls="--", color="tab:blue", alpha=0.6, label="baseline 0.781")
ax2.set_yticks(yy)
ax2.set_yticklabels(names, fontsize=8)
ax2.set_xlabel("OOD AUROC (factory-grouped 95% CI)")
ax2.set_title("B. Model shape vs equal-weight mean")
ax2.set_xlim(0.45, 1.0)
ax2.legend(fontsize=8, loc="lower left")
ax2.grid(alpha=0.3)

fig.suptitle("N1 — OOD accuracy on 160 human-labelled Egocentric-10K windows "
             "(22 bad / 47 factories)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(IMG, "n1_acc_adapt.png")
fig.savefig(out, dpi=140)
print("wrote", out)
