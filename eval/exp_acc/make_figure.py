#!/usr/bin/env python3
"""A1 figure: capacity sweep + ensembling AUROC, in-domain (LOCO) vs OOD (grouped)."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RES = os.path.join(ROOT, "eval", "results", "exp_acc")
OUT = os.path.join(ROOT, "docs", "reports", "images")
os.makedirs(OUT, exist_ok=True)

cap = json.load(open(os.path.join(RES, "A1.01.json")))
stack = json.load(open(os.path.join(RES, "A1.02.json")))
reg = json.load(open(os.path.join(RES, "A1.03.json")))

# capacity: best AUROC per model class per domain
models = ["logistic", "gbt", "mlp", "svm"]
best = {"gold": {}, "ood": {}}
for dom in ("gold", "ood"):
    for m in models:
        vals = [r["auroc"] for k, r in cap["domains"][dom]["rows"].items()
                if r["model"] == m and r["auroc"] is not None]
        best[dom][m] = max(vals) if vals else 0.0
    best[dom]["frozen_eg2"] = next(r["auroc"] for k, r in cap["domains"][dom]["rows"].items()
                                   if k == "baseline|eg2_score")
    best[dom]["frozen_2view"] = next(r["auroc"] for k, r in cap["domains"][dom]["rows"].items()
                                     if k == "baseline|frozen_two_view")

combos = ["rank_pixel+eg2", "rank_pixel+eg2+z3", "stack_meta_logistic5",
          "stack_meta_pixel_eg2_z3"]
comb = {"gold": {}, "ood": {}}
for dom in ("gold", "ood"):
    for c in combos:
        comb[dom][c] = stack["domains"][dom]["rows"][c]["auroc"]

labels = ["logistic", "gbt", "mlp", "svm", "frozen_eg2", "frozen_2view"]
x = np.arange(len(labels))
w = 0.38
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, dom, ttl in ((axes[0], "gold", "In-domain (IndEgo, LOCO)"),
                     (axes[1], "ood", "OOD-160 (factory-grouped CV)")):
    v = [best[dom][l] for l in labels]
    b = ax.bar(x - w / 2, v, w, label="best capacity model", color="#4C72B0")
    cv = [comb[dom][c] for c in combos]
    xb = np.arange(len(combos)) + len(labels) + 0.6
    b2 = ax.bar(xb, cv, w * 1.6, label="ensemble / stack", color="#DD8452")
    ax.axhline(0.95, ls="--", c="r", lw=1.2)
    ax.text(0.02, 0.952, "0.95 target", color="r", transform=ax.get_yaxis_transform(),
            fontsize=8, va="bottom")
    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.set_xticks(list(x) + list(xb))
    ax.set_xticklabels(labels + [c.replace("_", "\n") for c in combos],
                       rotation=45, ha="right", fontsize=7.5)
    ax.set_title(ttl)
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", alpha=0.25)
    for r in list(b) + list(b2):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.004,
                f"{r.get_height():.3f}", ha="center", fontsize=6.5)
axes[0].set_ylabel("AUROC")
axes[0].legend(loc="lower left", fontsize=8)
fig.suptitle("A1 — compute-free capacity + ensembling: best achievable AUROC vs 0.95", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, "exp_acc_capacity.png"), dpi=150)
print("wrote", os.path.join(OUT, "exp_acc_capacity.png"))
