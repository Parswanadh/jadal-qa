#!/usr/bin/env python3
"""Figure for A2: new-signal AUROC alone and combined vs baseline."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "eval", "results", "exp_acc")
OUT = os.path.join(ROOT, "docs", "experiments", "exp_acc", "images", "a2_signals.png")

s1 = json.load(open(os.path.join(RES, "A2.01.json")))
s2 = json.load(open(os.path.join(RES, "A2.02.json")))
s3 = json.load(open(os.path.join(RES, "A2.03.json")))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
gcol = {"flow": "#1f77b4", "hand": "#d62728", "fullres": "#2ca02c", "other": "#777777"}

# left: each signal alone
ax = axes[0]
sig = s1["signals"]
order = sorted(sig, key=lambda n: -sig[n]["gold_auroc"])
gy = [sig[n]["gold_auroc"] for n in order]
oy = [sig[n]["ood_auroc"] for n in order]
x = np.arange(len(order))
ax.bar(x - 0.2, gy, 0.4, color=[gcol[sig[n]["group"]] for n in order], alpha=0.95,
       label="gold (in-domain)")
ax.bar(x + 0.2, oy, 0.4, color=[gcol[sig[n]["group"]] for n in order], alpha=0.45,
       hatch="//", label="OOD")
b = s1["baseline_ensemble"]
ax.axhline(b["gold_loco_auroc"], color="k", ls="--", lw=1)
ax.axhline(b["ood_auroc"], color="k", ls=":", lw=1)
ax.text(len(order) - 0.5, b["gold_loco_auroc"] + 0.005, "baseline gold", ha="right", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([n.replace("flow_", "f:").replace("hand_", "h:").replace("fr_", "r:")
                    for n in order], rotation=90, fontsize=7)
ax.set_ylim(0.3, 1.0)
ax.set_ylabel("AUROC")
ax.set_title("A2.01 each new signal alone")
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", alpha=0.3)

# right: combined
ax = axes[1]
c = s3["candidates"]
md = s3["mean_domain_auroc"]
names = list(c.keys())
gv = [c[n]["gold_auroc"] for n in names]
ov = [c[n]["ood_auroc"] for n in names]
mv = [md[n] for n in names]
x = np.arange(len(names))
ax.bar(x - 0.25, gv, 0.25, label="gold")
ax.bar(x, ov, 0.25, label="OOD")
ax.bar(x + 0.25, mv, 0.25, label="mean-domain", color="gray", alpha=0.6)
ax.axhline(0.95, color="red", ls="--", lw=1)
ax.text(0, 0.955, "target 0.95", color="red", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([n.replace("ens+", "+").replace(" (baseline)", "\n(baseline)")
                    for n in names], rotation=20, fontsize=8)
ax.set_ylim(0.4, 1.0)
ax.set_title("A2.03 combined vs ensemble baseline")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130)
print("wrote", OUT)
