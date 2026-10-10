#!/usr/bin/env python3
"""Figure for N3 fusion + routing (reads the N3.01/02/03 JSONs)."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "eval", "results", "exp_acc")
IMG = os.path.join(ROOT, "docs", "experiments", "exp_acc", "images", "n3_fusion_routing.png")
os.makedirs(os.path.dirname(IMG), exist_ok=True)

s1 = json.load(open(os.path.join(RES, "N3.01.json")))
s2 = json.load(open(os.path.join(RES, "N3.02.json")))
s3 = json.load(open(os.path.join(RES, "N3.03.json")))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# --- Panel A: quality vs semantic AUROC ---
ax = axes[0]
cats = [("indego_loco", "auroc_all", "in:all"),
        ("indego_loco", "auroc_quality", "in:quality"),
        ("indego_loco", "auroc_semantic", "in:semantic"),
        ("ood160", "auroc_all", "ood:all"),
        ("ood160", "auroc_semantic", "ood:semantic")]
scorers = ["rules", "frozen", "adapted", "rules_quality", "rules_semantic"]
labels = {"rules": "rules (all)", "frozen": "frozen ens", "adapted": "adapted Z3",
          "rules_quality": "rules quality-only", "rules_semantic": "rules semantic-only"}
colors = {"rules": "#1f77b4", "frozen": "#ff7f0e", "adapted": "#2ca02c",
          "rules_quality": "#9edae5", "rules_semantic": "#c5b0d5"}
w = 0.15
xs = np.arange(len(cats))
for k, sc in enumerate(scorers):
    vals = []
    for dom, key, _ in cats:
        v = s1["domains"][dom]["scorers"][sc].get(key)
        vals.append(np.nan if v is None else v)
    ax.bar(xs + (k - 2) * w, vals, w, label=labels[sc], color=colors[sc])
ax.axhline(0.5, color="k", lw=0.8, ls="--")
ax.set_xticks(xs)
ax.set_xticklabels([c[2] for c in cats], rotation=20, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("AUROC (bad detection)")
ax.set_title("A. Quality-vs-semantic split")
ax.legend(fontsize=7, ncol=1, loc="upper right")

# --- Panel B: review% vs AUROC, single scorers + routers ---
ax = axes[1]
for dom, marker, tag in (("indego_loco", "o", "in-domain"), ("ood160", "s", "OOD")):
    sc = s2["domains"][dom]["scorers"]
    for name, r in sc.items():
        isrouter = name.startswith("route_")
        col = "#d62728" if isrouter else "#1f77b4"
        ax.scatter(r["review_share"], r["auroc"], marker=marker, c=col, alpha=0.7, s=45)
        if name in ("rules", "frozen", "adapted", "route_aspect_wide",
                    "route_fps_ge25", "route_quality_desc"):
            ax.annotate(name.replace("route_", "R:"), (r["review_share"], r["auroc"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
ax.axhline(0.5, color="k", lw=0.8, ls="--")
ax.set_xlabel("REVIEW share (calibrated band [0.4,0.6])")
ax.set_ylabel("AUROC")
ax.set_title("B. Routing: review vs accuracy\n(circles=in-domain, squares=OOD)")

# --- Panel C: fusion ---
ax = axes[2]
combos = ["frozen", "adapted", "rules", "rank_adapted+frozen", "rank_adapted+frozen+rules"]
x = np.arange(len(combos))
w = 0.38
for j, (dom, col, tag) in enumerate((("indego_loco", "#1f77b4", "in-domain"),
                                     ("ood160", "#ff7f0e", "OOD"))):
    vals = [s3["domains"][dom]["combos"][c]["auroc"] for c in combos]
    ax.bar(x + (j - 0.5) * w, vals, w, color=col, label=tag)
ax.axhline(0.5, color="k", lw=0.8, ls="--")
ax.set_xticks(x)
ax.set_xticklabels([c.replace("rank_", "rk:").replace("+", "\n+") for c in combos],
                   rotation=0, fontsize=7)
ax.set_ylim(0, 1.0)
ax.set_ylabel("AUROC")
ax.set_title("C. Rank-average fusion")
ax.legend(fontsize=8)

fig.suptitle("N3 — deterministic rules x adapted model x frozen ensemble: quality/semantic split, routing, fusion",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(IMG, dpi=130)
print("wrote", IMG)
