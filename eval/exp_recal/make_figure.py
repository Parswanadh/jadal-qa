#!/usr/bin/env python3
"""N5 figure: frozen-vs-sparse heads on gold + OOD (reads N5.01/N5.02 + oof_gold.npz)."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
OUTDIR = os.path.join(REPO, "eval", "results", "exp_recal")
IMG = os.path.join(REPO, "docs", "experiments", "exp_recal", "images")
os.makedirs(IMG, exist_ok=True)

r1 = json.load(open(os.path.join(OUTDIR, "N5.01.json")))
r2 = json.load(open(os.path.join(OUTDIR, "N5.02.json")))
z = np.load(os.path.join(OUTDIR, "oof_gold.npz"), allow_pickle=True)
y = z["y"]

fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))

# 1. score distributions (gold)
for tag, key, col in [("frozen features", "ens_frozen", "tab:blue"),
                      ("sparse + frozen heads", "frozen_heads_on_sparse", "tab:red"),
                      ("sparse + refit heads", "ens_sparse", "tab:green")]:
    ax[0].hist(z[key], bins=20, range=(0, 1), alpha=0.5, label=tag, color=col)
ax[0].axvline(0.4, color="k", ls=":", lw=1)
ax[0].axvline(0.6, color="k", ls=":", lw=1)
ax[0].set_title("Gold OOF score distribution")
ax[0].set_xlabel("usable_prob"); ax[0].legend(fontsize=7); ax[0].set_ylabel("windows")

# 2. verdict share at the factory_1080p30 profile thresholds (the L3 corpus path)
prof = r1["operating_point_profile_thresholds_gold"]["factory_1080p30"]
labels = ["accept", "reject", "review"]
x = np.arange(3); w = 0.26
for i, (tag, key, col) in enumerate([("frozen", "frozen_features", "tab:blue"),
                                     ("sparse+frozen heads", "sparse_frozen_heads", "tab:red"),
                                     ("sparse+refit", "sparse_refit", "tab:green")]):
    vals = [prof[key][l] / 150 for l in labels]
    ax[1].bar(x + (i - 1) * w, vals, w, label=tag, color=col)
ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
ax[1].set_title("Gold verdict share @ profile 0.75/0.25")
ax[1].legend(fontsize=7); ax[1].set_ylabel("share")

# 3. LOCO AUROC (gold)
rows = [("frozen feat. (nestedC)", r1["loco"]["frozen_features"]["ensemble_probavg"], "tab:blue"),
        ("sparse feat. (nestedC)", r1["loco"]["sparse_features_nestedC"]["ensemble_probavg"], "tab:green"),
        ("frozen feat. (fixedC)", r1["loco"]["frozen_features_fixedC"]["ensemble_probavg"], "tab:blue"),
        ("sparse feat. (fixedC)", r1["loco"]["sparse_features_fixedC"]["ensemble_probavg"], "tab:green")]
for i, (tag, v, col) in enumerate(rows):
    ax[2].barh(i, v["auroc"], color=col, alpha=0.8)
    lo, hi = v["ci95"]
    ax[2].plot([lo, hi], [i, i], "k-", lw=1.2)
    ax[2].text(v["auroc"] + 0.01, i, f"{v['auroc']:.3f}", va="center", fontsize=8)
ax[2].set_yticks(range(len(rows))); ax[2].set_yticklabels([r[0] for r in rows], fontsize=8)
ax[2].set_xlim(0.55, 1.0); ax[2].set_title("Gold LOCO ensemble AUROC (95% clip CI)")

# 4. OOD AUROC (grouped CI)
oo = [("frozen/frozen", r2["frozen_features"], "tab:blue"),
      ("frozen/sparse", r2["sparse_frozen_heads"], "tab:red"),
      ("refit/sparse", r2["sparse_refit"], "tab:green")]
for i, (tag, v, col) in enumerate(oo):
    ax[3].barh(i, v["auroc"], color=col, alpha=0.8)
    lo, hi = v["group_ci95"]
    ax[3].plot([lo, hi], [i, i], "k-", lw=1.2)
    ax[3].text(v["auroc"] + 0.01, i, f"{v['auroc']:.3f}", va="center", fontsize=8)
ax[3].axvline(0.5, color="gray", ls="--", lw=1)
ax[3].set_yticks(range(len(oo))); ax[3].set_yticklabels([r[0] for r in oo], fontsize=8)
ax[3].set_xlim(0.3, 0.85); ax[3].set_title("OOD-160 AUROC (factory-grouped CI)")

fig.suptitle("N5 — refit heads for the optimized sparse pipeline (pixel 3.33 fps, EG2 1 fps, 4 s prefix)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
p = os.path.join(IMG, "recal_gold_ood.png")
fig.savefig(p, dpi=130)
print("wrote", p)
