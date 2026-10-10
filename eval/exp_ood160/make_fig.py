#!/usr/bin/env python3
"""O1 figure: frozen PoC OOD on 160 labels - AUROC with window vs
factory-grouped CIs, and the shipped operating-point confusion matrix."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
R = json.load(open(os.path.join(ROOT, "eval", "results", "exp_ood160", "O1.01.json")))
IMG = os.path.join(ROOT, "docs", "experiments", "exp_ood160", "images")
os.makedirs(IMG, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

# --- panel 1: AUROC + CIs ---
ax = axes[0]
labels = ["ensemble\nusable_prob", "EG2 view", "pixel-26 view", "PoC cascade*"]
vals = [R["auroc"]["usable_prob"], R["auroc"]["p_eg2"], R["auroc"]["p_pixel"],
        R["poc_cascade_handover_160"]["auroc"]]
win = [R["auroc_ci95"]["usable_prob"]["window_bootstrap_optimistic"],
       R["auroc_ci95"]["p_eg2"]["window"],
       R["auroc_ci95"]["p_pixel"]["window"],
       R["poc_cascade_handover_160"]["ci_window_optimistic"]]
grp = [R["auroc_ci95"]["usable_prob"]["factory_grouped_bootstrap_honest"], None, None, None]
x = np.arange(len(labels))
for i, (v, w, g) in enumerate(zip(vals, win, grp)):
    ax.plot([i, i], w, color="#888", lw=7, alpha=0.5, solid_capstyle="round",
            label="window bootstrap (optimistic)" if i == 0 else None)
    if g:
        ax.plot([i], [v], "o", color="#1f77b4", ms=9)
        ax.plot([i, i], g, color="#1f77b4", lw=2.2, alpha=0.9,
                label="factory-grouped (honest)" if i == 0 else None)
    ax.plot([i], [v], "o", color="k", ms=5)
    ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
ax.axhline(0.5, color="crimson", ls="--", lw=1, label="chance")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0.25, 0.8)
ax.set_ylabel("OOD AUROC (good vs bad)")
ax.set_title("Frozen PoC OOD AUROC on 160 human labels")
ax.legend(fontsize=8, loc="upper left")
ax.text(0.02, 0.02, "*cascade: carried over from handover\n(not recomputed from bundle caches)",
        transform=ax.transAxes, fontsize=7, color="#555")

# --- panel 2: confusion matrix ---
ax = axes[1]
cm = R["confusion_matrix"]
cols = ["accept", "review", "reject"]
M = np.array([[cm["good"][c] for c in cols], [cm["bad"][c] for c in cols]], dtype=float)
row = M.sum(axis=1, keepdims=True)
Mn = M / np.where(row == 0, 1, row)
im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_xticklabels(cols)
ax.set_yticks(range(2)); ax.set_yticklabels([f"human GOOD (n={cm['good']['n']})",
                                             f"human BAD (n={cm['bad']['n']})"])
for i in range(2):
    for j in range(3):
        ax.text(j, i, f"{int(M[i,j])}\n{Mn[i,j]*100:.0f}%", ha="center", va="center",
                color="white" if Mn[i, j] > 0.5 else "black", fontsize=10)
ax.set_title(f"Shipped operating point (BA review=miss {R['operating_point']['balanced_accuracy_review_as_miss']:.3f})")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row fraction")

fig.tight_layout()
out = os.path.join(IMG, "ood160_auroc_confusion.png")
fig.savefig(out, dpi=150)
print("wrote", os.path.relpath(out, ROOT))
