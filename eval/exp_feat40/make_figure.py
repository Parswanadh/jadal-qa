"""Figure for F40: in-domain / OOD / hold-out AUROC (26 vs 40) + ablation.

Reads eval/results/exp_feat40/eval_feat40.json. Writes
docs/experiments/exp_feat40/images/feat40.png (and a copy next to the report).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = json.load(open(os.path.join(REPO, "eval", "results", "exp_feat40", "eval_feat40.json")))
OUT = os.path.join(REPO, "docs", "experiments", "exp_feat40", "images")
os.makedirs(OUT, exist_ok=True)
S = RES["sets"]


def val(key):
    return S[key]["auroc"], S[key]["ci95"]


fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

# panel 1: in-domain
ax = axes[0]
labels = ["26", "40"]
a26, c26 = val("in_domain_26"); a40, c40 = val("in_domain_40")
ax.bar(labels, [a26, a40], color=["#888", "#2b7"], width=0.6)
ax.errorbar(labels, [a26, a40],
            yerr=[[a26 - c26[0], a40 - c40[0]], [c26[1] - a26, c40[1] - a40]],
            fmt="none", ecolor="k", capsize=4)
ax.axhline(0.810, ls="--", c="r", lw=1, label="baseline 0.810")
ax.set_ylim(0.5, 1.0); ax.set_title("In-domain LOCO AUROC"); ax.legend(fontsize=8)

# panel 2: OOD + hold-out
ax = axes[1]
keys = ["ood_frozen_all14(stable8+z3_6)", "ood_all26", "ood_all40",
        "ood_expanded22(stable8+z3_6+new8)"]
names = ["frozen\n14", "26", "40", "expanded\n22"]
av = [S[k]["auroc"] for k in keys]
lo = [S[k]["auroc"] - S[k]["ci95"][0] for k in keys]
hi = [S[k]["ci95"][1] - S[k]["auroc"] for k in keys]
ax.bar(names, av, color=["#888", "#59f", "#2b7", "#f96"], width=0.65)
ax.errorbar(names, av, yerr=[lo, hi], fmt="none", ecolor="k", capsize=4)
ax.axhline(0.781, ls="--", c="r", lw=1, label="pooled baseline 0.781")
ax.set_ylim(0.4, 1.0); ax.set_title("OOD-160 AUROC (factory-grouped)"); ax.legend(fontsize=8)

# panel 3: hold-out
ax = axes[2]
keys = ["holdout_frozen_all14(stable8+z3_6)", "holdout_all26", "holdout_all40",
        "holdout_expanded22(stable8+z3_6+new8)"]
av = [S[k]["auroc"] for k in keys]
lo = [S[k]["auroc"] - S[k]["ci95"][0] for k in keys]
hi = [S[k]["ci95"][1] - S[k]["auroc"] for k in keys]
ax.bar(names, av, color=["#888", "#59f", "#2b7", "#f96"], width=0.65)
ax.errorbar(names, av, yerr=[lo, hi], fmt="none", ecolor="k", capsize=4)
ax.axhline(0.676, ls="--", c="r", lw=1, label="hold-out baseline 0.676")
ax.set_ylim(0.4, 1.0); ax.set_title("Hold-out 108 AUROC (scored once)"); ax.legend(fontsize=8)

fig.suptitle("F40: per-pixel features 26 -> 40", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "feat40.png"), dpi=130)
print("wrote", os.path.join(OUT, "feat40.png"))
