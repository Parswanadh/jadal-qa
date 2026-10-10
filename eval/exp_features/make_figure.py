"""Z3 figure: per-feature IndEgo vs OOD AUROC + combined-set comparison."""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "eval", "results", "exp_features")
IMG = os.path.join(REPO, "docs", "experiments", "exp_features", "images")
os.makedirs(IMG, exist_ok=True)

z = json.load(open(os.path.join(OUT, "Z3.01.json")))
rows = z["table"]
fams = json.load(open(os.path.join(OUT, "Z3.02.json")))["families"]
z3 = json.load(open(os.path.join(OUT, "Z3.03.json")))

order, colors = [], []
pal = {"stable": "#2a9d8f", "flip": "#e76f51", "chance": "#999999"}
for r in sorted(rows, key=lambda r: (r["class"] != "stable", -min(r["indego_auroc"], r["ood_auroc"]))):
    order.append(r)
fig, axes = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [2, 1]})
ax = axes[0]
y = np.arange(len(order))
ai = [r["indego_auroc"] for r in order]
ao = [r["ood_auroc"] for r in order]
ax.barh(y - 0.2, ai, 0.4, label="IndEgo AUROC", color="#457b9d")
ax.barh(y + 0.2, ao, 0.4, label="OOD-60 AUROC", color="#e9c46a")
ax.axvline(0.5, color="k", lw=0.8, ls="--")
ax.set_yticks(y)
ax.set_yticklabels([f"{r['feature']}  [{r['class']}]" for r in order], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("signed AUROC (label 1 = usable/good)")
ax.set_title("Z3 candidates: IndEgo vs Egocentric-10K OOD-60")
ax.legend(loc="lower right")

ax2 = axes[1]
combo = z3["combined"]
ks = [k for k in combo if not k.startswith("zscore")]
ks += [k for k in combo if k.startswith("zscore")]
labels, iv, ov = [], [], []
for k in ks:
    v = combo[k]
    labels.append(f"{k}\n({v['n_features']} feats)")
    iv.append(v.get("indego_loco_auroc", v.get("indego_auroc")))
    ov.append(v.get("ood_zeroshot_auroc", v.get("ood_auroc")))
yy = np.arange(len(ks))
ax2.barh(yy - 0.2, iv, 0.4, color="#457b9d", label="IndEgo")
ax2.barh(yy + 0.2, ov, 0.4, color="#e9c46a", label="OOD zero-shot")
ax2.axvline(0.415, color="#e76f51", ls=":", lw=1.2, label="frozen ensemble OOD 0.415")
ax2.axvline(0.5, color="k", lw=0.8, ls="--")
ax2.set_yticks(yy); ax2.set_yticklabels(labels, fontsize=8); ax2.invert_yaxis()
ax2.set_xlim(0, 1); ax2.set_xlabel("AUROC"); ax2.set_title("combined sets")
ax2.legend(loc="lower right", fontsize=7)
fig.tight_layout()
p = os.path.join(IMG, "z3_features.png")
fig.savefig(p, dpi=130)
print("wrote", p)
