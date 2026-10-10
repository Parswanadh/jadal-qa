#!/usr/bin/env python3
"""F50 — figure: 26 -> 50 features (in-domain, OOD, ablation, saturation).

Run: OMP_NUM_THREADS=2 python3 eval/gpu/make_fig_feat50.py
Writes docs/reports/images/exp_feat50.png
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(REPO, "eval", "results", "exp_feat50")
OUT = os.path.join(REPO, "docs", "reports", "images", "exp_feat50.png")


def j(name):
    return json.load(open(os.path.join(RES, name)))


def main():
    ind = j("in_domain.json")
    ood = j("ood.json")
    ab = j("ablation.json")
    hold = j("holdout.json")
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) in-domain LOCO
    a = ax[0, 0]
    vals = [ind["rows"]["f26"]["auroc"], ind["rows"]["f50"]["auroc"]]
    a.bar(["26 feats", "50 feats"], vals, color=["#888", "#2b6"])
    for i, v in enumerate(vals):
        a.text(i, v + 0.005, f"{v:.3f}", ha="center")
    a.axhline(0.810, ls="--", c="k", lw=1, label="documented 26 baseline 0.810")
    a.set_ylim(0.5, 0.95)
    a.set_ylabel("AUROC")
    a.set_title("(a) In-domain LOCO (7 IndEgo clips)")
    a.legend(fontsize=8)

    # (b) OOD adapted
    a = ax[0, 1]
    keys = ["frozen_ensemble_v1", "f26_all", "documented_baseline_stable8+transfer6",
            "documented_plus_18new", "f50_all"]
    labs = ["frozen ens.", "26 all", "doc 8+6", "doc+18", "50 all"]
    vals, lo, hi = [], [], []
    for k in keys:
        r = ood["rows"][k]
        vals.append(r["ood_auroc"])
        ci = r["ood_ci95_factory"]
        lo.append(r["ood_auroc"] - ci[0])
        hi.append(ci[1] - r["ood_auroc"])
    a.bar(labs, vals, yerr=[lo, hi], capsize=4,
          color=["#c66", "#888", "#48a", "#2b6"])
    for i, v in enumerate(vals):
        a.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    a.axhline(0.5, ls=":", c="k", lw=1)
    a.set_ylim(0.4, 1.0)
    a.set_ylabel("AUROC (factory-grouped 95% CI)")
    a.set_title("(b) OOD-160 adapted z-score")
    a.tick_params(axis="x", labelsize=8)

    # (c) family-only OOD AUROC
    a = ax[1, 0]
    fam = ab["family_only"]
    names = sorted(fam, key=lambda k: -fam[k]["ood_auroc"])
    v = [fam[k]["ood_auroc"] for k in names]
    a.barh([k.replace("only_", "") for k in names][::-1], v[::-1], color="#48a")
    a.axvline(0.5, ls=":", c="k", lw=1)
    for i, x in enumerate(v[::-1]):
        a.text(x + 0.005, i, f"{x:.3f}", va="center", fontsize=8)
    a.set_xlim(0.2, 0.9)
    a.set_xlabel("OOD AUROC (family alone)")
    a.set_title("(c) Which added families carry OOD signal")

    # (d) cumulative saturation
    a = ax[1, 1]
    cum = ab["cumulative_by_family_only_rank"]
    xs = list(range(1, len(cum) + 1))
    ys = [c["ood_auroc"] for c in cum]
    a.plot(xs, ys, "o-", color="#2b6")
    a.axhline(ood["rows"]["f26_all"]["ood_auroc"], ls="--", c="#888", lw=1,
              label=f"26 all = {ood['rows']['f26_all']['ood_auroc']:.3f}")
    a.axhline(0.781, ls=":", c="k", lw=1, label="documented adapted 0.781")
    for x, y, c in zip(xs, ys, cum):
        a.annotate(c["family_added"], (x, y), fontsize=7, rotation=30)
    a.set_xticks(xs)
    a.set_xlabel("# new families added (ranked)")
    a.set_ylabel("OOD AUROC")
    a.set_title("(d) Cumulative saturation")
    a.legend(fontsize=8)
    a.set_ylim(0.4, 0.95)

    fig.suptitle("F50 — per-pixel features 26 -> 50 (IndEgo LOCO + OOD-160 adapted; "
                 f"hold-out108 {hold['rows']['f50_all']['auroc']:.3f} vs "
                 f"{hold['rows']['f26_all']['auroc']:.3f})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
