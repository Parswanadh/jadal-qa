#!/usr/bin/env python3
"""Figure for Q2: deterministic rule-list transfer on 160 OOD labels.

Two panels: (a) per-rule OOD precision / good false-alarm (quality vs activity);
(b) review-band frontier (review share vs balanced accuracy, review as miss).
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(REPO, "eval", "results", "exp_phys")
OUT = os.path.join(REPO, "docs", "experiments", "exp_phys", "images")
os.makedirs(OUT, exist_ok=True)


def main():
    d = json.load(open(os.path.join(RES, "Q2.01.json")))
    rules = d["per_rule_transfer"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) per-rule transfer
    names = [r["rule"] for r in rules]
    prec = [r["ood_precision_bad"] or 0.0 for r in rules]
    fa = [r["ood_good_false_alarm"] for r in rules]
    cls = [r["class"] for r in rules]
    x = np.arange(len(names))
    colors = ["#2b8cbe" if c == "quality" else "#e34a33" for c in cls]
    ax[0].bar(x - 0.2, prec, 0.4, color=colors, label="OOD precision (bad|fires)")
    ax[0].bar(x + 0.2, fa, 0.4, color=colors, alpha=0.45,
              label="OOD good false-alarm")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, rotation=45, ha="right")
    ax[0].set_ylabel("rate")
    ax[0].set_title("(a) per-rule transfer: quality (blue) vs activity (red)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, axis="y")

    # (b) review frontier
    fr = d["review_band_frontier_active_rules"]
    rv = [f["review_share"] for f in fr]
    ba = [f["balacc_review_as_miss"] for f in fr]
    ax[1].plot(rv, ba, "o-", color="#2b8cbe")
    for f in fr:
        ax[1].annotate(f"w={f['band']}", (f["review_share"], f["balacc_review_as_miss"]),
                       fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[1].axhline(0.5, ls="--", color="grey", lw=1)
    ax[1].axvline(0.6, ls=":", color="red", lw=1)
    ax[1].set_xlabel("REVIEW share")
    ax[1].set_ylabel("balanced accuracy (review = miss)")
    ax[1].set_title("(b) review/error frontier (active rules)")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    p = os.path.join(OUT, "rule_transfer.png")
    fig.savefig(p, dpi=140)
    print("wrote", p)


if __name__ == "__main__":
    main()
