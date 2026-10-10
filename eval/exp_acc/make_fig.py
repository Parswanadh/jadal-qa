#!/usr/bin/env python3
"""N2 figure: review-vs-OOD-error frontier, adapted score vs frozen references."""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "eval/results/exp_acc")
OUT = os.path.join(ROOT, "docs/experiments/exp_acc/images")
os.makedirs(OUT, exist_ok=True)


def main():
    d2 = json.load(open(os.path.join(RES, "N2.02.json")))
    d3 = json.load(open(os.path.join(RES, "N2.03.json")))

    fig, ax = plt.subplots(figsize=(8.6, 5.8))

    def curve(rows, xk, label, color, **kw):
        pts = sorted((100 * r[xk], r["ood_ba"], r["ood_ba_ci95_grouped"]) for r in rows
                     if r["ood_ba"] == r["ood_ba"])
        x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        lo = np.array([p[2][0] for p in pts]); hi = np.array([p[2][1] for p in pts])
        ax.plot(x, y, label=label, color=color, **kw)
        ax.fill_between(x, lo, hi, alpha=0.10, color=color)

    # adapted global (within-regime) quantile
    curve(d2["frontiers"]["adapted_b"]["global"], "review_share_1546",
          "adapted(b) stable8+Z3-6, global quantile", "C0", lw=2.2)
    curve(d2["frontiers"]["adapted_a"]["global"], "review_share_1546",
          "adapted(a) Z3-6, global quantile", "C2", lw=1.2, ls="--")
    # frozen references
    fz = d3["frozen_reference_reproduced"]
    curve(fz["rankavg_band"], "review_share_1546",
          "frozen rank-avg band (O3)", "C1", lw=1.6)
    curve(fz["probavg_band"], "review_share_1546",
          "frozen prob-avg band (O3)", "C3", lw=1.2, ls=":")

    # conformal points (review% on 160 only -> drawn at their 160 review)
    conf = d3["conformal"]["rows"]
    cx = [100 * r["review_share_160"] for r in conf]
    cy = [r["ood_ba"] for r in conf]
    ax.scatter(cx, cy, marker="^", s=26, color="C4", zorder=4,
               label="adapted(b) split conformal (x=review% on 160)")

    ax.axhline(0.5, color="grey", lw=1, ls=":")
    ax.axvline(60.0, color="red", lw=1, ls=":")
    ax.text(60.6, 0.505, "review = 60% budget", color="red", fontsize=8)
    ax.text(1, 0.503, "chance (BA=0.5)", color="grey", fontsize=8)

    # O3 frozen reference point
    b = next(r for r in fz["rankavg_band"] if r["w"] == 0.16)
    ax.scatter([100 * b["review_share_1546"]], [b["ood_ba"]], color="black",
               marker="*", s=90, zorder=6)
    ax.annotate("O3 frozen ref\n44.2% / 0.708",
                (100 * b["review_share_1546"], b["ood_ba"]),
                textcoords="offset points", xytext=(6, -30), fontsize=8)

    # best adapted <=60
    ba = d3["best_adapted_under_review60"]
    ax.scatter([100 * ba["review_share_1546"]], [ba["ood_ba"]], color="C0",
               marker="*", s=120, zorder=6)
    ax.annotate("best adapted <=60%\n60.0% / 0.857", (60.0, ba["ood_ba"]),
                textcoords="offset points", xytext=(-95, 6), fontsize=8, color="C0")

    ax.set_xlabel("REVIEW share on the 1,546 Egocentric-10K windows  [%]  (workload, unlabelled)")
    ax.set_ylabel("OOD balanced accuracy on the 160 labelled windows")
    ax.set_title("N2 — adapted OOD score: review-vs-error frontier vs the frozen O3 frontier\n"
                 "factory-grouped 95% bands; OOD AUROC adapted 0.781 [0.634,0.895] vs frozen 0.566")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.40, 1.0)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "n2_operating_frontier.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


if __name__ == "__main__":
    main()
