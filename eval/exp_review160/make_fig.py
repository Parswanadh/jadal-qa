#!/usr/bin/env python3
"""Render the O3 review-vs-OOD-error frontier figure."""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "eval/results/exp_review160")
OUT = os.path.join(ROOT, "docs/experiments/exp_review160/images")
os.makedirs(OUT, exist_ok=True)


def main():
    d = json.load(open(os.path.join(RES, "O3.01.json")))
    band = d["band_frontier_fine"]
    gate = d["gate_frontier"]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    def curve(rows, xk, yk, yci, label, **kw):
        pts = [(100 * r[xk], r[yk], r[yci]) for r in rows]
        pts = [p for p in pts if p[1] == p[1]]
        pts.sort()
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        lo = np.array([p[2][0] for p in pts])
        hi = np.array([p[2][1] for p in pts])
        ax.plot(x, y, label=label, **kw)
        ax.fill_between(x, lo, hi, alpha=0.12)

    curve(band, "review_share_1546", "ood_ba", "ood_ba_ci95_grouped",
          "band, prob-avg (frozen score)", color="C0")
    curve(band, "review_share_1546_rankavg", "ood_ba_rankavg",
          "ood_ba_rankavg_ci95_grouped", "band, rank-avg", color="C1")
    curve(gate, "review_share_1546", "ood_ba", "ood_ba_ci95_grouped",
          "gate, prob-avg", color="C2", linestyle="--")
    curve(gate, "review_share_1546", "ood_ba_rankavg",
          "ood_ba_rankavg_ci95_grouped", "gate, rank-avg", color="C3",
          linestyle="--")

    ax.axhline(0.5, color="grey", lw=1, ls=":")
    ax.axvline(60.0, color="red", lw=1, ls=":")
    ax.text(60.6, 0.512, "review = 60% budget", color="red", fontsize=8)
    ax.text(1, 0.508, "chance (BA=0.5)", color="grey", fontsize=8)

    # recommended point
    rec = json.load(open(os.path.join(RES, "O3.02.json")))["recommendation"]
    ax.scatter([100 * rec["review_share_1546"]], [rec["ood_ba"]], color="black",
               zorder=5, s=45, marker="*")
    ax.annotate("recommended\n(band rank-avg w=%s)" % rec["param"]["w"],
                (100 * rec["review_share_1546"], rec["ood_ba"]),
                textcoords="offset points", xytext=(8, -28), fontsize=8)

    ax.set_xlabel("REVIEW share on the 1,546 Egocentric-10K windows  [%]  (workload, unlabelled)")
    ax.set_ylabel("OOD balanced accuracy on the 160 labelled windows")
    ax.set_title("O3 — review-vs-OOD-error frontier (frozen PoC score; 160 OOD labels, 22 bad, 47 factories)\ngrouped-bootstrap 95% bands; nothing fit on the OOD test")
    ax.set_xlim(0, 70)
    ax.set_ylim(0.30, 0.90)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    p = os.path.join(OUT, "o3_review160_frontier.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


if __name__ == "__main__":
    main()
