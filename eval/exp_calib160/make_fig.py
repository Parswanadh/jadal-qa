#!/usr/bin/env python3
"""P1 figure: reliability curves + review-vs-error frontier."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(REPO, "eval", "results", "exp_calib160", "P1.01.json")
IMGDIR = os.path.join(REPO, "docs", "experiments", "exp_calib160", "images")


def main():
    d = json.load(open(RES))
    tag = "(b) stable-8 + transferable-6"
    cal = d["recipes"][tag]["oof_calibration"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    # left: reliability (deciles of OOF P(bad)), recipe (b)
    for m, sty in (("platt", "o-"), ("isotonic", "s--"), ("beta", "^-.")):
        rel = cal[m]["reliability_deciles"]
        xs = [r["mean_pred"] for r in rel if r["n"] > 0]
        ys = [r["emp_bad_rate"] for r in rel if r["n"] > 0]
        ns = [r["n"] for r in rel if r["n"] > 0]
        ax[0].plot(xs, ys, sty, ms=4, label=f"{m} (Brier {cal[m]['brier']})")
        for x, y, n in zip(xs, ys, ns):
            if n >= 20:
                ax[0].annotate(str(n), (x, y), fontsize=6)
    ax[0].plot([0, 1], [0, 1], "k:", lw=1, label="ideal")
    ax[0].axhline(d["base_bad_rate"], color="gray", lw=1, linestyle=(0, (3, 3)),
                  label=f"base rate {d['base_bad_rate']}")
    ax[0].set(xlabel="mean predicted P(bad) (LFO-OOF decile)",
              ylabel="empirical bad rate", title="(b) reliability, n=160, n_bad=22 (LFO)")
    ax[0].legend(fontsize=7)
    ax[0].set_xlim(0, 0.7)

    # right: review% vs OOD err frontier on the 160
    for t, ls in (("(b) stable-8 + transferable-6", None),):
        for m, mk in (("platt", "o"), ("isotonic", "s"), ("beta", "^")):
            rows = d["recipes"][t]["frontiers"][m]
            ax[1].plot([r["review_rate"] * 100 for r in rows],
                       [r["err"] for r in rows], marker=mk, ms=3, lw=1,
                       label=f"adapted-{m}")
    fr = d["frozen_same160"]["frontier"]
    ax[1].plot([r["review_rate"] * 100 for r in fr], [r["err"] for r in fr],
               marker="D", ms=3, lw=1, label="frozen usable_prob (same 160)")
    ax[1].axhline(0.5, color="k", lw=1, linestyle=":")
    ax[1].axvline(60, color="gray", lw=1, linestyle=(0, (3, 3)))
    ax[1].set(xlabel="review % on the 160 (LFO-OOF, symmetric 0.5-band)",
              ylabel="OOD err = 1 - BA (decided)",
              title="joint review/error frontier (adapted: flat, fa=1.0)")
    ax[1].legend(fontsize=7)
    fig.suptitle("P1.01: calibrated adapted score cannot drive a symmetric 0.5-band "
                 "(n=160, 22 bad, 47 factories, LFO)", fontsize=9)
    fig.tight_layout()
    os.makedirs(IMGDIR, exist_ok=True)
    fig.savefig(os.path.join(IMGDIR, "p1_calib160.png"), dpi=120)
    print("wrote", os.path.join(IMGDIR, "p1_calib160.png"))


if __name__ == "__main__":
    main()
