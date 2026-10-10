"""E4 figures: overlaid histograms, EG2 cluster bars, regime deltas, gap scatter."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "exp_regimes")
FIG = os.path.join(RES, "figs")


def load():
    recs = []
    for fn in ("desc_indego.jsonl", "desc_ood.jsonl"):
        p = os.path.join(RES, fn)
        if os.path.exists(p):
            for line in open(p):
                r = json.loads(line)
                if r["ok"]:
                    recs.append(r)
    return recs


def main(top=8):
    recs = load()
    by = {}
    for r in recs:
        by.setdefault(r["set"], []).append(r["desc"])
    diag = json.load(open(os.path.join(RES, "diagnosis.json")))
    keys = [r["descriptor"] for r in diag["ranked_table"][:top]]
    n = len(keys)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 7))
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, keys):
        for s, lbl in (("indego", "IndEgo"), ("ood1_source", "OOD-src"),
                       ("ood1_transcode", "OOD-480p"), ("ego10k_sample", "Ego10K")):
            if s in by:
                v = np.array([d[k] for d in by[s]], float)
                ax.hist(v, bins=25, alpha=0.45, label=f"{lbl} (n={len(v)})", density=True)
        ax.set_title(k)
        ax.legend(fontsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("E4.01 regime descriptors: IndEgo vs Egocentric-10K")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "descriptors_hist.png"), dpi=90)
    print("wrote descriptors_hist.png")

    # EG2 cluster divergence
    try:
        eg = json.load(open(os.path.join(RES, "eg2_clusters.json")))
        x = np.arange(eg["k"])
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(x - 0.2, eg["indego_hist"], 0.4, label="IndEgo")
        ax.bar(x + 0.2, eg["ood60_hist"], 0.4, label="OOD-60")
        ax.set_title(f"EG2 cluster histograms (sym-chi2={eg['sym_chi2']})")
        ax.set_xlabel("cluster")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "eg2_clusters.png"), dpi=90)
        print("wrote eg2_clusters.png")
    except FileNotFoundError:
        print("eg2_clusters.json missing, skipped")

    # regime-vs-global deltas
    try:
        rg = json.load(open(os.path.join(RES, "regime_vs_global.json")))
        labels, deltas = [], []
        for view in ("feat26", "eg2"):
            for rn, r in rg[view]["regimes"].items():
                if "delta" in r:
                    labels.append(f"{view}:{rn}")
                    deltas.append(r["delta"])
        fig, ax = plt.subplots(figsize=(7, 0.5 * max(len(labels), 1) + 1.5))
        y = np.arange(len(labels))
        ax.barh(y, deltas)
        ax.set_yticks(y, labels, fontsize=7)
        ax.axvline(0.03, color="r", ls="--", label="adopt +0.03")
        ax.axvline(0, color="k")
        ax.set_xlabel("regime_model - global_model LOCO AUROC")
        ax.set_title("E4.02 regime-specific vs global (nested LOCO)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "regime_delta.png"), dpi=90)
        print("wrote regime_delta.png")
    except FileNotFoundError:
        print("regime_vs_global.json missing, skipped")


if __name__ == "__main__":
    main()
