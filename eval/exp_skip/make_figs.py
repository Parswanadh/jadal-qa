#!/usr/bin/env python3
"""Deterministic evidence figures for SKIP (Q2). Outputs PNGs <200 KB each.
  OMP_NUM_THREADS=2 python3 eval/exp_skip/make_figs.py \
     --results eval/results/exp_skip --out docs/experiments/exp_skip/images
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3, "font.family": "DejaVu Sans"})

KEEP = "#5aa469"; DROP = "#d9534f"; SPREAD = "#f0ad4e"; NEUT = "#e8e8e8"


def fig_timeline(out):
    """Drop-1/drop-2 rules across 5 sections (start/middle/end/evenly)."""
    rules = [("drop1-start", [0]), ("drop1-middle", [2]), ("drop1-end", [4]),
             ("drop1-evenly", "spread1"),
             ("drop2-start", [0, 1]), ("drop2-middle", [2, 1]), ("drop2-end", [3, 4]),
             ("drop2-evenly", "spread2")]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for row, (name, dropped) in enumerate(rules):
        y = len(rules) - 1 - row
        for s in range(5):
            x = s + 0.03
            if dropped == "spread1":
                # evenly spaced drops: alternate kept/dropped stripes inside each section
                drop_here = (s % 5) in {0}
                base = SPREAD if s == 0 else KEEP
                ax.add_patch(Rectangle((x, y + 0.12), 0.94, 0.76, facecolor=base,
                                       edgecolor="white", alpha=0.35 if s == 0 else 1.0))
                ax.plot([x, x + 0.94], [y + 0.5, y + 0.5], color=DROP, lw=1.4, ls=":")
                ax.text(x + 0.47, y + 0.5, "·↓·", ha="center", va="center", fontsize=7,
                        color=DROP)
            elif dropped == "spread2":
                ax.add_patch(Rectangle((x, y + 0.12), 0.94, 0.76, facecolor=SPREAD,
                                       edgecolor="white", alpha=0.35))
                ax.plot([x, x + 0.94], [y + 0.5, y + 0.5], color=DROP, lw=1.4, ls=":")
                ax.text(x + 0.47, y + 0.5, "·↓↓·", ha="center", va="center", fontsize=7,
                        color=DROP)
            else:
                col = DROP if s in dropped else KEEP
                ax.add_patch(Rectangle((x, y + 0.12), 0.94, 0.76, facecolor=col,
                                       edgecolor="white", alpha=0.9 if s in dropped else 1.0))
                ax.text(x + 0.47, y + 0.5, "drop" if s in dropped else "keep",
                        ha="center", va="center", fontsize=7,
                        color="white" if s in dropped else "#123")
        ax.text(5.15, y + 0.5, name, va="center", ha="left", fontsize=8)
    ax.set_xlim(0, 7.2); ax.set_ylim(-0.1, len(rules))
    ax.set_xticks([s + 0.5 for s in range(5)])
    ax.set_xticklabels([f"sec{i+1}" for i in range(5)])
    ax.set_yticks([]); ax.grid(False)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.set_title("Drop-1 / drop-2 of 5 sections — kept vs dropped (start / middle / end / evenly)")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_error_cost(s1, s2, out):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    fam = {"grid": ("#2c7fb8", "o"), "drop": ("#d9534f", "s"),
           "strat": ("#756bb1", "^"), "dense": ("#f0ad4e", "D")}
    def add(name, v, f):
        c, m = fam[f]
        ax.scatter(v["mean_frac"], v["mean_err_L"], c=c, marker=m, s=34, zorder=3)
    grid = []
    for n, v in s2.items():
        if n.startswith("uniform-every"):
            add(n, v, "grid"); grid.append((v["mean_frac"], v["mean_err_L"]))
    grid.sort(); gx, gy = zip(*grid)
    ax.plot(gx, gy, color=fam["grid"][0], lw=1.2, alpha=0.6, label="uniform every-k grid")
    for n, v in s1.items():
        if n != "full": add(n, v, "drop")
    for n, v in s2.items():
        if n.startswith(("strat-", "contig")): add(n, v, "strat")
        if n.startswith("dense-"): add(n, v, "dense")
    ax.axhline(0.05, color="grey", ls="--", lw=1); ax.text(0.36, 0.052, "err=0.05 tolerance", fontsize=7, color="grey")
    ax.axvline(0.515, color=fam["grid"][0], ls=":", lw=1)
    for n, v, dx, dy in [
        ("drop1-start", s1["drop1-start"], 0.01, 0.008),
        ("drop1-end", s1["drop1-end"], 0.01, -0.012),
        ("drop1-evenly-s0", s1["drop1-evenly-s0"], 0.005, 0.006),
        ("drop2-start", s1["drop2-start"], 0.005, 0.006),
        ("drop2-evenly-s2", s1["drop2-evenly-s2"], 0.005, 0.006),
        ("uniform-every-2", s2["uniform-every-2"], 0.008, 0.006),
        ("strat-probe(1/section)", s2["strat-probe(1/section)"], -0.02, -0.02),
        ("contig-probe(first 5)", s2["contig-probe(first 5)"], 0.005, 0.006),
        ("dense-from-strat(x1)", s2["dense-from-strat(x1)"], 0.008, 0.004)]:
        ax.annotate(n, (v["mean_frac"], v["mean_err_L"]),
                    xytext=(v["mean_frac"] + dx, v["mean_err_L"] + dy), fontsize=6.8)
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", label=k)
               for k, (c, m) in fam.items()]
    handles.insert(0, plt.Line2D([], [], color=fam["grid"][0], lw=1.2, label="uniform every-k"))
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    ax.set_xlabel("processed fraction of footage"); ax.set_ylabel("clip bad-fraction error (label oracle)")
    ax.set_title("Cost vs clip-level error — section drops (red) are dominated by the uniform every-2 grid")
    ax.set_xlim(0.15, 1.02); ax.set_ylim(-0.005, 0.24)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_perclip(s3, out):
    """Per-clip verdict agreement matrix for representative policies."""
    det = s3["per_clip_detail"]
    pols = ["drop1-start", "drop1-middle", "drop1-end", "drop1-evenly-s0",
            "drop2-start", "drop2-evenly-s0", "strat-probe(1/section)", "uniform-every-2"]
    clips = sorted(det["drop1-end"].keys())
    M = np.array([[det[p][c]["agree30"] for c in clips] for p in pols], float)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.imshow(M, cmap=matplotlib.colors.ListedColormap([DROP, KEEP]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(clips))); ax.set_xticklabels(clips, rotation=25, ha="right")
    ax.set_yticks(range(len(pols))); ax.set_yticklabels(pols)
    for i in range(len(pols)):
        for j in range(len(clips)):
            ax.text(j, i, "ok" if M[i, j] else "flip", ha="center", va="center",
                    color="white", fontsize=7)
    ax.set_title("Per-clip verdict agreement @0.30 (green=agrees, red=flips)")
    ax.grid(False)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_section_profile(s1, out):
    prof = [r["mean_bad_frac"] for r in s1["section_position_profile"]]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.bar([f"sec{i+1}" for i in range(5)], prof, color=["#5aa469", "#8cc08c", "#f0ad4e", "#e08a4e", "#d9534f"])
    ax.axhline(0.30, color="grey", ls="--", lw=1); ax.text(4.4, 0.31, "verdict thr 0.30", fontsize=7, color="grey", ha="right")
    for i, v in enumerate(prof): ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_ylabel("mean bad fraction"); ax.set_ylim(0, 0.8)
    ax.set_title("Bad rate rises towards the clip END (start is cleanest)")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval/results/exp_skip")
    ap.add_argument("--out", default="docs/experiments/exp_skip/images")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s1 = json.load(open(os.path.join(a.results, "SKIP.01.json")))["per_policy"]
    s2 = json.load(open(os.path.join(a.results, "SKIP.02.json")))["per_policy"]
    s1full = json.load(open(os.path.join(a.results, "SKIP.01.json")))
    s3full = json.load(open(os.path.join(a.results, "SKIP.03.json")))
    fig_timeline(os.path.join(a.out, "skip_timeline_drop_rules.png"))
    fig_error_cost(s1, s2, os.path.join(a.out, "skip_error_vs_cost.png"))
    fig_perclip(s3full, os.path.join(a.out, "skip_perclip_agreement.png"))
    fig_section_profile(s1full, os.path.join(a.out, "skip_section_profile.png"))
    for f in sorted(os.listdir(a.out)):
        print(f, os.path.getsize(os.path.join(a.out, f)), "bytes")


if __name__ == "__main__":
    main()
