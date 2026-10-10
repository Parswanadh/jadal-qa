"""Plot H3/E1.06 Pareto figures. Regen: python3 eval/exp_fps_res/plot_fps_res.py
Reads eval/results/exp_fps_res/E1.06.json, writes docs/experiments/exp_fps_res/images/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "eval", "results", "exp_fps_res")
IMG = os.path.join(ROOT, "docs", "experiments", "exp_fps_res", "images")

res = json.load(open(os.path.join(D, "E1.06.json")))
rows = res["rows"]
os.makedirs(IMG, exist_ok=True)

base = res["baseline"]["auroc"]

# 1. AUROC vs bytes, one line per fps
fig, ax = plt.subplots(figsize=(7, 4.5))
for fps in sorted(set(r["fps"] for r in rows)):
    rr = sorted([r for r in rows if r["fps"] == fps], key=lambda r: r["resolution_px"])
    xs = [r["bytes_per_window"] / 1024 for r in rr]
    ys = [r["auroc"] for r in rr]
    lo = [r["ci95"][0] for r in rr]
    hi = [r["ci95"][1] for r in rr]
    ax.errorbar(xs, ys, yerr=[[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]],
                marker="o", capsize=3, label=f"{fps} fps")
    for x, y, r in zip(xs, ys, rr):
        ax.annotate(f"{r['resolution_px']}p", (x, y), fontsize=7, xytext=(3, 4),
                    textcoords="offset points")
ax.axhline(base, color="k", ls="--", lw=1, label=f"baseline {base:.3f}")
ax.axhline(base - 0.02, color="k", ls=":", lw=1, label="baseline-0.02")
ax.set_xlabel("KiB / window (libx264 CRF23 re-encode at cell)")
ax.set_ylabel("LOCO AUROC (pixel-26, n=150, 7 clips)")
ax.set_title("H3 fps x resolution cross: accuracy vs bytes")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "pareto_bytes.png"), dpi=120)

# 2. Heatmap of AUROC over the grid
fps_vals = sorted(set(r["fps"] for r in rows), reverse=True)
res_vals = sorted(set(r["resolution_px"] for r in rows), reverse=True)
fig, ax = plt.subplots(figsize=(5.5, 4))
mat = [[next(r["auroc"] for r in rows if r["fps"] == f and r["resolution_px"] == s)
        for s in res_vals] for f in fps_vals]
im = ax.imshow(mat, vmin=0.70, vmax=0.85, cmap="RdYlGn")
for i, f in enumerate(fps_vals):
    for j, s in enumerate(res_vals):
        v = mat[i][j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9)
ax.set_xticks(range(len(res_vals)), [f"{s}p" for s in res_vals])
ax.set_yticks(range(len(fps_vals)), [f"{f} fps" for f in fps_vals])
ax.set_title("LOCO AUROC by (fps, resolution) cell")
fig.colorbar(im, ax=ax, label="AUROC")
fig.tight_layout()
fig.savefig(os.path.join(IMG, "heatmap_auroc.png"), dpi=120)

# plot data csv (numbers behind the figures)
with open(os.path.join(os.path.dirname(IMG), "plot_data.csv"), "w") as fh:
    fh.write("cell,fps,resolution_px,auroc,ci_lo,ci_hi,delta_vs_baseline,bytes_per_window,total_ms_per_window,within_budget\n")
    for r in rows:
        fh.write(f"{r['cell']},{r['fps']},{r['resolution_px']},{r['auroc']},{r['ci95'][0]},{r['ci95'][1]},"
                 f"{r['delta_vs_baseline']},{r['bytes_per_window']},{r['total_ms_per_window']},{r['within_budget']}\n")
print("wrote", IMG)
