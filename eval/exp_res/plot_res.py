"""Plots for the X1 resolution/compression sweep. Deterministic style, CPU-only.

Outputs (all < 200 KB):
  docs/experiments/exp_res/images/res_ladder.png     accuracy vs resolution & cost
  docs/experiments/exp_res/images/compression_ladder.png  AUROC vs bytes/window (CRF)
  docs/experiments/exp_res/images/sample_frames.png  one decoded frame per resolution
  docs/experiments/exp_res/plot_data.csv             tidy CSV of every config
Missing configs are simply skipped (partial runs are fine).
"""
import csv
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "eval", "results", "exp_res")
IMG = os.path.join(ROOT, "docs", "experiments", "exp_res", "images")
WORK = "/workspace/scratch/res_work"
NATIVE_AUROC = 0.8100

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})


def load():
    d = {}
    for f in sorted(glob.glob(os.path.join(RES, "r*_crf*.json"))):
        j = json.load(open(f))
        d[j["combo"]] = j
    return d


def main():
    os.makedirs(IMG, exist_ok=True)
    d = load()
    # tidy CSV
    cols = ["combo", "resolution_px", "crf", "auroc", "ci_lo", "ci_hi",
            "bytes_per_window", "total_ms_per_window", "feat_ms_per_window"]
    with open(os.path.join(ROOT, "docs", "experiments", "exp_res", "plot_data.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for name, j in d.items():
            w.writerow([name, j["resolution_px"], j["crf"], j["auroc"], j["ci95"][0],
                        j["ci95"][1], j["bytes_per_window"], j["total_ms_per_window"],
                        j["feat_ms_per_window"]])

    res_cfg = [(480, 23), (360, 23), (240, 23), (180, 23), (120, 23)]
    crf_cfg = [(480, 18), (480, 23), (480, 28), (480, 33)]

    # ---- res_ladder ----
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    xs, ys, los, his, bys = [], [], [], [], []
    for r, c in res_cfg:
        k = f"r{r}_crf{c}"
        if k not in d:
            continue
        j = d[k]
        xs.append(r); ys.append(j["auroc"]); los.append(j["ci95"][0]); his.append(j["ci95"][1])
        bys.append(j["bytes_per_window"] / 1024.0)
    if xs:
        ax[0].errorbar(xs, ys, yerr=[np.array(ys) - los, np.array(his) - np.array(ys)],
                       marker="o", capsize=3, lw=1.2, color="C0")
        ax[0].axhline(NATIVE_AUROC, ls="--", c="grey", lw=1, label=f"native 480p cache {NATIVE_AUROC:.3f}")
        ax[0].axhline(NATIVE_AUROC - 0.02, ls=":", c="red", lw=1, label="-0.02 band")
        ax[0].set_xlabel("encode resolution (px, square source)")
        ax[0].set_ylabel("LOCO AUROC (gold n=150)")
        ax[0].set_title("RES.01 resolution ladder (crf23)")
        ax[0].invert_xaxis(); ax[0].legend(fontsize=7)
        ax[1].plot(bys, ys, marker="o", color="C0")
        for x, y, r in zip(bys, ys, xs):
            ax[1].annotate(f"{r}p", (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax[1].set_xlabel("mean bytes / 10 s window (KiB)")
        ax[1].set_ylabel("LOCO AUROC")
        ax[1].set_title("accuracy vs cost")
        fig.savefig(os.path.join(IMG, "res_ladder.png"))
    plt.close(fig)

    # ---- compression_ladder ----
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    bs, ys, cs = [], [], []
    for r, c in crf_cfg:
        k = f"r{r}_crf{c}"
        if k not in d:
            continue
        j = d[k]
        bs.append(j["bytes_per_window"] / 1024.0); ys.append(j["auroc"]); cs.append(c)
    if bs:
        ax[0].plot(cs, ys, marker="s", color="C1")
        ax[0].axhline(NATIVE_AUROC - 0.02, ls=":", c="red", lw=1, label="-0.02 band")
        ax[0].axhline(NATIVE_AUROC, ls="--", c="grey", lw=1)
        ax[0].set_xlabel("H.264 CRF (480p)")
        ax[0].set_ylabel("LOCO AUROC")
        ax[0].set_title("RES.02 compression ladder")
        ax[0].legend(fontsize=7)
        ax[1].plot(bs, ys, marker="s", color="C1")
        for x, y, c in zip(bs, ys, cs):
            ax[1].annotate(f"crf{c}", (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax[1].set_xlabel("mean bytes / 10 s window (KiB)")
        ax[1].set_ylabel("LOCO AUROC")
        ax[1].set_title("accuracy vs bytes")
        fig.savefig(os.path.join(IMG, "compression_ladder.png"))
    plt.close(fig)

    # ---- sample frames per resolution ----
    sample = None
    for cand in ["480_23", "480_18"]:
        p = os.path.join(WORK, cand, "g001.mp4")
        if os.path.exists(p):
            sample = p
            break
    if sample:
        import subprocess
        src = json.load(open("/workspace/jadal_pod_bundle/labels/manifest_gold.json"))["items"]
        win = [i for i in src if i["id"] == "g001"][0]
        durs = win["end_s"] - win["start_s"]
        figs = []
        tags = []
        for r in [480, 360, 240, 180]:
            mp4 = os.path.join(WORK, f"{r}_23", "g001.mp4")
            if not os.path.exists(mp4):
                continue
            cmd = ["ffmpeg", "-v", "error", "-ss", str(durs / 2), "-i", mp4, "-frames:v", "1",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
            p = subprocess.run(cmd, stdout=subprocess.PIPE)
            w = h = r
            if len(p.stdout) < w * h * 3:
                continue
            a = np.frombuffer(p.stdout[:w * h * 3], dtype=np.uint8).reshape(h, w, 3)
            figs.append(a); tags.append(f"{r}p ({os.path.getsize(mp4)/1024:.0f} KiB)")
        if figs:
            n = len(figs)
            fig, axs = plt.subplots(1, n, figsize=(2.4 * n, 2.6))
            if n == 1:
                axs = [axs]
            for a, t, ax in zip(figs, tags, axs):
                ax.imshow(a); ax.set_title(t, fontsize=8); ax.axis("off")
            fig.suptitle("g001 mid-frame after re-encode (decode target is 160x90)", fontsize=9)
            fig.savefig(os.path.join(IMG, "sample_frames.png"), dpi=78)
            plt.close(fig)
    print("wrote images + plot_data.csv from", len(d), "configs")


if __name__ == "__main__":
    main()
