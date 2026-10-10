#!/usr/bin/env python3
"""R1 — figure: cumulative throughput + extrapolation, verdicts, per-stage latency."""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
OUTDIR = os.path.join(REPO, "eval", "results", "exp_opt_scale")
IMGDIR = os.path.join(REPO, "docs", "experiments", "exp_opt_scale", "images")


def main():
    rows = []
    for ln in open(os.path.join(OUTDIR, "windows.jsonl.partial")):
        try:
            r = json.loads(ln)["row"]
        except Exception:  # noqa: BLE001
            continue
        rows.append(r)
    meta = json.load(open(os.path.join(OUTDIR, "R1.01.json")))
    t0 = min(r["t_wall0"] for r in rows)
    t1 = np.array([r["t_wall1"] - t0 for r in rows])
    order = np.argsort(t1)
    t1 = t1[order]
    cum = np.arange(1, len(t1) + 1)
    n_manifest = meta["manifest"]["n_windows"]
    wpsec = meta["throughput"]["windows_per_s"]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].plot(t1, cum, lw=1.6, color="#1f77b4", label="completed windows")
    ax[0].plot([0, t1[-1]], [0, t1[-1] * wpsec], "--", color="gray", lw=1,
               label=f"fit {wpsec:.3f} win/s")
    tx = n_manifest / wpsec
    ax[0].axhline(n_manifest, color="green", ls=":", lw=1)
    ax[0].annotate(f"manifest {n_manifest}",
                   (0.02, n_manifest * 1.02), color="green", fontsize=8)
    ax[0].annotate(f"extrapolated full-corpus wall\n{tx/60:.0f} min ({tx/3600:.2f} h)",
                   (t1[-1] * 0.35, n_manifest * 0.72), color="green", fontsize=9)
    ax[0].set_xlabel("wall seconds (since run start)")
    ax[0].set_ylabel("cumulative windows scored")
    ax[0].set_title(f"R1 partial throughput — {len(rows)}/{n_manifest} windows")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].grid(alpha=0.3)

    vd = meta["verdict_distribution"]
    keys = ["accept", "reject", "review"]
    vals = [vd.get(k, 0) for k in keys]
    cols = ["#2ca02c", "#d62728", "#ff7f0e"]
    ax[1].bar(keys, vals, color=cols)
    for i, v in enumerate(vals):
        ax[1].text(i, v, f"{v}\n({v/max(sum(vals),1)*100:.0f}%)", ha="center", va="bottom",
                   fontsize=9)
    ax[1].set_title("Verdict distribution (recalibrated heads, descriptive)")
    ax[1].set_ylabel("windows")
    ax[1].grid(alpha=0.3, axis="y")

    st = meta["per_stage_ms"]
    names = [k for k in ["io_probe", "decode", "pixel_features", "descriptors",
                         "eg2_embed", "mosaic", "decision", "total"] if k in st]
    p50 = [max(st[k]["p50_ms"], 0.1) for k in names]
    p95 = [max(st[k]["p95_ms"], 0.1) for k in names]
    y = np.arange(len(names))
    ax[2].barh(y - 0.2, p50, height=0.4, label="p50", color="#1f77b4")
    ax[2].barh(y + 0.2, p95, height=0.4, label="p95", color="#ff7f0e")
    ax[2].set_yticks(y)
    ax[2].set_yticklabels(names)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("ms (log)")
    ax[2].set_title("Per-stage latency (contended pod)")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3, axis="x")

    fig.suptitle("R1 — optimized + recalibrated pipeline over the expanded 27.66 h corpus",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(IMGDIR, exist_ok=True)
    out = os.path.join(IMGDIR, "opt_scale.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
