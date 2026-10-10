#!/usr/bin/env python3
"""S1.01 — figure: throughput and resource use vs K isolated shard workers."""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
IMG = os.path.join(REPO, "docs", "experiments", "exp_shard", "images")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        REPO, "eval", "results", "exp_shard", "S1.01.json")
    d = json.load(open(src))
    rows = d["scaling"]
    K = [r["K"] for r in rows]
    wps = [r["windows_per_s"] for r in rows]
    hh = [r["footage_h_per_wall_h"] for r in rows]
    cores = [r["cpu_cores_used"] for r in rows]
    dec = [r["gpu_dec_pct_mean"] or 0 for r in rows]
    sm = [r["gpu_sm_pct_mean"] or 0 for r in rows]
    vram = [(r["vram_used_mib_max"] or 0) / 1024.0 for r in rows]

    os.makedirs(IMG, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    a = ax[0]
    a.plot(K, wps, "o-", color="#1f77b4", label="windows/s")
    a.set_xlabel("K  (isolated shard workers)")
    a.set_ylabel("windows/s", color="#1f77b4")
    a.tick_params(axis="y", labelcolor="#1f77b4")
    a.set_xticks(K)
    a2 = a.twinx()
    a2.plot(K, hh, "s--", color="#d62728", label="footage-h / wall-h")
    a2.set_ylabel("footage-h / wall-h", color="#d62728")
    a2.tick_params(axis="y", labelcolor="#d62728")
    # ideal linear scaling from K=1
    a.plot(K, [wps[0] * k for k in K], ":", color="gray", label="ideal linear")
    a.set_title("Throughput vs K (ceiling curve)")
    a.legend(loc="upper left", fontsize=8)

    b = ax[1]
    b.plot(K, cores, "o-", color="#2ca02c", label="CPU cores used")
    b.axhline(d.get("cgroup_cpu_quota_cores", 7.65), ls=":", color="#2ca02c",
              label="cgroup quota")
    b.plot(K, [x / 100.0 * 8 for x in dec], "^--", color="#ff7f0e",
           label="NVDEC dec% (/100×8)")
    b.plot(K, [x / 100.0 * 8 for x in sm], "v--", color="#9467bd",
           label="GPU sm% (/100×8)")
    b.plot(K, vram, "d--", color="#8c564b", label="VRAM used GB")
    b.set_xlabel("K  (isolated shard workers)")
    b.set_ylabel("cores  /  scaled %  /  GB")
    b.set_xticks(K)
    b.set_title("Resource use vs K")
    b.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    out = os.path.join(IMG, "exp_shard_scaling.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
