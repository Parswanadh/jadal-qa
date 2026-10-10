#!/usr/bin/env python3
"""Render the Y2 review-gate figure from the result JSONs (no recompute)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "eval/results/exp_review"
OUT = ROOT / "docs/experiments/exp_review/images"


def main() -> int:
    r1 = json.loads((RES / "Y2.01.json").read_text())
    r2 = json.loads((RES / "Y2.02.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {1: "#1f77b4", 2: "#ff7f0e", 5: "#d62728"}

    # Panel A: review% vs balanced accuracy (decided subset)
    for r in (1, 2, 5):
        for variant, ls, alpha in (("raw", "-", 1.0),
                                   ("loco_temperature", "--", 0.6)):
            rows = [m for m in r1["gold_sweep"]
                    if m["cost_ratio_fa_fr"] == r and m["variant"] == variant]
            xs = [100 * (1 - m["coverage"]) for m in rows]
            ys = [m["balanced_accuracy"] for m in rows]
            ax.plot(xs, ys, ls, color=colors[r], alpha=alpha,
                    label=f"gold cost {r}:1{'' if variant=='raw' else ' (LOCO T)'}")
    # OOD eval-only, cost 1
    rows = [m for m in r1["ood_sweep"]
            if m["cost_ratio_fa_fr"] == 1 and m["variant"] == "raw"]
    ax.plot([100 * (1 - m["coverage"]) for m in rows],
            [m["balanced_accuracy"] for m in rows],
            ":", color="black", marker=".", ms=4,
            label="OOD-60 (eval only, cost 1:1)")
    ax.axvline(60, color="grey", lw=1, alpha=0.6)
    ax.text(60.5, 0.52, "60% review\ntarget", fontsize=8, color="grey")
    ax.axhline(0.5, color="grey", lw=0.7, ls=":")
    # mark recommended (cost 1:1, coverage 0.45)
    rec_ba = next(m["balanced_accuracy"] for m in r1["gold_sweep"]
                  if m["cost_ratio_fa_fr"] == 1 and m["coverage"] == 0.45
                  and m["variant"] == "raw")
    ax.scatter([55], [rec_ba], color="green", zorder=5, s=45,
               label="recommended (55% review)")
    ax.set_xlabel("REVIEW share of the decided set (%)\n(= 100 - coverage)")
    ax.set_ylabel("balanced accuracy on decided windows")
    ax.set_ylim(0.4, 0.95)
    ax.set_title("A. Gating: review budget vs error (gold LOCO)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)

    # Panel B: pipeline levers on the 1,546 windows
    p = r2["pipeline"]
    labels = ["frozen policy\n(E8.01)", "a) gating\n(review=55%)",
              "b) + stage-0", "c) + temperature"]
    vals = [100 * p["base_frozen_policy"]["review_share"],
            100 * p["a_gating_alone"]["review_share"],
            100 * p["b_plus_stage0"]["review_share"],
            100 * p["c_plus_calibration"]["review_share"]]
    bars = ax2.bar(labels, vals, color=["#888", "#1f77b4", "#2ca02c", "#9467bd"])
    ax2.axhline(60, color="red", ls="--", lw=1.2, label="60% target")
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%",
                 ha="center", fontsize=9)
    ax2.set_ylabel("REVIEW share (%)")
    ax2.set_ylim(0, 85)
    ax2.set_title("B. Review share on the 1,546 Egocentric-10K windows")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Y2 review reduction — decision-theoretic gating + stage-0 triage",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"review_gate.{ext}", dpi=140)
    print("wrote", OUT / "review_gate.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
