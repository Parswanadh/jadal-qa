"""Render the decider report tables from result JSONs (stdlib only).

This is the generator behind the "GENERATED from eval/results/decider_baseline.json"
claim in docs/reports/decider.md: every number in the report tables comes out
of this script, never hand-copied.

Usage (from repo root)::

    PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/render_decider_report.py \
        [--baseline eval/results/decider_baseline.json] [--bench eval/results/bench_eg2.json]

Prints markdown: head metrics table, bound table, codec-shift line, bench line.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / "eval" / "results" / "decider_baseline.json"
DEFAULT_BENCH = REPO_ROOT / "eval" / "results" / "bench_eg2.json"


def render_baseline_tables(baseline: dict) -> str:
    lines = [
        "| head | split | n | acc | ECE | T |",
        "|---|---|---|---|---|---|",
    ]
    for head in ("hands_ok", "active_work"):
        for split in ("calibration", "test"):
            m = baseline["heads"][head][split]
            lines.append(
                f"| {head} | {split} | {m['n']} | {m['accuracy']:.4f} | "
                f"{m['ece']:.4f} | {m['temperature']:.4f} |")
    lines += [
        "",
        "| head | cal threshold | test accepted | test errors | error upper bound (1-delta, delta=0.1) |",
        "|---|---|---|---|---|",
    ]
    for head in ("hands_ok", "active_work"):
        b = baseline["heads"][head]["held_out_bound_at_cal_threshold"]
        lines.append(
            f"| {head} | {b['threshold']} | {b['accepted']} | {b['errors']} | {b['upper_bound']} |")
    return "\n".join(lines)


def render_codec_shift_line(baseline: dict) -> str:
    c = baseline.get("codec_shift")
    if not c:
        return "codec_shift: not measured"
    return (f"codec_shift n={c['n']}: mean={c['mean']:.4f} min={c['min']:.4f} "
            f"p5={c['p5']:.4f} p50={c['p50']:.4f}")


def render_bench_line(bench: dict | None) -> str:
    if not bench:
        return "bench: eval/results/bench_eg2.json not present"
    per = {r["budget"]: r for r in bench["per_budget"]}
    bits = []
    for budget in (70, 140, 280):
        r = per[budget]
        bits.append(f"{budget}: median {r['median_s_per_frame']:.4f} / "
                    f"p95 {r['p95_s_per_frame']:.4f} / {r['peak_vram_mb_batch1']:.1f} MB / "
                    f"batch-diff {r['batch1_vs_batch5_max_abs_diff']}")
    return ("bench (real frames, bf16, alone on GPU): " + "; ".join(bits) +
            f" -> {bench['go_nogo_140']} at 140 tokens "
            f"({bench['median_s_per_frame_at_140']:.4f} s/frame)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--bench", default=str(DEFAULT_BENCH))
    args = ap.parse_args()
    baseline = json.loads(Path(args.baseline).read_text())
    bench = json.loads(Path(args.bench).read_text()) if Path(args.bench).exists() else None
    print(render_baseline_tables(baseline))
    print()
    print(render_codec_shift_line(baseline))
    print()
    print(render_bench_line(bench))
    print()
    print(f"provenance: {baseline.get('provenance')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
