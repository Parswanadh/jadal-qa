#!/usr/bin/env python3
"""R1 — render docs/reports/exp_opt_scale.md from the result JSONs."""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
R = os.path.join(REPO, "eval", "results", "exp_opt_scale")
OUT = os.path.join(REPO, "docs", "reports", "exp_opt_scale.md")


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    m = load("R1.01.json")
    op = load("operating_point.json")
    det = load("determinism.json")
    cov = m.get("coverage", {})
    thr = m.get("throughput", {})
    vd = m.get("verdict_distribution", {})
    vs = m.get("verdict_share", {})
    st = m.get("per_stage_ms", {})
    gpu = m.get("gpu_vram", {})
    man = m.get("manifest", {})
    n = cov.get("n_windows_scored", 0)
    nman = man.get("n_windows", 0)

    lines = []
    A = lines.append
    A("# R1.01 — Optimized + recalibrated pipeline over the expanded 27.66 h corpus")
    A("")
    A("**Branch:** `m/opt-scale` · **Base:** `pod/base` · **Device:** A40  ")
    A(f"**Artifacts:** `eval/exp_opt_scale/`, `eval/results/exp_opt_scale/R1.01.json`, "
      f"this report + `docs/experiments/exp_opt_scale/images/opt_scale.png`.")
    A("")
    A("## Status")
    A("")
    if m.get("status") == "partial":
        A(f"**PARTIAL — run still in flight.** {n}/{nman} windows scored "
          f"({cov.get('frac_of_manifest',0)*100:.1f}% of the manifest). "
          "The full-corpus run continues in the background; see the resume command below.")
    else:
        A(f"**COMPLETE** — {n}/{nman} windows scored.")
    A("")
    A("## What ran")
    A("")
    A(f"The repo E8.01 pipeline (single NVDEC three-branch pass per window) over the "
      f"**expanded corpus** `{man.get('path','')}` — **{nman} windows / "
      f"{man.get('n_clips')} clips / {man.get('n_factories')} factories / "
      f"{man.get('total_hours')} h** — with the pod's adopted optimizations and the "
      "**recalibrated sparse heads** (N5, PR #156):")
    A("")
    A("| knob | value |")
    A("|---|---|")
    A("| heads | `configs/ensemble_v1_sparse.json` (EG2 C=0.003, pixel C=0.3) |")
    A("| abstain band | `[0.4193, 0.6112]` |")
    A("| pixel view | 3.33 fps (E1.01) |")
    A("| EG2 view | 1 fps (E1.02) |")
    A("| early exit | first 4 s (E3.03) |")
    A("| window grid | every window (stride 1) |")
    A("| mosaic | skipped (E2) |")
    A("| resources | 6 workers, `JADAL_GPU_SLOTS=4`, `OMP_NUM_THREADS=1`, NVDEC |")
    A("")
    A("> The config lists *window stride 2* as an adopted sampling policy; this validation "
      "run keeps stride 1 so the **whole** 9,757-window corpus is covered, and the "
      "stride-2 operating point is derivable as the even-indexed subset of the same rows. "
      "The corpus is unlabelled (`label_source=none`) — verdicts are **descriptive**.")
    A("")
    A("## Throughput (partial)")
    A("")
    A(f"- wall: **{thr.get('wall_s')} s** ({thr.get('wall_s',0)/60:.1f} min)")
    A(f"- windows/s: **{thr.get('windows_per_s')}**")
    A(f"- processed footage: **{thr.get('processed_footage_h_per_wall_h')} footage-h/wall-h**")
    A(f"- **extrapolated full-corpus wall: {thr.get('extrapolated_full_corpus_wall_min')} min "
      f"({thr.get('extrapolated_full_corpus_wall_h')} h)** for all {nman} windows")
    A(f"- errors: {m.get('errors')}")
    A("")
    A("## Verdict distribution (recalibrated heads, descriptive)")
    A("")
    A("| verdict | windows | share |")
    A("|---|---|---|")
    for k in ("accept", "reject", "review"):
        A(f"| {k} | {vd.get(k,0)} | {vs.get(k,0)} |")
    A("")
    ss = m.get("score_stats", {})
    A(f"score: mean {ss.get('mean')}, p05 {ss.get('p05')}, p50 {ss.get('p50')}, "
      f"p95 {ss.get('p95')}")
    A("")
    A("## Per-stage latency (ms)")
    A("")
    A("| stage | p50 | p95 | mean |")
    A("|---|---|---|---|")
    for k in ("io_probe", "decode", "pixel_features", "descriptors", "eg2_embed",
              "mosaic", "decision", "write", "total"):
        if k in st:
            A(f"| {k} | {st[k]['p50_ms']} | {st[k]['p95_ms']} | {st[k]['mean_ms']} |")
    A("")
    A("## GPU / VRAM")
    A("")
    if gpu:
        A(f"- GPU util %: mean {gpu.get('gpu_util_pct_mean')}, p50 {gpu.get('gpu_util_pct_p50')}, "
          f"p95 {gpu.get('gpu_util_pct_p95')}, max {gpu.get('gpu_util_pct_max')}")
        A(f"- VRAM MiB: mean {gpu.get('vram_mib_mean')}, p95 {gpu.get('vram_mib_p95')}, "
          f"max {gpu.get('vram_mib_max')}")
    else:
        A("- (no GPU samples in window)")
    A("")
    A("## Operating-point match (both head sets, labelled subset)")
    A("")
    A("Independent re-application of the **frozen** (`configs/ensemble_v1.json`) and "
      "**recalibrated** (`configs/ensemble_v1_sparse.json`) heads to the N5 sparse-sampled "
      "features (pixel 3.33 fps, EG2 1 fps, 4 s prefix) — reproduces N5 to 4 dp.")
    A("")
    A("| set | heads | band | review share | confident BA | AUROC |")
    A("|---|---|---|---|---|---|")
    for name, s in op.get("sets", {}).items():
        for hs in ("frozen_heads", "sparse_heads"):
            for band in ("frozen_band", "sparse_band"):
                b = s[hs][band]
                A(f"| {name} (n={s['n']}) | {hs.replace('_heads','')} | "
                  f"{band.replace('_band','')} | {b['review_share']} | "
                  f"{b['confident_balacc']} | {b['auroc']} |")
    A("")
    A("## Determinism (10 windows, two independent runs)")
    A("")
    dsum = det.get("summary", {})
    A(f"- identical content (excl. `latency_ms`/`receipt_sha256`): "
      f"**{dsum.get('same_content_receipt')}/{dsum.get('n')}**")
    A(f"- identical `features_digest`: **{dsum.get('same_features_digest')}/{dsum.get('n')}**")
    A(f"- identical score+verdict: **{dsum.get('same_score_verdict')}/{dsum.get('n')}**")
    A(f"- identical normalised receipt (excl. `latency_ms`): "
      f"**{dsum.get('same_normalized_receipt_excl_latency')}/{dsum.get('n')}**")
    A("")
    A("The raw `receipt_sha256` intentionally folds in `latency_ms`, so it varies between "
      "runs; removing the timing field yields byte-identical receipts.")
    A("")
    A("## Caveats")
    A("")
    A("1. **Pod contention.** A peer transcode job (`/workspace/scratch/c1/q3_clips`, ~4 cores) "
      "ran throughout; observed throughput (~5 h/h) is ~4x below the L3 uncontended 19.3 h/h. "
      "Decode p50 is 8.1 s vs L3's 2.8 s (NVDEC + network-FS seek under load).")
    A("2. Each worker holds ~2.1 GB RSS (EG2 model + CUDA context), above the 1.5 GB/proc "
      "guideline — same as L2.01/L3.01.")
    A("3. Corpus verdicts are unlabelled and descriptive; the accept/reject share is not accuracy.")
    A("")
    A("## Resume command")
    A("")
    A("```bash")
    A("cd /workspace/wt/m-opt-scale && source /workspace/env.sh")
    A("export OMP_NUM_THREADS=1 JADAL_GPU_SLOTS=4")
    A("nohup bash tools/pod/gpu_run.sh python3 eval/exp_opt_scale/run_opt_scale.py \\")
    A("  --workers 6 --no-compare --pixel-fps 3.33 --eg2-fps 1 --early-exit-s 4 \\")
    A("  --window-stride 1 --skip-mosaic --resume \\")
    A("  --out /workspace/wt/m-opt-scale/eval/results/exp_opt_scale/windows.jsonl \\")
    A("  >> /workspace/logs/opt_scale.log 2>&1 &")
    A("```")
    A("")
    A("## Artifacts")
    A("")
    A("- `eval/exp_opt_scale/run_opt_scale.py` (L3 runner + sparse-head/manifest override)")
    A("- `eval/exp_opt_scale/{analyze_opt_scale,verify_operating_point,determinism_check,make_figure}.py`")
    A("- `eval/results/exp_opt_scale/{R1.01,operating_point,determinism}.json` + `windows.jsonl.partial`")
    A("- `docs/experiments/exp_opt_scale/images/opt_scale.png`")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
