# R1.01 — Optimized + recalibrated pipeline over the expanded 27.66 h corpus

**Branch:** `m/opt-scale` · **Base:** `pod/base` · **Device:** A40  
**Artifacts:** `eval/exp_opt_scale/`, `eval/results/exp_opt_scale/R1.01.json`, this report + `docs/experiments/exp_opt_scale/images/opt_scale.png`.

## Status

**PARTIAL — run still in flight.** 830/9757 windows scored (8.5% of the manifest). The full-corpus run continues in the background; see the resume command below.

## What ran

The repo E8.01 pipeline (single NVDEC three-branch pass per window) over the **expanded corpus** `/workspace/jadal_pod_bundle/queues/manifest_scale_all.json` — **9757 windows / 391 clips / 85 factories / 27.661 h** — with the pod's adopted optimizations and the **recalibrated sparse heads** (N5, PR #156):

| knob | value |
|---|---|
| heads | `configs/ensemble_v1_sparse.json` (EG2 C=0.003, pixel C=0.3) |
| abstain band | `[0.4193, 0.6112]` |
| pixel view | 3.33 fps (E1.01) |
| EG2 view | 1 fps (E1.02) |
| early exit | first 4 s (E3.03) |
| window grid | every window (stride 1) |
| mosaic | skipped (E2) |
| resources | 6 workers, `JADAL_GPU_SLOTS=4`, `OMP_NUM_THREADS=1`, NVDEC |

> The config lists *window stride 2* as an adopted sampling policy; this validation run keeps stride 1 so the **whole** 9,757-window corpus is covered, and the stride-2 operating point is derivable as the even-indexed subset of the same rows. The corpus is unlabelled (`label_source=none`) — verdicts are **descriptive**.

## Throughput (partial)

- wall: **1509.0 s** (25.1 min)
- windows/s: **0.55**
- processed footage: **5.5 footage-h/wall-h**
- **extrapolated full-corpus wall: 295.6 min (4.93 h)** for all 9757 windows
- errors: 0

## Verdict distribution (recalibrated heads, descriptive)

| verdict | windows | share |
|---|---|---|
| accept | 210 | 0.253 |
| reject | 100 | 0.1205 |
| review | 520 | 0.6265 |

score: mean 0.4853, p05 0.2131, p50 0.3891, p95 0.8885

## Per-stage latency (ms)

| stage | p50 | p95 | mean |
|---|---|---|---|
| io_probe | 13.8 | 94.7 | 27.0 |
| decode | 7590.4 | 18100.9 | 8032.2 |
| pixel_features | 214.9 | 706.3 | 288.8 |
| descriptors | 1402.7 | 4538.7 | 1729.3 |
| eg2_embed | 463.8 | 1112.9 | 737.9 |
| mosaic | 0 | 0 | 0.0 |
| decision | 1.0 | 1.7 | 2.7 |
| write | 0 | 0 | None |
| total | 10002.5 | 23568.7 | 10819.1 |

## GPU / VRAM

- GPU util %: mean 12.37, p50 1.0, p95 90.0, max 100.0
- VRAM MiB: mean 13489.7, p95 16740.6, max 16942.0

## Operating-point match (both head sets, labelled subset)

Independent re-application of the **frozen** (`configs/ensemble_v1.json`) and **recalibrated** (`configs/ensemble_v1_sparse.json`) heads to the N5 sparse-sampled features (pixel 3.33 fps, EG2 1 fps, 4 s prefix) — reproduces N5 to 4 dp.

| set | heads | band | review share | confident BA | AUROC |
|---|---|---|---|---|---|
| gold150 (n=150) | frozen | frozen | 0.1867 | 0.7632 | 0.9204 |
| gold150 (n=150) | frozen | sparse | 0.1933 | 0.7856 | 0.9204 |
| gold150 (n=150) | sparse | frozen | 0.1867 | 0.9073 | 0.9332 |
| gold150 (n=150) | sparse | sparse | 0.1867 | 0.9015 | 0.9332 |
| ood160 (n=160) | frozen | frozen | 0.15 | 0.5563 | 0.6347 |
| ood160 (n=160) | frozen | sparse | 0.1313 | 0.5617 | 0.6347 |
| ood160 (n=160) | sparse | frozen | 0.2062 | 0.5042 | 0.5231 |
| ood160 (n=160) | sparse | sparse | 0.15 | 0.5006 | 0.5231 |

## Determinism (10 windows, two independent runs)

- identical content (excl. `latency_ms`/`receipt_sha256`): **10/10**
- identical `features_digest`: **10/10**
- identical score+verdict: **10/10**
- identical normalised receipt (excl. `latency_ms`): **10/10**

The raw `receipt_sha256` intentionally folds in `latency_ms`, so it varies between runs; removing the timing field yields byte-identical receipts.

## Caveats

1. **Pod contention.** A peer transcode job (`/workspace/scratch/c1/q3_clips`, ~4 cores) ran throughout; observed throughput (~5 h/h) is ~4x below the L3 uncontended 19.3 h/h. Decode p50 is 8.1 s vs L3's 2.8 s (NVDEC + network-FS seek under load).
2. Each worker holds ~2.1 GB RSS (EG2 model + CUDA context), above the 1.5 GB/proc guideline — same as L2.01/L3.01.
3. Corpus verdicts are unlabelled and descriptive; the accept/reject share is not accuracy.

## Resume command

```bash
cd /workspace/wt/m-opt-scale && source /workspace/env.sh
export OMP_NUM_THREADS=1 JADAL_GPU_SLOTS=4
nohup bash tools/pod/gpu_run.sh python3 eval/exp_opt_scale/run_opt_scale.py \
  --workers 6 --no-compare --pixel-fps 3.33 --eg2-fps 1 --early-exit-s 4 \
  --window-stride 1 --skip-mosaic --resume \
  --out /workspace/wt/m-opt-scale/eval/results/exp_opt_scale/windows.jsonl \
  >> /workspace/logs/opt_scale.log 2>&1 &
```

## Artifacts

- `eval/exp_opt_scale/run_opt_scale.py` (L3 runner + sparse-head/manifest override)
- `eval/exp_opt_scale/{analyze_opt_scale,verify_operating_point,determinism_check,make_figure}.py`
- `eval/results/exp_opt_scale/{R1.01,operating_point,determinism}.json` + `windows.jsonl.partial`
- `docs/experiments/exp_opt_scale/images/opt_scale.png`
