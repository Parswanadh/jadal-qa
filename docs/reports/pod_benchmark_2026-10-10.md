# Full-corpus pipeline benchmark — 2026-10-10 (pod A40)

**Status: MEASURED.** This is the end-to-end throughput benchmark of our PoC pipeline over the
entire Egocentric-10K corpus on the pod's NVIDIA A40.

## Headline (read this precisely)

> **4.404 hours of 1080p/30 fps HEVC footage (58 clips, 1,546 windows) processed in 801 s (13.4 min)
> → 19.8 footage-hours per wall-hour (≈20× real time).**
> Equivalently: **~3.0 min of wall time per footage-hour**, so **20 footage-hours ≈ 1 wall-hour**.

**This is NOT "20 hours of video in 13 minutes."** The corpus is 4.404 h; 13.4 min is the wall time
for that corpus. 19.8 h/h is the *rate*.

| metric | value |
|---|---|
| corpus | 4.404 h, 58 clips, 1,546 windows (1080p/30 fps HEVC) |
| wall time | **801 s (13.4 min)** |
| windows/s | **1.93** |
| **footage-h / wall-hour** | **19.79** (processed 19.29) |
| errors | 0 |

**Verdicts (automated):** accept **876 (56.7%)** · reject **40 (2.6%)** · review **630 (40.7%)**.
Note: 40.7% is routed to **human review** — the 19.8 h/h figure is the *automated processing* rate,
not a human-review rate.

## Configuration

- A40, `--workers 7` × `OMP_NUM_THREADS=1` (the pod's cgroup CPU quota is **7.65 cores**), `JADAL_GPU_SLOTS=6`.
- Decode: NVDEC (`-hwaccel cuda -c:v hevc_cuvid`) three-branch pass + `scale_cuda`.
- Adopted optimizations applied: **pixel view 3.33 fps** (E1.01), **EG2 view 1 fps** (E1.02),
  native 0.8 fps for descriptors, **4 s early exit** (E3.03), **mosaic off** (E2: dropped as a model input).
- **All windows processed** (`--window-stride 1`, `n_selected = n_manifest = 1,546`). The E3.01
  "every-2nd window" sampling policy was **not** applied to this run — the 19.8 h/h already covers
  the whole corpus; every-2nd would halve the window count and is a separate operating point.

## Atomic latency (per window, p50 / p95, share)

| stage | p50 ms | p95 ms | share |
|---|---|---|---|
| **decode (NVDEC 3-branch + hwdownload)** | **2,780** | **3,915** | **82%** |
| descriptors (native frames) | 298 | 492 | 9% |
| EG2 embed (1 fps) | 169 | 285 | 7% |
| pixel-26 features (3.33 fps) | 61 | 97 | 2% |
| io_probe | 1 | 3 | 0% |
| mosaic (skipped) | 0 | 0 | 0% |
| decision | 1 | 1 | 0% |
| **window total** | **3,357** | **4,626** | 100% |

**Bottleneck: decode (82%)** — the native 30 fps branch + the `hwdownload` (VRAM→RAM) that feeds the
CPU feature pass. The fps/mosaic optimizations removed the feature cost; decode is now the wall.

## Baseline vs optimized

| | baseline (4 workers, OMP=1, full fidelity) | **optimized** |
|---|---|---|
| windows/s | 0.80 | **1.93** |
| footage-h / wall-hour | 8.0 | **19.79** |
| pixel-26 p50 | 838 ms | **61 ms (−93%)** |
| EG2 p50 | 1,102 ms | **169 ms (−85%)** |
| mosaic | 203 ms | 0 (skipped) |
| decode p50 | 1,715 ms | 2,780 ms (unchanged path; now the bottleneck) |
| verdicts | accept 14% / reject 1% / review 85% | accept 56.7% / reject 2.6% / review 40.7% |

**End-to-end speedup: 2.5×.** The remaining gain is decode-bound.

## Honesty flags (must accompany any quote of these numbers)

1. **Rate vs corpus:** 19.8 footage-h *per wall-hour*; the 4.404 h corpus took 13.4 min. Do not state "20 h in 13 min".
2. **Review share:** 40.7% of windows still go to a human; throughput is automated processing.
3. **Verdict shift:** the optimized run's verdict mix (40.7% review) differs from the frozen baseline
   (75.5% review) because the fps subsampling shifts scores relative to the frozen abstain band — an
   **operating-point/accuracy change**, not just a speed-up. It needs recalibration (P1/Y1/O3) before
   quoting accuracy.
4. **No accuracy claim** is made from this run (the 1,546 windows are unlabelled); accuracy is measured
   separately on the 150 gold (LOCO) and the 160 OOD labels.

## Next steps (in flight)

- **Zero-copy GPU path (L4):** NVDEC → VRAM, GPU pixel-26, EG2 from device tensors (no `hwdownload`) —
  targets the 82% decode bottleneck; includes an **upload-compression** analysis (360p@CRF28 = ~40% bytes,
  X1) and NVDEC-vs-CPU flip points.
- **Deterministic physics model (Q1–Q5):** complete rule-list model + receipt determinism (Q3: **180/180
  identical receipts across 10 runs**) + full-corpus run.

## Corpus, labels and how to reproduce

**Corpus label.** Egocentric-10K (builddotai), 1080p/30 fps HEVC, **58 clips**,
**1,546 windows** (all windows; each window is one 10-s segment). Manifest
duration **4.404 h** (includes inter-window gaps); the 1,546 × 10 s of processed
footage is **4.294 h**. Windows are **unlabelled** — this is a throughput run,
not an accuracy run.

**Config label.** A40; `--workers 7` × `OMP_NUM_THREADS=1` (7.65-core cgroup
quota); `JADAL_GPU_SLOTS=6`; NVDEC 3-branch + `scale_cuda`;
`--pixel-fps 3.33 --eg2-fps 1 --early-exit-s 4 --window-stride 1 --skip-mosaic
--no-compare`. `run_meta.json.optimizations` records these exactly.

**Derivation (reproducible from the committed artifacts).**

```
footage_h_per_wall_h           = manifest.total_hours / (wall_s / 3600)
                               = 4.404 / (801.28 / 3600)      = 19.786
processed_footage_h_per_wall_h = (n_windows*10/3600) / (wall_s/3600)
                               = 4.294 / 0.2226               = 19.294
windows_per_s                  = n_windows / wall_s = 1546/801.28 = 1.929
```

All three match `run_meta.json` and `summary.json` exactly.

**How to reproduce the run.** The harness lives on branch `m/latency-fast`
(`eval/exp_latency_fast/`); it imports the repo E8.01 pipeline and only wraps
each stage with `time.perf_counter`:

```bash
source "${JADAL_ENV:-env.sh}"
export OMP_NUM_THREADS=1 JADAL_GPU_SLOTS=6
bash tools/pod/gpu_run.sh python3 eval/exp_latency_fast/run_fast.py \
      --workers 7 --pixel-fps 3.33 --eg2-fps 1 --early-exit-s 4 \
      --window-stride 1 --skip-mosaic --no-compare --resume
python3 eval/exp_latency_fast/analyze_fast.py
```

The per-stage breakdown (decode/descriptors/EG2/pixel/mosaic/decision) needs the
raw per-window JSONL (`m/latency-fast:eval/results/exp_latency_fast/windows.jsonl`,
1,546 rows), which is **not** committed here; the headline rates are fully
reproducible from the committed `run_meta.json` + `summary.json`.

**Provenance note.** `run_meta.json` records the pod-specific EG2 model path
(`pod_eg2_dir = /workspace/.cache/huggingface/...`) and the run commit
(`git_sha = ae2df45`, not this PR's head). Both are recorded evidence; exclude
them from any public snapshot.

## Artifacts

- `eval/results/exp_benchmark/summary.json` — this report's numbers (machine-readable)
- `eval/results/exp_benchmark/run_meta.json` — the runner's own metadata
- `eval/results/exp_benchmark/per_clip.csv` — per-clip window counts and timings
- raw per-window JSONL with atomic stage timings: branch `m/latency-fast`,
  `eval/results/exp_latency_fast/windows.jsonl.partial` (1,546 rows)
- baseline partial: branch `m/latency-full`, `eval/results/exp_latency_full/` (190 windows, PR #139)
