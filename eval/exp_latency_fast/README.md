# eval/exp_latency_fast — L3.01 full-corpus run at full resources + adopted optimizations

Runs the **repo's E8.01 pipeline** over all **1,546** Egocentric-10K scale
windows with `time.perf_counter` around every stage, at **full resources**
(`--workers 7` on the 7.65-core cgroup, `JADAL_GPU_SLOTS=6`, NVDEC three-branch
decode), with the pod's **adopted sampling optimizations** applied as flags.

It does **not** re-implement the pipeline: it imports `eval/exp_rescore/run_e8.py`
(decode, pixel-26, EG2, descriptors), `eval/exp_regimes/descriptors.py`,
`jadal/regimes/router.py`, `jadal/ensemble/scorer.py`, `eval/exp_rescore/contract.py`
and `jadal/mosaic/build.py`, and only inserts timing around them. **The scoring
code is unchanged**, so verdicts stay comparable to the frozen baseline.

This is the optimized sibling of `eval/exp_latency_full/` (L2.01, the frozen
baseline: 4 workers, all frames, 2 fps EG2, 10 s window, mosaic on).

## Adopted optimizations (flags / env)

| id | change | flag | env |
|---|---|---|---|
| E1.01 | pixel-26 view at 3.33 fps (~34 vs ~300 frames) | `--pixel-fps 3.33` | `JADAL_PIXEL_FPS` |
| E1.02 | EG2 view at 1 fps (10 vs 20 images) | `--eg2-fps 1` | `JADAL_EG2_FPS` |
| E3.03 | observe only the first 4 s of a 10 s window | `--early-exit-s 4` | `JADAL_EARLY_EXIT_S` |
| E3.01 | every-2nd-window grid | `--window-stride 2` | — |
| E2 | skip the mosaic evidence artifact | `--skip-mosaic` | `JADAL_SKIP_MOSAIC=1` |
| X1 | 360p@CRF28 source | — | not applied (see report) |

Defaults (`pixel_fps=0`, `eg2_fps=2`, `native_fps=0.8`, `early_exit_s=0`,
`stride=1`, mosaic on) reproduce the L2.01 baseline decode graph exactly; the
`tests/test_exp_latency_fast.py` suite asserts the ffmpeg filter graph for both
the defaults and the adopted settings.

## Stages (one JSONL row / window)

| # | stage | what is timed |
|---|---|---|
| 1 | `io_probe` | `open()` + `seek()` to the window byte offset + 64 KiB `read()` |
| 2 | `decode` | one NVDEC (`-hwaccel cuda`) + `scale_cuda` pass, 3 branches (pixel @ `--pixel-fps`, EG2 @ `--eg2-fps`, native descriptors @ `--native-fps`) |
| 3 | `pixel_features` | `jadal.gpu.features.cpu_window_features` (the 26 features) |
| 4 | `descriptors` | E4.01 regime descriptors on the native frames (`ffprobe` meta cached) |
| 5 | `eg2_embed` | EG2 L2-normalised mean embedding (`--eg2-fps` fps) |
| 6 | `mosaic` | `jadal.mosaic.build` — **skipped** with `--skip-mosaic` |
| 7 | `decision` | `ensemble_prob` + profile router + `verdict_for` + receipt |
| 8 | `write` | parent JSONL append + flush |

## Run / analyse

```bash
source /workspace/env.sh
export OMP_NUM_THREADS=1 JADAL_GPU_SLOTS=6
# full corpus, optimized per-window (stride 1), then the every-2nd policy (stride 2):
bash eval/exp_latency_fast/run_both.sh
python3 eval/exp_latency_fast/analyze_fast.py   # L3.01.json + report + 2 figures
```

Outputs: `eval/results/exp_latency_fast/{windows.jsonl,windows_stride2.jsonl,clips.jsonl,run_meta*.json,L3.01.json}`,
`docs/reports/exp_latency_fast.md`, `docs/experiments/exp_latency_fast/images/`.
