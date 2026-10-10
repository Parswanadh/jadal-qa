# eval/exp_latency_full — L2.01 full-corpus PoC pipeline run (atomic latency)

Runs the **repo's E8.01 pipeline** over **all 1,546** Egocentric-10K scale
windows with `time.perf_counter` around every stage, and writes one JSONL row
per window (`eval/results/exp_latency_full/windows.jsonl`). It does **not**
re-implement the pipeline: it imports `eval/exp_rescore/run_e8.py` (decode,
pixel-26, EG2, descriptors), `eval/exp_regimes/descriptors.py`,
`jadal/regimes/router.py`, `jadal/ensemble/scorer.py`, `eval/exp_rescore/contract.py`
and `jadal/mosaic/build.py`, and only inserts timing around them.

## Stages (one row / window)

| # | stage | what is timed |
|---|---|---|
| 1 | `io_probe` | `open()` + `seek()` to the window byte offset + 64 KiB `read()` |
| 2 | `decode` | repo path: ONE NVDEC (`-hwaccel cuda`) + `scale_cuda` pass, 3 branches (pixel 160x90 all frames, EG2 512x288 @2 fps, native 1920x1080 @0.8 fps) |
| 3 | `pixel_features` | `jadal.gpu.features.cpu_window_features` (the 26 features) |
| 4 | `descriptors` | E4.01 regime descriptors on the native frames (`ffprobe` meta cached) |
| 5 | `eg2_embed` | EG2 L2-normalised mean embedding, 2 fps (EmbeddingGemma-2, bf16) |
| 6 | `mosaic` | `jadal.mosaic.build` 8-tile 224 px mosaic, **only if the routed profile sets `evidence.mosaic`** |
| 7 | `decision` | `ensemble_prob` + profile router + `verdict_for` + `reasons_for` + receipt |
| 8 | `write` | parent JSONL append + flush (timed in the parent) |

`decode` also runs a **comparison** on the first window of every clip: CPU
`ffmpeg`/swscale vs explicit `ffmpeg -hwaccel cuda -c:v hevc_cuvid` + `scale_cuda`
(one 160x90 branch each).

## Reproduction

Verdicts / `score` / `features_digest` use the same code path as E8.01, so they
are identical to `eval/results/exp_rescore/E8.01.json`. Only `latency_ms` carries
more stages, so `receipt_sha256` differs from E8.01 — that is expected and
documented. The EG2 model dir is patched **in memory** to the pod cache
(`.../snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b`): a post-merge commit
reverted `configs/ensemble_v1.json` to the laptop path; the head *numbers* are
unchanged (verified against the frozen bundle artifact by `run_e8.load_config_checked`).

## Run / analyse

```bash
source /workspace/env.sh
export OMP_NUM_THREADS=1
bash tools/pod/gpu_run.sh python3 eval/exp_latency_full/run_full.py --workers 4 --resume
python3 eval/exp_latency_full/analyze_full.py     # L2.01.json + 2 figures
```

Outputs: `eval/results/exp_latency_full/{windows.jsonl,clips.jsonl,run_meta.json,L2.01.json}`,
`docs/reports/exp_latency_full.md`, `docs/experiments/exp_latency_full/images/`.
