# Z1 — mosaic delta + zero-copy GPU path + decode verdict

**Device:** A40 · **Branch:** `m/zerocopy` · **Runner:** `eval/exp_zerocopy/run_zerocopy.py`

> Mosaic is an evidence artefact, not a model input. Zero-copy: the repo decode is an out-of-process ffmpeg pipe (host memory only), so true VRAM-residency needs an in-process NVDEC binding; wherever that was not possible it is reported, not faked. All numbers carry n.

## Z1.01 Mosaic ON vs OFF

Same 30-window sample (spread over 30 factories), full repo pipeline run twice; only the mosaic build differs.

- Mosaic stage (the only difference between the two passes): mean 233.22 ms, p50 227.46 ms per window → **clean wall delta 233.22 ms/window**.
- Two independent passes cross-check: ON mean 5222.47 ms vs OFF mean 4475.84 ms → delta 746.62 ± 1730.65 ms (noisy: run-to-run variance / pod contention; the mosaic build is the only code difference).
- Verdicts identical: **True** (disagreements 0, `features_digest` mismatches 0, score mismatches 0).

![mosaic](../experiments/exp_zerocopy/images/mosaic_on_off.png)

## Z1.02 Zero-copy GPU path vs hwdownload

- Mode: `upload` (torchcodec available: True).
- **Blockers:** `{"torchcodec_runtime": "RuntimeError: Unsupported device: cuda"}`

| stage | hwdownload mean ms | GPU mean ms | speedup |
|---|---|---|---|
| decode | 1185.3 | 1224.38 | 0.968x |
| pixel_features | 821.57 | 18.46 | 44.516x |
| descriptors | 800.97 | 715.41 | 1.12x |
| eg2_embed | 581.26 | 393.09 | 1.479x |
| total | 3402.05 | 2356.13 | 1.444x |

- EG2 embedding cosine vs the repo `encode()` path: mean 1.0 (p50 1.0).
- Verdict agreement: 29/30 (0.967).
- PCIe bytes/window: base 80421120 vs GPU 93381120 → **saved -12960000 B/window** (-12.96 MB).
- Largest pixel-feature |Δ|: `radial_slope` 0.17597, `dup_hash_mean` 0.13611, `pca_top1_share` 0.02302, `illum_r` 0.00669, `illum_b` 0.00418, `illum_g` 0.00342.

![zerocopy](../experiments/exp_zerocopy/images/zerocopy_stages.png)

## Z1.03 Decode verdict (NVDEC vs CPU)

| arm | CPU p50 ms | NVDEC+hwdownload p50 ms | NVDEC no-download p50 ms | hwdownload cost |
|---|---|---|---|---|
| h264_480p_160x90 | 227.7 | 705.2 | 696.0 | 9.2 |
| h264_480p_native | 233.0 | 771.7 | 680.7 | 91.0 |
| hevc_1080p_160x90 | 4067.1 | 832.0 | 847.3 | -15.3 |
| hevc_1080p_native | 3209.4 | 4403.4 | 839.3 | 3564.1 |

![decode](../experiments/exp_zerocopy/images/decode_verdict.png)

## Upload-compressed input (deployment may choose the upload spec)

If upload-side encoding is allowed, the codec/resolution/quality is a free variable. Ladder on the same 8-window sample (feature/EG2 costs are resolution-invariant: the pipeline always works at 160x90 / 512x288, so only decode is spec-dependent).

| spec | bitrate kbps | B/frame | decode NVDEC p50 ms | decode NVDEC no-dl p50 ms | decode CPU p50 ms | total base ms | total GPU ms | footage-h/wall-h (GPU) |
|---|---|---|---|---|---|---|---|---|
| 1080p30_hevc | 4694.9 | 19562.1 | 904.0 | 804.4 | 3637.1 | 2589.0 | 1597.7 | 6.259 |
| 1080p_h264_crf23 | 12891.8 | 53715.8 | 1104.5 | 1040.9 | 3076.6 | 2510.2 | 1518.9 | 6.584 |
| 480p_h264_crf23 | 3190.9 | 13295.3 | 648.3 | 649.2 | 744.7 | 2075.7 | 1084.4 | 9.222 |
| 360p_h264_crf28 | 1067.8 | 4449.1 | 749.3 | 699.0 | 300.0 | 2123.6 | 1132.3 | 8.832 |
| 240p_h264_crf28 | 526.6 | 2194.4 | 1689.5 | 1667.4 | 779.8 | 3107.1 | 2115.8 | 4.726 |

- **Recommended upload spec:** `480p_h264_crf23` (best footage-h/wall-h among specs with a validated accuracy budget; X1 measured 360p@CRF28 = ~40% bytes with ΔAUROC < 0.02; 240p accuracy was **not** validated, so it is listed but not recommended).
- NVDEC-vs-CPU flip: `cpu_over_nvdec_p50` < 1 means CPU is faster at that spec; see the table column and Z1.03.
- **End-to-end extrapolation to the full corpus** (1,546 windows / 4.404 h) for the recommended combo `480p_h264_crf23` × NVDEC decode × GPU feature/embed (1084.4 ms/window): **27.9 min on 1 worker** (9.457 footage-h/wall-h), ~7.0 min at 4 workers ideal (37.828 footage-h/wall-h). ideal linear worker scaling; real scaling is GPU-bound (EG2 forward) and measured at 4 workers in L2.01.

![upload specs](../experiments/exp_zerocopy/images/upload_specs.png)

## Reproduction

```bash
source /workspace/env.sh
export OMP_NUM_THREADS=2 JADAL_GPU_SLOTS=4
bash tools/pod/gpu_run.sh python3 eval/exp_zerocopy/run_zerocopy.py --sample-n 30
python3 eval/exp_zerocopy/analyze_zerocopy.py
```

Verdicts are descriptive (the corpus is unlabelled). n=30 windows, one per factory; per-stage timings are wall-clock on a shared A40.

