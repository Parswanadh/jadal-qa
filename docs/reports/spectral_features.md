# Arm B — spectral features (RGB whole-video) + Arm C (hyperspectral check)

Worker: spectral-features branch (`feat/spectral-features`).
Method code: `jadal/spectral/{features,stream,dup}.py`. Runners:
`eval/spectral_features/{run_arm,run_dups,icvl_check}.py`.
Reproduce: `PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python
eval/spectral_features/run_arm.py --tiers T0,T1` (gold),
`--manifest manifest_synth.json --train-manifest manifest_gold.json` (synth),
`eval/spectral_features/run_dups.py` (dups).

## Method (arm B)

- **Whole-video, single pass.** One ffmpeg subprocess per segment (fast input
  seek + accurate output trim), scaled in C to the tier resolution, rgb24 pipe.
  **Every decoded frame** updates streaming accumulators (running spectra,
  Welford-style sums, per-frame scalars). No frame sampling.
- **T0** = 64×64 working resolution (cheapest). **T1** = 160×160.
  (Contract suggested 64×36/160×90 for 16:9; source here is square 480×480, so
  same pixel budgets in square form.)
- 40 scalar features in 8 families: `spatial_fft` (radial power slope, HF
  ratio), `spatial_dct` (whole-frame DCT HF energy), `wavelet` (db1 band
  ratios), `basic` (Laplacian/Sobel/brightness/contrast/under/over),
  `temporal` (per-pixel rFFT over time in 0.15–1/1–3/3–8 Hz bands, frozen
  fraction, flicker, global periodicity, frame-diff motion energy),
  `chroma` (Cr/Cb + R-G opponent + colorfulness), `skin` (YCrCb-rule area,
  its std/max, centroid motion — hand proxy, no MediaPipe), `compressed`
  (per-frame MV *counts* + key/non-key packet-size stats from a decode-free
  ffprobe pass).
- **Streaming:** `SpectralStream.feed()` takes 2-s BGR chunks, returns the
  running 40-vector (ring buffer = last 100 frames, O(1) in stream length).
- **Dup signature** (`dup.py`): 64-bit multi-scale DCT hashes (64px + 32px
  mean frames) + 32-float temporal fingerprint (mean-luma and diff series,
  resampled → rFFT). Similarity = 0.25·hash64 + 0.25·hash32 + 0.5·cosine(temp).
  NOT flip-invariant (by design — flips are non-duplicates per contract).
- Standalone head: logistic regression (C=1, per-fold StandardScaler), LOCO
  over clips for gold; gold-pooled head transferred to synth.

## Headline numbers

Label snapshot at run time: manifest_gold.json, n=79 (65 good / 14 bad),
only 3 of 7 clips labelled yet (desk, ecu_full, ecu_late). Protocol: LOCO,
clip-resampled bootstrap CI (2000x).

| set | tier | metric | value |
|---|---|---|---|
| gold human | T0 | LOCO AUROC (95% CI) | 0.637 (0.467, 0.693) |
| gold human | T1 | LOCO AUROC (95% CI) | 0.678 (0.500, 0.684) |
| synth (transfer, gold head) | T0 | AUROC clean-vs-degraded | 0.932 |
| synth (transfer, gold head) | T1 | AUROC clean-vs-degraded | 0.957 |
| dups (128 pairs, thr 0.84 on stratified half) | — | test P / R / F1 | 0.952 / 1.000 / 0.976 |

Gold verdict: **weak signal, CIs span 0.5** — with 14 bad labels over 3 clips
no honest claim beyond "temporal+skin carry the most signal" is supported.

## Ablation (LOCO AUROC; gold n=79)

| family | T0 leave-one-out | T0 only | T1 leave-one-out | T1 only |
|---|---|---|---|---|
| spatial_fft | 0.630 | 0.324 | 0.688 | 0.312 |
| spatial_dct | 0.640 | 0.351 | 0.681 | 0.345 |
| wavelet | 0.642 | 0.497 | 0.676 | 0.488 |
| basic | 0.690 | 0.403 | 0.711 | 0.391 |
| temporal | **0.563** | **0.681** | **0.545** | **0.729** |
| chroma | 0.686 | 0.614 | 0.691 | 0.613 |
| skin | 0.589 | **0.719** | 0.632 | **0.705** |
| compressed | 0.615 | 0.420 | 0.677 | 0.416 |
| full (all 40) | 0.637 | — | 0.678 | — |

Read: dropping `temporal` hurts most; `skin`-only and `temporal`-only beat the
full 40-feature model (small-data overfit — expected with 14 bad labels).
Spatial-FFT/DCT-only and compressed-only are at/below chance alone.

Synth transfer, family-only AUROC (gold head → synth, clean vs degraded):
spatial_dct 0.81/0.79, temporal 0.74/0.74, chroma 0.74/0.62, rest ≤0.56.
Per-degradation at the gold-Youden operating point (T0 thr≈0.84, T1 thr≈0.68;
n=6 each — point detection rates are noisy, AUROC_vs_clean is the robust col):

| degradation | T0 det | T0 auc | T1 det | T1 auc |
|---|---|---|---|---|
| black | 1.00 | 1.00 | 0.00 | 0.00 |
| darken | 1.00 | 1.00 | 0.67 | 0.67 |
| frozen | 1.00 | 1.00 | 1.00 | 1.00 |
| shake | 1.00 | 1.00 | 1.00 | 1.00 |
| overexpose | 0.67 | 0.42 | 0.33 | 0.47 |
| noise | 0.50 | 0.03 | 1.00 | 1.00 |
| static | 0.00 | 0.00 | 1.00 | 1.00 |
| blur | 0.00 | 0.00 | 0.67 | 0.67 |
| lowbitrate | 0.00 | 0.00 | 0.00 | 0.22 |

Key finding: the **activity-trained head is not a quality detector** — which
degradations it catches depends on tier/threshold (T0 misses blur/lowbitrate/
static entirely; T1 inverts black). Capture quality needs its own operating
point or head: the features DO separate them (spatial_dct family-only 0.81,
pooled transfer AUROC 0.93–0.96) but the gold head's threshold is wrong for
them. The harness should fit quality and activity heads separately.

## Throughput (measured, single core via `taskset -c 0`, 10-s segments)

| tier | total s/seg | RTF total | decode s (RTF) | math s (RTF) |
|---|---|---|---|---|
| T0 | 0.418 | **23.9×** | 0.334 (29.9×) | 0.046 (217×) |
| T1 | 0.527 | **19.0×** | 0.315 (31.7×) | 0.175 (57×) |

- Decode (H.264 480p→small) is the floor at ~30×: **100×/core is unreachable
  with full software decode on this CPU**; math-only T0 already runs at 217×.
- Cores for 10,000 h in 2 h (needs 5000× aggregate): T0 ≈ 209 cores,
  T1 ≈ 263 cores. In 3 h (3333×): 140 / 176 cores.
- VLM/EG2 scale note for the harness: at e.g. 5 s/segment a GPU arm does 720
  seg/h → 3.6M segments ≈ 5000 GPU-hours ≈ 7 months on one 4070 for 10,000 h.
- Edge: 100 devices × 100 h in 2 h needs 50×/device — T0 (23.9×) cannot; live
  1× streaming is easily met (chunk latency ~34 ms per 2-s chunk, RAM < 100 MB).
- Tiering recommendation: T0 screens all windows; expensive models only on
  T0-uncertain windows (triggered fraction to be measured by harness).

## Streaming

Per-2-s `chunk_scores` in every arm file. Chunk CPU latency mean ~34 ms
(T0, cv2 chunk decode excluded — decode is shared with recording ingest).
Time-to-decision at Youden point: median 1 chunk (2 s) — caveat: with n=6 and
an activity-trained head, early-chunk scores are uncalibrated, so this TTD is
optimistic; the missed degradations never cross. Per-degradation patterns
FLIP between tiers (e.g. black 1.00→0.00, static 0.00→1.00): with n=6 these
cells are noise-dominated — the robust read is the pooled AUROC (0.93/0.96)
plus the family-only table, not any single cell.

## Arm C — bounded hyperspectral feasibility (ICVL)

`eval/spectral_features/icvl_check.py`. The local ICVL cache contains **only
the dataset README (16 KB) — zero hyperspectral cubes**, so no downloads were
attempted and the check ran on synthetic Gaussian-response math (stated in the
script output): ridge RGB→spectrum R² = 0.991 visible / 0.707 red-edge /
0.685 NIR; RGB carries 99.2% of its response mass in visible, 0% in NIR;
k-means AMI(full, RGB) = 0.857.
**Verdict:** RGB cannot recover what it never measures (NIR/red-edge); ridge
"reconstruction" beyond visible is prior interpolation, not measurement — it
adds no usable information. Nothing here is evidence on Awign footage; a real
hyperspectral sensor study would be needed, which is out of scope. Never call
arm B hyperspectral — it is RGB-only.

## What arm B cannot do (explicit)

1. Cannot reach 100×/core on 1 CPU core with full H.264 decode (floor ~30×);
   needs ~200+ cores for the 2-h/10k-h target, or triage tiering.
2. Numeric per-block motion-vector *magnitudes* are unobtainable from the
   ffmpeg 6.1 CLI (only per-frame counts); magnitude comes from pixels.
3. The activity head does not detect blur/low-bitrate/static degradations —
   needs a separate quality operating point (features exist: dct_hf 0.81).
4. Gold evidence is thin (14 bad / 3 clips, CIs span 0.5); skin-only beating
   full-model signals overfitting, not a finding to deploy on.
5. Horizontal flips score 0.73 (temp term is flip-invariant); excluded from
   positive pairs by rule, but a symmetric scene could score higher.
6. No Awign footage tested — IndEgo research clips only; activity judgement
   may not transfer.
