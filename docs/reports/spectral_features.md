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

Label snapshot at run time: manifest_gold.json `snapshot_utc 2026-10-10T10:53:24Z`,
n=150 (100 good / 50 bad, all 7 clips). An earlier interim run used n=79; all
numbers below are from the n=150 snapshot unless noted. Protocol: LOCO,
clip-resampled bootstrap CI (2000x).

| set | tier | metric | value |
|---|---|---|---|
| gold human (n=150) | T0 | LOCO AUROC (95% CI) | 0.725 (0.598, 0.796) |
| gold human (n=150) | T1 | LOCO AUROC (95% CI) | 0.748 (0.624, 0.819) |
| synth, pooled gold head → synth (n=60) | T0 | transfer AUROC | 0.528 (0.444, 0.639) |
| synth, pooled gold head → synth (n=60) | T1 | transfer AUROC | 0.664 (0.611, 0.778) |
| dups (128 pairs, thr 0.84 on stratified half) | — | test P / R / F1 | 0.952 / 1.000 / 0.976 |

Correction (fix round 1): the previous version of this report quoted synth
0.932/0.957 as "transfer" — those were within-synth LOCO separability numbers,
not gold→synth transfer. True pooled-head transfer is 0.528/0.664 (T0 ≈
chance). The harness scorer should note the distinction.

Gold verdict: moderate signal, CIs clear 0.5 on n=150. `temporal`-only matches
the full model; no single family is critical (leave-one-out is flat).

## Ablation (LOCO AUROC; gold n=150)

| family | T0 leave-one-out | T0 only | T1 leave-one-out | T1 only |
|---|---|---|---|---|
| spatial_fft | 0.728 | 0.460 | 0.753 | 0.615 |
| spatial_dct | 0.726 | 0.537 | 0.750 | 0.545 |
| wavelet | 0.724 | 0.572 | 0.740 | 0.687 |
| basic | 0.743 | 0.410 | 0.756 | 0.438 |
| temporal | 0.728 | **0.738** | 0.766 | **0.740** |
| chroma | 0.735 | 0.279 | 0.769 | 0.277 |
| skin | 0.710 | 0.649 | 0.725 | 0.698 |
| compressed | 0.752 | 0.594 | 0.770 | 0.596 |
| full (all 40) | 0.725 | — | 0.748 | — |

Read: leave-one-out is flat (0.71–0.77) — families are redundant, none is
load-bearing alone. `temporal`-only ≈ full model at both tiers; `skin`-only
is second (0.65/0.70). `chroma`-only is inverted (0.28: ranks bad above good).

Synth transfer, family-only AUROC (pooled gold head per family → synth,
clean vs all degraded; n=60, 6 clean):
T0: spatial_dct **0.84**, basic 0.77, chroma 0.69, compressed 0.69,
spatial_fft 0.52, wavelet 0.49, temporal **0.44**, skin 0.43.
T1: spatial_dct **0.81**, basic 0.69, compressed 0.69, chroma 0.64,
temporal **0.52**, wavelet 0.50, skin 0.43, spatial_fft 0.41.
So `temporal` carries the gold-activity signal but does NOT transfer to
capture-quality degradations; `spatial_dct` transfers at both tiers.
Per-degradation with the pooled full head at the gold-Youden point
(T0 thr≈0.67, T1 thr≈0.66; n=6 each — cells are noisy, AUROC_vs_clean robust):

| degradation | T0 det | T0 auc | T1 det | T1 auc |
|---|---|---|---|---|
| black | 0.00 | 0.00 | 0.00 | 0.00 |
| darken | 0.33 | 0.39 | 0.67 | 0.67 |
| frozen | 0.00 | 0.00 | 1.00 | 0.94 |
| shake | 1.00 | 1.00 | 1.00 | 1.00 |
| overexpose | 0.67 | 0.64 | 0.33 | 0.47 |
| noise | 1.00 | 1.00 | 1.00 | 1.00 |
| static | 1.00 | 1.00 | 1.00 | 1.00 |
| blur | 0.67 | 0.61 | 0.67 | 0.67 |
| lowbitrate | 0.00 | 0.11 | 0.00 | 0.22 |

Key finding: the **activity-trained full head is not a quality detector**
(pooled transfer 0.53/0.66; black/frozen invert at T0). Capture quality needs
its own head: single-family heads trained on the same gold labels DO transfer
(spatial_dct 0.84/0.81 — blur/static/noise detected). The harness should fit
quality and activity heads separately; per-cell det rates flip with
tier/threshold at n=6, so read the AUROC columns, not single cells.

## Throughput (unpinned spot checks, single core via `taskset -c 0`, 10-s segments)

Two independent spot checks (N=3 segments each, tiers T0/T1); wall-clock varies
with machine load (other workers run concurrently), so treat as ranges:

| run | tier | total s/seg | RTF total | decode s (RTF) | math s (RTF) |
|---|---|---|---|---|---|
| quiet machine | T0 | 0.418 | **23.9×** | 0.334 (29.9×) | 0.046 (217×) |
| quiet machine | T1 | 0.527 | **19.0×** | 0.315 (31.7×) | 0.175 (57×) |
| loaded machine | T0 | 0.762 | **13.1×** | 0.562 (17.8×) | 0.110 (91×) |
| loaded machine | T1 | 1.143 | **8.7×** | 0.657 (15.2×) | 0.412 (24×) |

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

Per-2-s `chunk_scores` in every arm file (gold files: LOCO-fold heads, no TTD —
LOCO has no single operating point by design; synth files: pooled-head scores
plus `time_to_decision_chunks` at the gold-Youden point, fixed in round 1 to
use `threshold_youden`, previously a dead `"threshold"` key fell back to 0.5).
Chunk CPU latency mean ~34 ms
(T0, cv2 chunk decode excluded — decode is shared with recording ingest).
Time-to-decision at Youden point: median 1 chunk (2 s) — caveat: with n=6 and
an activity-trained head, early-chunk scores are uncalibrated, so this TTD is
optimistic; the missed degradations never cross. Per-degradation cells FLIP
with tier/threshold at n=6 — read the AUROC columns, not single cells.

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
3. The pooled activity head does not transfer to capture quality (transfer
   AUROC 0.53/0.66); single-family quality heads do (spatial_dct 0.84/0.81) —
   fit quality and activity heads separately.
4. Gold evidence is moderate (50 bad / 7 clips, CIs clear 0.5 but wide:
   T0 0.60–0.80); leave-one-out is flat, so family rankings rest on
   family-only heads, not ablations that hurt.
5. Horizontal flips score 0.73 (temp term is flip-invariant); excluded from
   positive pairs by rule, but a symmetric scene could score higher.
6. No Awign footage tested — IndEgo research clips only; activity judgement
   may not transfer.

## Fix round 1 (review of PR #52)

Addressed every Critical (none) / Important (1–4) / Minor (1–5) finding in
owned files only. Real command outputs:

- Important-1: report synth tables rewritten from committed
  `eval/results/spectral/features_B_{T0,T1}_synth.json`. T1 family-only is
  now spatial_dct 0.81, basic/compressed 0.69, chroma 0.64, temporal 0.52
  (≈ chance — stated, transfer story corrected). Minor-1: T0 overexpose auc
  cell fixed to 0.44.
- Honesty correction found while fixing: the old "synth transfer 0.932/0.957"
  were within-synth LOCO numbers mislabeled as transfer. `run_arm.py` no
  longer overwrites pooled transfer scores with a within-synth LOCO block;
  arm files carry `"protocol": "transfer: pooled head trained on
  manifest_gold.json"` and true transfer AUROC is T0 0.528 / T1 0.664.
- Important-2: `thr = transfer["threshold_youden"]` (was dead
  `transfer.get("threshold", 0.5)`). Verified live — TTD distribution is now
  non-degenerate: T0 {chunk1: 50, chunk2: 1, never: 9}, T1 {chunk1: 54,
  never: 6} over 60 synth items; gold files (150 items, LOCO, no operating
  point) carry 0 TTD by design.
- Important-3: `decode_pipe()` raises `FileNotFoundError` (ffmpeg rc≠0, stderr
  tail included) / `ValueError` (zero frames); `compressed_stats()` failures
  print `[spectral] compressed_stats failed for <path>: <err>` to stderr —
  no more silent all-zero features.
- Important-4: new `TestIOPipe` (real mp4 via cv2.VideoWriter through
  ffmpeg/ffprobe): 40 finite features, nonzero `i_size`/`p_count_frac`, 20
  decoded frames, and `(FileNotFoundError, ValueError)` on a missing file.
  Runs in 0.8 s.
- Minor-2: dead `dur = 12.0` removed from `run_dups.py`. Minor-3: `stream.py`
  docstring qualified (window-bounded temporal FFT, compressed features always
  0.0 in streaming). Minor-5: `--manifest/--train-manifest/--tiers/--out-suffix`
  validated (basenames, known tiers, no separators/`..`).
- Minor-4: 1-core RTF raw outputs committed here (spot checks, N=3 segs each):
  quiet machine → `T0 1-core: total 0.418 RTF 23.9 | decode 0.334 (RTF 29.9)
  | math 0.046 (RTF 217.3)` and `T1 1-core: total 0.527 RTF 19.0 | decode
  0.315 (RTF 31.7) | math 0.175 (RTF 57.2)`; loaded machine → `T0 1-core:
  total 0.762 RTF 13.1 | decode 0.562 (RTF 17.8) | math 0.110 (RTF 91.1)`
  and `T1 1-core: total 1.143 RTF 8.7 | decode 0.657 (RTF 15.2) | math 0.412
  (RTF 24.3)`. Command: `taskset -c 0 .../bin/python -c
  "from run_arm import extract_timed; ..."` over segments at 14.6/100/300 s.
  Table above labeled as ranges, not point claims.
- Gold re-ran on the completed label snapshot (n=150, 100 good/50 bad, 7
  clips): T0 0.725 (0.598, 0.796), T1 0.748 (0.624, 0.819).
- Full suite: `PYTHONNOUSERSITE=1 timeout 250 .../bin/python -m unittest
  discover -s tests 2>&1 | grep -E '^(Ran|OK|FAILED|ERROR|FAIL:)'` →
  `Ran 567 tests in 33.380s` / `OK (skipped=8)` (verbatim).
- Dups unchanged by this round (unsupervised, no gold dependence): test
  P=0.952/R=1.000/F1=0.976 stands.
