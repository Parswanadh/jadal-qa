# F50 — Per-pixel features 26 → 50: accuracy gain (and why there isn't one)

**Branch:** `m/feat50` · **Owner:** F50 · **Base:** `pod/base` · **Device:** CPU (ffmpeg decode + numpy/sklearn)
**Results:** `eval/results/exp_feat50/{in_domain,ood,ablation,holdout,cost}.json`
**Figure:** `docs/reports/images/exp_feat50.png`
**Reproduce:** `OMP_NUM_THREADS=2 python3 eval/gpu/run_features50.py && python3 eval/gpu/eval_feat50.py && python3 eval/gpu/cost_feat50.py`
**Code:** `jadal/gpu/features50.py`, `eval/gpu/run_features50.py`, `eval/gpu/eval_feat50.py`, `eval/gpu/make_fig_feat50.py` · **Tests:** `tests/test_feat50.py` (5 pass)

## TL;DR

Expanding the per-pixel feature vector from **26 → 50** gives **no accuracy gain**. It is a
small **loss** on both protocols, and it does **not** break the "features are not
domain-invariant" ceiling.

| protocol | 26 features | 50 features | Δ |
|---|---|---|---|
| In-domain LOCO AUROC (IndEgo, 7 clips) | **0.810** [0.683, 0.892] | 0.787 [0.653, 0.873] | **−0.023** (ns) |
| OOD-160 adapted z-score (factory-grouped) | **0.781**¹ [0.634, 0.895] | 0.512 (all-50) | **−0.269** |
| OOD-160, documented recipe + 18 new | 0.781 | **0.657** [0.478, 0.824] | **−0.123** [−0.224, −0.035], P(>0)=0.4 % |
| Hold-out108 (scored once) | **0.676**² | 0.457 (all-50) | **−0.219** |
| Hold-out108, documented recipe + 18 new | 0.676 | **0.612** | **−0.064** |
| Frozen ensemble (OOD reference) | — | 0.566 | — |

¹ documented adapted baseline (stable8 + transferable6), **reproduced exactly** by this harness.
² documented confirmatory O4(b) number, **reproduced exactly**.
ΔAUROC is paired, factory-grouped bootstrap (2000 resamples).

**Verdict: `drop`.** The 18 genuinely new features add no transferable signal. The only new
family with standalone OOD signal — finer DCT spectral bands (0.728 alone) — is already
saturated: adding it to the documented recipe moves 0.781 → 0.779 (ns). Every other family
is at or below chance OOD and drags the equal-weight adapted score down.

## What was added (24 features, 6 reused from cache + 18 new)

The 50 columns are **frozen-26 + frozen-Z3-transferable-6 + 18 newly computed**. The 6 Z3
transferable features are **reused from the frozen Z3 cache** (`z3_160.npz`, `confirm108 Xz3`),
not reimplemented (reimplementing them independently loses signal — see the note below).

| # | feature | definition | cost |
|---|---|---|---|
| 1 | `ct_fsize_std` ¹ | std/mean of per-frame compressed packet sizes | 1 ffprobe (~104 ms) |
| 2 | `mo_hand_motion_excess` ¹ | skin-mask-weighted mean \|Δluma\| − global mean \|Δluma\| | O(T·HW) |
| 3 | `inv_clip_frac` ¹ | fraction luma<0.02 or >0.98 | O(T) |
| 4 | `inv_laplacian_var` ¹ | variance of the 4-neighbour Laplacian (mid frame) | O(HW) |
| 5 | `inv_hist_entropy` ¹ | 32-bin luma histogram Shannon entropy | O(HW) |
| 6 | `inv_dct_hf_lf` ¹ | 8×8 DCT energy r≥4 / r≤1 | O(HW) |
| 7-10 | `dct_band_r1..r4plus` | fraction of 8×8 DCT AC energy in radial band 1/2/3/≥4 | O(HW) |
| 11-13 | `col_rg_ratio`, `col_bg_ratio`, `col_sat_std` | R/G, B/G channel ratios; saturation spread | O(T) |
| 14-17 | `lag1..4_diff` | mean \|luma[t]−luma[t−k]\|, k=1..4 | O(T·HW) |
| 18-19 | `noise_floor_med`, `noise_floor_p90` | per-pixel temporal-Δ std: median, p90 | O(T·HW) |
| 20-21 | `clip_low_5pct`, `clip_high_99` | fraction luma<0.05 / >0.99 | O(T) |
| 22-23 | `edge_density`, `lap_mean_abs` | fraction \|∇²\|>0.10; mean \|∇²\| | O(HW) |
| 24 | `hand_occupancy` | YCrCb skin-locus pixel fraction | O(T) |

¹ reused from the frozen Z3 cache (cols 26-31). Features 7-24 are computed here.
All computed on the same 160×90 decoded frames the frozen extractor uses; no hyperspectral
content (RGB only).

**Note on reimplementation.** Independently reimplementing the 6 Z3 features gives a
*weaker* set (my `ct_fsize_std` OOD AUROC 0.480 vs frozen 0.650; my `inv_dct_hf_lf`
0.706 vs frozen 0.244 — opposite sign), which is why the primary 50-feature set reuses the
cache. A variant `f50_mine_z3` (all 24 reimplemented) scores in-domain 0.804 and is included
in `in_domain.json` for transparency.

## OOD-160 results (138 good / 22 bad, 47 factories; factory-grouped 95 % CI)

| recipe | n feat | OOD AUROC | grouped CI | BA |
|---|---|---|---|---|
| frozen ensemble v1 | — | 0.566 | [0.424, 0.700] | 0.546 |
| 26 all (adapted z-score) | 26 | 0.386 | [0.231, 0.525] | 0.413 |
| **documented stable8+transfer6** | 14 | **0.781** | [0.634, 0.895] | 0.733 |
| documented + 18 new | 32 | 0.657 | [0.478, 0.824] | 0.651 |
| 24 new only (6 Z3 + 18) | 24 | 0.602 | [0.409, 0.784] | 0.616 |
| 18 newly computed only | 18 | 0.492 | [0.305, 0.648] | 0.575 |
| 50 all (adapted z-score) | 50 | 0.512 | [0.341, 0.658] | 0.507 |

Paired factory-grouped gains: **all-50 − all-26 = +0.126 [0.003, 0.261], 98 % > 0** (the
all-feature baseline is dragged below chance by sign-flipping features, so adding *anything*
"helps" it); **documented+18 − documented = −0.123 [−0.224, −0.035], 0.4 % > 0**
(significantly worse). The second comparison is the one that matters.

## Ablation / saturation (adapted OOD AUROC)

Family-alone (each of the 7 computed families used on its own):

| family | n | OOD AUROC | grouped CI |
|---|---|---|---|
| **dct_bands** | 4 | **0.728** | [0.595, 0.845] |
| **edge_lap** | 2 | **0.680** | [0.522, 0.824] |
| exposure_clip | 2 | 0.402 | [0.212, 0.586] |
| noise_floor | 2 | 0.395 | [0.242, 0.519] |
| temporal_lags | 4 | 0.379 | [0.237, 0.499] |
| hand | 1 | 0.331 | [0.201, 0.467] |
| colour | 3 | 0.269 | [0.140, 0.400] |

Cumulative, starting from the documented 14-feature recipe (adding families in rank order):

| added | n feat | OOD AUROC |
|---|---|---|
| — (documented) | 14 | **0.781** |
| + dct_bands | 18 | 0.779 |
| + edge_lap | 20 | 0.769 |
| + exposure_clip | 22 | 0.765 |
| + noise_floor | 24 | 0.747 |
| + temporal_lags | 28 | 0.700 |
| + hand | 29 | 0.693 |
| + colour | 32 | 0.657 |

**Saturation is immediate.** Only the DCT spectral bands (and, weakly, edge/laplacian)
carry OOD signal; the first addition is flat (0.781 → 0.779, well inside the CI) and every
subsequent family degrades the score. Leave-one-family-out on the all-50 score shows the
same ranking (dropping dct_bands costs the most: 0.512 → 0.443).

## Hold-out 108 (92 good / 16 bad, 44 factories) — scored ONCE, no tuning

| recipe | n feat | AUROC | grouped CI | BA |
|---|---|---|---|---|
| 26 all | 26 | 0.358 | [0.211, 0.505] | 0.379 |
| **documented stable8+transfer6** | 14 | **0.676** | [0.525, 0.827] | 0.648 |
| documented + 18 new | 32 | 0.612 | [0.453, 0.759] | 0.553 |
| 50 all | 50 | 0.457 | [0.286, 0.622] | 0.480 |

The documented 0.676 confirmatory number is reproduced exactly; adding the 18 new features
costs **−0.064** on the hold-out, consistent with the pooled-160 result. Scored once; no
feature re-selection, no threshold change, no re-fit.

## Cost (ms per 10-s window, CPU, 160×90, 100 frames)

| stage | ms/window |
|---|---|
| decode (ffmpeg, shared with the frozen 26) | 157 |
| frozen-26 kernels | 240 |
| **added 18 kernels** | **43** |
| `ct_fsize_std` ffprobe | 104 |
| **total added (kernels + ffprobe)** | **148** |

The added per-pixel kernels are cheap (43 ms); most of the marginal cost is the single
ffprobe for `ct_fsize_std` (104 ms). Measured on CPU, not GPU.

## Does this break the "features are not domain-invariant" ceiling (A3 / N1)?

**No.** The N1 diagnostic stands: features are not domain-invariant (IndEgo-vs-OOD
classifier AUROC 0.999). Adding 24 per-pixel features — including the full Z3 transferable
set, many finer DCT/spectral bands, colour ratios, temporal lags, noise-floor and
exposure-clipping statistics — produces **no transferable gain**. The best OOD score remains
the *selected* 14-feature adapted recipe (0.781 pooled / 0.676 hold-out). Naive expansion
to 50 features sits at 0.512 pooled / 0.457 hold-out, and even the carefully-selected
50-feature recipe (documented + 18) is significantly worse than the 14-feature baseline.

The one positive signal — finer DCT spectral bands (0.728 family-alone) — is already
captured by the existing `dct_energy`/`hf_ratio`/`radial_slope` features and saturates
immediately. More per-pixel features is not the lever; the ceiling is set by domain shift,
not by feature count.

## Honesty / caveats

- OOD n=160 (n_bad=22, 47 factories, ~1 window/factory), single labeller; grouped CI is
  honest but wide. Hold-out n_bad=16.
- The adapted z-score is unsupervised except for the per-feature sign (from IndEgo gold);
  the "documented" recipe's 6 transferable features were originally selected using OOD
  labels, so its 0.781 is mildly optimistic — but it is the *pre-existing* baseline, and
  the comparison (add-18 − documented) is paired and does not change.
- `ct_fsize_std` is compression telemetry, not per-pixel; included because the Z3
  transferable set is explicitly in scope.
- No labels edited; no GPU used; only this branch touched.
