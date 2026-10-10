# A2 — New full-resolution signals: dense flow, hand segmentation, full-res per-pixel

**Pod:** jadal-qa · **Branch:** `m/acc-signals` · **PR:** (link pending) base `main`

**Question.** With compute free, extract genuinely new signals at full resolution / all frames and measure the accuracy they add, alone and on top of the 26-feature + EG2 ensemble.

**Device:** CPU for extraction (ffmpeg decode, OpenCV Farneback flow, MediaPipe HandLandmarker) and CPU numpy/sklearn for scoring; no GPU used. **Git sha:** `f20013b677ab3e25b930d7ed23bac572dbb78134`.

**Results:** `eval/results/exp_acc/A2.01.json` (alone), `A2.02.json` (incremental), `A2.03.json` (combined). **Figure:** `docs/experiments/exp_acc/images/a2_signals.png`.

## Data & protocol

| set | n | good | bad | frames/window | protocol |
|---|---|---|---|---|---|
| IndEgo gold | 150 | 100 | 50 | 100 @480x480/10fps | leave-one-clip-out (7 clips) |
| OOD-160 Egocentric-10K | 160 | 138 | 22 | 300 @854x480/30fps | frozen ensemble; factory-grouped CI |
| hold-out-108 (round 3) | 108 | 92 | 16 | — | *not run*: the 480p transcodes (`label_windows3/`) are absent from this bundle; labels only |

## 1. Signals extracted (full resolution, all frames)

| group | features | method |
|---|---|---|
| flow | 13 | OpenCV Farneback dense flow on **every consecutive frame** of the full-res decode, downscaled to 256p for the flow solve; magnitude / divergence / curl statistics, active-area fraction, stability, static & peak fractions |
| hand | 5 | MediaPipe HandLandmarker (Tasks, CPU) on ~60 sampled frames/window (every 2nd gold / 6th OOD frame): hand-present fraction, hand-motion fraction, hand occupancy, mean #hands, hand-region flow excess |
| fullres | 6 | native 480x480 / 854x480 per-pixel temporal statistics: Laplacian sharpness (mean/p10/temporal CV), noise floor (MAD of median residual, mean/p90), high-frequency band energy fraction (256x256 centre FFT) |

Compute cost (single-thread ms/window, median):

| domain | decode | flow | hand | fullres | **total** |
|---|---|---|---|---|---|
| gold | 1084.9 | 9420.6 | 7103.9 | 1916.3 | **19993.9** |
| ood | 1297.5 | 5200.1 | 2846.8 | 2910.6 | **11892.8** |

Extraction wall time: gold 510.8s (6 procs), OOD 493.9s (6 procs); errors: gold 0, OOD 0.

## 2. Each signal alone (A2.01)

Bad-score AUROC (higher feature = more likely bad; sign fit on gold then applied to OOD).

| signal | group | gold AUROC | OOD AUROC [grouped 95% CI] |
|---|---|---|---|
| `fr_lap_var_p10` | fullres | 0.788 | 0.697 [0.518, 0.858] |
| `fr_sharp_temporal_cv` | fullres | 0.788 | 0.620 [0.470, 0.763] |
| `hand_n_mean` | hand | 0.755 | 0.633 [0.492, 0.776] |
| `hand_motion_frac` | hand | 0.765 | 0.610 [0.494, 0.721] |
| `hand_present_frac` | hand | 0.755 | 0.618 [0.473, 0.762] |
| `fr_lap_var_mean` | fullres | 0.695 | 0.645 [0.451, 0.818] |
| `flow_mag_p99` | flow | 0.753 | 0.566 [0.453, 0.668] |
| `flow_mag_temporal_std` | flow | 0.723 | 0.553 [0.465, 0.639] |
| `hand_occupancy_mean` | hand | 0.717 | 0.550 [0.408, 0.693] |
| `flow_mag_spatial_std` | flow | 0.748 | 0.502 [0.344, 0.649] |
| `flow_mag_p90` | flow | 0.730 | 0.502 [0.358, 0.627] |
| `flow_curl_abs_mean` | flow | 0.734 | 0.492 [0.331, 0.628] |
| `flow_div_abs_mean` | flow | 0.722 | 0.488 [0.321, 0.632] |
| `flow_mag_mean` | flow | 0.729 | 0.470 [0.339, 0.589] |
| `flow_curl_abs_p90` | flow | 0.715 | 0.473 [0.310, 0.617] |
| `flow_div_abs_p90` | flow | 0.711 | 0.477 [0.307, 0.629] |
| `flow_peak_frac` | flow | 0.527 | 0.577 [0.442, 0.707] |
| `flow_active_frac` | flow | 0.626 | 0.442 [0.316, 0.555] |
| `fr_hf_band_frac` | fullres | 0.569 | 0.497 [0.358, 0.644] |
| `flow_stability` | flow | 0.504 | 0.527 [0.352, 0.691] |
| `fr_noise_mean` | fullres | 0.556 | 0.468 [0.396, 0.515] |
| `fr_noise_p90` | fullres | 0.511 | 0.458 [0.386, 0.507] |
| `hand_flow_excess_mean` | hand | 0.504 | 0.412 [0.279, 0.567] |
| `flow_static_frac` | flow | 0.525 | 0.379 [0.247, 0.503] |

Baseline 26-feature + EG2 ensemble: gold LOCO 0.834, OOD 0.566 [0.424, 0.700].

## 3. Incremental value on top of the ensemble (A2.02)

Baseline ensemble: gold 0.834, OOD 0.566.

| model | gold AUROC | Δ gold (paired) | OOD AUROC [grouped CI] | Δ OOD (paired) |
|---|---|---|---|---|
| 26+ens+flow | 0.774 | — | — | — |
| 26+ens+fullres | 0.759 | — | — | — |
| 26+ens+hand | 0.802 | — | — | — |
| ens+flow | 0.809 | -0.028 [-0.052,-0.010] | 0.600 [0.470,0.741] | 0.038 [-0.059,0.164] |
| ens+fullres | 0.732 | -0.100 [-0.184,-0.019] | 0.544 [0.414,0.698] | -0.017 [-0.155,0.133] |
| ens+hand | 0.827 | -0.010 [-0.069,0.053] | 0.613 [0.513,0.724] | 0.050 [-0.033,0.151] |

## 4. Best combined set (A2.03)

| combination | gold AUROC | OOD AUROC [grouped CI] | mean-domain | Δ OOD (paired) |
|---|---|---|---|---|
| ens (baseline) | 0.834 | 0.566 [0.424,0.700] | 0.700 | — |
| ens+flow | 0.809 | 0.600 [0.470,0.741] | 0.705 | 0.038 |
| ens+flow+hand | 0.807 | 0.643 [0.522,0.781] | 0.725 | 0.083 |
| ens+flow+hand+fullres | 0.775 | 0.539 [0.383,0.710] | 0.657 | -0.019 |
| ens+fullres | 0.732 | 0.544 [0.414,0.698] | 0.638 | -0.017 |
| ens+hand | 0.827 | 0.613 [0.513,0.724] | 0.720 | 0.050 |

## 5. Gap to 95%

- Best combined configuration: **ens+flow+hand** — gold 0.807, OOD 0.643.
- Gap to 95% AUROC: in-domain **14.3 pts**, OOD **30.7 pts**.

## 6. Which signals move the needle

| signal | alone gold | alone OOD | added to ens: Δ gold | added to ens: Δ OOD | reading |
|---|---|---|---|---|---|
| **hand segmentation** | 0.755–0.765 | **0.610–0.633** | −0.010 | **+0.050** | the only new group that *transfers*: hand-presence/count separates NO_ACTIVE_WORK / HANDS_NOT_VISIBLE on Egocentric-10K |
| **dense flow** | 0.71–0.75 | 0.47–0.57 (≈chance) | −0.028 | +0.038 | strong in-domain, **does not transfer alone**; helps only in combination with hand |
| **full-res per-pixel** | **0.788** (lap_var_p10) | 0.62–0.70 | **−0.100** | −0.017 | best single in-domain signal, but **collinear with the ensemble** (which already encodes 160p sharpness) so adding it hurts |
| flow + hand | — | — | −0.027 | **+0.083** | best combined: OOD 0.566 → **0.643** [+0.522, +0.781], mean-domain 0.700 → **0.725** |

**Verdict (2026-10-10): partial / positive OOD, negative in-domain.** Hand/active-work
segmentation is the one genuinely new, transferable signal: it lifts OOD AUROC from 0.566 to
0.613 alone-added and to **0.643** when combined with dense flow (+0.083 paired, mean-domain
0.700 → 0.725). Dense flow is informative in-domain but near chance OOD on its own. Full-res
per-pixel sharpness is the best single in-domain feature (0.788) yet **hurts** when added to the
ensemble, because the ensemble already carries the 160×90 version of the same information
(collinearity). The new signals do **not** approach 95%: the gap is **14.3 pts in-domain** and
**30.7 pts OOD** for the best combination.

### Honesty caveats

- **Resolution scope.** Decode is full-res (native 480×480 / 854×480) and the full-res per-pixel
  features are computed at native resolution, but the *flow solve* is run at 256p (4× the
  160×90 baseline pixel density) and the *hand detector* runs on ~60 sampled frames/window, not
  every frame. "All frames" holds for flow (every consecutive frame) and per-pixel statistics;
  hands are subsampled.
- **OOD paired gains are not significant**: n_bad = 22 over 47 factories, so the +0.083 CI is
  [−0.06, +0.16]. The direction is consistent (hand alone, flow+hand) but underpowered.
- **`26+ens+*` rows are in-domain only** (the 26-feature view is unavailable on OOD in this
  bundle) and are *below* the ensemble baseline because the ensemble's own two-view logistic is
  a stronger representation than refitting a logistic on the raw 26 features.
- **Sign** of each single feature is fit on gold then applied to OOD (one bit per feature).
- **Hold-out-108 (round 3) was not scored**: its 480p transcodes (`label_windows3/`) are not in
  this bundle (labels only). It remains the confirmatory set.
- Extraction peak RAM ~1.6 GB/proc on OOD; 6 procs/domain. CPU only, no GPU lock taken.

### Compute cost (single-thread, median ms/window)

Gold ≈ **20.0 s** (flow 9.4 s, hand 7.1 s, fullres 1.9 s, decode 1.1 s); OOD ≈ **11.9 s**
(flow 5.2 s, hand 2.8 s, fullres 2.9 s, decode 1.3 s). Full extraction wall: gold 511 s /
OOD 494 s at 6 processes each. The flow solve dominates and is the obvious target for a GPU
port (RAFT-small measured at ~35 ms/pair at 854×480 on the A40 → ~10 s/window, no better than
CPU Farnebäck here; a half-res RAFT would be the win).

