# GRID — our own pipeline on a multi-frame picture (no vision-language model)

Track owner: DSH stream S1 (`dsh/grid-feats`), S2 (`dsh/grid-sweep`) and S0 (`dsh/infra`).
Base: `origin/main` at `53983d6` + the tooling restore (`dsh/infra`). Environment: `docs/pod/ENV_grid_auto.md`.
Every number below comes from a results JSON under `eval/results/exp_grid/` and a registry line in
`experiments/registry.jsonl`. Device for every number on this page: **CPU (numpy/sklearn/PIL), this
laptop (RTX 4070 Laptop 8 GB, 22 cores, 15 GB RAM)**; no GPU result is claimed anywhere.

## The question

The customer said a picture holding many frames ("multiple images in one image") can be analysed,
and that 12 frames in a 4x3 grid is their headline configuration. What had been tested before this
track was **only EmbeddingGemma-2 embeddings of mosaics** — an embedding model, not our pipeline and
not a decision path. The 26-number head could not be applied to a still mosaic at all, because about
10 of its 26 features are measured *along time at each pixel* (`temp_b0..b3`, `frozen_frac`,
`flicker_idx`, `motion_energy`, `global_shift`, `dup_hash_mean`, `zncc_self`, `rppg_pulse`).

So this track built **mosaic-native features** (`jadal/mosaic/features.py`, version
`mosaic-features-1.0.0`) in three ablatable groups:

| Group | What it measures | n features |
|---|---|---|
| `spatial` | per-tile luma percentiles, dark/bright fraction, radial spectrum slope, HF ratio, DCT energy, PCA residual, skin fraction, grey-world illuminant ratios, **brightness-calibrated Laplacian variance** (the blur statistic), entropy — pooled over tiles by mean/min/max/std | 60 |
| `intertile` | temporal *proxies* between consecutive tiles: mean absolute difference, near-identical-pair share (freeze/duplicate proxy), pHash Hamming share, ZNCC mean/min, phase-correlation shift mean/max, cross-tile brightness variance, flicker, sub-threshold pair count | 12 |
| `diff` | the same pooled statistics on the difference mosaic \|I_t − I_{t−1}\| | 7 |

The burned timestamp is masked out of every tile by default, and mid-grey letterbox padding is
detected and excluded, so a square IndEgo frame and a 16:9 factory frame are measured on their
content. 13 unit tests cover determinism, the timestamp mask, the temporal proxies, the difference
mosaic and the blur statistic (`tests/test_mosaic_features.py`).

## 0. Baseline gate (GRID.00) — PASS

The 26-number logistic head, leave-one-clip-out on the 150 IndEgo windows, reproduces the pod's
documented number exactly:

| | AUROC | 95 % CI (clip bootstrap, 2000, seed 0) | window bootstrap (**optimistic**) |
|---|---|---|---|
| 26-feature baseline, n=150 (100 good / 50 bad), ONE labeller, 7 clips | **0.810** | [0.683, 0.892] | [0.722, 0.889] |

good-vs-semantic-bad 0.797, good-vs-quality-bad 0.846, `pallet`/`pallet2` at chance (0.50) as
documented. Nothing in this track started until this passed. Two interpreters (the project venv and
system `python3.12`) print the same 0.810.

## 1. The mosaic as a DECISION input — killed

Identical protocol for every arm: mosaic built by `jadal/mosaic/build.py` unchanged
(`mosaic-1.0.0`, N=9 3x3 224 px, timestamps on, row-major time order), `StandardScaler` +
`LogisticRegression(C=0.3, class_weight='balanced')` — the shipped head recipe — leave-one-clip-out
over the 7 clips, pooled AUROC with a **clip-level** bootstrap CI, and every A-vs-B difference from a
**paired clip bootstrap** on the same windows.

| Arm | Features | AUROC | 95 % CI (clip) | Δ vs baseline 26 | paired CI |
|---|---|---|---|---|---|
| GRID.00 baseline | 26 shipped | 0.810 | [0.683, 0.892] | — | — |
| GRID.01 | `spatial` (60) | 0.6864 | [0.522, 0.772] | **−0.124** | [−0.202, −0.071] excludes 0 |
| GRID.01 secondary | `spatial`, mean pooling only (15) | 0.7122 | [0.504, 0.788] | −0.098 | — |
| GRID.02 | `spatial+intertile` (72) | 0.7058 | [0.514, 0.785] | −0.104 | — |
| GRID.03 | `+diff` (79) | 0.7002 | [0.503, 0.778] | −0.110 | — |

* **GRID.01**: the still picture is *significantly worse* than the full-window pipeline, not within
  the 0.02 the decision rule asks for. The mean-pooling capacity control (15 features) is 0.026
  *better* than the 60-feature pooled version, but the difference is not established
  ([−0.093, +0.072]) — the extra pooling dimensions do not earn their place at n=150.
* **GRID.02 — the pre-registered KILL-GATE FAILED.** Inter-tile features add **+0.019** AUROC
  (paired CI [−0.018, +0.053], does **not** exclude zero), below the pre-registered 0.02. Per the
  brief, the mosaic-as-decision-input line **stopped here**: no full GRID.04 sweep, no adoption path.
* **GRID.03** (run only as the small control the brief permits after a failed gate): the difference
  mosaic adds −0.006 [−0.016, +0.002] — nothing. This reproduces the earlier EmbeddingGemma-2
  finding from the other direction (motion mosaic 0.863 in-domain, 0.401 out of domain).

### Negative controls and canvas controls (all required, all reported)

| Control | Value | Required | Pass |
|---|---|---|---|
| Shuffled labels (20 draws) | mean 0.5035, max 0.6384 | ≈0.5 | ✅ |
| Random Gaussian features (20 draws) | mean 0.4695, max 0.5774 | ≈0.5 | ✅ |
| Null mosaic of flat grey tiles | 0.5000 | =0.5 | ✅ |
| **Text leak**: timestamps drawn *and not masked* vs no text and no mask | Δ −0.0058 [−0.027, +0.036] | no leak | ✅ no measurable leak |
| **Mask cost**: primary (drawn+masked) vs content reference (unmasked) | Δ −0.0178 [−0.048, +0.008] | ≈0 | ✅ masking costs nothing measurable |
| **Arrangement**: shuffled tile placement vs time order | Δ 0.0000 | — | the feature set is *arrangement-invariant by construction* |

The arrangement control is worth stating plainly: because the spatial features are pooled over tiles
and the inter-tile features are read in time order from the sidecar, **tile layout cannot matter** to
this feature set. That is not a bug, it is the answer to "does the spatial arrangement of tiles
help" — it does not, which is consistent with the repo's earlier withdrawal of the spatial
localisation claim.

## 2. GRID.04 — the customer's headline configuration, as a small control

After the failed kill-gate only a compact ladder was run. Same features, head, folds and comparisons
per arm.

| Arm | N | grid | tile px | canvas px | AUROC | 95 % CI | Δ vs N=9 | Δ vs baseline 26 |
|---|---|---|---|---|---|---|---|---|
| m2x2_t224 | 4 | 2x2 | 224 | 203 k | 0.6750 | [0.504, 0.762] | −0.025 | −0.135 |
| m3x3_t224 | 9 | 3x3 | 224 | 456 k | 0.7002 | [0.503, 0.778] | 0 | −0.110 |
| **m4x3_t224 (headline)** | **12** | **4x3** | **224** | **607 k** | **0.7044** | **[0.488, 0.786]** | +0.004 | **−0.106** |
| m4x4_t224 | 16 | 4x4 | 224 | 808 k | 0.6986 | [0.516, 0.764] | −0.002 | −0.111 |
| m4x3_t160 | 12 | 4x3 | 160 | 310 k | 0.7126 | [0.521, 0.785] | +0.012 | −0.097 |
| m4x3_t112 | 12 | 4x3 | 112 | 153 k | 0.7146 | [0.536, 0.774] | +0.014 | −0.095 |

Two honest readings: **frame count is flat** (N=4 through N=16 all within noise of each other), and
**smaller tiles are marginally better, not worse** — t112 beats t224 by 0.010 at a quarter of the
canvas pixels, which agrees with the earlier tile ladder (MR.01: 112–160 px as good as 224 at 3x3).
The customer's headline 12-frame 4x3 picture scores **0.7044**, about 0.11 below the full-window
pipeline on the same windows and folds.

The full GRID.04 sweep (N ∈ {4,6,9,12,16} × tile ∈ {112,160,224} × uniform vs motion-adaptive frame
selection × timestamps on/off) was **not run**, because the kill-gate failed and the brief says to
spend the freed compute elsewhere.

## 3. GRID.06 — does the mosaic transfer to factory footage?

OOD-160 = 160 human-labelled Egocentric-10K windows, **138 good / 22 bad, 47 factories, ONE
labeller**. **19 of the 22 bad windows are `NO_ACTIVE_WORK`** (the worker is not doing task-relevant
work); `HANDS_NOT_VISIBLE` (2) and `SHAKY` (1) are too few to report separately. Every number below
carries that context.

| Head | AUROC | 95 % CI (factory-grouped) | paired vs frozen 26 |
|---|---|---|---|
| Frozen 26-number shipped head (scores from the harden gate run) | 0.5878 | — | — |
| **Frozen IndEgo-trained mosaic head** | **0.3557** | [0.224, 0.515] | **−0.232 [−0.380, −0.077], excludes 0** |
| Mosaic features, 5-fold factory-grouped CV **trained on factory windows** | 0.6798 | [0.487, 0.845] | +0.092 [−0.149, +0.296], **not established** |

The frozen mosaic head does not merely fail to transfer — it **inverts**, exactly as the frozen
per-pixel head did (0.342) before it. Trained on factory footage the same features reach 0.68, in the
same band as the pod's OOD-trained recipes, and the paired comparison against the shipped head does
not establish a difference at n_bad=22. The honest conclusion is that **the missing ingredient is
factory labels, not the picture format** — which is the same conclusion the pod's `NEXT_EXPERIMENTS`
reached from the other direction.

## 4. GRID.05 / GRID.07 / GRID.09

See `docs/reports/grid_s0_parity.md` (S0 parity on synthetic defects, cost, and the
brightness-calibrated blur fix), owned by stream S2. Numbers live in
`eval/results/exp_grid/GRID.05.json`, `GRID.07.json` and `GRID.09.json`.

## 5. What was NOT done, and why

* The full GRID.04 sweep — kill-gate failed (pre-registered stopping rule).
* Motion-adaptive frame selection and the timestamps-on/off *inference* ablation across layouts —
  same reason; the timestamp leak was tested at N=9 and shows no measurable leak.
* GRID.08 (the 108-window round-3 hold-out) — **not run**. The hold-out has already been scored by
  two pre-registered configs (C1 PR #162, physics v2 PR #163), and nothing in this track produced a
  candidate worth spending the last permitted look on. The hold-out stays unspent.

## The one honest sentence the product may say

**A 4x3 picture of 12 timestamped frames scored by our own mosaic-native features reaches AUROC
0.704 [0.488, 0.786] on 150 IndEgo windows and 0.356 [0.224, 0.515] when the IndEgo-trained head is
applied unchanged to 160 labelled factory windows (n_bad=22, 19 of them no-active-work), which is
worse than the full-window pipeline (0.810 in-domain, 0.588 out of domain) and therefore is an
evidence picture, not a decision input; the same features trained on factory footage reach 0.680
[0.487, 0.845], so the missing ingredient is factory labels, not the picture format.**
