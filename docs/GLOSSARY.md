# Glossary — jadal-qa

Every term a reader needs to follow the experiments and the product, each with a
one-line definition and a pointer to the file or experiment where it is used. Terms are
grouped by area. Where a number is quoted, the source file is named.

Author: claim **X4** (documentation/glossary steward). Branch `m/docs-index` (base `pod/base`).
Companion documents: [`docs/EXPERIMENTS_INDEX.md`](EXPERIMENTS_INDEX.md) and
[`docs/experiments/index.csv`](experiments/index.csv).

---

## How to read the numbers

Every quantitative claim in this repo is meant to be read with four things attached, so
that a number never travels without them:

- **n** — the sample size. Window counts are the usual unit (e.g. `n=150`). A small
  sub-sample is not a result: `n_bad=9` is **exploratory**, and `n=8` (an early smoke run)
  is not usable.
- **label source** — *who* labelled the data. `human` means one human labeller
  (IndEgo gold, Egocentric-10K queues); `AI` means an AI model produced the labels
  (the 10k FACTORY frames, `gemini-2.5-flash`, **not** human); `none` means descriptive
  scores only, no accuracy.
- **protocol** — how the number was produced: the validation split
  (**LOCO** = leave-one-clip-out), the model and its regularisation (`logistic C=0.3`),
  the CI method (clip-bootstrap vs window-bootstrap), and whether the model was
  **frozen** before the test data was read (**pre-registered**).
- **CI** — a 95% confidence interval. For in-domain numbers it is a **clip-bootstrap
  95% CI** (resampling whole clips). For out-of-domain numbers it is a **window
  bootstrap**, which is **optimistic** because windows within a factory are correlated.

Rule of thumb used throughout: an alternative is only *adopted* when it is within **0.02
AUROC** of the reference on **both** regimes; a difference smaller than the CI is not a
result. Two regimes are never mixed in one number: **in-domain** (IndEgo 480p/10 fps
gold) and **out-of-domain** (Egocentric-10K 1080p/30 fps HEVC).

---

## Evaluation & statistics

| Term | Definition | Where used |
|---|---|---|
| **LOCO** (leave-one-clip-out) | Validation where each fold holds out one whole clip, so train and test never share a clip; the default protocol for IndEgo (7 clips). | `eval/standard/run_all.py`; E1.01, E9.01 (`docs/reports/exp_fps.md`, `docs/reports/evalharness.md`) |
| **AUROC** | Area under the ROC curve: probability a good window scores above a bad one; 0.5 = chance, 1.0 = perfect. | Every accuracy number, e.g. E9.01 pixel26 0.8100 / EG2 0.8480 |
| **clip-bootstrap 95% CI** | 95% confidence interval from resampling whole clips with replacement (B=1000-2000); respects within-clip correlation. | E1.01, E3.01, LB1 (`docs/reports/pod_verification.md`) |
| **window-bootstrap 95% CI** | Resampling windows (not clips); used only for OOD where there is ~1 window/factory — **optimistic**. | E1.03, E5.00, E5.03 |
| **balanced accuracy** | Mean of good-window and bad-window recall; used because classes are imbalanced (100 good / 50 bad). | LB2 (`docs/reports/poc_integration.md`); FACTS.md |
| **pre-registration / frozen** | A model whose bytes and hash are fixed *before* the test labels are read. | E5.04; LB4 `configs/frozen/ensemble_v1.prereg.json` |

## Features & models

| Term | Definition | Where used |
|---|---|---|
| **EG2 / EmbeddingGemma-2 window embedding** | `google/embeddinggemma-2` image embedding of a window's frames, averaged into one window vector; the strongest in-domain view (~0.85 LOCO). | E1.02, E2.02 (`docs/reports/exp_fps.md`, `docs/reports/exp_mosaic.md`) |
| **pixel-26 / 26 per-pixel features** | 26 deterministic features (noise, sharpness, exposure, spatial/temporal spectrum, motion, etc.) computed on decoded pixels; the second view (~0.81). | E1.01, E4.02, E5.03 (`eval/gpu/features.py`) |
| **per-pixel view** | The model that consumes the 26 per-pixel features; complement to the EG2 view. | E1.01; LB2 (fused view) |
| **two-view ensemble / prob-average** | Equal-weight average of the EG2 and pixel-26 probabilities; the frozen production scorer `ensemble_v1`. Switching to **rank-average** is a free upgrade (Y3.02). | E2.02, E5.00; LB2 fused 4 s 0.874; Y3.02 |
| **sign inversion** | A feature or clip whose good/bad direction flips between regimes or clips (e.g. `motion_energy`, clip `pallet`), so a frozen direction scores below chance. | E6.02, E5.03, E6.03 |
| **ego-motion magnitude** | Mean phase-correlation shift between frames — how much the wearer is moving; the strongest *motion* scalar, oriented AUROC 0.805 on IndEgo. | E6.01 (`eval/results/exp_motion_cue/E6.01.json`) |

## Mosaic & review artefacts

| Term | Definition | Where used |
|---|---|---|
| **mosaic** | One image built by tiling N frames sampled across a window, embedded once instead of N times. | E2.01 (`docs/reports/exp_mosaic.md`) |
| **tile** | One frame placed into a mosaic cell. | E2.01 |
| **tile px** | The edge length of one mosaic tile in pixels (224/160/112/96/64 in MR.01). Tiles of 112-160 px keep accuracy and are ~36% cheaper; <96 px degrades (0.762 at 96, 0.736 at 64). | X3 `MR.01` (`docs/reports/exp_mosaic_res.md`) |
| **canvas** | The full mosaic image surface (rows x cols x tile_px), the deterministic CPU cost proxy. The X3 equal-canvas duel holds canvas ~462-465k px while changing tile size (3x3/224 px 0.785 vs 4x4/168 px 0.845). | E2.01 / X3 `MR` (`eval/exp_mosaic_res/run_res.py`) |
| **sidecar** | The JSON written next to a mosaic PNG recording `grid`, `tile_px`, per-tile `t_s`/`src_frame_index`, and `mosaic_sha256`. | `docs/reports/exp_mosaic.md`; X3 `MR` (`docs/experiments/exp_mosaic_res/images/index.json`) |
| **timestamped tile** | A tile with `t=X.Xs` burned into its top-right corner; tested (E2.04) to be **not load-bearing** in a 3x3/224 px mosaic, but MR.03 measures **+0.021** when timestamps are on. | E2.04; X3 `MR.03` |

## Review, abstain & fusion (Y-track)

| Term | Definition | Where used |
|---|---|---|
| **abstain band** | The interval of scores around a decision threshold that returns REVIEW (uncertain) instead of accept/reject. Y1.01 sweeps a symmetric half-width `w` on the raw score: `w=0.16` (band `[0.34,0.66]`) cuts review to 53.4% at gold BA 0.846. The frozen E8.01 profile band is `[0.25,0.75]`. | Y1.01 (`docs/reports/exp_review_band.md`); `profiles/*.json` |
| **review share** | Fraction of windows the policy sends to REVIEW rather than accept/reject. Descriptive on unlabelled windows: E8.01 frozen band 75.5% (1,167/1,546); Y1 raw band `w=0.16` -> 53.4%; Y2 gating + stage-0 -> 54.0%. | E8.01; Y1.01; Y2.02 |
| **cost-margin gating** | Selective classification: decide the top-`coverage` fraction of windows with the largest distance from the cost-optimal threshold `p* = C_fa/(C_fa+C_fr)`, and review the rest. Symmetric 1:1 cost adopted; asymmetric 2:1 / 5:1 rejected. | Y2.01 (`eval/exp_review/run_y2.py`) |
| **coverage** | In cost-margin gating, the fraction of windows that receive an ACCEPT/REJECT decision (the remainder are REVIEWed). Y2 uses coverage 0.45 -> review 55.0%. | Y2.01 |
| **rank-average fusion** | Combining views by averaging their per-window **ranks** rather than probabilities. A free win for the existing two views: AUROC 0.834 -> 0.844 and profile-band review mass 0.653 -> 0.607, with no new inference. | Y3.02 (`docs/reports/exp_review_score.md`) |
| **LOW_MARGIN_PROFILE_REVIEW** | The verdict code the frozen profile emits for a window whose score falls inside the profile's abstain band (`factory_1080p30`: accept >=0.75 / reject <=0.25). It is the source of the E8.01 75.5% review share - the `[0.25,0.75]` band, not `[0.4,0.6]` (0.296). | Y3 honesty note; E8.01; `jadal/poc_stream/policy.py` |

## Regimes, profiles & policy

| Term | Definition | Where used |
|---|---|---|
| **regimes** | Distinct data types (IndEgo 480p/10 fps H.264 vs Egocentric-10K 1080p/30 fps HEVC) treated as separate populations. | E4.01 (`docs/reports/exp_regimes_pod.md`) |
| **regime descriptor** | A deterministic per-window measurement (resolution, fps, sharpness, noise sigma, luma stats, ego-motion, static fraction, ...) used to characterise a regime. | E4.01 (`eval/exp_regimes/descriptors.py`) |
| **profiles** | Per-regime pipeline configuration (sampling fps, window policy, feature set, model, thresholds, abstain band). | E4.04 `profiles/*.json` |
| **deterministic router** | `choose_profile(descriptors)`: rule-based selection of a profile with a margin and a safe `default`/`UNSEEN_REGIME` fallback. | E4.04 `jadal/regimes/router.py` |
| **CUSUM / SPRT** | Sequential change-point / sequential-probability-ratio tests; tested as change-driven samplers and as a collapse gate. As a sampler it never fired (equiv. to every-4) and was **dropped**. | E3.01; FACTS.md (breakthrough lab); `jadal/lab/early_reject.py` |
| **early exit (4 s)** | Stop analysing a 10 s window after 4 s once the score saturates; adopted because 4 s is within 0.01 of the 6 s peak (0.860). | E3.03 (`eval/results/exp_temporal/E3.03_early_exit.json`) |
| **every-2nd window policy** | Process every second window (grid `range(0,n,2)`); clip bad-fraction error 0.049 at ~51% cost — the adopted IndEgo sampling policy. | E3.01, E3.05, LB2 |
| **sparse / dense / stratified sampling** | *Sparse* = probe a subset; *dense* = densify where the score is near threshold; *stratified* = one probe per equal section. Every-2nd beats stratified/contiguous probe and section-drop-anywhere on IndEgo. | E3.01; `SKIP` (`eval/exp_skip/run_skip.py`) |
| **skip-first-M** | Dropping the first M minutes of a clip; **not supported** on either regime (IndEgo's start is the cleaner part). | E0.01, E3.02 |

## Data & labels

| Term | Definition | Where used |
|---|---|---|
| **IndEgo gold (150 windows)** | 150 ten-second windows from 7 IndEgo (Fraunhofer IPK) videos, 480p ~10 fps, labelled by **one human**: 100 usable / 50 bad. The in-domain test set. | E0.01, E1.01, E9.01; `$JADAL_BUNDLE/labels/manifest_gold.json` |
| **OOD (out-of-domain)** | Data from a different source/regime than training — here real Egocentric-10K factory footage vs IndEgo training. All IndEgo-trained models scored 0.4-0.53 AUROC OOD. | E1.03, E5.00-E5.04; FACTS.md 18:40 |
| **Egocentric-10K (58 clips / 1,546 windows)** | The real factory corpus on disk: 58 clips, 4.404 h, 47 factories, 1080p HEVC 30 fps, 1,546 non-overlapping 10 s windows, **no human labels**. | E8.01, E3.04; `$JADAL_BUNDLE/queues/manifest_scale.json` |
| **AI labels** | Labels produced by an AI model (`gemini-2.5-flash`) on the 10k FACTORY frames — **not human**; usable only as a model-vs-model prior. | E5.01; FACTS.md (Data) |

## Product & operations

| Term | Definition | Where used |
|---|---|---|
| **receipt / content hash** | Per-window record hashed with SHA-256 over canonical JSON; content-hashed, **not signed** (optional HMAC sidecar). `features_digest` hashes the `features{}` block. | E8.01; `docs/SPEC.md`; `jadal/` receipt builders |
| **funnel S0 / S1 / S2** | Three-stage pipeline: **S0** early exact rules (dark/blocked/frozen) at tiny resolution; **S1** whole-video per-pixel scoring; **S2** heavier tools (MediaPipe hands, EG2) only on uncertain windows. | `docs/ARCHITECTURE.md` §Stage 0/1/2; `docs/reports/funnel.md` |
| **verdict ACCEPT / REJECT / REVIEW** | Per-window decision from the frozen rule: score > 0.5 accept, < 0.4 reject, else review/uncertain. | E8.01, LB2 (`jadal/poc_stream/policy.py`) |
| **NVDEC** | NVIDIA hardware video decoder; on this pod it saturates at one engine and is faster than CPU on 1080p HEVC but slower on 480p. | E7.01 (`docs/reports/cost_model.md`); FACTS.md |
| **CRF** | Constant-Rate-Factor (x264 quality knob, lower = better). RES.02: CRF28 (ΔAUROC −0.014) at 57% of CRF23 bytes and CRF33 (Δ −0.015) at 36% both stay inside the 0.02 budget, so CRF is a bytes lever, not a compute lever. | `RES` (`eval/exp_res/run_res.py`; `docs/reports/exp_res.md`) |
| **bitrate** | Encoded bits per second; a deterministic regime descriptor, not a finding. | E4.01 |
| **cost model (hours of video per wall-hour)** | Throughput unit: footage-hours processed per wall-clock hour; per-profile 0.51-15.22 on the pod A40, every-2nd+4 s exit x3.88 vs full sampling. | E7.01 (`eval/results/exp_cost/E7.01.json`) |
| **telemetry** | The 1 Hz GPU/pod CSV (`logs/gpu_telemetry.csv`: power, utilisation, clocks, temperatures) joined to benchmark windows via `eval/gpu/telemetry.py::window_stats()`. | `docs/reports/gpu_pipeline.md` §7; `eval/gpu/e2e.py` |

## Follow-up experiment ids (X-track and Y-track)

| Term | Definition | Where used |
|---|---|---|
| **RES (X1)** | Resolution + compression sweep (480/360/240/180/120p, H.264 CRF 18-33) on IndEgo gold plus OOD resolution sensitivity; best shipping point **360p@CRF28** (0.8036, 257 KiB). | `m/res`, PR #117; `docs/reports/exp_res.md` |
| **SKIP (X2)** | Non-contiguous section sampling: can sections be dropped anywhere (middle/end/evenly)? Answer **no** - uniform-every-2 stays the best cell. | `m/skip`, PR #114; `docs/reports/exp_skip.md` |
| **MR (X3)** | Low-res/dense mosaic tiles (64-224 px, 2x2-6x6, equal-canvas duel); adopts the small-tile geometry for a mosaic pipeline but keeps the E2 verdict. | `m/mosaic-res`, PR #119; `docs/reports/exp_mosaic_res.md` |
| **Y1.01** | Review reduction by an abstain-band + calibration sweep; raw `w=0.16` -> 53.4% review, calibration a negative. | `m/review-band`, PR #118; `docs/reports/exp_review_band.md` |
| **Y2.01 / Y2.02** | Decision-theoretic cost-margin gating (Y2.01) + deterministic stage-0 overlay (Y2.02); 75.5% -> 54.0% review at gold BA 0.837. | `m/review-gate`, PR #116; `docs/reports/exp_review_gate.md` |
| **Y3.01 / Y3.02** | Cached-view inventory + two-view baseline reproduction (Y3.01); fusion / view-stacking sweep (Y3.02) - rank-average is a free win, more views is a clean negative. | `m/review-score`, PR #120; `docs/reports/exp_review_score.md` |

---

## Not to be confused with

- **hyperspectral** — the cameras are RGB (3 bands); we do per-pixel frequency/temporal
  analysis of RGB, not hyperspectral imaging (FACTS.md, Direction).
- **"hand-region motion"** — retired wording; the motion cue is largely "the wearer is
  not walking", so say *motion cues predict usable windows; part may be head stability*
  (FACTS.md 18:25; E6.01).
- **the two OOD numbers** — 0.529 is the **PoC cascade**; 0.415 is the **frozen ensemble
  `ensemble_v1`**. Never quote one for the other (LB1, `docs/reports/pod_verification.md`).
