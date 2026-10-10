# Y3 — Review reduction by making the score more separable (cheap cached-view ensembling)

**Branch:** `m/review-score` · **Owner:** Y3 (claim Y3) · **Device:** CPU, `OMP_NUM_THREADS=2`
**Timebox:** 30 min · **Status:** complete (clean negative + one cheap positive)
**Results:** `eval/results/exp_review/Y3.01.json`, `eval/results/exp_review/Y3.02.json`
**Code:** `eval/exp_review/run_y3.py` (deterministic, seed 7, all inputs read-only from `/workspace/jadal_pod_bundle`)
**Figure:** `docs/experiments/exp_review/images/y3_separability.png`

## Question

The frozen two-view PoC (pixel-26 + EG2@2fps, equal-weight probability average) is
not separable out-of-domain: **1,167 / 1,546 (75.5%)** of the unlabelled
Egocentric-10K windows get a REVIEW verdict (`/workspace/artifacts/windows_rescored.jsonl`).
Does adding more **cached** views — per-pixel series / physics features, extra
EG2 frame rates — or switching **probability-average → rank-average** cut the
ambiguous (REVIEW) share at equal or better AUROC?

## Headline honesty checks (do these first)

1. **The 75.5% is not the `[0.4,0.6]` abstain band.** Only **29.6%** of the 1,546
   re-scored windows have `score ∈ [0.4,0.6]`. The 75.5% is the
   `factory_1080p30` **profile** band `[0.25,0.75]` (75.4% of scores) produced by
   reason `LOW_MARGIN_PROFILE_REVIEW` (E4.04 profile routing), not by the core
   two-view abstain band. Score separability can only move the part of the 75.5%
   that lives in the wider profile band.
2. **The window scores on the 1,546 have `label_source: none`.** Their REVIEW
   share is a *workload* number, not an accuracy number. All in-domain accuracy
   below is on the 150-window IndEgo **human** gold.
3. **No EG2@1fps / EG2@0.5fps cache exists.** `caches/frames10k/segments.npz` is
   the only EG2 view (21 frames @ 0.5 s spacing = **2 fps**). That candidate is
   reported as absent and skipped.
4. **OOD accuracy is near chance for every view** (n = 60, 51 good / 9 bad):
   p_eg2 AUROC 0.586, p_pixel 0.342, probavg 0.415. n_bad = 9 ⇒ CIs are wide.
   OOD is evaluation only, never fit on.

## Protocol

* **In-domain:** 150-window IndEgo human gold (100 good / 50 bad, single
  labeller). **Nested leave-one-clip-out** over 7 clips: for each held-out clip,
  the logistic regularisation `C` is chosen by an *inner* LOCO on the 6 training
  clips only (`C ∈ {0.003,0.01,0.03,0.1,0.3,1.0}`), then refit and applied to the
  held-out clip. No held-out label touches any fit. `class_weight=balanced`.
* **Per-view OOF reproduces the frozen artifacts exactly:** nested-LOCO
  pixel-26 = 0.7916 and EG2 = 0.8458, matching `caches/ensemble/oof_probs.npz`
  (`p_pixel`, `p_eg2`); nested-LOCO two-view prob-average = 0.8336, matching
  `nested_loco.json:ensemble_probavg`.
* **Frozen anchor:** with the config's fixed `C` (pixel 0.3, EG2 0.01) the
  two-view prob-average gives **AUROC 0.8534** — reproducing
  `ensemble_v1.json:full_oof_auroc_resubstitution_free`.
* **Ambiguous fraction** = share of scores inside the fixed band (the REVIEW
  proxy). Reported on the deployed probability scale (`amb_raw`) and after a
  LOCO Platt calibration (`amb_cal`) so probability- and rank-average fusions are
  comparable. The total-stay statistic (Zhang 2009, `median_stay_w0.1`) is a
  monotone-invariant check: it is near-constant (~0.51–0.67) across every combo,
  i.e. most band changes are operating-point/calibration effects, not new
  information.
* **Clip-bootstrap** CI (2,000 resamples of the 7 clips) for every AUROC.

## Views available (all cached)

| view | dim | nested-LOCO AUROC | band `[0.4,0.6]` |
|---|---|---|---|
| `pixel26` (`caches/gpu/features.json`) | 26 | 0.7916 | 0.247 |
| `eg2` (`caches/frames10k/segments.npz`, 2 fps) | 768 | **0.8458** | **0.160** |
| `perpixel121` (`caches/perpixel/gold.npz`) | 121 | 0.8140 | 0.320 |
| `physics16` (`caches/perpixel/physics_feats.npz`) | 16 | 0.7608 | 0.147 |
| `theory18` (`caches/perpixel/theory_feats.npz`) | 18 | 0.6698 | 0.467 |
| `series16` (`caches/perpixel/series_gold.npz`, mean+std over time) | 16 | 0.7912 | 0.227 |
| EG2@1fps / @0.5fps | — | **absent** | — |

## Combo sweep (nested LOCO, 150 IndEgo gold)

Sorted by the profile-band ambiguity `[0.25,0.75]` (the band that actually drives
the 75.5% headline), then AUROC ↓.

| combo | fusion | AUROC [95% CI] | amb `[.4,.6]` raw | amb `[.4,.6]` cal | amb `[.25,.75]` raw |
|---|---|---|---|---|---|
| `eg2` alone | probavg | 0.846 [0.717, 0.915] | **0.160** | **0.093** | **0.420** |
| `eg2` alone | rankavg | 0.846 [0.718, 0.913] | 0.207 | 0.120 | 0.500 |
| `pixel26+eg2` (baseline) | rankavg | 0.844 [0.724, 0.924] | 0.180 | 0.167 | 0.607 |
| `pixel26+eg2` (baseline) | probavg | 0.834 [0.724, 0.913] | 0.213 | 0.140 | 0.653 |
| `+perpixel121` | rankavg | 0.837 [0.725, 0.916] | 0.207 | 0.147 | 0.587 |
| `+perpixel121+physics16` | rankavg | 0.829 [0.694, 0.906] | 0.193 | 0.140 | 0.580 |
| `+physics16` | probavg | 0.819 [0.687, 0.902] | 0.173 | 0.160 | 0.613 |
| `+series16` | probavg | 0.835 [0.694, 0.919] | 0.213 | 0.153 | 0.680 |
| `+perpixel121` | probavg | 0.832 [0.716, 0.917] | 0.227 | 0.147 | 0.700 |
| `+perpixel121+physics16` | probavg | 0.821 [0.688, 0.907] | 0.220 | 0.140 | 0.707 |
| `+perpixel121+physics16+series16+theory18` | probavg | 0.812 [0.683, 0.899] | 0.227 | 0.140 | 0.787 |
| `pixel26` alone | probavg | 0.792 [0.668, 0.886] | 0.247 | 0.167 | 0.613 |
| frozen anchor, fixed C (probavg) | probavg | **0.8534** [0.7295, 0.9285] | 0.173 | 0.127 | 0.427 |

Full table incl. `theory18` and `[0.25,0.75]` calibrated: `Y3.02.json`.

## OOD descriptive slice (n = 60 human-labelled, 51 good / 9 bad; evaluation only)

| score | AUROC | amb `[0.4,0.6]` raw |
|---|---|---|
| `p_eg2` | 0.586 | 0.133 |
| `p_pixel` | 0.342 | 0.033 |
| probavg (`usable_prob`, deployed) | 0.415 | 0.233 |
| rankavg | 0.468 | 0.333 |

IndEgo-trained views do not transfer (all ≤ 0.59). The `[0.4,0.6]` band on OOD
is *not* the 75.5% (see honesty check 1): the 75.5% lives in the profile band
`[0.25,0.75]` (0.754) on the unlabelled 1,546.

## Findings

1. **Answer: no.** Adding cached views does **not** reduce ambiguity at equal or
   better AUROC. Every extra view (`perpixel121`, `physics16`, `series16`,
   `theory18`) leaves AUROC flat-to-worse and **increases** the review-band mass
   (e.g. `[0.25,0.75]` 0.653 → 0.707 → 0.787 as views are stacked on the
   baseline). The monotone-invariant total-stay statistic is ~constant, so this
   is not an information gain; the weak pixel/per-pixel views drag the fused
   score toward the middle.
2. **The single view is stronger than the fusion.** EG2 alone dominates the
   two-view baseline on *both* axes: AUROC 0.834 → 0.846 and band `[0.4,0.6]`
   calibrated 0.140 → 0.093, profile-band raw 0.653 → 0.420. The pixel view is
   the separability drag.
3. **Cheapest deployable change with no new compute:** switch the frozen
   two-view fusion from probability-average to **rank-average**: AUROC
   0.834 → 0.844, `[0.4,0.6]` raw 0.213 → 0.180, `[0.25,0.75]` raw
   0.653 → 0.607. It uses the two probabilities already computed per window.
4. **All AUROC differences are inside the clip-bootstrap CI** (baseline
   [0.724,0.913] vs EG2 [0.717,0.915]); the AUROC edge is not statistically
   established. The ambiguity reductions are the firmer signal, but are partly
   operating-point effects (total-stay ≈ constant).

## Recommendation

* **Adopt (cheap, safe):** rank-average of the existing two views — zero new
  inference, lower REVIEW-band mass, AUROC not worse.
* **Consider (needs a 1-line change, still one view):** EG2-only scoring. It cuts
  the review-band mass the most (0.653 → 0.420 at `[0.25,0.75]`) and has the
  best AUROC, but it discards the CPU-cheap pixel head and does not change OOD
  transfer, so gate it on a held-out confirmation.
* **Do not adopt:** adding per-pixel / physics / series / theory views to the
  score (no AUROC gain, more REVIEW).
* **Reframe the 75.5%:** most of it is the `factory_1080p30` profile band
  `[0.25,0.75]`, not the `[0.4,0.6]` abstain band (29.6%). Narrowing the profile
  band is a *funnel/calibration* decision (see review-band pod), not a
  separability tune; score ensembling alone cannot remove the 75.5%.

## Reproduce

```bash
cd <worktree>
OMP_NUM_THREADS=2 python3 eval/exp_review/run_y3.py
# -> eval/results/exp_review/Y3.01.json, Y3.02.json
# -> docs/experiments/exp_review/images/y3_separability.png
```
