# O4 — Confirming the transferable-feature adaptation on the 160 OOD labels

Branch `m/adapt160` · experiments `O4.01–O4.02` (registry: `experiments/registry.jsonl`;
results: `eval/results/exp_adapt160/O4.01.json`, `O4.02.json`) · figure:
`docs/experiments/exp_adapt160/images/o4_adapt160.png`.
Every number carries `n`, `label_source`, `protocol`, `ci95`, `device`, `command`, `git_sha`.

**Goal.** Z3.03 reported OOD AUROC **0.84–0.85** for the adapted score, but on only
60 windows with `n_bad = 9` and a window-bootstrap CI. Re-measure on the full
**160 human-labelled** Egocentric-10K windows (138 good / 22 bad, 47 factories, q1=60 + q2=100)
with the **honest factory-grouped** CI, and re-check the review/error operating point.

**Evaluation only.** The only fitted quantities are the per-feature **sign** taken from
the 150 IndEgo gold windows and (for the Z1 arm) a logistic head on IndEgo. The per-domain
z-score is unsupervised (target mean/std). Nothing is threshold-tuned on the OOD labels.

> **Leakage / provenance (read first).** q1 (the first 60 windows) **informed feature
> selection and the Z3.03 recipe**, so the pooled-160 headline is **q1-contaminated** and is
> labelled as such throughout. q2 (100 windows, 13 bad) is the only clean hold-out; on its own
> it is **not separable from chance** (see below). O2.03 (`m/fewshot160`) later re-ran the sign
> table on all 160 labels and changed `global_shift` (stable → flip) and `frozen_frac`
> (flip → stable); O4 kept the pre-O2.03 set, so recipe (b2) reports (b) with `global_shift`
> dropped.

## TL;DR

On the **pooled 160 (q1 contaminated)** the adapted 6-feature score reaches **OOD AUROC
0.759 [0.597, 0.886]** (factory-grouped 95% CI) and **BA 0.709**; the stable-8+6
combination **0.781 [0.634, 0.895]** / BA **0.733**, and the same recipe with the flip
feature `global_shift` dropped **0.788 [0.648, 0.898]** / BA 0.729 — versus the frozen
ensemble **0.566 [0.424, 0.700]**. The paired factory-grouped gain over frozen is
**+0.19 / +0.21 / +0.22** with ≥99.4% of resamples > 0. **The headline is the pooled-160
number, and it is contaminated by q1.** On the clean **q2-only** hold-out the gain is much
weaker (**0.697 / 0.734 / 0.735**); the q2-only grouped CI **includes 0.5 for (a)** and
clears 0.5 only barely for (b)/(b2) (lower bound 0.540 / 0.537). **q2 alone is not
separable from chance.**

## OOD AUROC + balanced accuracy on the 160 (factory-grouped CI, honest)

| # | recipe | OOD AUROC | grouped 95% CI | window CI (optimistic) | OOD BA | IndEgo |
|---|---|---|---|---|---|---|
| a | **Z3 transferable-6** (per-domain z-score, IndEgo-signed) | **0.759** | **[0.597, 0.886]** | [0.647, 0.859] | **0.709** | 0.684 |
| b | **E5 stable-8 + Z3 transferable-6** (same rule) | **0.781** | **[0.634, 0.895]** | [0.682, 0.868] | **0.733** | 0.736 |
| b2 | E5 stable-8 **minus `global_shift`** + Z3-6 (same rule) | **0.788** | **[0.648, 0.898]** | [0.692, 0.871] | 0.729 | 0.727 |
| c | Z1 sharpness-3 head (logistic on IndEgo, zero-shot) | 0.692 | [0.541, 0.828] | [0.588, 0.792] | 0.493¹ | 0.578 |
| c2 | sharpness-3 z-score (like-for-like with a/b) | 0.684 | [0.535, 0.827] | [0.555, 0.798] | 0.641 | 0.634 |
| d | **frozen ensemble_v1** (`usable_prob`) | 0.566 | [0.424, 0.700] | [0.437, 0.694] | 0.546 | — |

¹ The Z1 head is an uncalibrated logistic probability; its **ranking** is 0.692 but its
probability is not centred at 0.5, so the fixed-threshold BA is near chance. Reported as-is.

Balanced accuracy uses a **fixed natural threshold** (0 for the symmetric z-score combos,
0.5 for the probability scorers) — no OOD threshold tuning. `(d)` reproduces the frozen
baseline of the O1 report (0.566; binary BA 0.546).

**Recipe (b) with and without the flip feature.** The O4.02 per-feature table classifies
`global_shift` as **flip** (IndEgo 0.376, OOD-160 0.509), yet it sits in the E5 stable-8
set that recipe (b) uses. Dropping it (recipe **b2**) does **not** hurt: pooled AUROC rises
0.781 → **0.788**, BA is essentially unchanged (0.733 → 0.729), and the paired gain over
frozen is +0.213 → **+0.220**. So the (b) headline does not depend on the flip feature, and
(b2) is the cleaner recipe to freeze.

**Paired factory-grouped gain vs frozen (same bootstrap resamples):**

| recipe | ΔAUROC vs frozen | 95% CI | frac. resamples > 0 |
|---|---|---|---|
| a · Z3 transferable-6 | **+0.193** | **[0.044, 0.343]** | **0.995** |
| b · stable-8 + 6 | **+0.213** | **[0.060, 0.367]** | **0.995** |
| b2 · stable-8 − `global_shift` + 6 | **+0.220** | **[0.076, 0.370]** | **0.997** |
| c · Z1 sharpness-3 head | +0.125 | [−0.014, 0.271] | 0.957 |
| c2 · sharpness-3 z-score | +0.116 | [−0.069, 0.305] | 0.878 |

The Z3 gains are **significant**; the Z1 sharpness-3 gain over frozen is suggestive but
its CI touches 0.

## q1 vs q2 — slices of the pooled per-domain z-score

The per-domain z-score is computed **once on the pooled 160** (`combo_scores`), and q1/q2
are **slices of that single pooled score** (`batch_split`). They are **not per-batch
z-scores**: no batch-specific mean/std is used for these rows. (This corrects the earlier
"per-batch z-score" wording, the stdout label and the figure title.)

| recipe | q1 (n=60, bad=9) | q2 (n=100, bad=13) |
|---|---|---|
| a · Z3 transferable-6 | 0.861 [0.745, 0.961] | 0.697 [0.464, 0.870] |
| b · stable-8 + 6 | 0.843 [0.720, 0.951] | 0.734 [0.540, 0.891] |
| b2 · stable-8 − `global_shift` + 6 | 0.854 [0.742, 0.948] | 0.735 [0.537, 0.891] |
| c · Z1 sharpness-3 head | 0.712 [0.540, 0.863] | 0.683 [0.483, 0.852] |
| c2 · sharpness-3 z-score | 0.682 [0.460, 0.903] | 0.679 [0.467, 0.863] |
| d · frozen ensemble_v1 | 0.415 [0.185, 0.634] | 0.657 [0.472, 0.807] |

- **q1 is not a hold-out** — it informed feature selection and the Z3.03 recipe, so the q1
  row (0.861) is an in-selection number, not a confirmation.
- **q2 is the only clean hold-out and is weaker**: 0.70 / 0.73 / 0.74. The q2-only grouped
  CI **includes 0.5 for (a)** ([0.464, 0.870]); for (b) and (b2) the lower bound is **0.540
  / 0.537**, i.e. it clears 0.5 only barely. **q2 alone is not separable from chance.**
- The frozen baseline shows the **opposite batch ordering** (q1 0.415 → q2 0.657). The
  adapted score is not simply "better everywhere"; the two batches differ in difficulty
  and the frozen score happened to be anti-correlated with the adapted one in q1.

## Selection provenance and the O2.03 change

- The six "transferable" features and their sign-stability / "≥0.03 from 0.5 in both
  domains" rule come from Z3.01/Z3.02, evaluated on IndEgo 150 **plus the OOD-60 human
  labels (q1)**; the Z3.03 recipe was chosen on those same 60. The stable-8 come from E5.03,
  also built on the 60 OOD windows. **So q1 is contaminated by construction.**
- O2.03 (`m/fewshot160`, finished 17:40, before O4.01 started 17:41) re-ran the sign table
  on **all 160** labels and changed two features: `global_shift` (stable → flip) and
  `frozen_frac` (flip → stable). O4 kept the old set; recipe (b2) above reports (b) with the
  affected `global_shift` removed, and the conclusion is unchanged.
- No O4 code step reads q2 labels for feature selection, so q2 remains a clean hold-out.

## Transduction and the threshold (deployability)

- The per-domain z-score uses the **target batch's own feature statistics**. It is
  unsupervised (no labels) but **transductive**: a new window cannot be scored without the
  rest of its batch. AUROC is therefore a transductive ranking measure.
- The BA threshold **0 is the pooled signed-mean centre** (each z-scored feature has mean 0
  over the 160), so the BA at 0.709 / 0.733 is tied to the OOD pool's centre and is **not a
  deployable decision boundary** without a batch-normalisation rule. Treat the z-score BA as
  descriptive, not operational.
- No OOD threshold tuning is performed; the claim is limited to ranking (AUROC) and the
  descriptive pooled-centre BA.

## Bootstrap accounting

Factory-grouped bootstrap is the honest CI (`n_boot = 2000`, `seed = 0`); a resample that
draws a single class is skipped. Totals for this run (`O4.01.json` `bootstrap_resamples`):
`group_boot` used 47,988 / skipped 12; `window_boot` used 12,000 / skipped 0;
`paired_group_boot` used 10,000 / skipped 0. The skips are negligible but are now reported
rather than dropped silently.

## Frozen recipe (before the confirmatory set)

The exact recipe (feature lists, IndEgo-only signs, z-score definition, threshold rule) is
frozen as `eval/exp_adapt160/o4_recipe.json` with SHA-256
`449c27e0fdd8c2bde7461e63a62dc67e080bfb4413c19cc876b3569dec3b416c`
(`eval/exp_adapt160/o4_recipe.sha256`; `run_adapt160.py` verifies the sidecar and embeds the
hash in `O4.01.json`). Plan: the new **400-window** confirmatory set (queue 3, `labels3.json`)
will be scored **once, unchanged**, and that single number will replace the pooled-160 claim.

## Per-feature transfer on the 160 (`O4.02`)

| feature | IndEgo AUROC | OOD-160 AUROC | class |
|---|---|---|---|
| ct_fsize_std | 0.620 | 0.649 [0.462, 0.818] | **stable** |
| mo_hand_motion_excess | 0.649 | 0.608 [0.481, 0.746] | **stable** |
| inv_clip_frac | 0.656 | 0.661 [0.451, 0.859] | **stable** |
| inv_laplacian_var | 0.641 | 0.718 [0.566, 0.852] | **stable** |
| inv_hist_entropy | 0.534 | 0.714 [0.558, 0.846] | **stable** |
| inv_dct_hf_lf | 0.310 | 0.244 [0.117, 0.389] | stable (both low) |
| radial_slope | 0.633 | 0.635 | stable |
| hf_ratio | 0.618 | 0.657 | stable |
| dct_energy | 0.631 | 0.700 | stable |
| luma_p05 | 0.392 | 0.442 | stable (both low) |
| global_shift | 0.376 | 0.509 | **flip** |
| illum_r | 0.490 | 0.255 | stable |
| illum_b | 0.528 | 0.686 | stable |
| zncc_self | 0.574 | 0.517 | stable |

All **6 transferable candidates remain stable** at 160. Across the full 20 Z3 candidates
the class counts shift slightly (9 stable / 10 flip / 1 chance at 160 vs 9 / 11 at 60):
`ct_fsize_cv` and `ct_iframe_ratio` lose their (weak) signal at 160, and `mo_affine_gain`
becomes "stable" only because both AUROCs sit just below 0.5. The **selection set is
unchanged** (O4 did not re-select on q2; O2.03's two changes are disclosed above).

## Does the gain hold? What remains unproven

**Holds (pooled, q1 contaminated).** At `n_bad = 22` the Z3 adapted score (a), its stable-8
combination (b), and the flip-dropped variant (b2) beat the frozen baseline with a
factory-grouped CI excluding 0.5 and a paired grouped gain of +0.19/+0.21/+0.22 (≥99.4% of
resamples > 0). The recipe reproduces Z3.03 on q1.

**Not proven.**
- **q2 alone is not separable from chance** (grouped CI [0.464, 0.870] for (a); [0.540,
  0.891] / [0.537, 0.891] for (b)/(b2)). The headline rests on pooling q1+q2; batch
  difficulty (or labeller drift) is confounded.
- The OOD positives are still only **22 windows over 47 factories**, and a few factories
  contribute up to 13 windows with mixed classes, so the grouped bootstrap is wide/lumpy.
- The **BA operating point** of the z-score combos (0.71/0.73) is descriptive only: it is
  tied to the pooled mean (threshold 0) and the per-domain z-score is transductive, so it
  does not transfer to a cost-aware accept/reject/review policy without a batch-normalisation
  rule.
- **IndEgo in-domain** stays modest (0.68–0.74); the adapted score is not a strong
  in-domain scorer, it is a *domain-agnostic* one.
- No **few-shot** calibration or self-training was re-run here; those numbers remain
  exploratory from Z1/Z3.

## Review point (step 3) — attempted, not completed in the timebox

The published review policy (Y1/Y2) is defined on the frozen **probability** `usable_prob`
(symmetric band half-width `w = 0.16`; gate deciding the top `c = 0.45` by margin
`|p − 0.5|`). The adapted Z3 score is an **uncalibrated signed z-score** (boundary 0), so
re-deriving the review point on it requires a calibration to `[0,1]` — and Z3.03 showed a
logistic calibration of these features **does not transfer** (raw logistic OOD AUROC ≈ 0.50).
Any review% quoted on the uncalibrated score would therefore be either trivial (a percentile
re-mapping) or misleading (a non-transferring calibration).

To do it properly one must score the **1,546 unlabelled E8 windows** with the Z3 extractor.
That needs a full-file `ffprobe` (for `ct_fsize_std`) plus a native-fps decode per window;
under the 40-min timebox and the current pod load this exceeded budget and was stopped after
the probe phase. Ready-to-run tooling is committed
(`eval/exp_adapt160/extract_e8_sample.py`, `review_point.py`):

```
export JADAL_BUNDLE=/path/on/pod
export JADAL_E8_JSONL=/path/to/windows_rescored.jsonl
OMP_NUM_THREADS=2 python3 eval/exp_adapt160/extract_e8_sample.py --n 100
OMP_NUM_THREADS=2 python3 eval/exp_adapt160/review_point.py
```

**Consequence for the claim:** the adapted score is currently a **ranking** contribution
(AUROC), not an operating-point contribution. The review/error trade-off of the shipped
system is unchanged and remains the frozen policy's (review 53–75%, OOD BA near chance).
This is the main item left unproven.

## Caveats

- Single human labeller; 160 windows / 47 factories; `n_bad = 22`. Egocentric-10K and
  IndEgo footage are third-party and never committed.
- The per-domain z-score for the pooled 160 uses the 160's own mean/std; the q1/q2 rows are
  **slices of that one pooled score** (not per-batch z-scores). This is unsupervised (no
  labels) but **transductive** — it uses the full unlabelled batch, and the threshold 0 is
  the pooled mean.
- Sign selection uses the full 150 IndEgo gold set (mild optimism); magnitudes are
  unsupervised. The Z1 head picks C by inner LOCO only.
- The Z3 features on q1 are read from the frozen `m/new-features` `candidates.npz`; q2 is
  recomputed with the identical extractor (ffprobe frame table + 160×90 decode + phase
  correlation / Lucas–Kanade + per-frame invariants). The pixel-26 (stable-8) come from
  `ood160_pixel.npz` (10 fps-effective view).

## Reproduce

Pod paths are read from the environment only (no pod paths are hard-coded in tracked code):

```
export JADAL_BUNDLE=/path/on/pod                 # labels + caches
export JADAL_Z3_CAND=/path/candidates.npz        # frozen Z3 batch-q1 candidates (extract160)
export JADAL_PIXEL160=/path/ood160_pixel.npz     # pixel-26 OOD input
OMP_NUM_THREADS=2 python3 eval/exp_adapt160/extract160.py
OMP_NUM_THREADS=2 python3 eval/exp_adapt160/run_adapt160.py
python3 eval/exp_adapt160/make_figure.py
```

Device: CPU (numpy/sklearn; ffmpeg/cv2 decode). No GPU.
