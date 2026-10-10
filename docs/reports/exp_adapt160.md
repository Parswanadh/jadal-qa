# O4 — Confirming the transferable-feature adaptation on the 160 OOD labels

Branch `m/adapt160` · experiments `O4.01–O4.02` (registry: `experiments/registry.jsonl`;
results: `eval/results/exp_adapt160/O4.01.json`, `O4.02.json`) · figure:
`docs/experiments/exp_adapt160/images/o4_adapt160.png`.
Every number carries `n`, `label_source`, `protocol`, `ci95`, `device`, `command`, `git_sha`.

**Goal.** Z3.03 reported OOD AUROC **0.84–0.85** for the adapted score, but on only
60 windows with `n_bad = 9` and a window-bootstrap CI. Confirm that result on the full
**160 human-labelled** Egocentric-10K windows (138 good / 22 bad, 47 factories, q1=60 + q2=100)
with the **honest factory-grouped** CI, and re-check the review/error operating point.

**Evaluation only.** The only fitted quantities are the per-feature **sign** taken from
the 150 IndEgo gold windows and (for the Z1 arm) a logistic head on IndEgo. The per-domain
z-score is unsupervised (target mean/std). Nothing is threshold-tuned on the OOD labels.

## TL;DR

The Z3 gain **holds at `n_bad = 22`**: the adapted 6-feature score reaches **OOD AUROC
0.759 [0.597, 0.886]** (factory-grouped 95% CI) and **BA 0.709**, and the stable-8+6
combination **0.781 [0.634, 0.895]** / BA **0.733**, versus the frozen ensemble
**0.566 [0.424, 0.700]**. Both grouped CIs lie entirely above 0.5 and the paired
factory-grouped gain over frozen is **+0.19 / +0.21 with 99.5% of resamples > 0**.
The gain is **weaker but still real in batch q2 alone** (0.70 / 0.73) — the q1 point
estimate (0.86 / 0.84) reproduces Z3.03.

## OOD AUROC + balanced accuracy on the 160 (factory-grouped CI, honest)

| # | recipe | OOD AUROC | grouped 95% CI | window CI (optimistic) | OOD BA | IndEgo |
|---|---|---|---|---|---|---|
| a | **Z3 transferable-6** (per-domain z-score, IndEgo-signed) | **0.759** | **[0.597, 0.886]** | [0.647, 0.859] | **0.709** | 0.684 |
| b | **E5 stable-8 + Z3 transferable-6** (same rule) | **0.781** | **[0.634, 0.895]** | [0.682, 0.868] | **0.733** | 0.736 |
| c | Z1 sharpness-3 head (logistic on IndEgo, zero-shot) | 0.692 | [0.541, 0.828] | [0.588, 0.792] | 0.493¹ | 0.578 |
| c2 | sharpness-3 z-score (like-for-like with a/b) | 0.684 | [0.535, 0.827] | [0.555, 0.798] | 0.641 | 0.634 |
| d | **frozen ensemble_v1** (`usable_prob`) | 0.566 | [0.424, 0.700] | [0.437, 0.694] | 0.546 | — |

¹ The Z1 head is an uncalibrated logistic probability; its **ranking** is 0.692 but its
probability is not centred at 0.5, so the fixed-threshold BA is near chance. Reported as-is.

Balanced accuracy uses a **fixed natural threshold** (0 for the symmetric z-score combos,
0.5 for the probability scorers) — no OOD threshold tuning. `(d)` reproduces the frozen
baseline of the O1 report (0.566; binary BA 0.546).

**Paired factory-grouped gain vs frozen (same bootstrap resamples):**

| recipe | ΔAUROC vs frozen | 95% CI | frac. resamples > 0 |
|---|---|---|---|
| a · Z3 transferable-6 | **+0.193** | **[0.044, 0.343]** | **0.995** |
| b · stable-8 + 6 | **+0.213** | **[0.060, 0.367]** | **0.995** |
| c · Z1 sharpness-3 head | +0.125 | [−0.014, 0.271] | 0.957 |
| c2 · sharpness-3 z-score | +0.116 | [−0.069, 0.305] | 0.878 |

The Z3 gains are **significant**; the Z1 sharpness-3 gain over frozen is suggestive but
its CI touches 0.

## q1 vs q2 (per-batch z-score)

| recipe | q1 (n=60, bad=9) | q2 (n=100, bad=13) |
|---|---|---|
| a · Z3 transferable-6 | 0.861 [0.745, 0.961] | 0.697 [0.464, 0.870] |
| b · stable-8 + 6 | 0.843 [0.720, 0.951] | 0.734 [0.540, 0.891] |
| c · Z1 sharpness-3 head | 0.712 [0.540, 0.863] | 0.683 [0.483, 0.852] |
| c2 · sharpness-3 z-score | 0.682 [0.460, 0.903] | 0.679 [0.467, 0.863] |
| d · frozen ensemble_v1 | 0.415 [0.185, 0.634] | 0.657 [0.472, 0.807] |

- **q1 reproduces Z3.03** (0.861 vs the published 0.852) — the recipe is stable.
- **q2 is weaker**: 0.70 / 0.73, and the q2-only grouped CI **includes 0.5**. So the gain
  holds in the pooled 160 and in q1, but in q2 alone it is **not separable from chance**.
- The frozen baseline shows the **opposite batch ordering** (q1 0.415 → q2 0.657). The
  adapted score is not simply "better everywhere"; the two batches differ in difficulty
  and the frozen score happened to be anti-correlated with the adapted one in q1.

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
| global_shift | 0.376 | 0.509 | flip |
| illum_r | 0.490 | 0.255 | stable |
| illum_b | 0.528 | 0.686 | stable |
| zncc_self | 0.574 | 0.517 | stable |

All **6 transferable candidates remain stable** at 160. Across the full 20 Z3 candidates
the class counts shift slightly (9 stable / 10 flip / 1 chance at 160 vs 9 / 11 at 60):
`ct_fsize_cv` and `ct_iframe_ratio` lose their (weak) signal at 160, and `mo_affine_gain`
becomes "stable" only because both AUROCs sit just below 0.5. The **selection set is
unchanged**.

## Does the gain hold? What remains unproven

**Holds.** At `n_bad = 22` the Z3 adapted score (a) and its stable-8 combination (b) beat
the frozen baseline with a factory-grouped CI excluding 0.5 and a paired grouped gain of
+0.19/+0.21 (99.5% of resamples > 0). The recipe reproduces Z3.03 on q1.

**Not proven.**
- **q2 alone is not separable from chance** (grouped CI [0.46, 0.87] / [0.54, 0.89]).
  The headline rests on pooling q1+q2; batch difficulty (or labeller drift) is confounded.
- The OOD positives are still only **22 windows over 47 factories**, and a few factories
  contribute up to 13 windows with mixed classes, so the grouped bootstrap is wide/lumpy.
- The **BA operating point** of the z-score combos (0.71/0.73) is honest only for the fixed
  threshold 0; it is not a calibrated decision boundary and does not transfer to a
  cost-aware accept/reject/review policy.
- **IndEgo in-domain** stays modest (0.68–0.74); the adapted score is not a strong
  in-domain scorer, it is a *domain-agnostic* one.
- No **few-shot** calibration or self-training was re-run here; those numbers remain
  exploratory from Z1/Z3.

## Caveats

- Single human labeller; 160 windows / 47 factories; `n_bad = 22`. Egocentric-10K and
  IndEgo footage are third-party and never committed.
- The per-domain z-score for the pooled 160 uses the 160's own mean/std; the q1/q2 rows
  z-score each batch with its **own** stats (each batch treated as its own target sample).
  This is unsupervised (no labels) but does use the full unlabelled batch.
- Sign selection uses the full 150 IndEgo gold set (mild optimism); magnitudes are
  unsupervised. The Z1 head picks C by inner LOCO only.
- The Z3 features on q1 are read from the frozen `m/new-features` `candidates.npz`; q2 is
  recomputed with the identical extractor (ffprobe frame table + 160×90 decode + phase
  correlation / Lucas–Kanade + per-frame invariants). The pixel-26 (stable-8) come from
  `ood160_pixel.npz` (10 fps-effective view).

## Reproduce

```
OMP_NUM_THREADS=2 python3 eval/exp_adapt160/extract160.py
OMP_NUM_THREADS=2 python3 eval/exp_adapt160/run_adapt160.py
python3 eval/exp_adapt160/make_figure.py
```

Device: CPU (numpy/sklearn; ffmpeg/cv2 decode). No GPU.
