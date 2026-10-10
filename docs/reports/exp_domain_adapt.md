# E5 — Domain adaptation (IndEgo ↔ Egocentric-10K factory)

Track E5, branch `feat/exp-domain-adapt`, experiments E5.00–E5.04
(registry: `experiments/EXPERIMENTS.md`; results: `eval/results/exp_domain/`).
All numbers come from a results JSON in this repo; every JSON carries
`n`, `label_source`, `protocol`, `ci95`, `device`, `command`, `git_sha`.

**Context (reproduced here as E5.00, `eval/results/exp_domain/E5.00.json`).** Every
IndEgo-trained model failed out-of-domain on 60 human-labelled Egocentric-10K windows
(51 good / 9 bad): the frozen ensemble `usable_prob` scores **0.415 AUROC [0.201, 0.656]**
on those windows (window bootstrap, optimistic). E5.00 recomputes this from the frozen
scores in the pod bundle `$JADAL_BUNDLE/caches/ensemble/ood_scores.json`; the earlier
report with CI [0.210, 0.647] (`docs/reports/ensemble_views.md`) lives on `main` / PR #79
and is **not on this branch**. The OOD set has **n_bad = 9** and about one window per
factory, so all out-of-domain numbers are **exploratory** and the window-bootstrap CI is
**optimistic**.

## TL;DR

| id | question | headline | verdict |
|---|---|---|---|
| E5.00 | does the frozen ensemble reproduce its OOD prior on the 60 human windows? | **0.415** AUROC [0.201, 0.656] (exploratory, n_bad=9) | cited prior |
| E5.01 | does a head trained on the 10k **AI-labelled FACTORY** frames carry window signal on the 150 human IndEgo windows? | LOCO AUROC **0.789–0.826** (pooled AI-frame heads) vs the 0.85 in-domain EG2-window-embedding baseline | maybe |
| E5.02 | how many human OOD labels to beat chance? | pixel-26 **0.756** [0.575, 0.902], two-view **0.767** [0.596, 0.904]; ~12 labels already ~0.71 (std 0.10); smallest n_bad in any train fold **7** | maybe |
| E5.03 | which of the 26 per-pixel features keep their sign? | **8 stable, 14 flip, 4 chance**; sign-stable, still weak out-of-domain (AUROC 0.64–0.70); motion/temporal-band flip | maybe |
| E5.04 | frozen re-score of the 160 OOD windows | labels2 **not delivered** → confirmatory eval **inconclusive** (n = 0); 160 windows scored descriptively only | inconclusive |

## E5.01 — in-domain heads from AI labels, pooled to human windows

Frame heads = StandardScaler + logistic (C = 1, no class weighting) on the frozen
EG2 embeddings of the 10k AI-labelled Egocentric-10K-Evaluation **FACTORY** frames
(`active_labor`, `hands_ok = hand_count ≥ 1`). Labels are **AI (gemini-2.5-flash
prompt), NOT human**. Frame-level 5-fold CV (AI-vs-AI, i.e. **model-vs-model**):
`active_labor` 0.911 [0.901, 0.920], `hands_ok` 0.957 [0.948, 0.965].

Pooled to each of the 150 human IndEgo windows over 21 frames (0.5 s spacing);
evaluated against the **human** gold labels with **leave-one-clip-out** over the
7 clips + clip-bootstrap CI (`E5.01.json`):

| window score | AUROC | clip-boot 95% CI | bal.acc@0.5 |
|---|---|---|---|
| mean p(active_labor) | 0.813 | [0.695, 0.864] | 0.685 |
| mean p(hands_ok) | 0.818 | [0.685, 0.893] | 0.560 |
| mean of both | **0.826** | [0.717, 0.881] | 0.615 |
| LOCO logistic on 12 pooled feats | 0.789 | [0.686, 0.862] | 0.735 |

Reference in-domain baseline: EG2 window-embedding logistic **0.85** (nested 0.846,
`caches/ensemble/nested_loco.json`); 26 per-pixel-feature logistic 0.81.
The pooled AI-frame heads are **below** that baseline, so E5.01 is not adopted as
an improvement. Two different claims have to be kept apart: the **frame-level**
0.911 / 0.957 5-fold CV is **AI-vs-AI (model-vs-model)** and is **not ground-truth
accuracy**; the **window-level** 0.789–0.826 *is* **scored against the human IndEgo
gold labels** (single labeller, LOCO over 7 clips). What it measures is how well a
score produced by an AI-label-trained head ranks 150 human-labelled windows — nothing
more.

Chosen in-domain recipe: keep the **mean of p(active_labor) and p(hands_ok)** as a
cheap prior view only (0.826 LOCO), documented as a weak signal; do not replace the
EG2-embedding view.

## E5.03 — the what-transfers table (signed per-feature AUROC)

Signed AUROC per feature (label 1 = usable/good). IndEgo = 150 human gold windows,
native ~10 fps, clip-bootstrap CI; OOD = 60 human windows, 10 fps effective
(480p transcode subsampled), window-bootstrap CI (optimistic). Full table with CIs
in `E5.03.json`. Classification rule (`e5lib.classify_transfer`, asserted in the
tests): **chance** iff **both** domains have |AUROC − 0.5| < 0.05; otherwise
**stable** if the two AUROCs are on the same side of 0.5, **flip** if opposite. A
feature informative in only one domain is therefore *not* `chance`.

| feature | IndEgo | OOD (10 fps) | class |
|---|---|---|---|
| radial_slope | 0.633 | 0.645 | stable |
| hf_ratio | 0.618 | 0.693 | stable |
| dct_energy | 0.631 | 0.702 | stable |
| luma_p05 | 0.392 | 0.473 | stable (both low) |
| global_shift | 0.376 | 0.461 | stable (both low) |
| illum_r | 0.490 | 0.163 | stable (both low) |
| illum_b | 0.528 | 0.763 | stable |
| zncc_self | 0.574 | 0.545 | stable |
| motion_energy | 0.308 | 0.721 | **flip** |
| temp_b0 | 0.534 | 0.240 | **flip** |
| temp_b1 | 0.337 | 0.584 | **flip** |
| temp_b2 | 0.287 | 0.695 | **flip** |
| skin_frac | 0.667 | 0.257 | **flip** |
| pca_resid | 0.449 | 0.693 | **flip** |
| pca_top1_share | 0.539 | 0.349 | **flip** |
| luma_mean | 0.509 | 0.292 | **flip** |
| luma_p50 | 0.517 | 0.224 | **flip** |
| luma_p95 | 0.499 | 0.723 | **flip** |
| bright_frac | 0.371 | 0.678 | **flip** |
| illum_g | 0.401 | 0.695 | **flip** |
| rppg_pulse | 0.470 | 0.675 | **flip** |
| frozen_frac | 0.624 | 0.497 | **flip** |
| temp_b3 | 0.500 | 0.500 | chance |
| flicker_idx | 0.500 | 0.500 | chance |
| dark_frac | 0.480 | 0.451 | chance |
| dup_hash_mean | 0.500 | 0.500 | chance |

Counts: **8 stable, 14 flip, 4 chance** (26 total). Only *feature-sign stability* was
measured. The **sharpness / spatial-spectrum** group (radial_slope, hf_ratio,
dct_energy) is sign-stable, but still **weak out-of-domain (OOD AUROC 0.64–0.70)**.
**Motion / temporal-band** features (motion_energy, temp_b0/1/2) carry the most weight
in-domain but **invert** out-of-domain, which is the mechanistic reason the frozen
IndEgo pixel view does not transfer. The illumination features do **not** hold as a
group: `illum_g` flips, and `illum_r` is labelled `stable` only because both domains
are near chance (|AUROC − 0.5| < 0.05 on the IndEgo side), so it is not a usable cue.
This is a description of feature sign, not a quality measure.

The OOD native (≈30 fps) column is stored per row as `class_native30`. Applied to the
same rule it **disagrees with the 10 fps-effective class for 3 of 26 features**
(`temp_b3`, `flicker_idx`: chance→flip; `dark_frac`: chance→stable); the other 23 agree.
So most flips are not an fps artefact, but those three near-chance features are
fps-sensitive and are not a reliable `chance`/`flip`/`stable` call
(`native30_vs_10fps_class_disagreements` in `E5.03.json`).

## E5.02 — few-shot on the human OOD labels (factory-grouped)

60 Egocentric-10K human windows (51 good / 9 bad, 47 factories), human labels
(single labeller, `ood_labels.json`). Protocol: `StratifiedGroupKFold(5, groups=factory)`,
C ∈ {0.01, 0.1, 1, 10} by inner grouped CV on train only, `class_weight=balanced`;
CI = window bootstrap (optimistic). **Smallest n_bad in any training fold = 7.**

| recipe | OOF AUROC | 95% CI | bal.acc@0.5 |
|---|---|---|---|
| logistic on 26 per-pixel feats | 0.756 | [0.575, 0.902] | 0.634 |
| logistic on EG2 window means (768-d) | 0.546 | [0.350, 0.736] | 0.520 |
| two-view probability average | **0.767** | [0.596, 0.904] | 0.614 |

Learning curve (grouped random subsampling, pixel-26, C = 1 balanced): ~12 windows →
0.709 (std 0.102; 76 % of repeats > 0.6), ~24 → 0.715 (std 0.081), ~36 → 0.703
(std 0.156), ~45 → 0.691 (std 0.274). Repeats per point are **17 / 20 / 20 / 19**
(not 20 everywhere): draws are dropped when a train/test side ends up single-class. At
the 12-window point a training fold can contain as few as **1 bad example**
(`min_n_bad_train: 1` in `E5.02.json`), so that point sits close to a single-example
signal. On *this* set **about a dozen human labels already lift a fresh head to ~0.7
AUROC**, but the wide spread and the near leave-one-out grouped geometry
(≈1 window/factory) make that an optimistic upper bound, and n_bad = 9 means every
estimate is unstable.

## E5.04 — frozen re-score of the 160 OOD windows

`ood_labels2.json` is **not delivered**: the 160 scored windows have **no labels**
(`label_source` in `E5.04.json` says so explicitly) and the confirmatory eval is
**inconclusive** (`labels_missing: true`, `n = 0`, `ci95 = "n/a (no labels)"`); **no
accuracy is computed**. Descriptive only, frozen `ensemble_v1`: 160 windows scored
(60 batch-1 `$JADAL_BUNDLE/caches/ensemble/ood_scores.json` + 100 batch-2
`ood_scores2.json`), mean usable_prob 0.560 (median 0.460); accept 71 / review 35 /
reject 54. Batch-1 mean 0.496, batch-2 mean 0.598. Config check: the frozen scorer
recorded sha `4da1068a…`; the local `configs/ensemble_v1.json` raw sha is `d62a95a…`
because `scripts/remap_paths.py` rewrote pod paths, but the canonical content (path
and provenance fields stripped) **matches** — `config_assert` and both observed shas
are stored in `E5.04.json`. `eval/exp_domain/watch_e5_04.sh` re-scores and runs the
confirmatory window-bootstrap eval automatically when `ood_labels2.json` appears.

## What did NOT transfer

- **Motion / temporal-band features** (motion_energy, temp_b0/1/2) — sign inversion
  IndEgo → Egocentric-10K (E5.03); the frozen IndEgo pixel view therefore does not transfer.
- **EG2 window means as a few-shot OOD head** — 0.546 [0.350, 0.736], indistinguishable
  from chance with 9 bad windows (E5.02).
- **AI-frame heads as a drop-in** — 0.789–0.826 LOCO (scored against human IndEgo
  labels) but from an AI-label-trained head, so it is agreement, not a transferable
  accuracy, and it is below the 0.85 in-domain baseline (E5.01).
- **The frozen ensemble out-of-domain** — 0.415 AUROC [0.201, 0.656] (reproduced in
  E5.00 from the frozen bundle scores; exploratory, n_bad = 9). This is why E5 exists.

## Caveats

- OOD = **60 human windows, n_bad = 9, ~1 window/factory, single labeller**; every OOD
  number is exploratory and its window-bootstrap CI is optimistic.
- IndEgo = 150 windows from 7 clips / ~3 wearers, single labeller; clip-bootstrap CI.
- E5.01: the frame-level 0.911/0.957 is **AI-vs-AI (model-vs-model)**; the window-level
  0.789–0.826 *is* scored against the human IndEgo gold labels, but comes from a head
  trained on AI labels — agreement, not ground truth.
- The 10k frame labels are **AI**, **not human**.
- E5.02 grouped CV with ≈1 window/factory is close to leave-one-out, so few-shot numbers
  are an optimistic bound for adaptation to this specific set, not a generalisation rate.
- Not done: an EG2 re-embed of a *human-labelled* OOD adaptation set (no such labels yet);
  the 100-window batch is only scored descriptively (E5.04).

Regenerate: `OMP_NUM_THREADS=1 python3 eval/exp_domain/run_e5.py all`
(prep: `python3 eval/exp_domain/prep_inputs.py --fetch-git`).
