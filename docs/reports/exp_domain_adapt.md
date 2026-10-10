# E5 — Domain adaptation (IndEgo ↔ Egocentric-10K factory)

Track E5, branch `feat/exp-domain-adapt`, experiments E5.01–E5.04
(registry: `experiments/EXPERIMENTS.md`; results: `eval/results/exp_domain/`).
All numbers come from a results JSON in this repo; every JSON carries
`n`, `label_source`, `protocol`, `ci95`, `device`, `command`, `git_sha`.

**Context (not redone here).** Every IndEgo-trained model failed out-of-domain on
60 human-labelled Egocentric-10K windows (51 good / 9 bad): the frozen ensemble
scores 0.415 AUROC [0.210, 0.647] on those windows (`caches/ensemble/ood_scores.json`,
report `docs/reports/ensemble_views.md`). The OOD set has **n_bad = 9** and about
one window per factory, so all out-of-domain numbers are **exploratory** and the
window-bootstrap CI is **optimistic**.

## TL;DR

| id | question | headline | verdict |
|---|---|---|---|
| E5.01 | does a head trained on the 10k **AI-labelled FACTORY** frames carry window signal on the 150 human IndEgo windows? | LOCO AUROC **0.789–0.826** (pooled AI-frame heads) vs the 0.85 in-domain EG2-window-embedding baseline | maybe |
| E5.02 | how many human OOD labels to beat chance? | pixel-26 **0.756** [0.575, 0.902], two-view **0.767** [0.596, 0.904]; ~12 labels already ~0.71 (std 0.10); smallest n_bad in any train fold **7** | maybe |
| E5.03 | which of the 26 per-pixel features keep their sign? | **8 stable, 14 flip, 4 chance**; sharpness/light stable, motion/temporal-band flipped | maybe |
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
an improvement. It is **model-vs-model agreement on AI targets, not ground-truth
accuracy**: the numbers say a factory-frame head produces a window score that
ranks IndEgo usable/bad windows above chance, nothing more.

Chosen in-domain recipe: keep the **mean of p(active_labor) and p(hands_ok)** as a
cheap prior view only (0.826 LOCO), documented as a weak signal; do not replace the
EG2-embedding view.

## E5.03 — the what-transfers table (signed per-feature AUROC)

Signed AUROC per feature (label 1 = usable/good). IndEgo = 150 human gold windows,
native ~10 fps, clip-bootstrap CI; OOD = 60 human windows, 10 fps effective
(480p transcode subsampled), window-bootstrap CI (optimistic). Full table with CIs
in `E5.03.json`. Classification: **stable / flip** require both |AUROC − 0.5| ≥ 0.05;
otherwise **chance**.

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

Counts: **8 stable, 14 flip, 4 chance** (26 total). The **sharpness / spatial-spectrum**
group (radial_slope, hf_ratio, dct_energy) is the only group that is stable *and*
informative in both domains. **Motion / temporal-band** features (motion_energy,
temp_b0/1/2) carry the most weight in-domain but **invert** out-of-domain, which is the
mechanistic reason the frozen IndEgo pixel view does not transfer. Illumination features
are unstable in magnitude (illum_r/illum_g flip) though the white-balance pair partly
holds. This is a description of feature sign, not a quality measure.

The OOD native (≈30 fps) column in `E5.03.json` agrees with the 10 fps column for
these features (rank-identical classes), so the flips are not an fps artefact.

## E5.02 — few-shot on the human OOD labels (factory-grouped)

60 Egocentric-10K human windows (51 good / 9 bad, 47 factories), human labels
(single labeller, `ood_labels.json`). Protocol: `StratifiedGroupKFold(5, groups=factory)`,
C ∈ {0.01, 0.1, 1, 10} by inner grouped CV on train only, `class_weight=balanced`;
CI = window bootstrap (optimistic). **Smallest n_bad in any training fold = 7.**

| recipe | OOF AUROC | 95% CI | bal.acc@0.5 |
|---|---|---|---|
| logistic on 26 per-pixel feats | 0.756 | [0.575, 0.902] | — |
| logistic on EG2 window means (768-d) | 0.546 | [0.350, 0.736] | — |
| two-view probability average | **0.767** | [0.596, 0.904] | — |

Learning curve (grouped random subsampling, 20 repeats, pixel-26, C = 1 balanced):
~12 windows → 0.709 (std 0.102; 76 % of repeats > 0.6), ~24 → 0.715 (std 0.081),
~36 → 0.703 (std 0.156), ~45 → 0.691 (std 0.274). On *this* set **about a dozen
human labels already lift a fresh head to ~0.7 AUROC**, but the wide spread and the
near leave-one-out grouped geometry (≈1 window/factory) make that an optimistic
upper bound, and n_bad = 9 means every estimate is unstable.

## E5.04 — frozen re-score of the 160 OOD windows

`ood_labels2.json` is **not delivered**. Result JSON records `labels_missing: true`,
`n = 0`, `ci95 = "n/a (no labels)"` — the confirmatory eval is **inconclusive** and
**no accuracy is computed**. Descriptive only, frozen `ensemble_v1`
(config sha `4da1068a…`, matches `configs/ensemble_v1.json`): 160 windows scored
(60 batch-1 `caches/ensemble/ood_scores.json` + 100 batch-2
`ood_scores2.json`), mean usable_prob 0.560 (median 0.460); accept 71 / review 35 /
reject 54. Batch-1 mean 0.496, batch-2 mean 0.598. `eval/exp_domain/watch_e5_04.sh`
re-scores and runs the confirmatory window-bootstrap eval automatically when
`ood_labels2.json` appears.

## What did NOT transfer

- **Motion / temporal-band features** (motion_energy, temp_b0/1/2) — sign inversion
  IndEgo → Egocentric-10K (E5.03); the frozen IndEgo pixel view therefore does not transfer.
- **EG2 window means as a few-shot OOD head** — 0.546 [0.350, 0.736], indistinguishable
  from chance with 9 bad windows (E5.02).
- **AI-frame heads as a drop-in** — 0.789–0.826 LOCO, below the 0.85 in-domain baseline;
  and model-vs-model, not ground truth (E5.01).
- **The frozen ensemble out-of-domain** — 0.415 [0.210, 0.647] (cited prior; the reason E5 exists).

## Caveats

- OOD = **60 human windows, n_bad = 9, ~1 window/factory, single labeller**; every OOD
  number is exploratory and its window-bootstrap CI is optimistic.
- IndEgo = 150 windows from 7 clips / ~3 wearers, single labeller; clip-bootstrap CI.
- E5.01 is **model-vs-model agreement** (AI head labels vs human eval labels), **not accuracy**.
- The 10k frame labels are **AI**, **not human**.
- E5.02 grouped CV with ≈1 window/factory is close to leave-one-out, so few-shot numbers
  are an optimistic bound for adaptation to this specific set, not a generalisation rate.
- Not done: an EG2 re-embed of a *human-labelled* OOD adaptation set (no such labels yet);
  the 100-window batch is only scored descriptively (E5.04).

Regenerate: `OMP_NUM_THREADS=1 python3 eval/exp_domain/run_e5.py all`
(prep: `python3 eval/exp_domain/prep_inputs.py --fetch-git`).
