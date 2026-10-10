# Deterministic physics rule-list model (Q1.01) — in-domain LOCO

**Claim.** A *fully deterministic* rule list — fixed integer thresholds, no
learned weights, no probabilities — separates usable from unusable IndEgo
windows with LOCO AUROC **0.822** (95% CI 0.715–0.861). This AUROC is an
**optimistic in-domain estimate**: the candidate rule list, the features and the
directions were chosen by the author with all 150 labels visible (the *Leakage*
section below; only the thresholds and the operating point are nested in LOCO).

The **honest full-population headline** is balanced accuracy **0.710**
(REVIEW counted as accept) / **0.590** (REVIEW as reject) at 40 % coverage. The
0.808 figure is **decided-only** and must not be the headline.

Against the **like-for-like** 42-feature logistic (26 full + 16 physics, AUROC
0.787) the rule list leads by **+0.035**, but the paired clip-bootstrap CI
contains 0 — **no difference is established** at n=150.

Code: `eval/exp_phys/phys_rules.py`, `eval/exp_phys/run_exp_phys.py`.
Results: `eval/results/exp_phys/Q1.01.json`, `rule_table.json`, `roc_phys.png`.
Protocol: leave-one-clip-out (LOCO) over the 7 IndEgo clips, 150 human-labelled
windows (100 good / 50 bad), clip-bootstrap 95% CI (2,000 resamples).
Data root: `$JADAL_DATA` (required; cached features only, no GPU, no decode).

## Why this model
The product spec requires transparent, mathematics-first decisions ("rule list
with exact thresholds"); logistic regression and boosted trees are only
comparison baselines. The earlier naive rule list was degenerate (single
tripwires, no score). This is the real one: a fixed candidate rule list whose
thresholds are *derived from the training clips only*, with an integer score and
an explicit ACCEPT / REJECT / REVIEW decision carrying evidence.

## Rule table (frozen, all-clip thresholds for deployment)
Thresholds are the quantiles **p01/p05/p10/p25** (mirrored to **p99/p95/p90/p75**
for high-is-bad features) of each feature on the 7 clips, quantised to 4 dp.
`rule_table.json` is the machine-readable table; a quality rule that crosses its
**extreme** (first) threshold forces REJECT, softer crossings add to the score.

| rule | feature | op | thresholds (extreme → mild) | strength |
|---|---|---|---|---|
| TOO_DARK | full.luma_mean | < | 0.0723, 0.1845, 0.2360, 0.2678 | quality |
| TOO_DARK | full.dark_frac | > | 0.7398, 0.3709, 0.2394, 0.1949 | quality |
| OVEREXPOSED | full.bright_frac | > | 0.0979, 0.0459, 0.0251, 0.0057 | quality |
| BLUR | full.hf_ratio | < | 0.0047, 0.0077, 0.0097, 0.0134 | quality |
| BLUR | full.dct_energy | < | 0.0002, 0.0035, 0.0049, 0.0068 | quality |
| FROZEN_STATIC | full.temp_b0 | < | 0.8563, 4.7671, 7.6483, 9.7065 | quality |
| FROZEN_STATIC | full.frozen_frac | > | 0.0223, 0.0041, 0.0024, 0.0003 | quality |
| NO_ACTIVE_WORK | phys.flow_act | > | 0.8914, 0.7970, 0.7661, 0.6394 | weak |
| NO_ACTIVE_WORK | phys.flow_med | > | 2.2787, 2.0763, 1.9475, 1.3544 | weak |
| NO_ACTIVE_WORK | full.motion_energy | > | 0.0734, 0.0669, 0.0624, 0.0502 | weak |
| HANDS_NOT_VISIBLE | full.skin_frac | < | 0.0003, 0.0046, 0.0066, 0.0161 | weak |
| DUPLICATE | phys.dhash_loop_min | > | 12.0, 10.0, 9.1, 7.0 | weak |
| DUPLICATE | phys.mad_mean | > | 19.7433, 15.8866, 14.1781, 12.1200 | weak |
| CODING_NOISE | phys.frm_bytes_med | > | 10000.07, 8914.25, 8247.05, 6957.25 | weak |
| CODING_NOISE | phys.p_size_med | > | 9904.74, 8849.20, 8120.60, 6758.88 | weak |
| CODING_NOISE | phys.snr_db | > | 18.9924, 16.8766, 16.0496, 14.8676 | weak |

`UNDECODABLE` is a tripwire: any non-finite core feature (luma_mean,
motion_energy, luma_p05) scores 10⁶ and forces REJECT. It fires 0 times in
domain (the single labelled UNDECODABLE window has finite cached features — a
documented limitation of the cached view) and exists to guard the decode path.

**Direction justification (data, not weights) — with a caveat.** Residual optical
flow after ego-motion removal (`flow_act`/`flow_med`, physics spec 6) is *higher*
on bad windows (all-window median 0.59 bad vs 0.27 good): camera/head motion
without stable work. This is the opposite sign to the ALGO_SPEC's `flow_act`
claim. **Caveat:** this evidence is an *all-window* statistic and the direction
was fixed in `CANDIDATE_RULES` with the labels visible — it is **not** a
training-fold sign, so the LOCO comparison is optimistic. The fully nested
per-fold sign+feature selection is the `loco_selected` ablation (0.762), which is
worse.

## Decision
```
score = number of crossed thresholds (integer; +1 per crossing)
if any quality rule crosses its extreme (p01/p99) threshold:  REJECT (quality_rule)
elif score >= T_reject:                                      REJECT (score>=T_reject)
elif score >= 1:                                             REVIEW (weak_rule)
else:                                                        ACCEPT
```
`T_reject` is the smallest integer with ≤ 5 % false-reject on **training-good**
windows (per fold). Every decision emits `{rule, feature, value, op, threshold}`.

## LOCO results

| metric | value |
|---|---|
| **AUROC deterministic score (optimistic)** | **0.8224** (95% CI 0.7149–0.8605) |
| logistic 26-feat, same folds (not like-for-like) | 0.8016 |
| logistic 42-feat (26 full + 16 physics, like-for-like) | 0.7870 |
| EG2 two-view ensemble (cached) | 0.8458 |
| per-pixel conv (cached) | 0.7916 |
| two-view rank-avg ensemble (cached) | 0.8336 |

**Paired clip-grouped bootstrap on the AUROC difference** (2,000 shared
resamples, seed 0; delta = rules − baseline, both oriented higher = more bad).
Every CI contains 0, so **no difference is established** at n=150:

| comparison | delta | 95% CI |
|---|---|---|
| rules − logistic 26-feat | +0.0208 | [−0.1014, +0.0921] |
| rules − logistic 42-feat | +0.0354 | [−0.0846, +0.1108] |
| rules − EG2 ensemble | −0.0234 | [−0.1393, +0.0650] |
| rules − rank-avg ensemble | −0.0112 | [−0.1355, +0.0605] |

### Honest headline: full-population balanced accuracy

| operating point | BA (review-as-accept) | BA (review-as-reject) | BA (decided only) | REVIEW share | coverage | false-reject (good) | bad recall |
|---|---|---|---|---|---|---|---|
| fr5 conservative (5 % FR budget) | **0.710** | **0.590** | 0.808 | 0.60 | 0.40 | 0.10 | 0.52 |
| ba balanced (max train BA) | 0.710 | 0.590 | 0.678 | 0.38 | 0.62 | 0.32 | 0.74 |

The headline is **0.710 / 0.590** (REVIEW counted, 40 % coverage); **0.808 is
decided-only** (the 60 % REVIEW windows are excluded) and overstates the model.
The conservative point still beats the training-BA-optimal threshold on
*decided-only* accuracy: the BA-optimal threshold (6–8) over-rejects on held-out
clips (FR 0.32). **Caveat:** both operating points were selected on the same 150
windows, so this is a descriptive, not a held-out, comparison.

## Per-rule signal (LOCO)
| rule | fires | precision | recall | indicator AUROC | AUROC if dropped |
|---|---|---|---|---|---|
| BLUR | 60 | 0.55 | 0.66 | 0.695 | 0.802 |
| CODING_NOISE | 63 | 0.54 | 0.68 | 0.695 | 0.808 |
| NO_ACTIVE_WORK | 54 | 0.57 | 0.62 | 0.695 | 0.805 |
| DUPLICATE | 55 | 0.55 | 0.60 | 0.675 | 0.815 |
| HANDS_NOT_VISIBLE | 41 | 0.51 | 0.42 | 0.610 | 0.822 |
| TOO_DARK | 44 | 0.41 | 0.36 | 0.550 | 0.798 |
| OVEREXPOSED | 38 | 0.45 | 0.34 | 0.565 | 0.830 |
| FROZEN_STATIC | 66 | 0.33 | 0.44 | 0.500 | 0.826 |

**Which rules carry the signal:** dropping BLUR, TOO_DARK, NO_ACTIVE_WORK or
HANDS_NOT_VISIBLE *lowers* AUROC (they carry signal). Dropping OVEREXPOSED or
FROZEN_STATIC *raises* it slightly (0.830 / 0.826) — they add a little noise in
this corpus; FROZEN_STATIC is kept because it is a required spec tripwire that
only matters for frozen video (absent from gold). The strongest individual
rules are the motion/activity family (NO_ACTIVE_WORK) and BLUR.

## Ablation: per-fold feature selection fails
Selecting the top-12 features *per fold* by training univariate AUROC gives
**0.762** — far below the fixed candidate list. Clip-specific features win
inside a training fold and do not transfer. The fixed, spec-justified candidate
list is what generalises.

## Leakage (what was fit where)

This is the most important caveat in the report. Read it before quoting any
number:

* **Not nested (optimistic):** the candidate rule list (`CANDIDATE_RULES`), the
  choice of features, and every `op`/`side` direction were chosen by the author
  **with all 150 gold labels visible**. The evidence for the `flow_act`/
  `flow_med` direction is an all-window median (0.59 bad vs 0.27 good), not a
  training-fold sign. Only the **thresholds** and the **operating point** are
  re-derived inside each LOCO fold. So **0.8224 is an optimistic in-domain
  estimate**, not a nested-LOCO number.
* **Nested alternative:** a fully nested per-fold feature + sign selection
  (`loco_selected`, top-12 by training univariate AUROC) gives **0.7618** — the
  honest nested number, and it is *worse*. The fixed, spec-justified candidate
  list is what generalises; the fixed list is not what is being validated.
* **Like-for-like:** the rule list uses 26 full + 16 physics features. The
  `logistic_26feat` baseline uses only the 26 full features, so it is **not**
  like-for-like; the `logistic_42feat` baseline (same 42 features, median
  imputation for the 3 `snr_db` NaNs fit on the train fold) is. Against the
  like-for-like baseline the paired delta is +0.035, CI [−0.085, +0.111].
* **What is clean:** labels are never read by the scorer; the reject threshold
  uses a pre-registered 5 % training-good false-reject budget; all thresholds
  are training-fold quantiles.

## Honest limits and cost of determinism
* **In-domain only.** 150 windows, 7 clips, one labeller. No OOD transfer was
  tested; prior evidence says IndEgo-trained scorers do not transfer to
  Egocentric-10K (0.4–0.53 AUROC).
* **Full-population accuracy is 0.710 / 0.590**, not 0.808; 60 % of windows are
  REVIEW at the conservative point. The decided-only 0.808 covers only 40 % of
  windows.
* **No AUROC difference is established** vs any baseline at n=150 (all paired
  CIs contain 0). Do not claim the rule list beats the logistic baseline; do not
  claim the ensemble beats it.
* Integer scores create ties, which cost a little AUROC versus a continuous
  learned score.
* REVIEW share is high (0.60) at the conservative point because most windows
  cross at least one mild threshold; this is the intended abstain behaviour, not
  a decision.

## Per-window scores (traceability)

`Q1.01.json:per_window_scores` has one row per gold window —
`{id, clip, y, score, decision_fr5, decision_ba, p_logistic26, p_logistic42,
p_eg2, p_pixel, p_ens}`. Every number above (LOCO AUROC, clip-bootstrap CI,
paired delta, balanced accuracy) is re-derivable from those rows plus the `clip`
field. The paired deltas are in
`Q1.01.json:paired_clip_bootstrap_auroc_difference`.

## Reproduce
```bash
JADAL_DATA=/path/to/pod_bundle OMP_NUM_THREADS=2 python3 eval/exp_phys/run_exp_phys.py
OMP_NUM_THREADS=2 python3 -m pytest tests/test_exp_phys_rules.py -q
```
