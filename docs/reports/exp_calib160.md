# P1 — Calibrate the adapted OOD score on 160 labels + joint review/error frontier

Branch `m/calib160` · experiment `P1.01` · figure:
`docs/experiments/exp_calib160/images/p1_calib160.png` ·
results: `eval/results/exp_calib160/P1.01.json` ·
code: `eval/exp_calib160/run_p1.py`, `make_fig.py`.
Every number carries `n`, `label_source`, `protocol`, `ci95`, `device`, `command`.

**Goal.** O4's adapted score is an uncalibrated signed z-score, so the frozen
symmetric band cannot be applied. Calibrate score → P(bad) under
leave-factory-out (LFO) grouped CV and sweep the symmetric band on the
calibrated probability.

**Data.** 160 human-labelled Egocentric-10K windows, 138 good / 22 bad,
47 factories (single labeller). Adapted scores recomputed exactly per O4
(O4 AUROCs reproduced to < 0.005: (a) 0.759, (b) 0.781).

## 1. Calibration under LFO (pooled out-of-fold P(bad); base rate 0.1375)

| recipe | method | Brier [grouped 95% CI] | log-loss [CI] | OOF AUROC |
|---|---|---|---|---|
| (b) stable-8+6 | **beta** | **0.1124 [0.069, 0.157]** | **0.360 [0.238, 0.479]** | 0.753 |
| (b) stable-8+6 | Platt | 0.1128 [0.069, 0.158] | 0.365 [0.247, 0.484] | 0.750 |
| (b) stable-8+6 | isotonic | 0.1169 [0.072, 0.162] | 0.523 [0.257, 0.894] | 0.698 |
| (a) Z3-6 | beta | 0.1127 | 0.370 | 0.717 |
| (a) Z3-6 | Platt | 0.1138 | 0.373 | 0.718 |
| (a) Z3-6 | isotonic | 0.1172 | 0.545 | 0.649 |
| — | predict base rate | 0.1186 | 0.400 | — |
| — | raw sigmoid(score) | 0.29–0.31 | 0.78–0.81 | — |

Beta ≈ Platt > isotonic (isotonic log-loss much worse: lumpy LFO bins go
overconfident). All three beat the base-rate predictor on Brier and beat raw
sigmoid by ~2×. But calibration compresses hard toward the base rate:
max P(bad) 0.56–0.68, median P(bad) of bad windows only 0.19–0.25, and
**0/22 bad windows cross 0.5** (Platt, beta; 1/22 isotonic-(a)).

## 2. Frontier: symmetric band on calibrated P(good) — degenerate

Sweep w = 0…0.30 (accept ≥ 0.5+w, reject ≤ 0.5−w), review% on the 160 (LFO-OOF):

| w (Platt-(b), representative) | review% | n decided | OOD BA | err | false-accept |
|---|---|---|---|---|---|
| 0.00 | 0.0 | 160 | 0.496 | 0.504 | 1.00 |
| 0.10 | ~3 | ~155 | 0.50 | 0.50 | 1.00 |
| 0.20 | ~8 | ~147 | 0.50 | 0.50 | 1.00 |
| 0.30 | 21.2 | 126 | 0.50 | 0.50 | 1.00 |

All three calibrators × both recipes: **false-accept = 1.0 at every w**
(no bad window is ever rejected; decided-bad are all accepted), BA flat at
≈ 0.5. **There is no interior operating point: the argmin over
{err : review ≤ 60%} is meaningless (every row ties at err ≈ 0.5).**
Full grid (31 widths × 3 methods × 2 recipes, with grouped BA CIs) is in
`P1.01.json` → `recipes.*.frontiers`.

Why: with a 13.75% base rate, honest P(bad) values live far below 0.5, so a
symmetric 0.5-band can only accept-or-review, never reject. The adapted score
**ranks** (OOF AUROC 0.70–0.75) but its calibrated probabilities cannot drive
the frozen-style band policy.

## 3. Frozen comparison (same 160, same band machinery)

Frozen `usable_prob` band on the same 160 is non-degenerate:
op at **w = 0.19 → review 56.2%, BA 0.634 [0.446, 0.831]**, false-accept 0.55
(full grid in `P1.01.json` → `frozen_same160`). Frozen spreads probability
mass across 0.5 (however miscalibrated); calibrated-adapted does not.
O3's headline (rank-avg + w = 0.16 → 44.2% review, BA 0.708) measures review
on the **1,546 unlabelled windows** — review denominators differ, so review%
is not comparable; decided-window BA/false-accept on the 160 are.

## 4. What would be needed (illustrative, NOT recommended)

Single-threshold accept/reject on OOF P(bad) at theory-derived cost cutoffs
1/(1+C) — threshold values from cost theory, choice among them still
test-set-selected (`illustrative_cost_thresholds` in the JSON):
cut 0.25 (1:3) → BA 0.58–0.67, fa 0.55–0.73;
cut 0.20 (1:4) → BA 0.61–0.70, fa 0.41–0.55.
Signal exists at prevalence-aware cutoffs; the symmetric 0.5-band cannot reach it.

## 5. Verdict + honest caveats

- **Result: negative for the specified policy.** Calibration (beta/Platt) works
  as probability estimation (Brier 0.112 vs 0.119 base rate) but the joint
  review/error frontier under a symmetric 0.5-band collapses (BA 0.5,
  false-accept 1.0 everywhere). **Do not ship a 0.5-band on the calibrated
  adapted score.** Next step would be a cost-aware threshold (miss cost ≈ 3–4×
  false-accept cost), fitted on separate data — not on these 160.
- **n_bad = 22, 47 factories, one labeller;** grouped CIs wide (Brier CI spans
  ~0.07–0.16). Calibration is fitted on the target labels (post-hoc) —
  LFO keeps it factory-honest, but it is still the test domain.
- Adapted OOF AUROC here (0.70–0.75) sits below O4's in-sample 0.76–0.78:
  per-fold calibrators perturb pooled ranking slightly; ranking ≠ calibration.
- q2-only weakness from O4 stands (not re-examined here).
- CPU only, `OMP_NUM_THREADS=2`; deterministic (seed 11, fixed beta inits).

## Files

| file | role |
|---|---|
| `eval/exp_calib160/run_p1.py` | O4-exact rescore, LFO Platt/isotonic/beta, frontier, frozen comparison |
| `eval/exp_calib160/make_fig.py` | reliability + frontier figure |
| `eval/results/exp_calib160/P1.01.json` | all numbers (frontiers, CIs, reliability deciles, illustrative cuts) |
| `docs/experiments/exp_calib160/images/p1_calib160.png` | figure |
