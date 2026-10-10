# Y2 — Review reduction: decision-theoretic gating + stage-0 triage

**Branch** `m/review-gate` · **owner** monitor (Y2) · **device** CPU (cached signals only)
**Results** `eval/results/exp_review/Y2.01.json`, `Y2.02.json`, `Y2.recommend.json`
**Figure** `docs/experiments/exp_review/images/review_gate.png|svg`
**Reproduce** `OMP_NUM_THREADS=2 python3 eval/exp_review/run_y2.py && python3 eval/exp_review/make_fig.py`

## TL;DR

The frozen PoC policy (symmetric band `score>=0.75` accept / `<=0.25` reject /
else review) leaves **75.5%** of the 1,546 Egocentric-10K windows in REVIEW.
Replacing the band with **decision-theoretic gating** — decide the most
cost-confident windows, review the rest — reaches **55.0% REVIEW** at the
recommended operating point (coverage 45%, false-accept:false-reject = 1:1),
with a **gold-LOCO balanced accuracy of 0.837** on the decided windows
(clip-bootstrap 95% CI **[0.645, 0.937]**, n=150). Deterministic **stage-0**
triage decides 15 more windows (39 quality fires in total) and drops REVIEW to
**54.0%**; the gold-LOCO temperature lever does **not** change the review share
and slightly *hurts* the low-coverage operating point, so we recommend it OFF.

> **Honesty.** The 1,546 Egocentric-10K windows are **unlabelled**
> (`label_source: none`): their review share is descriptive, **not accuracy**.
> The IndEgo-trained score transfers poorly to the OOD-60 (AUROC **0.415**,
> n_bad=9): the measured OOD balanced accuracy at the recommended point is
> **0.496 ≈ chance**. Every OOD number below is **evaluation-only** (nothing is
> fit on the OOD-60).

## Data & protocol

| set | n | labels | protocol |
|---|---|---|---|
| gold LOCO | 150 (100 good / 50 bad, 7 clips) | human (single labeller) | cached clip-grouped OOF `p_ens` from `caches/ensemble/oof_probs.npz`; `AUROC=0.834` measured here |
| OOD-60 | 60 (51 good / 9 bad) | human | frozen `usable_prob` from `caches/ensemble/ood_scores.json`; **evaluation only** |
| rescored | 1,546 | **none** | frozen `score` from `/workspace/artifacts/windows_rescored.jsonl`; descriptive |

**Gating.** For positive-class probability `p` and costs `C_fa, C_fr`, the
cost-optimal decision boundary is `p* = C_fa/(C_fa+C_fr)`: ACCEPT when `p>=p*`,
else REJECT. The review set is the `1-coverage` fraction with the smallest
cost-margin `|p-p*|`; the number of reviews is therefore a direct budget knob
(`review = 1 - coverage`). Cost ratios **1:1, 2:1, 5:1** set `p*` = 0.5, 0.667,
0.833. Ties are broken by stable sort (deterministic).

## 1. Decision-theoretic gating — `Y2.01.json`

### Gold LOCO (n=150), raw `p_ens`

| cost | coverage | review | BA | FP (false accept) | FN (false reject) | weighted err |
|---|---|---|---|---|---|---|
| 1:1 | 0.40 | **0.60** | 0.818 | 4 | 7 | 0.183 |
| 1:1 | **0.45** | **0.55** | **0.837** | 4 | 7 | 0.162 |
| 1:1 | 0.50 | 0.50 | 0.840 | 4 | 8 | 0.160 |
| 1:1 | 0.60 | 0.40 | 0.852 | 5 | 8 | 0.144 |
| 1:1 | 1.00 | 0.00 | 0.795 | 11 | 19 | 0.200 |
| 2:1 | 0.45 | 0.55 | 0.636 | 1 | 19 | 0.309 |
| 5:1 | 0.45 | 0.55 | 0.500 | 0 | 26 | 0.382 |

The full sweep (all coverages × all costs × raw vs LOCO-temperature) is in the
JSON. Two honest observations:

* **Review helps.** Balanced accuracy *rises* as we decide more from 0% to
  ~55% coverage (0.795 → 0.855) and only falls at full coverage: the windows
  left in REVIEW are the genuinely hard ones. Review is not just lost coverage.
* **Asymmetric costs don't pay off here.** With `C_fa:C_fr = 2:1` the best
  measured weighted error is 0.309 vs **0.162** for the 1:1 config — i.e. even
  *under a 2:1 cost model the 1:1 threshold wins*, because the score is not
  sharp enough for a `p*=2/3` boundary and the reject-heavy rule creates too
  many false rejects. At 5:1 the rule degenerates to reject-almost-everything
  (BA 0.50). Recommended cost ratio is therefore **1:1**.

### OOD-60 (n=60, n_bad=9) — evaluation only

| cost | coverage | review | BA | FP | FN |
|---|---|---|---|---|---|
| 1:1 | 0.45 | 0.55 | 0.496 | 3 | 9 |

Every OOD point sits in **0.42–0.50 BA** (AUROC of the score 0.415): the
decisions are **near chance**. This is the same "no transfer" result E1.03,
E5.02 and E8.01 already reported; we do not claim OOD accuracy.

### Descriptive application to the 1,546 windows (cost 1:1)

| coverage | review share | accept | reject | review |
|---|---|---|---|---|
| 0.40 | 0.600 (Wilson [0.576, 0.624]) | 348 | 270 | 928 |
| **0.45** | **0.550 ([0.525, 0.575])** | 353 | 343 | 850 |
| 0.50 | 0.500 | 360 | 413 | 773 |
| 0.60 | 0.400 | 374 | 554 | 618 |

The clip-level review share tracked E8.01's 75.5% at the frozen band; here we
choose the budget directly.

## 2. Stage-0 triage — `Y2.02.json`

Deterministic rules over the **cached descriptor features** (dark /
over-exposed / frozen / static; no decode, no `UNDECODABLE` failures — all
1,546 windows decoded):

| code | feature test | synthetic fire | gold fire |
|---|---|---|---|
| TOO_DARK | `luma_mean<0.06` or `dark_frac>0.85` | black 6/6, darken 6/6 | 1 |
| OVEREXPOSED | `luma_mean>0.80` or `bright_frac>0.25` | overexpose 6/6 | 0 |
| FROZEN | `frozen_frac>0.50` or `zncc_self>0.90` | frozen 6/6 | 2 |
| NO_ACTIVE_WORK | `motion_energy<1e-4` | black 6/6 | 0 |

The rules **catch every synthetic black / darken / overexpose / frozen window**
(6/6 each) and **never fire on clean / blur / noise / shake / lowbitrate /
static** — a clean, physics-anchored definition.

**On the 150 human gold windows the rules fire only 3 times** (2 bad, 1 good →
precision_bad 0.67, recall_bad 0.04, n=3 — indicative only). This matches the
funnel's own certification (`configs/funnel_thresholds.gold150.provisional.json`:
*no quality-reject rule certified on gold*). We therefore treat stage-0 as a
**low-yield, unverifiable** extra and say so.

**On the 1,546 windows** the rules fire on **39 (2.5%)** — 39 `OVEREXPOSED`,
1 `FROZEN`; 15 of those lie inside the gating REVIEW set. We cannot validate
those 39 (no labels; OOD decisions near chance), so the review reduction is a
count, not a measured accuracy.

## 3. Breakdown (a) → (b) → (c) on the 1,546 windows

| lever | rule | review | share |
|---|---|---|---|
| frozen E8.01 | symmetric band `[0.25,0.75]` | 1,167 | **75.5%** |
| (a) gating | decide top-45% by cost-margin | 850 | **55.0%** |
| (b) + stage-0 | + deterministic quality REJECT on reviews | 835 | **54.0%** |
| (c) + temperature | + gold-LOCO temperature | 835 | 54.0% |

* **(a) is the only lever that matters**: −20.5 points of review.
* **(b)** buys −0.9 points; honest but small and unverified.
* **(c) is a no-op on review share** (temperature is a monotone transform) and
  on gold it *lowers* the low-coverage BA (0.837 → 0.781 at coverage 0.45)
  because the per-clip temperature is unstable (T ∈ [0.55, 1.39] across 7
  clips; global T=0.87). **Do not use it** for the review decision.

## Recommendation

**Adopt** `gate(score, coverage=0.45, cost_ratio_fa_fr=1)` with the
deterministic stage-0 REJECT rules layered on the review set:

* REVIEW **75.5% → 54.0%** (gating alone 55.0%).
* In-domain error at the recommended point (gold LOCO, n=150): BA **0.837**
  [0.645, 0.937], false accepts **4/100** good windows (FP), false rejects
  **7/50** (FN) on the decided subset.
* Temperature OFF; asymmetric costs OFF (1:1 wins empirically).
* **Residual unreliability:** OOD decisions are near chance (BA 0.496, AUROC
  0.415, n_bad=9). The review reduction is real; the *correctness* of the
  autonomous ACCEPT/REJECT out-of-domain is **not established**.

## Files

| file | role |
|---|---|
| `eval/exp_review/gate.py` | metrics, calibration, gating, stage-0 rules, bootstrap |
| `eval/exp_review/run_y2.py` | writes `Y2.01/02.json` + `Y2.recommend.json` |
| `eval/exp_review/make_fig.py` | renders `review_gate.png|svg` |
| `tests/test_exp_review.py` | 12 tests, <1 s, no network |
| `eval/results/exp_review/Y2.*.json` | results (n + label source + protocol + command) |
