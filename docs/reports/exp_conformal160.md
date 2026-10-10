# Z2 — Conformal prediction sets + risk control — re-run on the 160 OOD labels

> **Supersedes the n=60 version (PR #126, branch `m/conformal`).**
> This is a **NEW branch** (`m/conformal-160`); the original PR branch is untouched.
> What changed: the **out-of-domain evaluation** now uses the merged **160**
> human-labelled Egocentric-10K windows (**138 good / 22 bad, 47 factories**;
> q1=60 + q2=100) instead of the old **60** (51 good / 9 bad), and the primary CI is
> the **factory-grouped** bootstrap. The IndEgo fitting side (n=150, 7 clips) and its
> LOCO frontier are **unchanged** — nothing is fit on OOD.

**Branch:** `m/conformal-160` (base `pod/base`) · **Owner:** M5
**Results:** `eval/results/exp_conformal160/Z2.01_160.json`, `Z2.02_160.json`, `Z2.03_160.json`
**Code:** `eval/exp_conformal160/run_z2_160.py` (+ `conformal_risk.py` copied verbatim) · **Tests:** `tests/test_exp_conformal160.py`
**Data:** IndEgo human gold OOF probs `caches/ensemble/oof_probs.npz` (n=150, 7 clips);
OOD-160 frozen scores `caches/ensemble/ood_scores{,2}.json` + `labels/ood160_labels.json`.
**Device:** CPU only, seed 7. **Reproduce:** `OMP_NUM_THREADS=2 python3 eval/exp_conformal160/run_z2_160.py`

---

## TL;DR (160 vs the old 60)

| claim | 60-label | **160-label** | verdict |
|---|---|---|---|
| frozen score OOD AUROC | 0.415 (n_bad=9) | **0.566 [0.4125, 0.7042]** (grouped) | still not separable from chance |
| IndEgo split-conformal @ cov 0.85 | review 12.0 %, cov 85.3 % | **identical** (fit side unchanged) | holds in-domain |
| IndEgo rule **applied OOD** @ cov 0.85 | no OOD coverage reported | **coverage 55.6 %** (target 85 %), review 11.2 % | **guarantee does NOT transfer** |
| frozen review-vs-error frontier on OOD | not computed | **all grouped CIs include 0.5** | no OOD operating point |

## 1. IndEgo frontier (fit side, unchanged — PR #126)

Split-conformal, clip-grouped leave-one-clip-out on `p_ens`; review share =
non-singleton set rate. Re-verified on this branch, identical to PR #126:

| coverage target | review % | empirical coverage | error \| decided | marginal false-accept | false-accept \| accept |
|---|---|---|---|---|---|
| 0.70 | 15.3 | 70.7 | 16.5 | 5.3 | 9.9 |
| **0.80** | **3.3** | 80.0 | 18.6 | 6.0 | 10.0 |
| **0.85** | **12.0** | **85.3** | 16.7 | **4.0** | 7.3 |
| 0.90 | 47.3 | 89.3 | 20.3 | 4.0 | 14.6 |

Operating points (IndEgo): minimum review with marginal FA ≤ 5 % → **coverage 0.85,
review 12.0 %**; selective FA ≤ 5 % → never (min ≈ 7.3 %); error|decided ≤ 5 % → never.
The CRC/LTT result is also unchanged: with K=7 clips a 5 % false-accept bound is
**infeasible** (floor UCB 0.66); the guarantee is data-limited by the number of
exchangeable clips.

## 2. The IndEgo-calibrated conformal rule on OOD-160 (evaluation only)

`q` is calibrated on **IndEgo only** (all 150 windows) and applied to the 160 OOD
frozen scores. **Nothing is fit on OOD.** Coverage = P(true label in the conformal set);
CIs = factory-grouped bootstrap.

| coverage target | q (IndEgo) | review % | **empirical coverage** | coverage 95% CI (grouped) | error \| decided | marginal FA |
|---|---|---|---|---|---|---|
| 0.30 | 0.744 | 63.7 | 30.6 | — | 15.5 | 3.1 |
| 0.50 | 0.694 | 56.9 | 32.5 | — | 24.6 | 3.8 |
| 0.70 | 0.580 | 18.8 | **41.9** | [0.314, 0.527] | 48.5 | 4.4 |
| 0.80 | 0.504 | 1.2 | **47.5** | [0.360, 0.592] | 51.9 | 5.0 |
| **0.85** | 0.445 | 11.2 | **55.6** | **[0.428, 0.676]** | 50.0 | 4.4 |
| 0.90 | 0.347 | 44.4 | 80.6 | [0.719, 0.882] | 34.8 | 4.4 |
| 0.95 | 0.216 | 72.5 | 96.9 | — | 11.4 | 2.5 |
| 0.99 | 0.056 | 99.4 | 100.0 | — | 0.0 | 0.0 |

**The recommended IndEgo operating point (coverage target 0.85, review 12.0 %)
gives only 55.6 % empirical coverage on OOD-160** (grouped CI [0.428, 0.676], far
below 0.85), with 50 % error among decided windows. The distribution-free guarantee
**does not transfer** because there is no target exchangeability: the IndEgo score
does not rank the OOD windows (AUROC 0.566). The nominal target is only met in the
degenerate regime (coverage 0.95–0.99 → almost everything is reviewed).

## 3. Frozen review-vs-error frontier on OOD-160 (grouped; no fitting)

A-priori symmetric band `|p−0.5| ≥ w` and top-c coverage gating on the frozen
`usable_prob`; OOD balanced accuracy among decided windows, factory-grouped CI.

| band w | review % (160) | OOD BA (decided) | grouped 95% CI |
|---|---|---|---|
| 0.00 | 0.0 | 0.546 | [0.439, 0.655] |
| 0.04 | 6.9 | 0.565 | — |
| 0.10 | 26.3 | 0.524 | — |
| 0.16 | 45.0 | 0.577 | [0.403, 0.742] |
| **0.18** | **52.5** | **0.612** | **[0.424, 0.814]** |
| 0.20 | 57.5 | 0.614 | [0.414, 0.816] |
| 0.30 | 75.0 | 0.486 | — |

Coverage gating: c=1.0 → BA 0.546; c=0.6 → 0.550; c=0.5 → 0.583; c=0.4 → 0.594.

**Every grouped CI contains 0.5.** Reducing review does not buy OOD accuracy on the
frozen score — the frontier is flat around chance. (O3, PR #125, reaches the same
conclusion with a different review denominator.)

## What changed vs the n=60 version (PR #126)

1. The OOD evaluation now covers **160 windows / 22 bad** with **factory-grouped**
   CIs (the old report only had an OOD-60 note).
2. The old OOD-60 AUROC **0.415** becomes **0.566** on 160 (the old value was noise at
   n_bad=9); both CIs contain 0.5.
3. New, explicit: the IndEgo-calibrated rule's **OOD coverage collapses to 55.6 % at
   target 0.85** — the guarantee is in-domain only.
4. The IndEgo frontier / CRC numbers are **unchanged** (fit side).

## Honesty ledger

| claim | source | n | label source | protocol |
|---|---|---|---|---|
| conformal frontier | Z2.01_160 | 150 | human gold | clip-grouped LOCO, clip-bootstrap CI |
| OOD transfer of the rule | Z2.01_160 | 160 (22 bad) | human gold (OOD-160) | **evaluation only**, grouped CI |
| nominal vs empirical coverage OOD | Z2.02_160 | 160 | human gold | q from IndEgo, applied OOD |
| frozen review-vs-error frontier | Z2.03_160 | 160 | human gold | a-priori band/gate sweep, grouped CI |
| CRC/LTT bound | PR #126 | 150 | human gold | LTT on 7 clips (unchanged) |

**Reminder:** never fit `q`/`t` on OOD; n_bad=22 over 47 factories; the frozen score is
near chance out-of-domain, so OOD verdicts are descriptive, not guaranteed.
