# N2 — Operating point on the adapted OOD score: can REVIEW *and* error fall together?

**Track:** N2 (operating point) · **Owner:** N2 · **Branch:** `m/acc-operating`
**Base:** `pod/base` · **Device:** CPU only, `OMP_NUM_THREADS=2` (all scores cached; no GPU)
**Results:** `eval/results/exp_acc/N2.01.json`, `N2.02.json`, `N2.03.json`
**Figure:** `docs/experiments/exp_acc/images/n2_operating_frontier.png`
**Code:** `eval/exp_acc/run_n2.py` (seed 7, 2,000 factory-grouped bootstrap draws), `eval/exp_acc/make_fig.py`
**Registry:** N2.01, N2.02, N2.03

**Reproduce**
```bash
cd /workspace/wt/m-acc-operating
N2_Z3_INPUTS=/workspace/wt/m-adapt160/eval/results/exp_adapt160/inputs \
  OMP_NUM_THREADS=2 python3 eval/exp_acc/run_n2.py
OMP_NUM_THREADS=2 python3 eval/exp_acc/make_fig.py
```

---

## 0. Question, data, and the honest denominator

The frozen PoC score is near chance out-of-domain (OOD AUROC **0.566** [0.424, 0.700]); O3
(#125) showed that cutting REVIEW below 60 % on it is a **workload** win, not an accuracy win
(best frozen point: rank-avg band `w=0.16` → **44.2 % review, OOD BA 0.708** [0.521, 0.832]).
The O4 adapted score (`stable-8 + Z3 transferable-6`, per-domain z-score, IndEgo-signed) ranks
much better: **OOD AUROC 0.781 [0.634, 0.895]**. N2 asks whether that ranking can be turned into
an operating point that is better on **both** axes.

* **OOD error** = `1 − balanced accuracy` on **DECIDED** windows of the **160** human-labelled
  Egocentric-10K windows (138 good / 22 bad, **47 factories**), pre-registered test, nothing fit
  on the evaluated factory. CIs are **factory-grouped** bootstraps (2,000 draws, seed 7).
* **REVIEW %** is the share of the **1,546** unlabelled E8 windows left in REVIEW. For the
  quantile rules this is **exact by construction** (`1−2q` globally; the per-factory count for the
  within-factory rule), so it needs no extra inference. For fixed-threshold rules (band/conformal)
  the 1,546 share needs the adapted score on the 1,546, which is **not cached** — those rows are
  flagged.

**AUROC (grouped 95 % CI), 160 windows**

| score | AUROC | grouped 95 % CI |
|---|---|---|
| **adapted (b) stable-8 + Z3-6** | **0.781** | **[0.634, 0.895]** |
| adapted (a) Z3-6 | 0.759 | [0.597, 0.886] |
| frozen `usable_prob` | 0.566 | [0.424, 0.700] |

---

## 1. Class-prior / base-rate correction (N2.01) — does not rescue the 0.5-band

The adapted score is an uncalibrated signed z-score (boundary 0), not a probability. We
calibrated it to `P(good)` two ways and then swept a symmetric `0.5`-band `w = 0…0.30`:

1. **Leave-factory-out Platt** on the 160 (prior = target bad rate 13.75 %) — the P1 setting.
2. **Source→target prior shift**: Platt fit on the 150 IndEgo gold windows (bad rate 33.3 %),
   then the logit offset `logit(π_target) − logit(π_source) = −1.143` applied to the OOD score.

| calibration | max P(bad) | bad rejected at 0.5 | full-decision BA (w=0) | verdict |
|---|---|---|---|---|
| LFO Platt | **0.60** | **0/22** | 0.496 | 0.5-band collapses |
| prior-shifted | **0.579** | **0/22** | 0.496 | collapses *further* |

Because the OOD bad rate (13.75 %) is **lower** than IndEgo's (33.3 %), base-rate correction pushes
the posterior **toward GOOD**, so `P(bad)` compresses more and no bad window ever crosses 0.5. This
reproduces P1 (#137) and adds the mechanism: **a 0.5-probability band is the wrong decision rule at
a 13.75 % prior** — the frontier has to be re-derived on the score's *rank*, which is exactly what
the quantile rule does. (The frontier itself is unchanged by any monotone calibration: it is the
score's ROC.)

---

## 2. Rank / quantile decision rule (N2.02) — a non-collapsing frontier exists

Deterministic, prior-free: **reject the lowest `q`, accept the highest `q`, review the middle**
(the *within-regime* / whole-cohort quantile). On the 1,546 this reviews exactly `1−2q`.

| q | REVIEW % (1,546) | OOD BA [grouped 95 % CI] | err | n decided | FA (marg.) | FR (marg.) |
|---|---|---|---|---|---|---|
| 0.10 | 80.0 | 0.820 [0.618, 0.946] | 0.180 | 32 | 0.0 | 5.6 |
| 0.15 | 70.0 | 0.800 [0.660, 0.911] | 0.200 | 48 | 0.0 | 10.0 |
| **0.20** | **60.0** | **0.808 [0.683, 0.911]** | 0.192 | 64 | 0.0 | 12.5 |
| 0.25 | 50.0 | 0.798 [0.688, 0.893] | 0.202 | 80 | 0.0 | 16.9 |
| **0.28** | **44.0** | **0.793 [0.686, 0.891]** | 0.207 | 88 | 0.0 | 19.4 |
| 0.30 | 40.0 | 0.751 [0.584, 0.879] | 0.249 | 96 | 0.6 | 21.9 |
| 0.35 | 30.0 | 0.732 [0.588, 0.854] | 0.268 | 112 | 1.2 | 25.0 |
| 0.50 | 0.0 | 0.711 | 0.289 | 160 | 13.8 | 0.0 |

**A non-collapsing frontier exists and it dominates the frozen one.** The adapted score reaches
BA **0.793 at 44 % review** — *lower* review and *higher* BA than the frozen O3 reference
(0.708 at 44.2 %), and BA **0.751 at 40 % review**. This is the ROC of a 0.781-AUROC score, made
reachable by a prior-free rank threshold.

**Asymmetric 2-D sweep** (accept top-`a`, reject bottom-`r`) finds the error-minimising point:

| rule | REVIEW % | OOD BA [CI] | err | n decided | FA (sel.) | FR (sel.) |
|---|---|---|---|---|---|---|
| **accept 25 %, reject 15 %** | **60.0** | **0.857 [0.746, 0.939]** | 0.143 | 64 | 0.0 | 0.667 |
| symmetric q=0.20 | 60.0 | 0.808 [0.683, 0.911] | 0.192 | 64 | 0.0 | 0.625 |

The asymmetric optimum buys BA by **accepting more than it rejects** (bads are rare), at the cost of
a higher selective false-reject (66.7 % of rejects are good) — a workload/labour trade-off, not a
free lunch. It is also the **argmax over 169 grid points** on the test set, so it is optimistically
biased; the symmetric `q=0.28` point is the more robust recommendation.

**Within-factory quantile (the deployment-local variant):** the E8 factories hold 17–119 windows,
so a within-factory quantile is well defined in production — but the **160 evaluation set holds
only 1–4 windows per factory**, so the within-factory rule can decide almost nothing and its BA on
the 160 is ≈ chance (0.50–0.56, CIs spanning 0.24–0.77). This is an **evaluation limitation**, not a
refutation: the rule is deployable but the 160 labels cannot validate it. Reported for
completeness in `N2.02.json`.

---

## 3. Split / Mondrian conformal on the adapted score (N2.03)

LAC conformal, **leave-one-factory-out** (`score_good = s`, `score_bad = −s`; `q` = the
`⌈(n+1)α⌉`-th order statistic of the calibration conformity scores). Coverage is measured on the
held-out factory. The decision map reduces to a symmetric band `|s| < |q|` around the natural
boundary 0.

| nominal cov. | empirical cov. | median q | REVIEW % (160) | OOD BA [CI] | n decided |
|---|---|---|---|---|---|
| 0.95 | 0.944 | −0.768 | 88.1 | 0.679 [0.500, 0.885] | 19 |
| 0.90 | 0.900 | −0.614 | 75.6 | 0.742 [0.565, 0.883] | 39 |
| 0.85 | 0.850 | −0.389 | 53.1 | 0.806 [0.689, 0.907] | 75 |
| 0.80 | 0.788 | −0.283 | 42.5 | 0.753 [0.580, 0.881] | 92 |
| 0.70 | 0.700 | −0.122 | 15.6 | 0.749 [0.625, 0.861] | 135 |
| **0.65** | **0.650** | **−0.032** | **4.4** | **0.728** [0.616, 0.835] | 153 |
| 0.60 | 0.600 | +0.030 | 5.6 | 0.729 [0.616, 0.835] | 151 |

**The marginal coverage guarantee holds on held-out factories** (empirical ≈ nominal to ±0.01).
REVIEW is U-shaped with a minimum where `q` crosses 0 (nominal ≈ 0.65): below it the empty-set band
widens; above it the singleton band widens. At nominal 0.65 the adapted score decides **95.6 %** of
the 160 with BA 0.728. Note that REVIEW % here is on the **160** — the fixed conformal threshold's
1,546 share is not computable from cached data (flagged in the table).

---

## 4. The joint frontier — one table

REVIEW % is on the 1,546 unless marked `†` (160 only). FA/FR are marginal shares of all windows.

| method | score | REVIEW % | OOD BA [grouped 95 % CI] | false-accept | false-reject |
|---|---|---|---|---|---|
| frozen rank-avg band w=0.16 (**O3 ref**) | frozen | 44.2 | 0.708 [0.521, 0.832] | 0.6 | 22.5 |
| frozen prob-avg band w=0.17 | frozen | 57.4 | 0.600 [0.440, 0.767] | 4.4 | 11.2 |
| adapted(b) global-quantile q=0.10 | adapted | 80.0 | 0.820 [0.618, 0.946] | 0.0 | 5.6 |
| adapted(b) global-quantile q=0.15 | adapted | 70.0 | 0.800 [0.660, 0.911] | 0.0 | 10.0 |
| adapted(b) global-quantile q=0.20 | adapted | 60.0 | 0.808 [0.683, 0.911] | 0.0 | 12.5 |
| adapted(b) global-quantile q=0.25 | adapted | 50.0 | 0.798 [0.688, 0.893] | 0.0 | 16.9 |
| **adapted(b) global-quantile q=0.28** | adapted | **44.0** | **0.793 [0.686, 0.891]** | 0.0 | 19.4 |
| **adapted(b) global-quantile q=0.30** | adapted | **40.0** | **0.751 [0.584, 0.879]** | 0.6 | 21.9 |
| adapted(b) global-quantile q=0.35 | adapted | 30.0 | 0.732 [0.588, 0.854] | 1.2 | 25.0 |
| **adapted(b) asymmetric a=0.25 r=0.15** | adapted | **60.0** | **0.857 [0.746, 0.939]** | 0.0 | 10.0 |
| adapted(b) within-factory q=0.30 | adapted | 40.8 | 0.556 [0.243, 0.769] | 1.2 | 12.5 |
| adapted(b) conformal nominal 0.85 † | adapted | 53.1 † | 0.806 [0.689, 0.907] | 0.0 | 15.0 |
| adapted(b) conformal nominal 0.65 † | adapted | 4.4 † | 0.728 [0.616, 0.835] | 0.0 | 5.6 |
| adapted(b) LFO-Platt band † | adapted | (160) | 0.500 [0.500, 0.500] | 13.8 | 0.0 |

**Best point minimising OOD error at REVIEW ≤ 60 %:** `asymmetric accept 25 % / reject 15 %` →
**REVIEW 60.0 %, OOD BA 0.857 [0.746, 0.939]** (err 0.143). Robust symmetric alternative: `q=0.20`
→ **60.0 % review, BA 0.808 [0.683, 0.911]**.

**Does any point beat the frozen frontier (44.2 % review, BA 0.708)?** **Yes — strictly, on both
axes.** **3** grid points have *both* lower review and higher BA than the O3 reference:

| point | REVIEW % | OOD BA [CI] |
|---|---|---|
| adapted(b) global-quantile q=0.28 | 44.0 | 0.793 [0.686, 0.891] |
| adapted(b) global-quantile q=0.30 | 40.0 | 0.751 [0.584, 0.879] |
| adapted(b) global-quantile q=0.35 | 30.0 | 0.732 [0.588, 0.854] |

**Paired** factory-grouped bootstrap of the BA difference (adapted `q=0.28` vs frozen rank-avg
`w=0.16`, same resamples): mean **+0.092**, 95 % CI **[−0.053, 0.281]**, 87.3 % of resamples > 0 —
directionally robust but **not** significant at n_bad = 22.

---

## 5. Honest answer

**A joint improvement IS achievable with the current score/labels.** The driver is purely the
OOD **ranking** (AUROC 0.781 vs 0.566): the adapted score's quantile frontier sits uniformly above
the frozen frontier, so at *any* review budget ≤ 60 % it gives lower OOD error, and at *equal*
review (≈44 %) it gives BA 0.793 vs 0.708. This is not a calibration or prior effect — the
base-rate/Platt route collapses (§1); it is the score's separability, exposed by a prior-free rank
threshold (§2). The conformal construction gives the same frontier with a **stated coverage
guarantee** that holds on held-out factories (§3).

### Caveats (must travel with the numbers)

1. **The quantile is transductive** — computed on the deployment cohort (the 1,546, or a factory
   batch), exactly like O3's rank-avg. Its 1,546 REVIEW % is exact; its OOD BA is estimated on the
   160. Per-factory quantiles (well defined at 17–119 windows/factory) cannot be validated on the
   160 (1–4 windows/factory).
2. **The optimum is selected on the test set.** The 160 is the pre-registered test and nothing is
   *fit* on it, but the reported points are argmaxes over a grid; the asymmetric 0.857 is the most
   optimistic. All top candidates' CIs overlap.
3. **n_bad = 22, 47 factories, one labeller.** Grouped CIs are wide and the paired BA-difference CI
   contains 0. The frozen score's own AUROC CI contains 0.5; the adapted score's lower bound
   (0.634) clears it but not comfortably.
4. **REVIEW % is a workload number** on unlabelled windows; accuracy is only measurable on the 160.
5. **Fixed-threshold rules (band/conformal) lack a 1,546 REVIEW %** because the adapted score on the
   1,546 is not cached (decoding it exceeded the timebox); those rows are flagged `†`.

---

**PR:** `m/acc-operating` → `pod/base` · **Branch sha:** see PR
**Artifacts:** `eval/results/exp_acc/N2.01.json`, `N2.02.json`, `N2.03.json`;
`docs/experiments/exp_acc/images/n2_operating_frontier.png`
