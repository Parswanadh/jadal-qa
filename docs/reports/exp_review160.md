# O3 — Honest review-vs-error frontier on the 160-window OOD label set

**Registry ids:** O3.01 (frontier) · O3.02 (operating point) · **Track:** O3 · **Owner:** monitor (O3)
**Branch:** `m/review160` · **Base:** `pod/base` @ `f20013b` · **Device:** CPU only, `OMP_NUM_THREADS=2` (all scores cached)
**Results:** `eval/results/exp_review160/O3.01.json`, `O3.02.json`
**Figure:** `docs/experiments/exp_review160/images/o3_review160_frontier.png`
**Code:** `eval/exp_review160/run_o3.py`, `eval/exp_review160/make_fig.py` (deterministic, seed 7, 2,000 bootstrap draws)

**Reproduce**
```bash
cd /workspace/wt/m-review160 && OMP_NUM_THREADS=2 python3 eval/exp_review160/run_o3.py
cd /workspace/wt/m-review160 && OMP_NUM_THREADS=2 python3 eval/exp_review160/make_fig.py
```

## 1. What is measured (and what is not)

* **REVIEW%** is the share of the **1,546** rescored Egocentric-10K windows left in
  REVIEW (`label_source: none`). This is a **workload / descriptive** number, **not accuracy**.
* **OOD error** is `1 − balanced accuracy (BA)` computed **only over DECIDED windows** of the
  **160** human-labelled Egocentric-10K windows (138 good / 22 bad, **47 factories**), the merge of
  `ood_labels.json` (60) + `ood_labels2.json` (100). This is the **pre-registered OOD test:
  evaluation only, nothing is fit on it**.
* 95% CIs are **factory-grouped** bootstraps: the 47 factories are resampled with replacement and
  **all windows of a sampled factory enter together**, respecting within-factory correlation.
  With `n_bad = 22` the intervals are wide.

Two levers are swept honestly, plus a no-new-inference fusion:

1. **Symmetric abstain band** half-width `w ∈ [0, 0.30]` (`accept if score ≥ 0.5+w`,
   `reject if score ≤ 0.5−w`, else REVIEW).
2. **Coverage gating** `c ∈ [0.40, 1.00]`: decide the top-`c` fraction by cost-margin `|p−0.5|`
   (cost 1:1), review the rest. `review% on 1546 = 1 − c` exactly.
3. **Rank-average fusion** of the two cached views, `0.5·(rank(p_eg2)+rank(p_pixel))` — no new inference.

## 2. Reproduction of the frozen policy scores on the 160 OOD windows

| check | value |
|---|---|
| OOD windows matched to frozen scores | **160 / 160** (60 from `ood_scores.json` + 100 from `ood_scores2.json`) |
| `usable_prob` == ½(p_eg2 + p_pixel) | max abs diff **5.0e−5** (4-dp rounding) |
| stored verdicts (rule `accept ≥ 0.6`, `reject ≤ 0.4`, else review) reproduced | **160 / 160** → accept **64**, reject **54**, review **42** |
| E8 profile rule (`factory_1080p30` `[0.25,0.75]`, `default` `[0.8,0.2]`) reproduced on the 1,546 | **1,546 / 1,546** → accept 296, reject 83, review 1,167 |

**OOD separability of the frozen score (160 windows, 138 good / 22 bad):**

| score | OOD AUROC [grouped 95% CI] |
|---|---|
| `usable_prob` (frozen, prob-average) | **0.566** [0.422, 0.702] |
| `p_eg2` | 0.589 |
| `p_pixel` | 0.520 |
| rank-average (fused) | 0.563 [0.396, 0.705] |

Every AUROC CI contains 0.5. At full decision (`w = 0`) the frozen score gives OOD
BA **0.546** [0.439, 0.655]; the frozen `factory_1080p30` profile band `[0.25,0.75]`
(w = 0.25, ~75 % review) decides 57/160 and gives OOD BA **0.603**. The frozen score is
**near chance out-of-domain**.

## 3. Frontier — band sweep (`w = 0…0.30`), review ≤ 60 %

Review is increasing in `w`, so the widest band ≤ 60 % is `w = 0.17` (the next grid point,
`w = 0.18`, already gives 60.7 %). OOD BA on decided windows, grouped CI:

| w | review% (1,546) | OOD BA [grouped 95% CI] | n decided (OOD) |
|---|---|---|---|
| 0.00 | 0.00 | 0.546 [0.439, 0.655] | 160 |
| 0.02 | 2.65 | 0.543 [0.424, 0.657] | 154 |
| 0.04 | 5.82 | 0.565 [0.441, 0.692] | 149 |
| 0.06 | 10.93 | 0.538 [0.394, 0.678] | 140 |
| 0.08 | 19.92 | 0.536 [0.372, 0.685] | 130 |
| 0.10 | 29.56 | 0.524 [0.374, 0.674] | 118 |
| 0.12 | 37.00 | 0.536 [0.383, 0.697] | 110 |
| 0.14 | 45.08 | 0.543 [0.380, 0.705] | 98 |
| **0.16** | **53.36** | **0.577 [0.415, 0.740]** | 88 |
| **0.17** | **57.44** | **0.600 [0.440, 0.767]** | 82 |
| 0.18 | 60.74 | (review > 60 % — excluded) | 76 |

On the frozen (prob-avg) score **every** OOD BA in the ≤ 60 % region (0.52–0.60) has a
grouped CI that **contains 0.5**. Reducing review to ≤ 60 % does not buy OOD accuracy on this
score.

## 4. Frontier — coverage gating (`c = 0.40…1.00`)

Decide the top-`c` by cost-margin; `review% = 1 − c`. Two OOD evaluations:
**(i)** the same coverage applied per-dataset (the Y2 convention), and
**(ii)** the honest deployment variant — the margin **cutoff taken from the 1,546** and applied
unchanged to the 160 (this shifts how many OOD windows get decided).

| coverage | review% (1,546) | OOD BA (per-dataset) [CI] | n dec | OOD BA (1,546-cutoff) [CI] |
|---|---|---|---|---|
| 0.40 | 60.00 | 0.594 [0.406, 0.833] | 64 | 0.587 [0.408, 0.762] |
| **0.45** | **55.00** | **0.629 [0.444, 0.822]** | 72 | 0.587 [0.427, 0.752] |
| 0.50 | 50.00 | 0.583 [0.404, 0.754] | 80 | 0.571 [0.405, 0.736] |
| 0.55 | 45.00 | 0.577 [0.415, 0.740] | 88 | 0.543 [0.380, 0.705] |
| 0.60 | 40.00 | 0.550 [0.387, 0.712] | 96 | 0.543 [0.387, 0.704] |

Gating is the stronger lever: at 55 % review it reaches OOD BA **0.629** vs the band's
**≈0.587** at a comparable 55 % budget (`w = 0.165`, 55.6 % review), and the deployment-honest
cutoff variant still sits at 0.587 — again **inside chance** (CI includes 0.5).

## 5. Rank-average fusion (no new inference) — at *equal* review

Global AUROC does **not** improve (`usable_prob` 0.566 vs rank-avg 0.563 — the rank fusion is
*not* more separable). But at the **same review budget** under gating it decides a smoother set:

| coverage | review% | prob-avg OOD BA [CI] | rank-avg OOD BA [CI] |
|---|---|---|---|
| 0.40 | 60.0 | 0.594 [0.406, 0.833] | **0.674** [0.455, 0.826] |
| 0.45 | 55.0 | 0.629 [0.444, 0.822] | **0.690** [0.472, 0.833] |
| 0.50 | 50.0 | 0.583 [0.404, 0.754] | **0.694** [0.506, 0.827] |
| 0.55 | 45.0 | 0.577 [0.415, 0.740] | 0.630 [0.437, 0.788] |

At equal review the rank fusion moves OOD BA up ~0.09–0.11 and it **also lowers the ambiguous
share** on the 1,546 (raw band `[0.34,0.66]`, `w = 0.16`: prob-avg **53.4 %** review → rank-avg
**44.2 %**). This repeats Y3's "free win" but the AUROC is flat, so it is an **operating-point /
scale effect, not new information** (the rank transform spreads the score distribution; ranks
are a **batch/transductive** statistic — they depend on the cohort).

## 6. Recommended operating point

**Minimise OOD error subject to review ≤ 60 %** on the measured frontier gives:

> **Rank-average fusion + symmetric band `w = 0.155–0.16`**
> REVIEW **42.8–44.2 %** on the 1,546 (≤ 60 % ✓) ·
> OOD BA **0.708** [0.521, 0.832] (err 0.292), **85 / 160** decided.

Robust, equal-review alternative (avoids the rank-band scale artifact — review is then set by a
budget, not by an arbitrary rank threshold):

> **Rank-average fusion + coverage gating `c = 0.45`**
> REVIEW **55.0 %** · OOD BA **0.690** [0.472, 0.833].

If the score fusion must stay frozen (prob-average):

> **Coverage gating `c = 0.45`** — REVIEW **55.0 %**, OOD BA **0.629** [0.444, 0.822];
> or **band `w = 0.17`** — REVIEW **57.4 %**, OOD BA **0.600** [0.440, 0.767].

Recommended point vs alternatives (all ≤ 60 % review):

| operating point | review% (1,546) | OOD BA [grouped CI] | n dec |
|---|---|---|---|
| **band w=0.155, rank-avg (recommended)** | **42.8** | **0.708 [0.521, 0.832]** | 85 |
| band w=0.16, rank-avg | 44.2 | 0.708 [0.521, 0.832] | 85 |
| gate c=0.45, rank-avg | 55.0 | 0.690 [0.472, 0.833] | 72 |
| gate c=0.45, prob-avg (frozen score) | 55.0 | 0.629 [0.444, 0.822] | 72 |
| band w=0.17, prob-avg (frozen score) | 57.4 | 0.600 [0.440, 0.767] | 82 |
| gate c=0.40, prob-avg, 1,546-cutoff | 60.0 | 0.587 [0.408, 0.762] | 78 |

### Honesty caveats (must travel with the number)

1. **The "optimum" is selected on the test set.** The 160 OOD windows are the pre-registered
   test; nothing was *fit* on them, but the recommended point is the **argmax over the grid**,
   so its BA is an optimistically-biased order statistic. The ROs of the top points
   (`w = 0.155` and `w = 0.16`) are **identical** (BA 0.708, 85 decided) and all top candidates'
   CIs overlap heavily.
2. **`n_bad = 22`, 47 factories, one labeller.** Grouped CIs are wide; the recommended point's
   lower bound (0.521) only **just** clears chance. The frozen score's **global AUROC is 0.566
   [0.422, 0.702] — the CI contains 0.5**, so "OOD skill" is not established.
3. **Rank-average does not improve separability** (AUROC 0.563 vs 0.566); its BA gain is an
   operating-point effect and it is **transductive** (rank within the batch/cohort), which is a
   deployment concern.
4. **REVIEW% is a workload number** on unlabelled windows; accuracy is only measurable on the 160.
5. **47 factories is not an error bar on 160 independent windows** — the grouped bootstrap is the
   honest one, and it is what is reported everywhere above.

## 7. Explicit answer to "can review go ≤ 60 %?"

**Yes.** REVIEW can be cut from the frozen 75.5 % to ≤ 60 % (and lower) with **no refit**:
band `w = 0.16` → 53.4 %, band `w = 0.17` → 57.4 %, gating `c = 0.45` → 55.0 %, and with rank
fusion as low as 42.8 %.

**But on the frozen (prob-average) score the OOD error does not improve** — OOD BA stays
**0.52–0.60** with **every** grouped CI containing 0.5, i.e. **at chance**. The reduction is a
**workload win, not a demonstrated accuracy win**. The rank-average variant nominally lifts OOD
BA to ~0.69–0.71, but it does not lift AUROC and the gain is inside the noise / a test-set
selection effect. We therefore ship the review reduction as a workload improvement and state
plainly that **autonomous ACCEPT/REJECT quality out-of-domain remains unestablished**.

## Files

| file | role |
|---|---|
| `eval/exp_review160/run_o3.py` | metrics, band/gating sweeps, rank fusion, grouped bootstrap |
| `eval/exp_review160/make_fig.py` | renders the frontier figure |
| `eval/results/exp_review160/O3.01.json` | reproduction + full band/gate/rank frontier |
| `eval/results/exp_review160/O3.02.json` | recommended operating point + candidates + conclusion |
| `docs/experiments/exp_review160/images/o3_review160_frontier.png` | figure |
