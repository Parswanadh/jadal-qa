# AUTO policy report — forced-decision error, second-view resolver, risk-coverage frontier (draft v1, S4 stream)

Track AUTO, stream S4 (`dsh/auto-policy`), 2026-10-18. Owner: dsh-s4-auto.
Device: **CPU only** (numpy/sklearn on the builder's laptop; RTX 4070 Laptop 8 GB, 22 cores, 15 GB RAM — no GPU was used by any experiment in this stream).
Interpreter: `/home/parshu/envs/jadal/bin/python` (Python 3.12.15, sklearn 1.9.1, numpy 2.5.2).
Bundle: `JADAL_BUNDLE=/home/parshu/projects/jadal-qa-wt/handover/jadal_pod_bundle/pod_bundle`.

This file is the S4 stream's own record. It is a draft for the lead to fold into the AUTO memo; the
AUTO.08 spec, the receipt fields and the end-to-end table are the lead's section, not this one.

**Registry ids:** `AUTO.00`, `AUTO.01`, `AUTO.03`, `AUTO.04`, `AUTO.07` (registered with `tools/exp.py new`
before each run, closed with `exp.py finish`).
**Results JSON (every number below comes from one of these):**
`eval/results/auto/AUTO.00.json`, `AUTO.01.json`, `AUTO.03.json`, `AUTO.04.json`, `AUTO.07.json`,
each carrying `n`, `n_bad`, `label_source`, `protocol`, `ci95`, `ci95_type`, `device`, `cost`,
`command`, `git_sha`, `seed=0`, `config_sha256`.
**Tests:** `tests/test_dsh_auto.py` (9 tests, < 5 s, no network).

## 0. What the shipped funnel does, and what this stream replaced

The shipped cascade sends a window to a human when `|p - 0.5| < 0.25`, i.e. **accept iff p ≥ 0.75,
reject iff p ≤ 0.25, else review** (`jadal/poc/cascade.py::decide_window`, MARGIN = 0.25; independently
reproduced by `dsh/task`, PR #173, 160/160 OOD windows). Every policy in this stream is the **forced**
variant of that funnel: the review band is replaced by a decision, so coverage is 100 %.

- **P1 `s1_forced_0.5`** — the shipped 26-number head score, accept iff p ≥ 0.5.
- **P2 `youden_perfold`** — same score, threshold = Youden (balanced-accuracy maximiser) chosen on the
  six *training* clips' inner-LOCO probabilities of each outer fold.
- **P3 `band_plus_second_view`** — S1 outside the band, the cached EmbeddingGemma-2 second view inside it
  (threshold 0.5 in both regions). IndEgo only.

In-domain data IndEgo-150: n = 150, n_bad = 50, 7 clips, **ONE human labeller**, 26 cached per-pixel
features, leave-one-clip-out (LOCO), clip-level bootstrap 2000 draws seed 0.
Out-of-domain data OOD-160: **n = 160, n_good = 138, n_bad = 22, 47 factories, ONE human labeller**
(19 of the 22 bad windows are `NO_ACTIVE_WORK`, 2 `HANDS_NOT_VISIBLE`, 1 `SHAKY`), scores **frozen**
from the original portal's 26-number head. Every OOD interval below is **factory-grouped**; where a
window bootstrap is shown it is marked optimistic and ignores within-factory correlation.

## 1. AUTO.00 — forced-decision baseline (100 % coverage)

### 1.1 In-domain, IndEgo-150 (LOCO, clip-grouped CI)

Pooled AUROC 0.810 [0.683, 0.892] (clip bootstrap; the window bootstrap 0.722–0.889 is optimistic).

| policy | balanced acc | 95 % CI (clip) | false-accept | false-reject | accept share |
|---|---|---|---|---|---|
| P1 forced at 0.5 | **0.740** | [0.631, 0.789] | 0.260 | 0.260 | 0.580 |
| P2 Youden on training folds | **0.755** | [0.646, 0.797] | 0.240 | 0.250 | 0.580 |
| P1, threshold chosen on all 150 OOF labels — *in-sample contrast, not a policy* | 0.785 | — | 0.280 | 0.150 | 0.660 |

Per-fold Youden thresholds (min / median / max): 0.359 / 0.501 / 0.558 — the training folds put the
threshold close to 0.5 on six of seven clips (`desk` is the outlier at 0.359).
The shipped band holds **48.0 %** of IndEgo windows (clip CI [0.347, 0.570]).

### 1.2 OOD-160 (frozen shipped head, factory-grouped CI)

> **n = 160, n_bad = 22, 19 of the 22 bad are NO_ACTIVE_WORK**, 47 factories, one labeller.

| quantity | value | 95 % CI |
|---|---|---|
| AUROC (all bad) | 0.588 | [0.450, 0.728] factory-grouped |
| AUROC, window bootstrap — **optimistic** | 0.588 | [0.470, 0.711] |
| AUROC, `NO_ACTIVE_WORK` bad only (n = 157, 19 bad) | 0.576 | [0.427, 0.739] factory-grouped |
| AUROC, `HANDS_NOT_VISIBLE` only (2 bad) | 0.853 | [0.711, 0.939] |
| AUROC, `SHAKY` only (1 bad) | 0.290 | [0.156, 0.412] |

| policy | balanced acc | 95 % CI (factory) | false-accept | false-reject | accept share |
|---|---|---|---|---|---|
| P1 forced at 0.5 | **0.556** | [0.448, 0.673] | 0.409 | 0.478 | 0.506 |
| P2 (threshold 0.396, chosen on IndEgo-150 OOF and transferred) | 0.552 | [0.426, 0.684] | 0.455 | 0.442 | 0.544 |
| P2 in-sample on the OOD labels — *contrast, not a policy* | 0.584 | — | 0.455 | 0.377 | 0.600 |

The shipped band holds **13.8 %** of OOD-160 windows. **The lower bound of the OOD-160 balanced accuracy
at 100 % coverage is 0.448, below the brief's 0.70 tolerance**, so the brief's decision rule points at
"tiered automation with audit" (§5).

### 1.3 Round-3 108-window hold-out — the 26-number single-head arm is SKIPPED

There is **no per-window score file for the 26-number single head** on the 108 round-3 windows, and
producing one would mean running a model on a consumed hold-out. That arm is therefore **skipped**, not
proxied. Read-only, from the C1 PR's committed inputs (`origin/m/confirm`,
`eval/results/exp_confirm/inputs/confirm108_ensemble.json`; **nothing was scored in this stream**), the
frozen *two-view* shipped ensemble was scored on the 108 once (n = 108, n_bad = 16, 44 factories) —
a **different model** from the 26-number head, reported only because it is the shipped cascade's score:

| frozen score | AUROC (factory-grouped) | forced at 0.5: balanced acc | FAR | FRR |
|---|---|---|---|---|
| `usable_prob` (frozen two-view ensemble) | 0.690 [0.534, 0.823] | 0.640 | 0.188 | 0.533 |
| `p_eg2` alone | 0.733 [0.564, 0.874] | 0.587 | 0.750 | 0.076 |
| `p_pixel` alone | 0.625 [0.474, 0.765] | 0.591 | 0.188 | 0.630 |

The window bootstraps for these rows (e.g. 0.541–0.820 for `usable_prob`) are optimistic. The hold-out is
**consumed** (C1 PR #162, physics v2 PR #163) and cannot support adoption; adoption needs a fresh batch.

## 2. AUTO.01 — second-view resolver on the review band (IndEgo only)

Second view: `caches/frames10k/segments.npz`, EmbeddingGemma-2 window embeddings, 150 windows × 21 frame
slots × 768-d at ~2 fps. The 21st slot is NaN padding on 145/150 windows and was excluded by using each
window's `counts`. Representation: mean over the valid frame embeddings; same StandardScaler +
LogisticRegression(C = 0.3, balanced) recipe and the same LOCO scheme. 72 of 150 windows fall in the band.

| quantity | value |
|---|---|
| AUROC in the band, S1 (26 features) | 0.820 |
| AUROC in the band, EG2 | 0.811, clip CI [0.683, 0.882] (window bootstrap 0.698–0.912, **optimistic**) |
| paired EG2 − S1 in the band | **−0.010 [−0.115, +0.052]** — interval includes zero, **difference not established** |
| cross-check: the repo's frozen nested-LOCO EG2 view | AUROC 0.846 overall / 0.848 in-band; Pearson r = 0.909 vs this view (the repo's artifact, not this stream's result) |

100 %-coverage policies (paired clip bootstrap against P1 on the same windows):

| policy | balanced acc | 95 % CI (clip) | FAR | FRR | accept share | paired Δ vs P1 |
|---|---|---|---|---|---|---|
| P1 forced at 0.5 | 0.740 | [0.631, 0.789] | 0.260 | 0.260 | 0.580 | — |
| hybrid: EG2 decides the band | 0.740 | [0.625, 0.806] | 0.340 | 0.180 | 0.660 | **0.000 [−0.044, +0.059]** — not established |
| hybrid with the band decided by a rank-average of the two OOF scores | **0.800** | [0.672, 0.890] | 0.120 | 0.280 | 0.520 | **+0.060 [+0.015, +0.136]** — excludes zero |

The rank-average is the only arm whose paired interval excludes zero. Two honesty controls were run
because a rank-average decided at rank 0.5 accepts the top half of the band by construction:
against **P1's own ranks cut at 0.5** (same coverage) it wins by **+0.025 [+0.007, +0.050]**, but against
**EG2's ranks cut at 0.5** it is **+0.025 [−0.028, +0.077]** — not established. In other words the win is
mostly a change of operating point (FAR 0.26 → 0.12, FRR 0.26 → 0.28 inside a 48 %-wide band), not a
demonstrated gain from fusing the two views. The brief's Y3 rank-average note is therefore *partly*
confirmed here: it improves on the forced 0.5 rule, but the incremental value over ranking by the
S1 score alone is small and only established against one of the two controls.

## 3. AUTO.03 — risk-coverage frontier and the bound

Coverage = share of windows decided automatically; risk = error rate **among the auto-decided windows**
(a false accept and a false reject both count). At coverage c the c-fraction with the largest confidence
`|score − threshold|` is decided automatically and the rest go to a human. Intervals resample clips
(IndEgo) or factories (OOD-160) and **re-select inside every resample**.

IndEgo-150 (LOCO, clip-grouped), balanced accuracy / error at each coverage — full table in the JSON:

| coverage | P1 0.5 | P2 Youden | P3 band+EG2 |
|---|---|---|---|
| 100 % | 0.740 / 0.260 | 0.755 / 0.247 | 0.740 / 0.233 |
| 80 % | 0.844 / 0.167 | 0.821 / 0.175 | 0.763 / 0.208 |
| 60 % | 0.826 / 0.178 | 0.823 / 0.167 | 0.808 / 0.167 |
| 40 % | 0.771 / 0.233 | 0.804 / 0.200 | 0.800 / 0.133 |
| 20 % | 0.796 / 0.200 | 0.768 / 0.233 | 0.812 / 0.100 |

On OOD-160 the frontier is reported for P1 and P2 only: **P3 is unavailable out of domain** because the
cached second-view embeddings cover the 150 IndEgo windows and nothing else, and this stream did not run
a model to make more.

### 3.1 Conformal-style bound under exchangeability, assumptions listed

λ is chosen from **training folds only** — per outer fold from the six training clips' inner-LOCO
probabilities (IndEgo), and from the IndEgo-150 out-of-fold scores for OOD-160 — by the CRC selection
rule `(n/(n+1))·risk_hat(λ) + 1/(n+1) ≤ α` on a 101-point confidence grid. The primary pick is the
smallest λ (largest coverage) that passes; a conservative monotone-envelope pick and an uncorrected
pick are recorded in the JSON for contrast. **No test label enters the selection.**

Assumptions, listed:
1. **Exchangeability** of calibration and deployment windows; in-domain the split is at **clip** level, so
   the honest exchangeable unit is the clip, not the window.
2. **One labeller** — label noise is not modelled and no inter-rater variance is in any interval.
3. Calibration scores are **inner-LOCO out-of-fold**: no calibration window's own clip trained its model.
4. **Intra-clip correlation breaks window-level exchangeability.** The repo's own record
   (`docs/FACTS.md`, on main) is that a nominal 2.7 % false-reject became **46 %** on held-out clips with
   intra-clip correlation 0.58 and effective n ≈ 11. That is the repo's measurement, not re-measured here.
5. **OOD-160 is a different distribution.** No target exchangeability holds, so an IndEgo-selected λ
   carries no guarantee on OOD-160; those rows are descriptive.
6. The CRC rule bounds the **expected** risk of the selected rule under a monotone risk curve. It is a
   statement about the calibration draw, not a per-window certificate, and not a licence to skip the audit.

Maximum coverage the training folds support, and what it actually delivered on held-out clips:

| dataset | target risk | policy | λ selection | coverage achieved | 95 % CI (grouped) | realized risk on clips |
|---|---|---|---|---|---|---|
| IndEgo | 5 % | P3 band+EG2 | 5/7 folds feasible | **7.3 %** | [1.1 %, 16.9 %] | 18.2 % |
| IndEgo | 5 % | P1 | 1/7 folds feasible | 2.0 % | [0.0 %, 6.1 %] | 33.3 % |
| IndEgo | 10 % | P3 band+EG2 | 5/7 | 13.3 % | [4.0 %, 27.5 %] | 25.0 % |
| IndEgo | 10 % | P2 | 3/7 | 8.7 % | [0.0 %, 22.8 %] | 30.8 % |
| IndEgo | 20 % | P3 band+EG2 | 6/7 | 46.0 % | [23.2 %, 80.8 %] | 30.4 % |
| IndEgo | 20 % | P1 | 5/7 | 28.0 % | [6.1 %, 63.9 %] | 33.3 % |

**The bound does not survive the held-out clips**: at every target the realized risk is several times α
(18–36 % against targets of 5–20 %). This reproduces the repo's own 2.7 % → 46 % finding in the AUTO
setting and is the strongest reason not to read these rows as a deployment guarantee.

OOD-160 with λ transferred from IndEgo: **infeasible at 5 % and 10 %**; at 20 %, P1 selects λ = 0.073
giving 95.0 % coverage [92.1 %, 97.9 %] with a realized OOD risk of **46.1 %** (and P2 λ = 0 gives 100 %
coverage, 44.4 % risk). The guarantee does not transfer; the numbers are descriptive only.

The one construction that does not rely on cross-domain exchangeability is a **factory-level split inside
OOD-160** (47 factories cut in half by seed 0, one half calibrates, the other evaluates):

| target | split | calibration (n / n_bad) | coverage on eval | 95 % CI (factory) | realized risk |
|---|---|---|---|---|---|
| 5 % | A → B, P2 | 72 / 9 | 13.6 % | [3.5 %, 27.2 %] | 8.3 % |
| 5 % | B → A, P2 | 88 / 13 | 13.9 % | [1.7 %, 27.3 %] | 0.0 % |
| 10 % | A → B, P2 | 72 / 9 | 42.0 % | [27.4 %, 58.8 %] | 10.8 % |
| 10 % | B → A, P2 | 88 / 13 | 25.0 % | [6.5 %, 40.9 %] | 5.6 % |
| 20 % | A → B, P2 | 72 / 9 | 43.2 % | [28.1 %, 60.0 %] | 13.2 % |
| 20 % | B → A, P2 | 88 / 13 | 44.4 % | [17.5 %, 63.2 %] | 9.4 % |

P1 (fixed 0.5) was **infeasible at all three targets** in this split. These rows are the only ones in this
section where the realized risk sits near the target, and they rest on the exchangeability of factories —
an assumption with 47 units and 22 bad windows in total, not a comfortable one.

## 4. AUTO.04 — cost-sensitive thresholds (false accept : false reject)

The threshold minimises `c_fa·(bad accepted) + c_fr·(good rejected)` on the training folds only
(IndEgo: the six training clips' inner-LOCO probabilities, per fold; OOD: the IndEgo-150 OOF scores).
The analytic Bayes threshold `c_fa/(c_fa+c_fr)` is arithmetic, not measurement.

IndEgo-150 (threshold min/median/max across folds; cost per 1000 windows in that row's own units):

| ratio | thresholds | balanced acc | FAR | FRR | accept | cost/1000 | at fixed 0.5 |
|---|---|---|---|---|---|---|---|
| 1:1 | 0.186 / 0.417 / 0.501 | 0.680 | 0.480 | 0.160 | 0.720 | 267 | **260** |
| 3:1 | 0.500 / 0.725 / 0.768 | 0.650 | 0.120 | 0.580 | 0.320 | 507 | **433** |
| 1:3 | 0.009 / 0.133 / 0.219 | 0.540 | 0.880 | 0.040 | 0.933 | **373** | 607 |

The direction is right (3:1 accepts less, 1:3 accepts more), but on held-out clips the tuned threshold
only pays for itself at **1:3**; at 1:1 and 3:1 the fixed 0.5 rule has the lower cost. The per-fold
thresholds are unstable (0.19–0.50 at 1:1), which is what six training clips buy.

OOD-160 (threshold transferred from IndEgo, factory-grouped):

| ratio | threshold | balanced acc | FAR | FRR | cost/1000 | AUROC context |
|---|---|---|---|---|---|---|
| 1:1 | 0.396 | 0.552 | 0.455 | 0.442 | 444 | 0.588 [0.450, 0.728] |
| 3:1 | 0.642 | 0.565 | 0.364 | 0.507 | 588 | same |
| 1:3 | 0.118 | 0.562 | 0.500 | 0.377 | 1044 | same |

All three are near chance; the cost ordering across ratios is not comparable (the units differ), which is
why each row carries the fixed-0.5 cost on the same cost function.

## 5. AUTO.07 — audit policy (all rows arithmetic, not measurement)

Recommended deployment form: **auto-decide everything; send every low-confidence (band) window and a
uniform random audit sample to humans; track error and verdict mix per site; alarm on drift.**

Measured inputs (from AUTO.00's scores and labels): IndEgo band share **0.480** [0.347, 0.570] clip CI,
in-band forced error 0.333, out-of-band forced error 0.192. OOD-160 band share **0.138**, in-band forced
error 0.545, out-of-band forced error **0.457**.
The audit is modelled over 10,000 video hours = 3.6 M ten-second windows (360 windows per video hour,
arithmetic from the window definition). **`T_review` — human seconds per window — is not measured
anywhere in the bundle or the repo; it is an assumed parameter**, and the hours column scales linearly in
it (20 / 30 / 60 s are all in the JSON).

| dataset | audit rate | human hours per 10,000 video hours (T = 30 s) | detected-error rate | escaped errors per 10k windows | caught per 10k windows |
|---|---|---|---|---|---|
| IndEgo-150 | 2 % | 15,000 | 0.623 | 1,000 | 20 |
| IndEgo-150 | 5 % | 15,900 | 0.635 | 1,000 | 50 |
| IndEgo-150 | 10 % | 17,400 | 0.654 | 1,000 | 100 |
| OOD-160 | 2 % | 4,725 | 0.177 | 3,938 | 79 |
| OOD-160 | 5 % | 5,625 | 0.202 | 3,938 | 197 |
| OOD-160 | 10 % | 7,125 | 0.244 | 3,938 | 394 |

Every row is labelled `arithmetic`. The human workload is dominated by the **band**, not the audit: on
IndEgo the band plus a 2 % audit already costs ≈ 1.5 human-hours per video-hour at 30 s/window. The audit
catches a proportionate `a` of the errors that escape the auto-decision; because 45.7 % of OOD-160's
out-of-band decisions are wrong, 2 % catches ≈ 79 of the 3,938 escaped errors per 10,000 windows.
Drift arithmetic per site (10,000 windows/day, assumed scenario): 200 audited windows at 2 % give
`P(≥1 detected error) = 1.000` at both measured out-of-band error rates; **15 windows** suffice for a 95 %
chance of seeing at least one at the IndEgo rate (5 at the OOD rate), while estimating a site's error to
±5 percentage points needs **239** (IndEgo) to **382** (OOD) audited windows. The brief's
"accept share moved from 19 % to 57 %" is quoted in the brief, was **not** re-measured here, and is the
reason per-site drift tracking is specified rather than a number this file relies on.

## 6. Recommendation (S4 input to the AUTO.08 spec)

1. **Do not claim fully automated.** The measured 100 %-coverage balanced accuracy on OOD-160 is 0.556
   with a factory-grouped lower bound of **0.448**, below the brief's 0.70 tolerance ⇒ **tiered automation
   with audit**, with the error and its interval printed next to any claim.
2. **The second view does not fix the band.** EG2 in the band gives no established improvement
   (0.000 [−0.044, +0.059]) and its in-band AUROC is not distinguishable from S1's.
3. **The only lever that moved the 100 %-coverage error** is the rank-average inside the band
   (+0.060 [+0.015, +0.136] paired, clip-grouped), and even there the increment over ranking by S1 alone
   is +0.025 [+0.007, +0.050] — real but small, and bought by rejecting more good windows.
4. **Do not deploy the conformal-style bound as a guarantee.** It is stated as a
   *conformal-style bound under exchangeability, assumptions listed* (§3.1); it failed on held-out clips
   and did not transfer to OOD. The only rows that tracked their target were factory-split within OOD,
   on 47 factories and 22 bad windows.
5. **Shape of the funnel to test next:** the band is the cost driver and the audit is nearly free by
   comparison; a narrower band plus a resolver that is actually better than S1 inside it, then a random
   audit, is the configuration the arithmetic supports. New labels alone are not the lever for
   `NO_ACTIVE_WORK` (19 of 22 OOD bad windows); a hand-visibility / motion signal is.

## 7. Limits, and what could not be done

- **Round-3 108 for the 26-number single head: skipped.** No per-window score file exists without running
  a model on a consumed hold-out; the read-only frozen-ensemble arm in §1.3 is a *different* model.
- **P3 (band + second view) on OOD-160: not available.** The cached EmbeddingGemma-2 window embeddings
  cover the 150 IndEgo windows only; no model was run to make more.
- **`T_review` is assumed, not measured.** The AUTO.07 hours columns are arithmetic and scale linearly in it.
- **n_bad = 22 out of domain, 19 of them one reason.** Every OOD interval is factory-grouped and wide; the
  reason-stratified rows for `HANDS_NOT_VISIBLE` (2 windows) and `SHAKY` (1 window) are descriptive only.
- **One labeller.** No inter-rater agreement exists, so no interval here includes label noise.
- **Windows inside a clip are correlated.** Clip-grouped intervals are the honest ones in-domain; every
  window-bootstrap interval in this report is marked optimistic. The AUTO.03 frontier's low-coverage rows
  rest on a handful of windows and their intervals are correspondingly wide.
- The hybrid policies are built on the **forced** 0.5 rule, not on the shipped outside-band rule
  (accept iff p ≥ 0.75); they are a strengthened funnel with the band decided instead of reviewed, not the
  shipped cascade with one stage swapped.

## 8. Reproduce

```bash
export JADAL_BUNDLE=/home/parshu/projects/jadal-qa-wt/handover/jadal_pod_bundle/pod_bundle
export OMP_NUM_THREADS=2
for s in auto00 auto01 auto03 auto04 auto07; do
  nice -n 10 /home/parshu/envs/jadal/bin/python tools/dsh/$s.py
done
OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python tests/test_dsh_auto.py
```
Total ≈ 32 s of CPU on this laptop, 0 GPU, no network. Helpers shared with the rest of the pass live in
`tools/dsh/auto_util.py`, which imports `tools/dsh/common.py` (one owner: S0) and adds only a generic
group-resampling bootstrap, rank helpers, cost/Youden threshold selection and the CRC λ selection.
