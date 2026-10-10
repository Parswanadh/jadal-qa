# A3 — What does it take to reach 95% accuracy? Label/factory budget + per-factory specialization

**Pod:** jadal-qa · **Branch:** `m/acc-labelbudget` · **Owner:** A3 · **Timebox:** 35 min · **CPU only**
**Results:** `eval/results/exp_acc/A3.01.json`, `A3.02.json`, `A3.03.json`, `label_plan_95.json`
**Figure:** `docs/reports/images/exp_acc_labelbudget.png`
**Reproduce:** `OMP_NUM_THREADS=2 python3 eval/exp_acc/run_acc_budget.py`

## TL;DR (the honest answer)

**0.95 accuracy is not reachable with the current cached signals at any realistic label budget — and
0.90 is out of reach too.** The best current recipe is the *label-free* adapted score (frozen O4:
E5 stable-8 + Z3 transferable-6, per-domain z-score, IndEgo-signed mean). It already sits at the
feature ceiling: **BA 0.733 / error 0.267 / AUROC 0.781 on the pooled 160**, but only **error 0.352 /
AUROC 0.676 on the 108 confirmatory hold-out** (it does **not** replicate — C1.01). A saturating fit to
the label curve gives an **asymptote e∞ ≈ 0.33** (error = 1 − balanced accuracy). Labels buy a few
points and then plateau; **0.10 and 0.05 error are not within the model family.**

Certification is a *factory-count* problem, not a model problem. The corpus now has **85 factories**
(up from 47), so labelling all 85 certifies only a marginal error of **r ≈ 0.20 (grid-union) / 0.13
(single threshold)** at 95% confidence. To *certify* 0.90 accuracy you need **~342 factories**; 0.95
needs **~1,365** (grid-union). The corpus is **~16× too small** for a 0.05 bound.

| target | within model family? | labels/factories to reach (model) | factories to certify @95% (grid-union) |
|---|---|---|---|
| **error ≤ 0.20 (80% BA)** | yes (≈ plateau) | ~85 × 4 = 340 labels | **86** (have 85) |
| **error ≤ 0.10 (90% acc)** | **no** (e∞ ≈ 0.33) | ∞ — labels do not reach it | **342** |
| **error ≤ 0.05 (95% acc)** | **no** | ∞ — needs new signals | **1,365** |

---

## 0. Data & protocol

| set | n | good/bad | factories | role |
|---|---|---|---|---|
| IndEgo gold | 150 | 100/50 | 7 clips | signs + in-domain (LOCO) |
| OOD-160 Egocentric-10K | 160 | 138/22 | 47 | fit / grouped-CV curve |
| **Hold-out 108** (round-3) | 108 | 92/16 | 44 | **evaluate only, scored once** |
| unlabelled corpus (`manifest_scale_all.json`) | 9,757 | — | **85** (391 clips, 27.66 h) | certification clusters |

*Best current recipe* = frozen O4 adapted score, optionally augmented with a logistic on
`[adapted, 26 per-pixel features]` fit on k human labels. All evaluation is **factory-grouped**
(`StratifiedGroupKFold(5, groups=factory)`): a factory is never in both train and test. Error is
`1 − balanced accuracy @0.5` (the honest metric; plain accuracy is dominated by the 86% good prior).
No labels edited; no tuning on the hold-out.

---

## 1. A3.01 — accuracy-vs-labels curve, by strategy, extrapolated

Best-recipe curve (logistic on `[adapted, 26 feats]`, factory-grouped CV, error = 1 − BA):

| k labels | random | breadth (new factories) | depth (existing factories) | **diversity/maximin** |
|---:|---:|---:|---:|---:|
| 5 | 0.451 | 0.529 | **0.321** | 0.580 |
| 10 | 0.436 | 0.460 | 0.563 | 0.423 |
| 20 | 0.401 | **0.380** | 0.375 | 0.433 |
| 40 | 0.374 | **0.351** | 0.568 | 0.409 |
| 60 | 0.372 | **0.316** | 0.412 | **0.316** |
| 80 | 0.338 | 0.331 | 0.361 | **0.283** |
| 120 | 0.309 | 0.333 | 0.311 | 0.371 |
| 160 | 0.307 | 0.307 | 0.307 | 0.307 |
| **label-free adapted** | — | — | — | **0.267** |

Read: **diversity/maximin** wins at mid budget (k = 80, 0.283), **breadth** at low budget
(k = 20–60); **depth is harmful** (0.57 at k = 40 — overfits a few factories); random is a reasonable
default. All gaps are within noise (n = 160, n_bad = 22, n_test ≈ 32/fold). This reproduces N4's
ordering. The 26-feature logistic alone (N4 continuity) gives the same picture (best point 0.2575 at
k = 80 diversity; 0.342 at full 160).

**Saturating fit** (best strategy, `error(k) = e∞ + a·exp(−k/τ)`, 300 fold-bootstrap resamples):

- **e∞ = 0.326**, a = 0.253, τ = 20.6 — 95% CI on e∞ **[0.0, 0.384]** (the lower bound is a
  bootstrap artefact; the point estimate and every observed k ≤ 160 point say the floor is ≈ 0.27–0.33).
- **k for error 0.10: unreachable** at the point estimate; optimistic bootstrap median ~0.7–1.5k
  labels (fitting noise). **k for error 0.05: unreachable** (optimistic ~1.0–2.2k).
- The label-free adapted score (0.267) already beats the fitted k = 160 point (0.307): **more labels
  do not beat the zero-shot transferable score.** This is N1's result, reconfirmed.

**Independent hold-out check (108, scored once):**

| scorer | AUROC | error (1 − BA) |
|---|---:|---:|
| adapted O4 (label-free) | 0.676 | 0.352 |
| frozen ensemble_v1 | **0.690** | 0.360 |
| logistic fit on all 160 labels | 0.619 | 0.348 |

The pooled-160 0.781 **does not replicate** (0.676); the frozen ensemble is equally weak. The
*deployable* ceiling is ~0.68–0.69 AUROC / ~0.35 error, i.e. **balanced accuracy ≈ 0.65**, not 0.95.

---

## 2. A3.02 — per-factory specialization (few labels per factory)

Per-factory calibration on `m` labels from the factory (Platt, or a shrunk intercept with the global
model as prior), evaluated on the remaining windows of the same factory, pooled over 40 seeds:

| labels/factory m | 0 (global) | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| per-factory Platt: BA | **0.733** | 0.518 | 0.507 | 0.570 | 0.545 |
| shrunk (λ = 2): BA | 0.733 | 0.578 | 0.560 | 0.576 | 0.566 |
| per-factory accuracy | 0.638 | 0.515 | 0.513 | 0.586 | 0.617 |

**Per-factory specialization HURTS.** With ~3.4 windows/factory the per-factory calibration is fit on
1–4 points, which is pure noise; even partial pooling (shrinkage) does not recover the global model.
This matches N1's failure of per-factory test-time normalization (0.727 → 0.383). The **labels-per-
factory needed** to make per-factory calibration break even is **≫ 4**, and to *certify* a per-factory
error of 0.05 you would need **~1,365 labels per factory** (§3) — infeasible.

**Verdict:** per-factory specialization is not the route to 95%; the global label-free score is
already at the signal ceiling, and per-factory fitting only adds variance.

---

## 3. A3.03 — certification with the 85-factory corpus (95% confidence)

Cluster-Hoeffding UCB on the marginal error rate, `K_min = ⌈ ln(1/δ′) / (2(r − R̂)²) ⌉`,
δ = 0.05, R̂ = 0, δ′ = δ (single pre-registered threshold) or δ/46 (union over the LTT grid):

| target error r | factories, single threshold | factories, grid-union |
|---:|---:|---:|
| 0.05 (95% acc) | 600 | **1,365** |
| 0.10 (90% acc) | 150 | **342** |
| 0.15 | 67 | 152 |
| 0.20 (80% BA) | 38 | **86** |

**Certifiable today:**

| clusters labelled | certifiable r, grid-union | certifiable r, single |
|---|---:|---:|
| 47 (today) | 0.269 | 0.179 |
| **85 (corpus, all labelled)** | **0.200** | **0.133** |

So the new corpus (85 factories) **does** buy the 0.20 bound that N4 asked for (86 factories) — but
only just, and it certifies **0.20, not 0.05**. A 0.05 (95% accuracy) certificate needs **1,365
factories** (grid-union) / 600 (single): the corpus is ~16× too small. Window-level counts (9,757)
are anti-conservative under within-factory correlation; the factory is the honest exchangeable unit.
To certify each *factory* at 0.05 you would need ~1,365 labels per factory.

---

## 4. What else is needed for 95% (honest ceiling)

Labels and per-factory calibration are **not** the bottleneck — the cached 26 per-pixel + 20
capture-telemetry features are. The feature set is **not domain-invariant** (IndEgo-vs-OOD classifier
AUROC 0.999; OOD160-vs-pool 0.902), and the OOD failures are **semantic** (NO_ACTIVE_WORK 19/22,
HANDS_NOT_VISIBLE 2, SHAKY 1), which sharpness/telemetry features cannot separate. To raise the
ceiling:

- **A1 — new signals.** Full-resolution / motion-cue / hand-visibility signals (tracks `m/mosaic-res`,
  `m/exp-motion-cue`, `m/new-features-160`) are the only route to a higher ceiling; mosaic work shows
  resolution alone is not the lever (480p→mosaic geometry is accuracy-neutral). A semantic
  hand/active-work detector is what the 19/22 NO_ACTIVE_WORK errors need.
- **A2 — capacity.** Larger models/heads (track `m/acc-capacity`) are bounded by the *same* cached
  features; capacity without new signals cannot cross ~0.70 BA.
- **Ceiling statement:** with the current data and signals the honest ceiling is **error ≈ 0.27–0.35
  (BA 0.65–0.73)**, and the deployable hold-out number is ~0.65 BA. **95% accuracy is not a labeling
  problem; it requires a new signal family and is not achievable by labels alone.**

---

## 5. Concrete label plan — `label_plan_95.json`

| phase | goal | factories × windows | labels | expected error (BA) | certifies |
|---|---|---:|---:|---|---|
| 1 | cover all factories + certify r = 0.20 | 85 × 4 | **340** | ~0.28–0.32 (plateau) | error ≤ 0.20 @95% (grid-union, 86 factories) |
| 2 | per-factory calibration + denser model | 85 × 8 | 680 | ~0.25–0.30 | error ≤ 0.15 @95% (152 factories) |
| 3 | **reach 0.90 / 0.95 — needs NEW SIGNALS** | ≥342 / ≥1,365 factories | ≥342·4 / ≥1,365·4 | 0.10 / 0.05 **only if** new signals raise the ceiling | 0.90 @95% (342); 0.95 @95% (1,365) |

**Which windows:** diversity/maximin over the 26 regime features, one window per **new** factory first
(breadth), then fill to 4/factory. **Do NOT** sample by model uncertainty (Z4: anti-correlated,
AUROC 0.349). Priority new factories (largest, unlabelled): `factory_017`, `factory_074`,
`factory_002`, `factory_004`, `factory_006`, `factory_008`, `factory_007`, `factory_005`, … (38 new
factories available). Never edit existing label files.

---

## Caveats

- n = 160 / n_bad = 22 (and 108 / 16) → wide CIs; the curve differences are suggestive, not
  significant. The hold-out was scored once and not used for any tuning.
- The saturating fit is a 3-parameter model on 10 noisy points; the e∞ bootstrap CI is wide and its
  lower bound (0) is not credible. We report the point estimate and the honest "not within family"
  reading, not a precise k.
- Certification is a closed-form data requirement, not a measured guarantee; it assumes factory-level
  exchangeability.
- CPU only, cached inputs, no GPU, no decode, no label edits.

## Reproduce

```
OMP_NUM_THREADS=2 python3 eval/exp_acc/run_acc_budget.py
```
Inputs (read-only): `labels/ood160_labels.json`, `labels/manifest_gold.json`,
`labels/round3/ood_labels3_108_with_factory.json`, `queues/manifest_scale_all.json`,
`eval/results/exp_acc/inputs/ood160_pixel.npz`, `inputs/confirm108_features.npz`,
`inputs/confirm108_ensemble.json`, `/workspace/wt/m-adapt160/.../z3_160.npz`.
