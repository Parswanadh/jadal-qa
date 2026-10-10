# A1 — Compute-free capacity + ensembling: how close to 95%?

**Pod:** jadal-qa · **Branch:** `m/acc-capacity` · **Task:** A1
**Device:** CPU (numpy/sklearn; cached features only, **no GPU**) · **Git sha:** `f20013b677ab3e25b930d7ed23bac572dbb78134`

**Question.** With compute treated as free, what is the highest accuracy reachable by
*model capacity + ensembling/stacking*, and how far is it from 95%?

**Results:** `eval/results/exp_acc/A1.01.json` (capacity sweep), `A1.02.json`
(ensembling/stacking), `A1.03.json` (regimes + hold-out + cost + verdict).
**Figure:** `docs/reports/images/exp_acc_capacity.png`.
**Reproduce:** `OMP_NUM_THREADS=2 python3 eval/exp_acc/run_acc_capacity.py && python3 eval/exp_acc/make_figure.py`

## Data & protocol

| set | n | good | bad | protocol |
|---|---|---|---|---|
| IndEgo gold | 150 | 100 | 50 | leave-one-clip-out (7 clips) |
| OOD-160 Egocentric-10K | 160 | 138 | 22 | grouped CV over 47 factories |
| Confirmatory 108 | 108 | 92 | 16 | **scored once, never tuned** |

All features are cached: pixel-26 (`caches/gpu/features.json`), EG2 (768-d mean of
`caches/frames10k/segments.npz`), physics-16 + theory-18 (`caches/perpixel/*`), the
adapted Z3-20 (m/adapt160, m/new-features), the frozen two-view per-view scores
(`caches/ensemble/*`), the deterministic rules (tools/phys, LOCO on gold / frozen on
OOD) and E4 descriptors (`desc_*.jsonl`). **Model CV is nested and grouped**: outer
leave-one-clip-out (gold) / GroupKFold(10)-by-factory (OOD), inner GroupKFold(≤4)
hyperparameter selection, standardise + median-impute, AUROC on the out-of-fold
good-score, BA at the fold-local max-BA threshold, grouped bootstrap CI (n_boot=2000).
Baselines (frozen scorers) reproduce the prior numbers exactly (gold rules 0.8222,
gold two-view 0.8336, OOD two-view 0.5660, hold-out two-view 0.6895).

## 1. Capacity sweep (A1.01)

Best AUROC per model class across feature sets (full tables in `A1.01.json`):

| model (best feature set) | in-domain AUROC [CI] | in-domain BA | OOD AUROC [CI] | OOD BA |
|---|---|---|---|---|
| logistic (gold eg2_768 / OOD z3_20) | **0.8256** [0.705, 0.892] | 0.705 | **0.7688** [0.649, 0.873] | 0.684 |
| gradient-boosted trees (gold theory18 / OOD all62) | 0.7884 [0.611, 0.888] | 0.695 | 0.6940 [0.536, 0.834] | 0.604 |
| small MLP (gold pixel26 / OOD z3_20) | 0.7998 [0.648, 0.882] | 0.760 | 0.7661 [0.643, 0.862] | 0.685 |
| SVM-RBF (gold eg2_768 / OOD z3_20) | 0.8042 [0.637, 0.879] | 0.735 | 0.7477 [0.622, 0.855] | 0.555 |
| *frozen* EG2 view (no fitting) | **0.8458** [0.715, 0.915] | 0.770 | 0.5891 | 0.517 |
| *frozen* two-view (no fitting) | 0.8336 [0.719, 0.913] | 0.795 | 0.5660 | 0.553 |

**Reading.** Non-linear capacity does **not** help in-domain: every fitted model
(GBT/MLP/SVM, and even a logistic on the 768-d EG2 embedding) scores *below* the
frozen EG2 view (0.846). OOD, the best capacity model (logistic on the 20 adapted Z3
features, 0.769) edges the prior adapted score (0.759) but the grouped CI is wide
([0.649, 0.873]). The binding constraint is not the model class.

## 2. Ensembling / stacking (A1.02)

Base scores are out-of-fold by construction (frozen OOF two-view; LOCO rules;
grouped-CV logistic on Z3-20). Rank-average = mean of per-domain rank01 good-scores;
stack = nested logistic meta-learner over the base scores.

| recipe | in-domain AUROC [CI] | in-domain BA | OOD AUROC [CI] | OOD BA |
|---|---|---|---|---|
| EG2 view (best single, in-domain) | **0.8458** [0.715, 0.915] | 0.770 | 0.5891 | 0.517 |
| Z3-20 logistic (best single, OOD) | 0.7620 | 0.695 | **0.7688** [0.649, 0.873] | 0.764 |
| rank pixel+eg2 | 0.8443 [0.725, 0.922] | 0.785 | 0.5634 | 0.526 |
| rank pixel+eg2+z3 | 0.8298 [0.696, 0.915] | 0.785 | 0.6989 [0.535, 0.832] | 0.658 |
| rank all 5 (pixel,eg2,frozen,rules,z3) | 0.8440 [0.722, 0.916] | 0.770 | 0.6207 | 0.605 |
| stack meta logistic (5 bases) | 0.8356 [0.741, 0.886] | 0.725 | 0.7082 [0.542, 0.845] | 0.711 |
| stack meta (pixel,eg2,z3) | 0.8196 [0.709, 0.903] | 0.770 | 0.7236 [0.558, 0.861] | 0.711 |

Paired factory-grouped bootstrap gains vs the frozen two-view:
- OOD `stack_meta(pixel,eg2,z3)` **+0.157 [+0.044, +0.280]**, 99.6% of resamples > 0.
- OOD `stack_meta(5)` **+0.141 [+0.025, +0.268]**, 99.1% > 0.
- In-domain `stack_meta(5)` **−0.001 [−0.065, +0.046]**, 56% > 0 (a tie).

**Reading.** Stacking helps **OOD** (it pulls the adapted Z3 view up and down-weights
the non-transferring pixel view) but is a **tie in-domain**: no combination beats the
single frozen EG2 view, and the in-domain meta-stack CI straddles 0. **0.95 is not
cleared in either domain**; the in-domain best is 0.846 and the OOD best 0.769.

## 3. Regime-specific models + routing (A1.03)

E4-descriptor regimes; `route_aspect_wide` = adapted when width/height ≥ 1.3 else
rules; `route_quality_desc` = rules on dark/blur/clipped windows else adapted;
`per_regime_heads` = separate logistic heads for the quality-looking vs normal regime.

| policy | in-domain AUROC [CI] | OOD AUROC [CI] |
|---|---|---|
| route_aspect_wide | 0.8222 [0.713, 0.855] | 0.7688 [0.649, 0.873] |
| route_quality_desc | 0.7589 [0.588, 0.862] | 0.6849 [0.572, 0.794] |
| per_regime_heads (quality) | 0.7420 [0.600, 0.826] | 0.7190 [0.588, 0.836] |

**Reading.** The aspect router simply picks the domain-appropriate single scorer
(IndEgo is 480×480 → rules; the OOD/108 windows are 16:9 → adapted) and therefore
**cannot exceed the best single scorer in each domain**. Routing by the *content*
quality descriptor is a negative result (0.759 in-domain / 0.685 OOD) and per-regime
heads are worse still. Aspect ratio is a dataset shortcut, not a semantic detector
(as flagged in N3).

## 4. Compute cost — so "compute-free" is honest

| term | cost |
|---|---|
| pixel-26 feature | ~0.05 s/window (CPU, cached) |
| physics-16 feature | ~0.02 s/window (CPU, cached) |
| Z3-20 feature (decode) | ~0.30 s/window (CPU) |
| **EG2 encode (GPU)** | **~7–9 s/window** (measured `latency_ms` in `ood_scores*.json`, 20 frames) |
| model training, grouped-CV logistic on any feature set | **0.03–0.11 s total** |
| model training, GBT/MLP/SVM sweep | seconds–minutes (still trivial vs EG2) |
| whole A1 run (all three experiments, CPU) | **364 s** |

**Reading.** Feature *extraction* is the only real compute, and EG2 dominates it by
~2 orders of magnitude. **Model capacity is effectively free**: the entire sweep plus
nested stacking trains in well under a minute. So "compute-free" is honest — and it
buys **+0.0 in-domain** and **+0.01 OOD AUROC** over the frozen views. The ceiling is
not compute.

## 5. Confirmatory hold-out 108 (scored once, never tuned)

⚠️ **Contamination flag.** All **44** factories in the 108 hold-out also occur in the
OOD-160 (different windows, same factories). Any recipe *trained on OOD-160* is
therefore factory-leaked and **is not a clean generalisation number**. Those rows are
marked ✗ and excluded from the verdict.

| recipe | clean? | AUROC | grouped CI | BA |
|---|---|---|---|---|
| **rank pixel+eg2+z3-six** | ✓ | **0.7568** | [0.613, 0.873] | 0.720 |
| rank pixel+eg2 | ✓ | 0.7337 | [0.591, 0.845] | 0.683 |
| EG2 view (frozen) | ✓ | 0.7334 | [0.565, 0.874] | 0.647 |
| frozen two-view (prior C1 recipe) | ✓ | 0.6895 | [0.534, 0.823] | 0.610 |
| Z3-six signed z-score (O4 recipe a) | ✓ | 0.6746 | [0.523, 0.822] | 0.647 |
| pixel view (frozen) | ✓ | 0.6247 | [0.474, 0.765] | 0.647 |
| rules (frozen table) | ✓ | 0.5061 | [0.316, 0.683] | 0.523 |
| Z3-20 logistic trained on OOD-160 | ✗ leaked | 0.7942 | [0.673, 0.894] | 0.720 |
| stack meta trained on OOD-160 | ✗ leaked | 0.7154 | [0.570, 0.847] | 0.720 |
| all-42 logistic trained on OOD-160 | ✗ leaked | 0.6590 | [0.442, 0.863] | 0.610 |

**Reading.** The best **clean** hold-out recipe is a rank-average of the two frozen
views plus the IndEgo-signed Z3-six score: **AUROC 0.757 [0.613, 0.873]** — above the
pre-registered frozen two-view (0.690) but with heavily overlapping CIs. The
apparently strong 0.794 from an OOD-trained logistic is **factory leakage** and must
not be reported as transfer. This is a *descriptive* menu read once; selecting its max
is itself a mild selection effect, so the honest claim is "≈0.69–0.76, CI overlapping".

## Verdict

| quantity | value | gap to 0.95 |
|---|---|---|
| **best in-domain AUROC** | **0.8458** (frozen EG2 view; BA 0.770) | **−0.104** |
| **best OOD AUROC** | **0.7688** (Z3-20 logistic, grouped CV) | **−0.181** |
| best *clean* hold-out AUROC | 0.7568 (rank pixel+eg2+z3six) | −0.193 |

**Best achievable accuracy today (compute-free): in-domain AUROC ≈ 0.85, OOD AUROC
≈ 0.77.** The gap to 0.95 is ≈ 0.10 in-domain and ≈ 0.18 OOD.

**Why — data/labels and domain shift, not model capacity.**
1. **Capacity is not the constraint.** GBT, MLP and SVM all score *below* the frozen
   EG2 view in-domain and below a simple logistic in OOD. Adding parameters does not
   move the ceiling; the whole sweep trains in <1 min.
2. **In-domain is limited by the semantic sub-problem and sample size.** N3.01 showed
   quality-bad windows are near-separable (rules AUROC 0.957) while semantic-bad
   (NO_ACTIVE_WORK / HANDS_NOT_VISIBLE, 37/50 bad windows) sits at 0.77–0.82. With
   n=150 from 7 clips (correlated windows) the CI is ±0.10, so even a real +0.02 gain
   is unmeasurable.
3. **OOD is limited by domain shift + a tiny bad count.** The frozen IndEgo views
   invert on Egocentric-10K (0.57); only capture-telemetry-style features transfer
   (Z3 0.77). With n_bad=22 over 47 factories the grouped CI is ±0.11. No amount of
   ensembling manufactures signal that is not in the labels.
4. **Ensembling is a tie, not a win.** Rank-averaging and nested stacking are within
   CI of the best single scorer in-domain and give only a modest, CI-overlapping OOD
   improvement.

### Honesty caveats
- The 108 hold-out shares all factories with OOD-160 → OOD-trained recipes are leaked
  and excluded; only IndEgo-trained/frozen recipes are clean.
- OOD and 108 CIs are wide (n_bad = 22 and 16); differences of <0.1 AUROC are not
  resolvable.
- The 108 menu was read once, but reporting its max is a mild selection effect.
- Feature-extraction costs are order-of-magnitude from cached `latency_ms`, not a
  fresh benchmark; no GPU job was run for A1 (all features cached).
- Rules use a fold-local table on gold and the frozen table on OOD/108; no labels edited.
