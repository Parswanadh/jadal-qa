# Independent re-run of GRID.01 / GRID.02 (and GRID.03, GRID.00's baseline)

Stream **S5** verification duty, 2026-10-18. Registry: **`TASK.07`**. Results:
`eval/results/task/TASK.07.json`. Code: `tools/task/verify_grid01.py`. Claim source: `dsh/grid-feats`
(`eval/results/exp_grid/GRID.01.json`, `GRID.02.json`, `GRID.03.json`). **Verdict: reproduced exactly.**

This page is linked from `docs/reports/task_completion.md`. Mine is always the first number in every
pair, theirs the second.

## 1. What "independent" means here

Written by S5, which did not build the mosaic features and does not own the GRID track.

* `eval/exp_grid/run_grid.py` is **not imported**.
* `tools/dsh/common.py` is **not used** for AUROC, the LOCO probabilities, the clip bootstrap or the
  paired bootstrap — all four are re-implemented in `verify_grid01.py` from the protocol description.
  The re-implementation is cross-checked against `sklearn.metrics.roc_auc_score`: max absolute
  difference **2.2204e-16** over the three headline scores, so the AUROC arithmetic is not a place an
  error could hide.
* `common.load_gold()` **is** imported, but only so that my own derivation (below) can be asserted
  equal to it — a bug in that loader cannot hide behind this verification.
* The claim's own config, its pre-registration record and the feature artifact are checked, not trusted.

## 2. Data integrity before any number is computed

| Check | Result |
|---|---|
| My own gold derivation vs `common.load_gold()` (ids, y, clip, 26 features) | **all equal** (feature columns are alphabetically ordered here and canonically ordered there; the column-map check passes and a column permutation cannot change a scaler+LR fit) |
| Cache's **own** `label`/`clip` fields vs the manifest, all 150 windows | **0 disagreements**; the cache holds 210 windows, 60 of them the synthetic degradations that are not part of the gold set |
| Mosaic artifact re-hashed with a re-implementation of `jadal/mosaic/features.py::feature_digest` | **150/150 per-window digests match**, top-level digest recomputes to the declared `600e6eb4…7315`, `feature_version` = `mosaic-features-1.0.0`, 79 unique feature names, no duplicates |

So the vectors being verified are the ones the declared feature version produces from the declared
layout, and the labels are the ones in the bundle — not a stale copy.

## 3. The load-bearing numbers: mine first, theirs second

| Arm | Features | My AUROC [clip CI] | Their AUROC [clip CI] | Agree? |
|---|---|---|---|---|
| GRID.00 baseline (`GRID.00.json`) | 26 shipped | **0.8100 [0.683, 0.892]** | 0.8100 [0.683, 0.892] | exact |
| GRID.01 primary | `spatial` pooled (60) | **0.6864 [0.522, 0.772]** | 0.6864 [0.522, 0.772] | exact |
| GRID.01 secondary | `spatial` mean-only (15) | **0.7122 [0.504, 0.788]** | 0.7122 [0.504, 0.788] | exact |
| GRID.02 primary | `spatial+intertile` (72) | **0.7058 [0.514, 0.785]** | 0.7058 [0.514, 0.785] | exact |
| GRID.03 primary | `+diff` (79) | **0.7002 [0.503, 0.778]** | 0.7002 [0.503, 0.778] | exact |

Paired clip bootstraps (2000 draws, seed 0; the A-vs-B difference on the **same** resampled clips):

| Comparison | Mine | Theirs | Agree? |
|---|---|---|---|
| GRID.01 `spatial` − baseline 26 | **−0.12360 [−0.20164, −0.07104]** excludes 0 | −0.1236 [−0.20164, −0.07104] excludes 0 | exact |
| GRID.02 `spatial+intertile` − `spatial` | **+0.01940 [−0.01803, +0.05316]** includes 0 | +0.01940 [−0.018027, +0.053160] includes 0 | exact |
| GRID.03 `+diff` − `spatial+intertile` | **−0.00560 [−0.01615, +0.00168]** | −0.00560 [−0.016153, +0.001677] | exact |
| GRID.01 mean-only − pooled | **+0.02580 [−0.07223, +0.09269]** (see §5.1 for their sign) | −0.02580 [−0.09269, +0.07223] | same magnitude |

Every interval also survives a different RNG stream: recomputing the clip bootstrap with **10 000
draws at seed 12345** gives 0.810 [0.679, 0.892], 0.6864 [0.517, 0.770], 0.7058 [0.504, 0.786] and a
paired delta interval of [−0.01908, +0.05276]. The intervals are not bootstrap-noise artefacts.

## 4. Does the kill gate survive?

**Yes.** The pre-registered rule is "stop the mosaic-as-decision-input line if the intertile group
adds < 0.02 AUROC over spatial-only".

* My delta: **+0.019400**, against a threshold of 0.02 → **FAIL** (margin **0.0006 AUROC**).
* Claimed delta: +0.0194 → FAIL. Same decision.
* `run_grid.py` evaluates the gate on AUROCs already rounded to 4 dp; recomputed on raw values the
  outcome is **identical**, so the rounding is not load-bearing here.
* The paired interval **[−0.018, +0.053] includes zero**, so the stop does not rest on the arbitrary
  0.02 bar alone: the intertile group's added value is not distinguishable from nothing at this n.

The gate margin is genuinely thin (0.0006), and that is worth stating plainly rather than presenting
0.0194 < 0.02 as comfortable. The *conclusion* is better supported than the *gate*: three independent
signals agree that the mosaic adds nothing as a decision input — the point estimate misses the bar, the
paired interval includes zero, and a 15-feature mean-only capacity control (0.7122) already matches or
beats both richer mosaic variants (§5.3).

## 5. Findings beyond the reproduced numbers

### 5.1 A sign convention in `GRID.01.json` that reads backwards

`secondary_mean_pooling.delta_vs_primary` records **−0.0258**. That value is
`AUROC(primary 60) − AUROC(secondary 15)` = 0.6864 − 0.7122, so a *negative* number means the primary
is *worse*. Read as "the secondary's delta against the primary" it would be **+0.0258**, and a reader
taking the field name at face value would state the control's result backwards.

`docs/reports/grid_pipeline.md` reads it correctly ("the mean-pooling capacity control (15 features)
is 0.026 *better* than the 60-feature pooled version"), so **no conclusion is affected** — this is a
field-naming hazard, not an arithmetic error. Flagged because the same phrase would be wrong in a
memo or a slide.

### 5.2 Per-clip AUROC: the pooled number hides a below-chance clip

| Clip | n | good/bad | baseline 26 | `spatial` 60 | `spatial+intertile` 72 |
|---|---|---|---|---|---|
| `desk` | 29 | 16/13 | 0.8029 | 0.6058 | 0.6587 |
| `ecu_full` | 48 | 43/5 | 0.9488 | 0.7721 | 0.7535 |
| `ecu_late` | 14 | 10/4 | 0.6500 | 0.6500 | 0.6500 |
| `logistics4` | 17 | 14/3 | 0.9048 | 0.6905 | 0.7143 |
| `organise` | 19 | 9/10 | 0.9111 | 0.6444 | 0.6889 |
| `pallet` | 17 | 6/11 | 0.5000 | **0.1970** | **0.1667** |
| `pallet2` | 6 | 2/4 | 0.5000 | 0.5000 | 0.2500 |

The mosaic head is **worse than chance on `pallet`** (0.197 for `spatial`, 0.167 for
`spatial+intertile`) where the 26-feature baseline is at chance (0.500). `pallet` carries 11 of the
50 bad windows. Per-clip AUROCs at n = 6-48 are noisy and this is a description, not a new claim —
but the pooled 0.6864/0.7058 should not be read as "the mosaic is a bit weaker everywhere"; on at
least one clip it is actively inverted.

### 5.3 The capacity control is the strongest mosaic result in the file

`spatial` mean-only (15 features) = **0.7122**, above both the 60-feature pooled primary (0.6864) and
the 72-feature `spatial+intertile` (0.7058). Their own JSON records this (as a control, with the sign
issue of §5.1). It is a fourth reason the mosaic line stops: the extra pooling dimensions and the
whole intertile group are not earning their place.

## 6. Pre-registration: what is verifiable and what is not

| Check | GRID.01 | GRID.02 |
|---|---|---|
| PREREG config re-hashes to its declared `config_sha256` | **yes** (`557752c9…`) | **yes** (`29901019…`) |
| That hash equals the config hash in the results JSON | **yes** | **yes** |
| `frozen_before_run` | `true` (self-reported) | `true` (self-reported) |
| `frozen_utc` | `2026-10-18` (date only) | `2026-10-18` (date only) |
| Ordering independently verifiable from git | **no** | **no** |

PREREG.json and the results JSON were added in the **same commit** (`3444f35`), so git history cannot
show that the config was frozen before the run. The registry lines do show the runs registered at
06:38 and closed at 06:43 (minute resolution), and the GRID.02 registration text pre-states the
expected *direction* ("intertile proxies add LESS than 0.02 … the pre-registered kill-gate"), which is
a prediction that makes the gate harder to pass rather than easier — evidence of good faith, not proof
of ordering. Recorded as `ordering_independently_verifiable: false` in the results JSON rather than
papered over.

## 7. What this re-run does NOT establish

This reproduces the GRID arithmetic; it does not validate the GRID protocol. The same 150 in-domain
windows from **one** labeller, **7** clips, and a mosaic feature family designed by the stream that
then evaluated it. Exact agreement means the code, the committed vectors and the labels line up — it
says nothing about whether the 26-feature baseline's advantage would hold on fresh footage, on more
factories, or with a second labeller.

The out-of-domain side of this track was **not** re-run here. One of its numbers has nevertheless
been reproduced by a different route: GRID.06's frozen-26 baseline on the 160 OOD windows is
**0.5878 [0.450, 0.728]** (factory-grouped bootstrap), and `TASK.00(a)` — written separately, for the
`NO_ACTIVE_WORK` question, with its own loader and bootstrap code — measured the same quantity on the
same 160 windows as **0.5878 [0.4503, 0.7284]**. GRID.06's frozen *mosaic* head, 0.3557
[0.224, 0.515], was not re-run; it is below chance and is quoted here only as the claim.

## 8. The one honest sentence for this verification

> S5 recomputed GRID.01, GRID.02 and GRID.03 from the committed feature vectors with independently
> written metrics and reproduced every published number and interval exactly, and the pre-registered
> kill gate still fails — the intertile group adds +0.0194 AUROC (paired CI [−0.018, +0.053],
> includes zero) against a 0.02 bar — so the decision to stop the mosaic-as-decision-input line
> stands, on a margin of 0.0006 that is thin in itself but is backed by a capacity control that
> matches the richer mosaic variants and by one clip where the mosaic head is below chance.
