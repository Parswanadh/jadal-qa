# Z1 — OOD error reduction (stable features, alignment, per-clip norm, few-shot) — re-run on the 160 OOD labels

> **Supersedes the n=60 version (PR #121, branch `m/domain-adapt`).**
> This is a **NEW branch** (`m/domain-adapt-160`); the original PR branch is untouched.
> What changed: the out-of-domain set is now the merged **160** human-labelled
> Egocentric-10K windows (**138 good / 22 bad, 47 factories**; q1=60 + q2=100)
> instead of the old **60** (51 good / 9 bad), and the primary CI is the
> **factory-grouped** bootstrap (honest). Protocol is otherwise identical.

**Branch:** `m/domain-adapt-160` (base `pod/base`) · **Owner:** M5
**Results:** `eval/results/exp_domain_adapt160/Z1.01_160.json`, `Z1.02_160.json`, `Z1.03_160.json`
**Code:** `eval/exp_domain_adapt160/run_z1_160.py` (+ `adaptlib.py`) · **Tests:** `tests/test_exp_domain_adapt160.py`
**Data:** IndEgo = 150 human-gold windows / 7 clips (LOCO); OOD = 160 human windows,
138 good / 22 bad, 47 factories (`labels/ood160_labels.json`), 10 fps-effective view.
**Device:** CPU (numpy/sklearn); GPU not used.
**Reproduce:** `OMP_NUM_THREADS=2 python3 eval/exp_domain_adapt160/run_z1_160.py`

**View note.** The 10 fps-effective view is used for all 160 windows. The q2=100
source transcodes are natively 10 fps, so a native-30fps column is only defined for
the old q1=60 windows; it is **not recomputed** here (stated, not hidden). The
EG2-768 window-mean row is likewise **not recomputed** for q2 (needs a GPU EG2
embed); the old n=60 value (0.480) is unchanged and omitted from the 160 table.

---

## TL;DR (160 vs the old 60)

| method | 60-label OOD [window CI] | **160-label OOD [factory-grouped 95% CI]** | verdict |
|---|---|---|---|
| frozen `ensemble_v1` | 0.415 [0.201, 0.656] | **0.566** [0.445, 0.674] | cited (near chance) |
| raw · all-26 · 10fps | 0.534 [0.315, 0.743] | **0.586** [0.475, 0.710] | baseline |
| raw · stable-8 · 10fps | 0.551 [0.340, 0.745] | **0.596** [0.480, 0.723] | weak |
| **raw · sharpness-3 · 10fps** | **0.706** [0.518, 0.867] | **0.708** [0.588, 0.823] | **best zero-shot — HOLDS** |
| stable8 + coral_s2t · 10fps | 0.678 [0.497, 0.838] | **0.624** [0.496, 0.740] | 2nd zero-shot |
| per-clip z · 10fps | 0.283 | **0.364** [0.239, 0.463] | drop (hurts) |
| few-shot pixel-26, k=12 | 0.712 | **0.589** ± 0.165 | optimistic at 60 |
| few-shot pixel-26, k=24 | 0.757 | **0.630** ± 0.131 | optimistic at 60 |
| self-training (Ego10K corpus) | 0.51–0.61 | **0.547–0.575** | drop (no gain) |

## Z1.01 — Stable-feature selection (160)

Head fit on IndEgo (C by inner LOCO), scored zero-shot on OOD-160.

| method | IndEgo AUROC (LOCO) | OOD-160 AUROC | grouped 95% CI | window 95% CI (optimistic) | OOD BA |
|---|---|---|---|---|---|
| frozen `ensemble_v1` (no fitting) | 0.85 (baseline) | 0.566 | [0.445, 0.674] | [0.437, 0.694] | — |
| raw · all-26 | 0.774 | 0.586 | [0.475, 0.710] | [0.465, 0.703] | 0.525 |
| raw · stable-8 (sharpness/light) | 0.519 | 0.596 | [0.480, 0.723] | [0.477, 0.711] | 0.561 |
| **raw · sharpness-3** | 0.578 | **0.708** | **[0.588, 0.823]** | [0.602, 0.810] | 0.493¹ |

¹ uncalibrated logistic probability; ranking 0.708 but a fixed 0.5 threshold gives BA near
chance (no OOD tuning).

**The Z1 headline survives at n_bad=22:** the 3-feature sharpness head
(`radial_slope`, `hf_ratio`, `dct_energy`), trained on IndEgo and applied zero-shot,
reaches **0.708 with a grouped CI [0.588, 0.823] that excludes 0.5** — the only
zero-shot recipe here that does. The old 0.706 was not a small-sample fluke. The
frozen ensemble improves from 0.415 (noise at n_bad=9) to 0.566, still not separable
from chance.

## Z1.02 — Unsupervised alignment + per-clip normalisation (no labels)

| method | IndEgo AUROC | OOD-160 AUROC | grouped 95% CI |
|---|---|---|---|
| raw | 0.774 | 0.586 | [0.475, 0.710] |
| domain_std | 0.774 | 0.453 | [0.329, 0.582] |
| domain_whiten (ZCA) | 0.751 | 0.522 | [0.409, 0.638] |
| coral_s2t | 0.600 | 0.441 | [0.345, 0.529] |
| coral_t2s | 0.774 | 0.493 | [0.409, 0.580] |
| per-clip z (global target norm) | 0.588 | 0.364 | [0.239, 0.463] |
| per-clip rank (global target norm) | 0.526 | 0.461 | [0.358, 0.575] |
| stable-8 + domain_whiten | 0.585 | 0.576 | [0.468, 0.695] |
| stable-8 + coral_s2t | 0.563 | 0.624 | [0.496, 0.740] |

**Negative result reproduced at 160: no unsupervised alignment or per-clip
normalisation beats raw features out-of-domain.** The best alignment arm
(stable-8 + coral_s2t, 0.624) still has a grouped CI containing 0.5 and is below the
sharpness-3 head. Per-clip z remains the worst (0.364).

## Z1.03 — Few-shot curve and self-training (POST-HOC on OOD labels)

Factory-grouped: whole factories are train or test (never both); k labelled windows;
C by inner grouped CV.

| k labels | OOD-160 AUROC (mean ± std) | repeats | min n_bad in a train fold |
|---|---|---|---|
| 5 | 0.523 ± 0.170 | 30 | 1 |
| 12 | 0.589 ± 0.165 | 44 | 1 |
| 24 | 0.630 ± 0.131 | 56 | 1 |
| **48** | **0.664 ± 0.083** | 60 | 3 |
| 80 | 0.653 ± 0.091 | 60 | 4 |

The n=60 few-shot curve (k=12 → 0.712, k=24 → 0.757) was **optimistic**: at n_bad=22
the curve is lower and flatter (k=48 → 0.664). With 22 bad windows the estimate is
still unstable, but the "≈12 labels already match the label-free sharpness-3 head"
claim from the 60-label report **does not hold** — it now takes ≈48 labels to reach
the sharpness-3 level.

**Self-training** (pseudo-label high-confidence unlabelled Ego10K corpus windows,
E8 n=1546; add once with weight 0.5), evaluated on OOD-160: raw 0.575/0.574 vs base
0.586; domain_std 0.569/0.547 vs base 0.525 — **no gain** (all grouped CIs overlap
the base).

## Recommendation (updated)

1. **Label-free OOD view — keep:** the 3-sharpness-feature logistic head, IndEgo-trained,
   zero-shot. **OOD-160 AUROC 0.708 [0.588, 0.823]** vs frozen 0.566 and all-26 0.586.
   Keep the full-26 head for in-domain (LOCO 0.774).
2. **With human labels:** a factory-grouped logistic on the 26 pixel features;
   **≈48 labels → 0.66** (post-hoc) — more labels are needed than the 60-label run implied.
3. **Do not ship:** unsupervised CORAL/whitening/per-clip normalisation (no gain or
   worse), and self-training (no gain).

## What changed vs the n=60 version (PR #121)

1. **The sharpness-3 headline is confirmed** (0.706 → 0.708) with an honest
   factory-grouped CI that excludes 0.5 at n_bad=22.
2. **The frozen baseline is 0.566, not 0.415** (the old number was noise at n_bad=9).
3. **The few-shot curve is lower/flatter**; "12 labels suffice" is retracted.
4. Alignment / per-clip negatives and the self-training negative are reproduced.

## Caveats

- OOD = **160 windows, 22 bad, 47 factories (~1 window/factory), single labeller**;
  factory-grouped CI is honest but wide; every OOD number is exploratory.
- IndEgo = 150 windows / 7 clips / ~3 wearers, single labeller; clip-bootstrap CI.
- Native-30fps and EG2-768 views not recomputed for q2 (see view note).
- "Sign-stable" = IndEgo sign == OOD sign (E5.03), not a quality measure.
