# Z1 — OOD error reduction (stable features, alignment, per-clip norm, few-shot)

**Pod:** jadal-qa, claim **Z1**, owner **monitor**. Branch `m/domain-adapt` (base `pod/base`).
**Registry:** `experiments/Z1.01`–`Z1.03`; **results:** `eval/results/exp_adapt/Z1.01..03.json`;
**figure:** `docs/experiments/exp_adapt/images/z1_ood_error_reduction.png`.
**Device:** CPU only (numpy/sklearn/scipy); GPU not used. Code: `eval/exp_adapt/`.

The task was to **cut the out-of-domain error** of the frozen PoC, which scores near
chance on the human-labelled Egocentric-10K windows (frozen ensemble `ensemble_v1`
**0.415 AUROC [0.201, 0.656]**, E5.00; IndEgo-trained pixel head **0.534 / 0.379**).
Every OOD number here is **exploratory**: the OOD set has **n_bad = 9**, about
**one window per factory**, one labeller, and the window-bootstrap CI is
**optimistic**. We never fit on OOD labels without marking it **post-hoc**.

## TL;DR

| method | IndEgo AUROC (LOCO) | OOD AUROC [95% CI] | verdict | note |
|---|---|---|---|---|
| frozen `ensemble_v1` (PoC prior) | 0.85 (baseline) | **0.415** [0.201, 0.656] | cited | no fitting |
| raw · all-26 · OOD **10fps** | 0.774 | 0.534 [0.315, 0.743] | baseline | near chance |
| raw · all-26 · OOD native | 0.774 | 0.379 [0.185, 0.598] | baseline | worse native |
| raw · **stable-8** · 10fps | 0.519 | 0.551 [0.340, 0.745] | weak | light feats drag it |
| **raw · sharpness-3 · 10fps** | 0.578 | **0.706 [0.518, 0.867]** | **best zero-shot** | no labels |
| raw · EG2-768 window means | 0.449 | 0.480 [0.451, 0.500] | drop | chance |
| domain_std · 10fps | 0.774 | 0.510 [0.303, 0.714] | drop | no gain |
| coral_t2s · 10fps | 0.774 | 0.547 [0.338, 0.757] | drop | ≈ raw |
| stable8 + coral_s2t · 10fps | 0.645 | 0.678 [0.497, 0.838] | maybe | 2nd zero-shot |
| coral_s2t → Ego10K corpus | 0.575 | 0.368 (10fps) | drop | fps confound |
| per-clip z · 10fps | 0.588 | 0.283 [0.113, 0.466] | drop | hurts |
| per-clip rank · 10fps | 0.526 | 0.512 [0.361, 0.673] | drop | chance |
| few-shot pixel-26, **k = 12** | post-hoc | **0.717** [native] / 0.712 | maybe | ±0.10 |
| few-shot pixel-26, **k = 24** | post-hoc | **0.763** / 0.757 | maybe | ±0.09 |
| self-training (Ego10K corpus) | — | 0.51–0.61 | drop | no real gain |

Full 22-row Z1.02 table and all Z1.01/Z1.03 arms are in the results JSON.

## Protocol

* **In-domain (IndEgo)** = leave-one-clip-out (LOCO) over the 7 clips; C chosen by
  inner LOCO on the training clips only; clip-bootstrap CI. Human labels
  (`labels/manifest_gold.json`, 150 windows, 100 good / 50 bad, single labeller).
* **Out-of-domain (OOD-60)** = 60 human-labelled Egocentric-10K windows
  (`labels/ood_labels.json`, 51 good / 9 bad, 47 factories, single labeller). A head
  is fit on IndEgo and scored on OOD-60; window-bootstrap CI (**optimistic**).
* Two OOD feature views: **native (~30 fps)** and **10fps-effective** (matches
  IndEgo's ~10 fps; the E5.03 primary view). Zero-shot methods are reported on both.
* Features = the 26 per-pixel features (E5 recipe). "sign-stable" = IndEgo sign ==
  OOD-10fps sign (E5.03): 8 stable / 14 flip / 4 chance.

## Z1.01 — Stable-feature selection

The E5.03 finding is that motion/temporal-band features carry the most in-domain
weight but **invert** out-of-domain, while the spatial-spectrum ("sharpness") group is
sign-stable. Restricting the head accordingly:

* **all-26** transfer is near chance (0.534 at 10 fps, 0.379 native).
* **stable-8** (sharpness + light) only reaches 0.551 — the light features
  (`luma_p05`, `global_shift`, `illum_r`, `zncc_self`) are individually weak and drag
  the head down.
* **sharpness-3** (`radial_slope`, `hf_ratio`, `dct_energy`) reaches **0.706
  [0.518, 0.867]** — a large cut in OOD error (AUROC 0.534 → 0.706, i.e. error
  0.47 → 0.29). The cost is in-domain: LOCO drops 0.774 → 0.578.
* The dense **EG2-768** head is at chance out-of-domain (0.480) — the embedding
  direction does not transfer either.

This is the headline: **a 3-feature sharpness head, trained on IndEgo and applied
zero-shot, is the best label-free OOD pipeline we found.**

## Z1.02 — Unsupervised alignment + per-clip normalisation (no labels)

All alignment transforms use only unlabelled target features; no OOD label is touched.

* **per-domain standardisation** 0.510 (10fps) / 0.499 (native) — no gain.
* **per-domain ZCA whitening** 0.371 / 0.330 — **worse** (whitening amplifies
  near-constant features).
* **CORAL t→s** 0.547 / **s→t** 0.519 — at best ≈ raw.
* **per-clip z** on IndEgo with a global OOD z: 0.283 — **worse** (removing
  within-clip variance destroys the usable sharpness signal).
* **per-clip rank**: 0.512 — chance. As flagged, per-factory norm on OOD is degenerate
  (~1 window/factory), so OOD is normalised globally; the concept cannot be fairly
  tested out-of-domain on this set.
* **stable-8 + coral_s2t** = **0.678 [0.497, 0.838]** — the best alignment arm, but
  still below sharpness-3.
* **CORAL onto the unlabelled Ego10K E8 corpus** (1546 windows, 47+ factories, native
  30 fps): 0.366–0.512 — **no gain**; the corpus is at native ~30 fps while IndEgo is
  ~10 fps, so this arm is confounded (not a clean test of alignment).

**Negative result: no unsupervised alignment or per-clip normalisation beat raw
features out-of-domain.**

## Z1.03 — Few-shot curve and self-training

**Few-shot (POST-HOC — fits on OOD labels).** Factory-grouped: whole factories are
either train or test (never both); k labelled windows; C by inner grouped CV.

| k labels | pixel-26 native | pixel-26 10fps | min n_bad in a train fold |
|---|---|---|---|
| 5 | 0.655 ± 0.147 | 0.649 ± 0.152 | 1 |
| 12 | 0.717 ± 0.099 | 0.712 ± 0.099 | 1 |
| 24 | **0.763 ± 0.088** | 0.757 ± 0.099 | 1 |

(Reproduced E5.02's ~0.71 at 12 and ~0.76 at 24.) **~12 labels already match the
label-free sharpness-3 head (0.706); ~24 are needed to clearly exceed it.** The
sharpness-3 few-shot head is *worse* (0.626 at k=24) — with labels the model can
exploit the OOD-correct signs of the flip features, which 3 features cannot express.

**Self-training (honest).** Pseudo-label high-confidence unlabelled Ego10K windows
(E8 corpus, n=1546), add once with weight 0.5, retrain, re-evaluate on OOD-60:

* raw: 0.399 (native) / 0.514 (10fps) vs base 0.379 / 0.534 — **no gain**.
* domain_std-aligned: 0.488 (native) / 0.608 (10fps) vs base 0.477 / 0.590 — at most
  **+0.01–0.02, well inside the CI**. Pseudo-labels are ~1520/1546 "confident", i.e.
  the head is not confident enough to select a useful subset; the corpus is also at a
  different fps.

**Blend:** 0.5·(pixel-26 few-shot + sharpness-3 zero-shot) gave 0.724 (k=12) / 0.740
(k=24) — **did not beat few-shot alone** (0.717 / 0.763).

## Recommendation

1. **Label-free OOD view — adopt:** a logistic head on the **3 sharpness features**
   (`radial_slope`, `hf_ratio`, `dct_energy`), trained on IndEgo, applied zero-shot.
   Measured OOD **AUROC 0.706 [0.518, 0.867]** vs frozen **0.415** and all-26 **0.534**.
   Keep the full-26 head for in-domain (LOCO 0.774).
2. **When human labels are available — few-shot:** a factory-grouped logistic on the
   **26 pixel features**; **≈12 labels → 0.72**, **≈24 labels → 0.76** (post-hoc).
3. **Do not ship:** unsupervised CORAL/whitening/per-clip normalisation (no gain or
   worse), EG2-768 transfer (chance), and self-training (no real gain).

## What is NOT established

* **The sharpness-3 OOD gain is not firmly established.** The CI lower bound is
  **0.518** with **n_bad = 9** and ~1 window/factory; the window-bootstrap CI is
  optimistic. This is exploratory, not a transfer rate.
* The **IndEgo cost is real**: LOCO falls 0.774 → 0.578 when only sharpness features
  are used. No method reached the in-domain 0.85.
* **No unsupervised alignment helped**, and the Ego10K-corpus arm is confounded by fps
  (IndEgo ~10 fps vs corpus native ~30 fps); a clean re-encode of the corpus at 10 fps
  would be needed to test alignment fairly.
* **Per-clip/factory normalisation could not be fairly evaluated out-of-domain**:
  OOD-60 has ~1 window/factory, so the per-clip concept collapses.
* **Few-shot numbers are post-hoc** and near leave-one-out (≈1 window/factory); the
  min n_bad in any training fold is **1**, so k = 5–12 sit close to a single example.
* Self-training was tested only with one round and the E8 corpus; other thresholds /
  rounds were not explored. The measured effect is within noise.
* The E8 `windows_rescored.jsonl` corpus has **no human labels**; it was used only
  unlabelled.

## Reproduce

```bash
export JADAL_BUNDLE=/workspace/jadal_pod_bundle
export JADAL_E5_INPUTS=/workspace/wt/exp-domain-adapt/eval/results/exp_domain/inputs
OMP_NUM_THREADS=2 python3 eval/exp_adapt/run_adapt.py
```

Inputs, method definitions and the data-source table are in `eval/exp_adapt/README.md`.
