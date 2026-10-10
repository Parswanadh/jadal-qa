# F30 — Per-pixel feature vector 26 → 30: accuracy gain

**Branch:** `m/feat30` (base `pod/base`) · **Experiment:** `F30.01`, `F30.02`
**Results:** `eval/results/exp_feat30/F30.01.json`, `eval/results/exp_feat30/F30.02_cost.json`
**Figure:** `docs/reports/images/exp_feat30.png` · **Code:** `jadal/gpu/features30.py`,
`eval/exp_feat30/run_feat30.py`, `eval/exp_feat30/measure_cost.py` · **Tests:** `tests/test_feat30.py` (5 pass)
**Device:** CPU only (numpy / sklearn / scipy; cached features, no decode for the eval).

## Question

Does expanding the 26-feature E5 per-pixel vector with 4 more per-pixel features
improve accuracy — in-domain (LOCO) and out-of-domain (grouped CI) — at what cost?

## The 4 added features

The four decode-based **Z3 transferable invariants** (the per-pixel members of the
Z3 six-feature set; the other two, `ct_fsize_std` and `mo_hand_motion_excess`, are
container / multi-frame and not per-pixel). Definitions in `jadal/gpu/features30.py`,
byte-compatible with the Z3 extractor `eval/exp_features/extract.py::_invariants_step`
(reproduced to ≤1.6e-2 relative on sampled OOD windows):

| feature | definition | cost |
|---|---|---|
| `inv_clip_frac` | fraction of luma pixels at the exposure rails (≤2/255 or ≥253/255), mean over frames | one O(HW) compare / frame |
| `inv_laplacian_var` | variance of the 3×3 Laplacian of luma (focus/blur), mean over frames | one O(HW) Laplacian / frame |
| `inv_hist_entropy` | Shannon entropy (nats) of the 32-bin luma histogram, mean over frames | one O(HW) histogram / frame |
| `inv_dct_hf_lf` | DCT high-band (r>20) / low-band (r<8) energy ratio of 64×64 luma, mean over frames | one 64×64 DCT / frame |

**Cost (F30.02, 16 OOD windows, 160×90, ~10 fps, median):** decode 637 ms · 26 feats
229 ms · **added 4 = 57.4 ms/window (25.1 % of the 26-feature cost)** · 30 feats total 284 ms.

## Headline

Baselines reproduce **exactly** on this harness: 26-feature in-domain LOCO **0.8100**,
frozen OOD ensemble **0.5660**, adapted pooled-160 (`stable8+z3six`) **0.7810**,
confirmatory hold-out **0.6760**.

| recipe | 26 feat | **30 feat** | Δ(30−26) |
|---|---|---|---|
| **in-domain LOCO** (C=0.3 bal., 7 clips, n=150) | 0.8100 [0.683, 0.892] | **0.8118** [0.687, 0.895] | **+0.0018** (paired clip +0.0023 [−0.005, +0.009], 74 % > 0) |
| in-domain BA | 0.7400 | **0.7550** | +0.0150 |
| **OOD-160 adapted per-domain z-score** (full vector) | 0.3864 [0.231, 0.525] | **0.4954** [0.331, 0.625] | **+0.1090** (paired grouped +0.1102 [0.027, 0.199], 99.4 % > 0) |
| OOD BA | 0.4127 | **0.4509** | +0.0382 |
| **OOD-160 frozen logistic** (fit all IndEgo) | 0.5596 [0.425, 0.694] | 0.5590 [0.422, 0.698] | −0.0006 (ns) |
| **hold-out 108 adapted z-score** (scored once) | 0.3580 [0.211, 0.505] | **0.3995** [0.234, 0.566] | **+0.0415** |
| hold-out 108 frozen logistic | 0.6094 [0.457, 0.750] | 0.5611 [0.411, 0.717] | −0.0483 |

**Read:** the 4 features change the in-domain head almost not at all (+0.0018 AUROC,
+0.015 BA, ns) and the frozen OOD head not at all; the **adapted per-domain z-score
recipe on the full vector gains +0.109 AUROC OOD (+0.042 on the hold-out)**, but the
full 26/30 vector is a weak transfer recipe (0.50 OOD) because the non-transferable
features dominate the equal-weight mean.

## Per-feature ablation

**In-domain LOCO — add-one gain vs 26** (all within noise, |Δ| ≤ 0.005):
`inv_clip_frac` +0.0036 · `inv_hist_entropy` +0.0032 · `inv_dct_hf_lf` −0.0016 ·
`inv_laplacian_var` −0.0048.

**OOD adapted z-score — add-one gain vs 26** (each feature helps):
`inv_dct_hf_lf` **+0.0322** · `inv_laplacian_var` +0.0233 · `inv_hist_entropy` +0.0194 ·
`inv_clip_frac` +0.0098. Drop-one losses from the 30-vector: `dct_hf_lf` 0.048,
`laplacian_var` 0.033, `hist_entropy` 0.033, `clip_frac` 0.026.

**Per-feature transfer (IndEgo / OOD-160 / hold-out AUROC):**

| feature | IndEgo | OOD-160 [grouped CI] | hold-out 108 | sign |
|---|---|---|---|---|
| `inv_clip_frac` | 0.656 | 0.661 [0.451, 0.859] | 0.586 | stable |
| `inv_laplacian_var` | 0.641 | 0.718 [0.566, 0.852] | 0.630 | stable |
| `inv_hist_entropy` | 0.534 | 0.714 [0.558, 0.846] | 0.539 | stable |
| `inv_dct_hf_lf` | 0.310 | 0.244 [0.117, 0.389] | 0.330 | stable (both low) |

## The per-pixel subset is enough

Because the 4 added invariants *are* the per-pixel part of the curated `stable8+z3six`
recipe (0.781 / 0.676), the interesting question is whether the per-pixel features alone
suffice. Scored with the same adapted z-score recipe:

| per-pixel subset | OOD-160 [grouped CI] | hold-out 108 [grouped CI] | BA OOD / hold-out |
|---|---|---|---|
| 26 full vector | 0.386 [0.231, 0.525] | 0.358 [0.211, 0.505] | 0.413 / 0.379 |
| **30 full vector** | 0.495 [0.331, 0.625] | 0.400 [0.234, 0.566] | 0.451 / 0.438 |
| `stable8 + new4` (12) | 0.786 [0.646, 0.895] | 0.660 [0.506, 0.815] | 0.695 / 0.601 |
| **`new4` only (4)** | **0.793 [0.639, 0.903]** | **0.692 [0.543, 0.827]** | **0.704 / 0.624** |
| baseline `stable8+z3six` (14, curated) | 0.781 [0.634, 0.895] | 0.676 [0.525, 0.827] | 0.733 / 0.648 |

The **4 per-pixel invariants alone match / slightly beat the curated 14-feature
baseline** on both OOD (0.793 vs 0.781) and the hold-out (0.692 vs 0.676), with all
per-pixel features and no container telemetry. The differences are inside the grouped
CIs (n_bad = 22 / 16).

## Verdict

**maybe.** Adding the 4 invariants to the 26-feature vector is free in-domain
(+0.0018 AUROC, ns) and improves the adapted OOD z-score by **+0.109 AUROC**
(99.4 % > 0) / **+0.042** on the hold-out, but does **not** beat the frozen ensemble
or the curated adapted baseline when using the full vector. The 4 features carry all
the transferable signal by themselves (`new4` only ≈ curated 14). Cost is small
(+57 ms/window, +25 % over the 26 features). Recommend keeping the 4 features in the
per-pixel extractor.

## What is NOT established

* OOD n_bad = 22 (160) and 16 (hold-out) over ~1 window/factory, single labeller; the
  factory-grouped CI is honest but wide, and the **hold-out CIs include 0.5**.
* The per-feature **signs are fit on IndEgo gold** (mild optimism), as in the Z3/N1
  baselines.
* The 4 features are *already* in the Z3 `six` set; the "gain" for OOD is a gain of the
  full-vector recipe, not a new signal class. The `new4`-only arm was defined a priori
  (the Z3 invariant family), not selected on the hold-out.
* No threshold tuning anywhere; the hold-out was scored **once** in the same pass.

## Reproduce

```bash
export JADAL_BUNDLE=/workspace/jadal_pod_bundle
OMP_NUM_THREADS=2 python3 eval/exp_feat30/run_feat30.py
OMP_NUM_THREADS=2 python3 eval/exp_feat30/measure_cost.py
OMP_NUM_THREADS=2 python3 eval/exp_feat30/make_figure.py
python3 -m pytest tests/test_feat30.py -q
```

Inputs (read-only caches): `caches/gpu/features.json` (gold 26) +
`/workspace/wt/m-adapt160/.../z3_160.npz` (gold/OOD-160 invariants) +
`/workspace/wt/m-fewshot160/.../ood160_pixel.npz` (OOD-160 26) +
`/workspace/wt/m-confirm/.../confirm108_features.npz` (hold-out 26 + invariants) +
`caches/ensemble/ood_scores{,2}.json` (frozen).
