# O4 OOD recipe — FROZEN (sha256)

**Status: FROZEN — frozen before confirmatory labels; no OOD fitting.**
Frozen on 2026-10-10. Owner: M4 (`m/recipe-freeze`, base `pod/base`).
Source of truth: `m/adapt160` (PR #131) and its dependencies `m/new-features` (PR #127, Z3)
and `m/fewshot160` (PR #130, E5). This document freezes the *recipe* so that the next
(confirmatory) OOD label batch cannot retro-change it.

> **frozen before confirmatory labels; no OOD fitting.**
> The only fitted quantity is the per-feature **sign**, taken from **IndEgo only**.
> The per-domain z-score is **unsupervised** (target mean/std). Nothing is threshold-tuned
> or fitted on OOD labels.

---

## 1. Frozen recipe components

### 1.1 Feature list

**Z3 transferable-6** (`m/new-features`, Z3.03; selection: stable ∧ min|AUROC−0.5| ≥ 0.03 in both domains):

| # | feature |
|---|---|
| 1 | `ct_fsize_std` |
| 2 | `mo_hand_motion_excess` |
| 3 | `inv_clip_frac` |
| 4 | `inv_laplacian_var` |
| 5 | `inv_hist_entropy` |
| 6 | `inv_dct_hf_lf` |

**E5 stable-8** (`m/fewshot160`, O2.03; sign-stable per-pixel features):

| # | feature |
|---|---|
| 1 | `radial_slope` |
| 2 | `hf_ratio` |
| 3 | `dct_energy` |
| 4 | `luma_p05` |
| 5 | `global_shift` |
| 6 | `illum_r` |
| 7 | `illum_b` |
| 8 | `zncc_self` |

The combined recipe uses the **14** features in the order `stable-8 ++ transferable-6`.

### 1.2 Sign source — **IndEgo only** (never OOD)

- **Rule:** `sign_j = +1 if AUROC_IndEgo(feature_j) >= 0.5 else -1` (label 1 = usable/good).
- **Fit on:** IndEgo human gold, **150 windows (100 good / 50 bad, 7 clips)**
  (`$JADAL_BUNDLE/caches/gpu/features.json` + `labels/manifest_gold.json`).
- **Never fit on OOD.** The sign vector is a pure IndEgo quantity.

Sign vector for the combined-14 (and the Z3-6 subset):

| feature | IndEgo AUROC | sign |
|---|---|---|
| `radial_slope` | 0.6334 | +1 |
| `hf_ratio` | 0.6182 | +1 |
| `dct_energy` | 0.6306 | +1 |
| `luma_p05` | 0.3924 | −1 |
| `global_shift` | 0.3761 | −1 |
| `illum_r` | 0.4898 | −1 |
| `illum_b` | 0.5278 | +1 |
| `zncc_self` | 0.5738 | +1 |
| `ct_fsize_std` | 0.6202 | +1 |
| `mo_hand_motion_excess` | 0.6492 | +1 |
| `inv_clip_frac` | 0.6564 | +1 |
| `inv_laplacian_var` | 0.6412 | +1 |
| `inv_hist_entropy` | 0.5342 | +1 |
| `inv_dct_hf_lf` | 0.3100 | −1 |

Z3-6 sign vector: `{ct_fsize_std:+1, mo_hand_motion_excess:+1, inv_clip_frac:+1, inv_laplacian_var:+1, inv_hist_entropy:+1, inv_dct_hf_lf:-1}`.

### 1.3 z-score rule (per-domain, unsupervised)

- Per feature, per domain: `z_j = (x_j − mean_domain(x_j)) / max(std_domain(x_j), 1e-9)`.
- Statistics: **mean and std**, computed on **each target domain's own windows**.
- **Unsupervised** — uses no labels. IndEgo uses IndEgo's mean/std; OOD uses the target
  sample's own mean/std. Pooled OOD-160 uses the 160's own mean/std; q1/q2 rows z-score
  each batch with its own stats.
- `std` floor `1e-9`.

### 1.4 Aggregation

- **Equal-weight mean** over the selected features:
  `score(x) = (1/n) · Σ_j sign_j · z_j(x)`, with uniform weight `1/n`.
- The score is **signed and symmetric around 0**.

### 1.5 Threshold

- **Value: `0.0`** for the z-score combos.
- **Applied as:** predict **usable/good** iff `score > 0.0`, else not-usable
  (`pred = (score > thr)`), used for balanced accuracy.
- **Fixed natural threshold — not calibrated, not tuned.** No OOD threshold tuning.
  (Probability scorers such as the frozen ensemble use `0.5`.)

---

## 2. Provenance

| item | value |
|---|---|
| source branch | `m/adapt160` |
| source PR | #131 — [O4] Confirm transferable-feature adaptation on the 160 OOD labels |
| source head sha | `526830c74e1aec7b66b813a3498a88a3af5d2bd8` |
| PR base | `pod/base` @ `f20013b677ab3e25b930d7ed23bac572dbb78134` |
| experiments | O4.01, O4.02 (registry: adopt) |
| source report | `docs/reports/exp_adapt160.md` |
| source results | `eval/results/exp_adapt160/O4.01.json`, `O4.02.json` |
| source code | `eval/exp_adapt160/run_adapt160.py`, `extract160.py`, `z3extract.py` |
| dependency (Z3) | `m/new-features` @ `148c9f965ba15f6fab0691057d454896bf794c22` (PR #127, Z3.01–03) |
| dependency (E5) | `m/fewshot160` @ `4b7f334b5089ca5f246468e3f686821d0a26c9f1` (PR #130, O2.01–03) |

### 2.1 Headline evaluation numbers (as measured by O4 on the 160 OOD labels)

| recipe | features | OOD AUROC | grouped 95% CI | OOD BA | IndEgo |
|---|---|---|---|---|---|
| a · Z3 transferable-6 | 6 | **0.759** | [0.597, 0.886] | 0.709 | 0.684 |
| b · E5 stable-8 + Z3 transferable-6 | 14 | **0.781** | [0.634, 0.895] | 0.733 | 0.736 |
| d · frozen `ensemble_v1` (`usable_prob`) | — | 0.566 | [0.424, 0.700] | 0.546 | — |

Paired factory-grouped gain vs frozen: (a) **+0.193 [0.044, 0.343]** (99.5% > 0);
(b) **+0.213 [0.060, 0.367]** (99.5% > 0). Data: 160 windows, 138 good / 22 bad, 47 factories
(q1=60 + q2=100), single labeller; factory-grouped bootstrap (n_boot=2000, seed=0).

**Unproven (unchanged by this freeze):** q2 alone (0.70/0.73) is not separable from chance
(grouped CI includes 0.5); 22 bad over 47 factories; BA at threshold 0 is a fixed natural
threshold, not a calibrated decision boundary; the review/error operating point of the
adapted score is not established.

---

## 3. ensemble_v1 integrity check across PR branches

Method: `git show origin/<branch>:configs/ensemble_v1.json | sha256sum` vs
`git show origin/main:configs/ensemble_v1.json | sha256sum`.

- `origin/main` sha256: `4da1068a78d5d15021f4e167d6aa24fd2e0bb08aa88f83c83c2f167cc086fe31`
- `origin/pod/base` sha256: `d62a95a473a31085444371f83003b4b0bd020bb0f2f5c13a379d20d9bded8533` (differs from main **only** by the laptop→pod path remap
  of `embed_model_dir` / `embed_command`; no model coefficients differ)

| PR | branch | sha256 | status vs main |
|---|---|---|---|
| #83 | `feat/exp-regimes-pod` | `4da1068a78d5…` | MATCH |
| #87 | `feat/exp-domain-adapt` | `d62a95a473a3…` | differs from main (= pod/base baseline) |
| #88 | `feat/exp-fps-pod` | `4da1068a78d5…` | MATCH |
| #89 | `feat/exp-motion-cue` | `4da1068a78d5…` | MATCH |
| #90 | `feat/exp-cost-model` | `4da1068a78d5…` | MATCH |
| #95 | `feat/exp-mosaic` | `4da1068a78d5…` | MATCH |
| #113 | `feat/exp-rescore` | `4da1068a78d5…` | MATCH |
| #114 | `m/skip` | `d62a95a473a3…` | differs from main (= pod/base baseline) |
| #115 | `m/docs-index` | `d62a95a473a3…` | differs from main (= pod/base baseline) |
| #116 | `m/review-gate` | `d62a95a473a3…` | differs from main (= pod/base baseline) |
| #117 | `m/res` | `d62a95a473a3…` | differs from main (= pod/base baseline) |
| #118 | `m/review-band` | `d62a95a473a3…` | differs from main (= pod/base baseline) |
| #119 | `m/mosaic-res` | `d62a95a473a3…` | differs from main (= pod/base baseline) |
| #120 | `m/review-score` | `d62a95a473a3…` | differs from main (= pod/base baseline) |
| #121 | `m/domain-adapt` | `d62a95a473a3…` | differs from main (= pod/base baseline) |
| #122 | `m/clip-sequential` | `d62a95a473a3…` | differs from main (= pod/base baseline) |
| #125 | `m/review160` | `d62a95a473a3…` | differs from main (= pod/base baseline) |
| #126 | `m/conformal` | `d62a95a473a3…` | differs from main (= pod/base baseline) |
| #127 | `m/new-features` | `d62a95a473a3…` | differs from main (= pod/base baseline) |
| #128 | `m/ood160-eval` | `d62a95a473a3…` | differs from main (= pod/base baseline) |
| #130 | `m/fewshot160` | `d62a95a473a3…` | differs from main (= pod/base baseline) |
| #131 | `m/adapt160` | `d62a95a473a3…` | differs from main (= pod/base baseline) |
| #137 | `m/calib160` | `d62a95a473a3…` | differs from main (= pod/base baseline) |
| #139 | `m/latency-full` | `4da1068a78d5…` | MATCH |

**Result: no listed branch modifies `configs/ensemble_v1.json` relative to its base.**
Every branch whose copy differs from `main` carries the identical **`pod/base` baseline**
(sha `d62a95a473…`), i.e. the inherited path remap — **not a branch edit**. No branch
deviates from both `main` and `pod/base`. The frozen ensemble config is untouched by all
listed branches.

---

## 4. Pod-path audit (env-vars rule)

Rule: tracked `*.py` / `*.sh` / `*.json` must not hardcode pod/laptop paths; they must read
`JADAL_BUNDLE`, `JADAL_DATA`, `JADAL_EXP_DIR` from the environment. Scanned paths:
`/workspace/jadal_pod_bundle`, `/workspace/`, `/home/parshu`. Excluded: `docs/`, `results/`,
`data/` (and the generated artifact `eval/gold/segments.json`). This is a **report only** —
no other branch was modified.

Summary: **124 files**, 304 unique file:line matches;
39 files already read an env var with a hardcoded fallback default (`env-fallback`),
85 files are **hardcoded** (need the fix). Env-var target counts (a file may need several):
`JADAL_DATA(legacy /home/parshu)` 67 · `JADAL_BUNDLE` 53 · `JADAL_EXP_DIR` 24 ·
`JADAL_DATA` 8 · model-cache 8 · other 5.

> **Special case — `configs/ensemble_v1.json`:** it contains `/home/parshu/...` and
> `/workspace/...` paths, but §3 shows it must **not** be edited on feature branches.
> Its paths are consumed via code (e.g. `eval/ensemble_score_windows.py`,
> `eval/frames10k/*`); fix the readers to use env vars, or remap in the base repo — do not
> edit the frozen config on a branch.
> **Special case — `eval/gold/segments.json`** (excluded here): a generated data artifact
> holding clip paths; regenerate from env, don't hand-edit.

Full list (file · line ranges · required env var · hardcoded? · branches):

| file | lines | env var | state | branches |
|---|---|---|---|---|
| `configs/ensemble_v1.json` | 11,2499-2500 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/bench_eg2.py` | 12,17,46 | JADAL_BUNDLE+JADAL_DATA(model cache)+JADAL_DATA | **hardcoded** | 24 branches |
| `eval/compute_gold_l0.py` | 22 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/edge_bench.py` | 314 | JADAL_BUNDLE+JADAL_DATA | **hardcoded** | 24 branches |
| `eval/ensemble/ood_heads.py` | 13 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/ensemble_determinism.py` | 14 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/ensemble_embed_ood.py` | 15 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/ensemble_nested_loco.py` | 38-40 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/ensemble_pixel_ood.py` | 14 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/ensemble_score_windows.py` | 4 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/ensemble_transfer_probe.py` | 24-25,95,109,119,121 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/exp_adapt/adaptlib.py` | 16,21-22 | JADAL_BUNDLE+JADAL_EXP_DIR | env-fallback | m/domain-adapt |
| `eval/exp_adapt160/extract160.py` | 36,40 | JADAL_BUNDLE+JADAL_EXP_DIR | env-fallback | m/adapt160 |
| `eval/exp_adapt160/extract_e8_sample.py` | 31-32 | JADAL_BUNDLE+JADAL_EXP_DIR | env-fallback | m/adapt160 |
| `eval/exp_adapt160/review_point.py` | 34 | JADAL_EXP_DIR | env-fallback | m/adapt160 |
| `eval/exp_adapt160/run_adapt160.py` | 41,46 | JADAL_BUNDLE+JADAL_EXP_DIR | env-fallback | m/adapt160 |
| `eval/exp_adapt160/z3extract.py` | 26 | JADAL_BUNDLE | env-fallback | m/adapt160 |
| `eval/exp_calib160/run_p1.py` | 33,36,39 | JADAL_BUNDLE+JADAL_EXP_DIR | env-fallback | m/calib160 |
| `eval/exp_clip/make_report.py` | 6,41 | JADAL_DATA/EXP_DIR | **hardcoded** | m/clip-sequential |
| `eval/exp_clip/run_clip.py` | 26,46,50 | JADAL_BUNDLE+JADAL_EXP_DIR | env-fallback | m/clip-sequential |
| `eval/exp_conformal/conformal_risk.py` | 34-35 | JADAL_BUNDLE+JADAL_EXP_DIR | env-fallback | m/conformal |
| `eval/exp_cost/run_e7.py` | 46,49,51,198,461-462,464 | JADAL_BUNDLE+JADAL_EXP_DIR+JADAL_DATA(model cache) | env-fallback | feat/exp-cost-model |
| `eval/exp_domain/e5lib.py` | 13 | JADAL_BUNDLE | env-fallback | feat/exp-domain-adapt |
| `eval/exp_domain/watch_e5_04.sh` | 10 | JADAL_BUNDLE | env-fallback | feat/exp-domain-adapt |
| `eval/exp_features/eflib.py` | 16 | JADAL_BUNDLE | env-fallback | m/new-features |
| `eval/exp_features/extract.py` | 26 | JADAL_BUNDLE | env-fallback | m/new-features |
| `eval/exp_fewshot/o2lib.py` | 24 | JADAL_BUNDLE | env-fallback | m/fewshot160 |
| `eval/exp_fewshot/prep_inputs.py` | 41 | JADAL_EXP_DIR | **hardcoded** | m/fewshot160 |
| `eval/exp_fps/fps_lib.py` | 9,12,16 | JADAL_BUNDLE+JADAL_DATA(legacy) | env-fallback | 24 branches |
| `eval/exp_fps/merge_e1_01.py` | 5,83 | JADAL_DATA(legacy) | **hardcoded** | 4 branches |
| `eval/exp_fps/merge_e1_04.py` | 11,97 | JADAL_DATA(legacy) | **hardcoded** | 4 branches |
| `eval/exp_fps/run_all.sh` | 6,9 | JADAL_BUNDLE | env-fallback | feat/exp-fps-pod |
| `eval/exp_fps/run_eg2.py` | 8,20 | JADAL_BUNDLE+JADAL_DATA(legacy) | **hardcoded** | 23 branches |
| `eval/exp_fps/run_fps.py` | 16,18 | JADAL_DATA(legacy) | **hardcoded** | 4 branches |
| `eval/exp_fps/run_ood.py` | 13,25-26 | JADAL_BUNDLE+JADAL_DATA(legacy) | **hardcoded** | 23 branches |
| `eval/exp_latency_full/analyze_full.py` | 22,24-26,268 | JADAL_BUNDLE+JADAL_EXP_DIR | env-fallback | m/latency-full |
| `eval/exp_latency_full/run_full.py` | 29,61,74 | JADAL_BUNDLE+JADAL_DATA(model cache) | env-fallback | m/latency-full |
| `eval/exp_mosaic/run_variants.py` | 44,47,50 | JADAL_BUNDLE+JADAL_EXP_DIR+JADAL_DATA(model cache) | env-fallback | feat/exp-mosaic |
| `eval/exp_mosaic_res/image_proofs.py` | 22 | JADAL_BUNDLE | env-fallback | m/mosaic-res |
| `eval/exp_mosaic_res/run_res.py` | 38,41,44 | JADAL_BUNDLE+JADAL_EXP_DIR+JADAL_DATA(model cache) | env-fallback | m/mosaic-res |
| `eval/exp_motion_cue/motion_cue.py` | 121 | JADAL_BUNDLE | env-fallback | feat/exp-motion-cue |
| `eval/exp_motion_cue/run_e6.py` | 20,41 | JADAL_BUNDLE | env-fallback | feat/exp-motion-cue |
| `eval/exp_ood160/run_ood160_eval.py` | 43 | JADAL_BUNDLE | env-fallback | m/ood160-eval |
| `eval/exp_regimes/analyze_regimes.py` | 15,87,93,101,108,124-125,148-149,156,160,215-216,223,227,305-306 | JADAL_BUNDLE+JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/exp_regimes/compute_descriptors.py` | 21,25,35,41,67 | JADAL_BUNDLE+JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/exp_regimes/make_results.py` | 12 | JADAL_BUNDLE | **hardcoded** | feat/exp-regimes-pod |
| `eval/exp_regimes/reencode_spec.py` | 20-21 | JADAL_BUNDLE | **hardcoded** | feat/exp-regimes-pod |
| `eval/exp_res/make_report.py` | 148-150 | JADAL_DATA/EXP_DIR | **hardcoded** | m/res |
| `eval/exp_res/plot_res.py` | 24,119 | JADAL_BUNDLE+JADAL_EXP_DIR | **hardcoded** | m/res |
| `eval/exp_res/run_ood_res.py` | 32 | JADAL_BUNDLE | env-fallback | m/res |
| `eval/exp_res/run_res.py` | 26,29 | JADAL_BUNDLE+JADAL_EXP_DIR | env-fallback | m/res |
| `eval/exp_rescore/contract.py` | 74 | JADAL_EXP_DIR | **hardcoded** | feat/exp-rescore,m/latency-full |
| `eval/exp_rescore/run_e8.py` | 17,46,51 | JADAL_BUNDLE+JADAL_EXP_DIR | env-fallback | feat/exp-rescore,m/latency-full |
| `eval/exp_rescore/write_results.py` | 4,25,34-35,37,91,180,247,320 | JADAL_EXP_DIR | env-fallback | feat/exp-rescore,m/latency-full |
| `eval/exp_review/run_y1.py` | 36-37,273,339 | JADAL_BUNDLE+JADAL_EXP_DIR | **hardcoded** | m/review-band |
| `eval/exp_review/run_y2.py` | 15,312,314 | JADAL_BUNDLE+JADAL_EXP_DIR | **hardcoded** | m/review-gate |
| `eval/exp_review/run_y3.py` | 51,315 | JADAL_BUNDLE+JADAL_EXP_DIR | env-fallback | m/review-score |
| `eval/exp_review160/run_o3.py` | 38-40 | JADAL_BUNDLE+JADAL_EXP_DIR | **hardcoded** | m/review160 |
| `eval/exp_skip/run_skip.py` | 25-26 | JADAL_BUNDLE | **hardcoded** | m/skip |
| `eval/exp_temporal/run_e3.py` | 11,13,20-27,238,307,352,381 | JADAL_BUNDLE+JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/exp_temporal/write_e305.py` | 70 | JADAL_DATA(legacy) | **hardcoded** | 4 branches |
| `eval/frames10k/embed10k_loop.sh` | 4,11,13,22 | JADAL_BUNDLE+JADAL_DATA(legacy)+JADAL_DATA | **hardcoded** | 24 branches |
| `eval/frames10k/embed_frames.py` | 5,26 | JADAL_DATA(model cache) | **hardcoded** | 24 branches |
| `eval/frames10k/embed_segments.py` | 18,64 | JADAL_BUNDLE+JADAL_DATA(model cache) | **hardcoded** | 24 branches |
| `eval/frames10k/stage2_loop.sh` | 4,8,21 | JADAL_BUNDLE+JADAL_DATA(legacy)+JADAL_DATA | **hardcoded** | 24 branches |
| `eval/freeze_ensemble.py` | 86,108-111 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/gpu/ablate.py` | 14,70 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/gpu/cascade.py` | 4,30 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/gpu/determinism.py` | 14 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/gpu/e2e.py` | 5-6,18,129-130 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/gpu/factory.py` | 3,25,59,155,167 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/gpu/holdc.py` | 6-7,22 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/gpu/microbench.py` | 4-5,22,388-389 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/gpu/nvdec_probe.py` | 11,55 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/gpu/nvdec_probe.sh` | 3-4 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/gpu/run_features.py` | 4,19-20 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/gpu/stream.py` | 22,157 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/gpu/telemetry.py` | 1,12 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/gpu/throughput.py` | 25 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/make_splits.py` | 12,61 | JADAL_BUNDLE+JADAL_DATA | **hardcoded** | 24 branches |
| `eval/perpixel/audit.py` | 12,32,35 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/perpixel/cost.py` | 4,18,72-73 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/perpixel/detectors.py` | 5,34 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/perpixel/freeze.py` | 19,43 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/perpixel/freeze_conv.py` | 25 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/perpixel/freeze_v2.py` | 34 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/perpixel/make_table.py` | 9-10,25,124-125 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/perpixel/physics_run.py` | 4,26 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/perpixel/rules_fix.py` | 11,27 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/perpixel/theory_clean.py` | 6,34 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/perpixel/train_conv_frozen.py` | 26 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `eval/spectral_features/icvl_check.py` | 4,40 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/spectral_features/run_arm.py` | 31-33 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/spectral_features/run_dups.py` | 15 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `eval/train_heads.py` | 13,53,62 | JADAL_BUNDLE+JADAL_DATA | **hardcoded** | 24 branches |
| `jadal/decision.py` | 9,63 | JADAL_DATA(model cache) | **hardcoded** | 24 branches |
| `jadal/poc/cascade.py` | 84 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `jadal/poc_stream/replay.py` | 69-70 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `opencode.json` | 25 | JADAL_DATA/EXP_DIR | **hardcoded** | 24 branches |
| `scripts/demo.sh` | 5,17,23,25,38-39,366 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `scripts/edge_check.py` | 259 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `scripts/gen_api_examples.py` | 11 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `scripts/make_demo_clips.py` | 13,74 | JADAL_DATA(legacy) | env-fallback | 24 branches |
| `scripts/pico_show.py` | 36-37 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `scripts/poc_demo.sh` | 7,11,17,19 | JADAL_DATA(legacy) | env-fallback | 24 branches |
| `scripts/poc_record.py` | 76 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `scripts/poc_stream_replay.py` | 6,22 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `scripts/poc_summary.py` | 21 | JADAL_DATA(legacy) | env-fallback | 8 branches |
| `sim/dock_sim.py` | 9-10 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `tests/test_exp_fps.py` | 8,31 | JADAL_DATA(legacy) | **hardcoded** | 8 branches |
| `tests/test_exp_rescore.py` | 21 | JADAL_BUNDLE | env-fallback | feat/exp-rescore,m/latency-full |
| `tools/exp.py` | 7 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `tools/gold_label/build_gold_set.py` | 4,11,24 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `tools/gold_label/label_server.py` | 13 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `tools/pod/autosave.sh` | 5,7 | JADAL_EXP_DIR | env-fallback | 24 branches |
| `tools/pod/bootstrap.sh` | 4 | JADAL_DATA/EXP_DIR | **hardcoded** | 24 branches |
| `tools/pod/gpu_run.sh` | 8,11 | JADAL_DATA/EXP_DIR | **hardcoded** | 24 branches |
| `tools/pod/install_kit.sh` | 5 | JADAL_BUNDLE | env-fallback | 24 branches |
| `tools/pod/launch_workers.sh` | 7,10 | JADAL_EXP_DIR | env-fallback | 24 branches |
| `tools/port/baseline_logistic_loco.py` | 6 | JADAL_BUNDLE | env-fallback | 24 branches |
| `tools/port/ensemble_views_exploratory.py` | 6 | JADAL_BUNDLE | env-fallback | 24 branches |
| `tools/port/make_label_queue.py` | 9,23 | JADAL_BUNDLE+JADAL_DATA | **hardcoded** | 24 branches |
| `web/mock/make_mock.py` | 4 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |
| `web/test_dashboard.py` | 4 | JADAL_DATA(legacy) | **hardcoded** | 24 branches |

---

## 5. Reproduce / verify

```
# recipe JSON (machine-readable)
sha256sum configs/ood_recipe_o4.json
# -> be321ec13654d57021353aa4128b3357a99286a82fe88f12257f5b58824a0f4d

# this document
sha256sum docs/reports/ood_recipe_o4_frozen.md
```

Regenerate the underlying numbers:
```
OMP_NUM_THREADS=2 python3 eval/exp_adapt160/extract160.py
OMP_NUM_THREADS=2 python3 eval/exp_adapt160/run_adapt160.py
python3 eval/exp_adapt160/make_figure.py
```

---

## Frozen artifact hashes

| artifact | sha256 |
|---|---|
| recipe JSON `configs/ood_recipe_o4.json` | `be321ec13654d57021353aa4128b3357a99286a82fe88f12257f5b58824a0f4d` |
| recipe doc `docs/reports/ood_recipe_o4_frozen.md` | `613041d32b420f9b742edf18c1f0a46570c90fb209c71fd0beaae56a26473b35` |

**Doc-hash convention:** the `recipe doc` sha256 above is the sha256 of this file truncated
just **before** the `## Frozen artifact hashes` footer (i.e. of the body above the `---`
separator preceding that marker). Verify with:

```
python3 - <<'EOF'
import hashlib
raw = open('docs/reports/ood_recipe_o4_frozen.md','rb').read()
marker = b'\n---\n\n## Frozen artifact hashes\n'
print(hashlib.sha256(raw[:raw.index(marker)]).hexdigest())
EOF
```

**frozen before confirmatory labels; no OOD fitting.**
