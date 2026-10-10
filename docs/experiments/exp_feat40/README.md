# exp_feat40 — per-pixel features 26 → 40

Branch `m/feat40` (off `pod/base`). Registry id **F40.01**
(`experiments/registry.jsonl`). Results `eval/results/exp_feat40/`. Report
`docs/reports/exp_feat40.md`. Figure `docs/experiments/exp_feat40/images/feat40.png`.

## What changed

`jadal/gpu/features.py` grows `FEATURE_NAMES` from 26 to 40. The 14 added
per-pixel features (`NEW_FEATURE_NAMES`) and their definitions / cost:

| # | feature | definition | cost (per 10-s window) |
|---|---------|------------|------------------------|
| 1 | `ct_fsize_std` | std of per-frame ffprobe packet sizes in the window (compression telemetry; **reused from the frozen Z3 cache**) | 1 ffprobe pass (~20–80 ms) |
| 2 | `mo_hand_motion_excess` | mean over frames of mean\|Δframe\| inside the YCrCb skin/hand mask **minus** the global mean\|Δframe\| (Z3; frozen cache) | O(T·H·W), one skin mask/frame |
| 3 | `inv_clip_frac` | fraction of luma pixels ≤2/255 or ≥253/255 (exposure clipping; Z3 cache) | O(T·H·W) |
| 4 | `inv_laplacian_var` | mean over frames of `cv2.Laplacian(gray).var()` (edge/sharpness; Z3 cache) | O(T·H·W) |
| 5 | `inv_hist_entropy` | mean over frames of the 32-bin luma histogram entropy (Z3 cache) | O(T·H·W) |
| 6 | `inv_dct_hf_lf` | DCT HF/LF energy ratio on a 64×64 resize of the middle frame (Z3 cache) | O(T·64²) |
| 7 | `spec_b1` | fraction of 2-D FFT energy in radial shell 0.10–0.20 (finer spectral band) | O(3·H·W·log) |
| 8 | `spec_b2` | shell 0.20–0.35 | as above |
| 9 | `spec_b3` | shell 0.35–0.50 | as above |
| 10 | `col_rg_ratio` | mean over frames of mean(R)/mean(G) (colour-channel ratio) | O(T·H·W) |
| 11 | `col_bg_ratio` | mean over frames of mean(B)/mean(G) | O(T·H·W) |
| 12 | `temp_lag2` | mean over frames of mean\|frame_t − frame_{t−2}\| (temporal lag 2) | O(T·H·W) |
| 13 | `temp_lag3` | lag 3 | O(T·H·W) |
| 14 | `noise_floor` | mean over frames of the median absolute deviation of the Laplacian (per-pixel noise floor) | O(T·H·W) |

Features 1–6 are the **Z3 transferable set**; they are reused verbatim from the
frozen caches (`m-adapt160/…/z3_160.npz`, `m-confirm/…/confirm108_features.npz`)
so the frozen OOD recipe is reproduced exactly. Features 7–14 are the **8 fresh
features**, computed by `eval/gpu/run_features40.py` (one decode per window).

`cpu_new_features()` implements all 14 for the standalone 40-feature extractor;
`cpu_window_features()` now returns 40 names.

## Protocol

* **In-domain**: LOCO over the 7 IndEgo clips, logistic (C=0.3,
  `class_weight='balanced'`), per-fold `StandardScaler`; AUROC + clip-bootstrap
  95% CI (2 000), balanced accuracy @ 0.5.
* **OOD**: adapted per-domain-z-score recipe (O4): per-feature sign from the 150
  IndEgo gold windows, unsupervised per-domain z-score, equal-weight mean,
  threshold 0.0; factory-grouped bootstrap 95% CI (2 000). n=160, 22 bad / 47
  factories (optimistic — one window per clip).
* **Hold-out 108**: scored **once**, no tuning. n=108, 16 bad / 44 factories.
* Every number carries n / label source (human) / protocol / 95% CI / device
  (CPU A40 host) / command.
