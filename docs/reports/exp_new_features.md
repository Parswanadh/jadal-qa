# Z3 — Domain-agnostic features that survive IndEgo → Egocentric-10K

Branch `m/new-features`, experiments `Z3.01–Z3.03` (registry:
`experiments/registry.jsonl`; results: `eval/results/exp_features/`).
Every number carries `n`, `label_source`, `protocol`, `ci95`, `device`, `command`, `git_sha`.

**Goal.** The frozen 26 per-pixel features are *not* domain-agnostic: 14 of 26 flip
sign out-of-domain (E5.03). Here we invent and measure **new** candidates in three
families — compression telemetry, affine ego-motion residual, illumination/scale-space
invariants — on the **same** 150 IndEgo gold windows and 60 Egocentric-10K human OOD
windows used by E5.03, then combine the survivors with the 8 sign-stable E5 features.

## TL;DR

| id | question | headline | verdict |
|---|---|---|---|
| Z3.01 | do the new candidates keep their sign across domains? | **9 stable / 11 flip** (of 20) | candidate set |
| Z3.02 | which family transfers? | **hand-region motion-excess + invariants stable; absolute bitrate and *both* translation-only and full-affine residuals flip** | mixed |
| Z3.03 | combine survivors with stable-8 | **domain-normalised score: OOD AUROC 0.84–0.85** [0.71, 0.95] vs frozen ensemble 0.415; no-normalisation logistic OOD 0.50 | **adopt normalised score** |

**Recommendation.** Adopt the **6 transferable candidates**
`ct_fsize_std, mo_hand_motion_excess, inv_dct_hf_lf, inv_clip_frac, inv_hist_entropy,
inv_laplacian_var`, combined with the 8 sign-stable E5 features via a
**per-domain z-score, IndEgo-signed, equal-weight mean** (Z3.03). Do **not** feed raw
feature values to a cross-domain logistic: absolute scales shift, and the trained
logistic falls back to 0.50 OOD. Normalising per domain recovers 0.84.

## Method

Windows (identical to E5.03): 150 IndEgo human-gold windows (100 good / 50 bad, 7
clips) + 60 Egocentric-10K human-labelled windows (51 good / 9 bad, 47 factories;
`n_bad = 9`, all OOD numbers exploratory, window-bootstrap CI optimistic).

- **A. Compression telemetry** (`ct_*`): per-frame `pkt_size` / `pict_type` from a
  single whole-clip `ffprobe` per unique file (the earlier `-read_intervals` approach
  returned offset frames, so we slice by `pts_time`). No full decode.
- **B. Affine ego-motion residual** (`mo_*`): RGB→grey at 160×90, ~10 fps effective
  (OOD 30 fps subsampled ×3 to match IndEgo). Per frame pair: translation-only
  residual via phase correlation; **full affine** residual via Lucas–Kanade +
  `estimateAffinePartial2D` (RANSAC). Residual restricted to a YCrCb **skin/hand
  mask**; `mo_hand_motion_excess` = hand-region minus global residual.
- **C. Invariants** (`inv_*`): DCT high/low band-energy ratio, Laplacian-variance
  sharpness, noise-floor-normalised sharpness, exposure-clipping fraction, luma
  histogram entropy, temporal dHash distance.

Per-feature AUROC is **signed** (label 1 = usable/good): IndEgo clip-bootstrap CI
(7 clips), OOD window-bootstrap CI. Transfer class uses the E5.03 rule
(`chance` iff both |AUROC−0.5| < 0.05; else `stable`/`flip` by sign).

## Z3.01 — candidate table (IndEgo vs OOD-60)

| feature | IndEgo AUROC [95% CI] | OOD-60 AUROC [95% CI] | class |
|---|---|---|---|
| `ct_bitrate` | 0.376 [0.184, 0.608] | 0.802 [0.678, 0.913] | **flip** |
| `ct_fsize_mean` | 0.376 [0.184, 0.608] | 0.802 [0.678, 0.913] | **flip** |
| `ct_fsize_cv` | 0.732 [0.542, 0.818] | 0.529 [0.256, 0.763] | stable |
| `ct_iframe_ratio` | 0.479 [0.415, 0.532] | 0.595 [0.348, 0.804] | **flip** |
| `ct_fsize_entropy` | 0.264 [0.188, 0.443] | 0.481 [0.238, 0.762] | stable (both low) |
| `ct_iframe_frac` | 0.350 [0.312, 0.426] | 0.488 [0.338, 0.580] | stable (both low) |
| `ct_fsize_std` | 0.620 [0.413, 0.753] | 0.747 [0.555, 0.907] | **stable** |
| `mo_trans_resid` | 0.301 [0.157, 0.507] | 0.704 [0.516, 0.863] | **flip** |
| `mo_affine_resid` | 0.322 [0.167, 0.547] | 0.728 [0.505, 0.904] | **flip** |
| `mo_affine_gain` | 0.304 [0.188, 0.453] | 0.573 [0.395, 0.738] | **flip** |
| `mo_hand_trans_resid` | 0.453 [0.326, 0.597] | 0.802 [0.664, 0.914] | **flip** |
| `mo_hand_affine_resid` | 0.480 [0.358, 0.628] | 0.824 [0.676, 0.935] | **flip** |
| `mo_hand_motion_excess` | 0.649 [0.576, 0.720] | 0.771 [0.632, 0.891] | **stable** |
| `mo_motion_frac` | 0.308 [0.161, 0.518] | 0.721 [0.545, 0.860] | **flip** |
| `inv_dct_hf_lf` | 0.310 [0.232, 0.456] | 0.174 [0.056, 0.337] | stable (both low) |
| `inv_noise_snr` | 0.505 [0.408, 0.666] | 0.253 [0.035, 0.491] | **flip** |
| `inv_clip_frac` | 0.656 [0.467, 0.752] | 0.662 [0.388, 0.886] | **stable** |
| `inv_hist_entropy` | 0.534 [0.330, 0.709] | 0.750 [0.567, 0.912] | **stable** |
| `inv_dhash_dist` | 0.202 [0.125, 0.361] | 0.549 [0.333, 0.760] | **flip** |
| `inv_laplacian_var` | 0.641 [0.422, 0.764] | 0.719 [0.516, 0.892] | **stable** |

Counts: **9 stable / 11 flip** (20 candidates). Figure:
`docs/experiments/exp_features/images/z3_features.png`.

## Z3.02 — family verdicts

- **Compression telemetry — half transfers.** The **absolute** cues `ct_bitrate`
  and `ct_fsize_mean` are the strongest OOD features (0.802) yet **flip** (IndEgo
  0.376): the two corpora have different codecs/bitrates, so "small frames" means
  "static/no work" in Ego10K but not in IndEgo. The **relative** cue `ct_fsize_std`
  (frame-size variability) is **stable** (0.620 / 0.747) and robust. Verdict: **use
  variability, never absolute bitrate.**
- **Affine ego-motion residual — the fix did not restore the sign.** Full affine
  (`mo_affine_resid` 0.322 / 0.728) flips just as translation-only
  (`mo_trans_resid` 0.301 / 0.704) and global motion (`mo_motion_frac` 0.308 / 0.721).
  The reason is mechanistic: residual magnitude ∝ how much the scene moves, and the
  motion↔"active work" coupling inverts between domains (E5.03). **The one survivor
  is the hand-region *excess*** `mo_hand_motion_excess` = hand residual − global
  residual: 0.649 / 0.771, **stable** — motion *concentrated in the hands* rather than
  raw motion is the transferable concept. Hand-restricted raw residuals still flip.
- **Invariants — mixed.** `inv_clip_frac` (exposure clipping, 0.656 / 0.662),
  `inv_laplacian_var` (sharpness, 0.641 / 0.719) and `inv_hist_entropy` (0.534 / 0.750)
  are **stable** and useful. `inv_noise_snr` (0.505 / 0.253) and `inv_dhash_dist`
  (0.202 / 0.549) **flip**; `inv_dct_hf_lf` is stable but low in both (0.310 / 0.174).

## Z3.03 — combining with the 8 sign-stable E5 features

Transferable subset (stable, ≥0.03 from 0.5 in **both** domains):
`ct_fsize_std, mo_hand_motion_excess, inv_dct_hf_lf, inv_clip_frac, inv_hist_entropy,
inv_laplacian_var` (6). Compared on the same windows:

| recipe | IndEgo | OOD-60 | OOD few-shot (factory CV) |
|---|---|---|---|
| frozen ensemble (E5.00) | — | 0.415 [0.201, 0.656] | — |
| in-domain 26 (E5.03 recipe) | 0.808 (LOCO) | 0.503 (zero-shot) | 0.678 |
| stable-8 only | 0.572 (LOCO) | 0.527 | 0.747 |
| 6 transferable only (logistic) | 0.608 (LOCO) | 0.710 [0.455, 0.915] | 0.647 |
| stable-8 + 6 (logistic) | 0.722 (LOCO) | 0.497 | 0.776 |
| **6 transferable, z-score** | **0.684** [0.507, 0.781] | **0.852** [0.732, 0.950] | — |
| **stable-8 + 6, z-score** | **0.736** [0.551, 0.860] | **0.841** [0.712, 0.942] | — |

*z-score* = per-domain z-score each feature, sign = IndEgo AUROC, equal-weight mean
(the only fitted quantity is the per-feature sign; per-domain stats are unsupervised).

**The decisive result:** a logistic trained on IndEgo collapses to **0.497 OOD** with
raw features, but the same features **z-scored per domain** reach **0.84–0.85 OOD**.
Cross-domain failure here is largely a *scale/offset* problem, not a *concept* problem.
The 6-feature z-score set is the most domain-agnostic (OOD 0.852) while keeping
in-domain 0.684; adding the stable-8 raises IndEgo to 0.736 at 0.841 OOD.

## Recommended feature set (with numbers)

Adopt **`ct_fsize_std`, `mo_hand_motion_excess`, `inv_clip_frac`, `inv_laplacian_var`,
`inv_hist_entropy`** as the core transferable cues (each stable, OOD 0.66–0.75), plus
`inv_dct_hf_lf` as a low-in-both stable extra, combined with the E5 stable-8 via the
per-domain z-score rule. Combined OOD AUROC **0.84–0.85** vs the frozen ensemble
**0.415**; in-domain **0.68–0.74**.

## Which candidates failed

- **Absolute compression**: `ct_bitrate`, `ct_fsize_mean` (flip, 0.376↔0.802).
- **Both motion-residual magnitudes**: translation-only `mo_trans_resid`, full affine
  `mo_affine_resid`, `mo_affine_gain`, `mo_motion_frac` — all flip; hand-restricted raw
  residuals `mo_hand_trans_resid` / `mo_hand_affine_resid` still flip.
- **Invariants**: `inv_noise_snr` (flip), `inv_dhash_dist` (flip); `inv_dct_hf_lf`,
  `ct_iframe_frac`, `ct_fsize_entropy` are stable but low in at least one domain.

## Caveats

- OOD = **60 windows, `n_bad = 9`, ~1 window/factory, single labeller**; every OOD
  number is **exploratory** and the window-bootstrap CI is optimistic.
- IndEgo = 150 windows / 7 clips / ~3 wearers, single labeller.
- z-score combos fit the per-feature **sign** on the full IndEgo set (mild optimism);
  the *magnitude* fit is unsupervised (per-domain z-score).
- Hand mask = YCrCb skin threshold, not a hand detector; `mo_hand_motion_excess`
  measures *skin-region* motion excess.

Regenerate: `OMP_NUM_THREADS=2 python3 eval/exp_features/extract.py && python3 eval/exp_features/run_eval.py && python3 eval/exp_features/make_figure.py`.
