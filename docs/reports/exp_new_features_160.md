# Z3 — Domain-agnostic features (IndEgo → Egocentric-10K) — re-run on the 160 OOD labels

> **Supersedes the n=60 version (PR #127, branch `m/new-features`).**
> This is a **NEW branch** (`m/new-features-160`); the original PR branch is untouched.
> What changed: the out-of-domain set is now the merged **160** human-labelled
> Egocentric-10K windows (**138 good / 22 bad, 47 factories**; q1=60 + q2=100)
> instead of the old **60** (51 good / 9 bad), and the primary CI is the
> **factory-grouped** bootstrap (honest; windows within a factory are correlated).
> Protocol is otherwise identical. The 60-label numbers were **optimistic**: the
> headline z-score AUROC falls from 0.85 to **0.76–0.78**, but the gain over the
> frozen ensemble still holds with a grouped CI that excludes 0.

**Branch:** `m/new-features-160` (base `pod/base`) · **Owner:** M5
**Results:** `eval/results/exp_features160/Z3.01_160.json`, `Z3.02_160.json`, `Z3.03_160.json`
**Code:** `eval/exp_features160/run_z3_160.py` · **Tests:** `tests/test_exp_features160.py`
**Data:** 160 Egocentric-10K windows, 138 good / 22 bad, 47 factories, single labeller
(`labels/ood160_labels.json`); IndEgo = 150 human-gold windows, 7 clips.
**Device:** CPU (numpy/sklearn); decode done by the O/Z agents, no GPU here.
**Reproduce:** `OMP_NUM_THREADS=2 python3 eval/exp_features160/run_z3_160.py`

---

## TL;DR (160 vs the old 60)

| id | question | 60-label headline | **160-label headline (grouped CI)** | verdict |
|---|---|---|---|---|
| Z3.01 | do the candidates keep their sign? | 9 stable / 11 flip (of 20) | **9 stable / 10 flip / 1 chance** | candidate set unchanged |
| Z3.02 | which family transfers? | hand-excess + invariants stable | **same** (raw motion + absolute bitrate still flip) | mixed |
| Z3.03 | combine survivors | z-score **0.852 [0.732, 0.950]** (60) | **0.759 [0.597, 0.886]** (6-feat) · **0.781 [0.634, 0.895]** (stable-8+6) | **adopt, smaller effect** |

Paired factory-grouped gain vs frozen (same resamples): 6-feat **+0.193 [0.044, 0.343]**,
stable-8+6 **+0.213 [0.060, 0.367]**; both 99.5 % of resamples > 0.

---

## Z3.01 — candidate table (IndEgo vs OOD-160)

Signed AUROC (label 1 = usable/good). IndEgo CI = clip-bootstrap (7 clips);
OOD-160 CI = **factory-grouped bootstrap** (honest). Class rule = E5.03
(`chance` iff both |AUROC−0.5| < 0.05; else `stable`/`flip` by sign).

| feature | IndEgo | OOD-60 (old) | **OOD-160 [grouped 95% CI]** | class (160) | old class (60) |
|---|---|---|---|---|---|
| `ct_bitrate` | 0.376 | 0.802 | **0.609** [0.485, 0.726] | flip | flip |
| `ct_fsize_mean` | 0.376 | 0.802 | **0.648** [0.526, 0.767] | flip | flip |
| `ct_fsize_cv` | 0.732 | 0.529 | **0.480** [0.330, 0.625] | **flip** | stable |
| `ct_iframe_ratio` | 0.479 | 0.595 | **0.537** [0.389, 0.677] | chance | flip |
| `ct_fsize_entropy` | 0.264 | 0.481 | **0.489** [0.373, 0.617] | stable | stable |
| `ct_iframe_frac` | 0.350 | 0.488 | **0.473** [0.332, 0.614] | stable | stable |
| `ct_fsize_std` | 0.620 | 0.747 | **0.649** [0.462, 0.818] | stable | stable |
| `mo_trans_resid` | 0.301 | 0.704 | **0.666** [0.542, 0.791] | flip | flip |
| `mo_affine_resid` | 0.322 | 0.728 | **0.727** [0.614, 0.833] | flip | flip |
| `mo_affine_gain` | 0.304 | 0.573 | **0.493** [0.370, 0.650] | stable | flip |
| `mo_hand_trans_resid` | 0.453 | 0.802 | **0.695** [0.582, 0.817] | flip | flip |
| `mo_hand_affine_resid` | 0.480 | 0.824 | **0.744** [0.657, 0.832] | flip | flip |
| `mo_hand_motion_excess` | 0.649 | 0.771 | **0.608** [0.481, 0.746] | stable | stable |
| `mo_motion_frac` | 0.308 | 0.721 | **0.646** [0.524, 0.778] | flip | flip |
| `inv_dct_hf_lf` | 0.310 | 0.174 | **0.244** [0.117, 0.389] | stable (both low) | stable |
| `inv_noise_snr` | 0.505 | 0.253 | **0.342** [0.171, 0.546] | flip | flip |
| `inv_clip_frac` | 0.656 | 0.662 | **0.661** [0.451, 0.859] | stable | stable |
| `inv_hist_entropy` | 0.534 | 0.750 | **0.714** [0.558, 0.846] | stable | stable |
| `inv_dhash_dist` | 0.202 | 0.549 | **0.526** [0.375, 0.689] | flip | flip |
| `inv_laplacian_var` | 0.641 | 0.719 | **0.718** [0.566, 0.852] | stable | stable |

**Counts:** **9 stable / 10 flip / 1 chance** at 160 (was 9 stable / 11 flip at 60).
Two candidates change class: `ct_fsize_cv` (stable→flip: its IndEgo signal 0.732 is
strong but the OOD direction is 0.480) and `ct_iframe_ratio` (flip→chance: 0.479/0.537,
both within 0.05 of chance). The **selection set of 6 transferable candidates is
unchanged** (see Z3.03). O4 (PR #131) reports the same counts.

## Z3.02 — family verdicts (160)

- **Compression telemetry — half transfers.** Absolute `ct_bitrate` / `ct_fsize_mean`
  are strong OOD (0.61 / 0.65) but **flip** (IndEgo 0.376): use *variability*
  (`ct_fsize_std`, stable 0.620/0.649), never absolute bitrate.
- **Affine ego-motion residual — the fix does not restore the sign.** Full affine
  (`mo_affine_resid` 0.322/0.727) flips just like translation-only
  (`mo_trans_resid` 0.301/0.666) and global motion (`mo_motion_frac` 0.308/0.646);
  hand-restricted raw residuals still flip. The one survivor is the hand-region
  **excess** `mo_hand_motion_excess` (0.649/0.608, stable).
- **Invariants — mixed.** `inv_clip_frac` (0.656/0.661), `inv_laplacian_var`
  (0.641/0.718) and `inv_hist_entropy` (0.534/0.714) are stable; `inv_noise_snr`
  and `inv_dhash_dist` flip; `inv_dct_hf_lf` is stable but low in both.

## Z3.03 — combining with the 8 sign-stable E5 features

Transferable subset (stable, ≥0.03 from 0.5 in **both** domains):
`ct_fsize_std, mo_hand_motion_excess, inv_dct_hf_lf, inv_clip_frac, inv_hist_entropy,
inv_laplacian_var` (6 — **identical to the n=60 set**).

| recipe | IndEgo | OOD-60 (old) | **OOD-160 zero-shot [grouped 95% CI]** | OOD-160 few-shot (grouped CV) |
|---|---|---|---|---|
| frozen ensemble (E5.00) | — | 0.415 | **0.566** [0.424, 0.700] | — |
| in-domain 26 (E5.03 recipe) | 0.774 (LOCO) | 0.503 | **0.580** [0.444, 0.718] | 0.689 [0.510, 0.847] |
| stable-8 only | 0.519 (LOCO) | 0.527 | **0.579** [0.433, 0.731] | 0.722 [0.596, 0.841] |
| 6 transferable only (logistic) | 0.606 (LOCO) | 0.710 | **0.664** [0.455, 0.858] | 0.724 [0.547, 0.865] |
| stable-8 + 6 (logistic) | 0.656 (LOCO) | 0.497 | **0.546** [0.392, 0.715] | 0.711 [0.559, 0.844] |
| **6 transferable, per-domain z-score** | **0.684** | **0.852** | **0.759** [0.597, 0.886] | — |
| **stable-8 + 6, per-domain z-score** | **0.736** | **0.841** | **0.781** [0.634, 0.895] | — |

*z-score* = per-domain z-score each feature, sign = IndEgo AUROC, equal-weight mean
(the only fitted quantity is the per-feature sign; per-domain stats are unsupervised).
Few-shot = `StratifiedGroupKFold(5, groups=factory)` OOF, C by inner grouped CV;
smallest `n_bad` in any training fold = **17** (was 7 at n=60).

**The decisive result holds:** a logistic on raw features collapses OOD
(0.546–0.664), while the same features **z-scored per domain** reach **0.76–0.78**.
Cross-domain failure is largely a *scale/offset* problem. But the n=60 magnitude
(0.85) was **optimistic**: on 160 it is **0.76–0.78**, still clearly above the frozen
0.566 (paired grouped gain excludes 0).

## What changed vs the n=60 version (PR #127)

1. **Headline magnitude drops** 0.852 → 0.759 (6-feat) and 0.841 → 0.781 (stable-8+6):
   the old CIs were window-bootstrap (optimistic) and n_bad=9.
2. **OOD CI is now factory-grouped** (honest); the old window CIs are kept in the JSON.
3. **Two per-feature classes change** (`ct_fsize_cv`, `ct_iframe_ratio`); the
   recommended **transferable set of 6 is unchanged**.
4. The frozen baseline is **0.566** on 160, not 0.415 (the old 0.415 was noise at
   n_bad=9; see O1/PR #128).

## Caveats (unchanged, still binding)

- OOD = **160 windows, 22 bad, 47 factories (~1 window/factory), single labeller**;
  factory-grouped CI is honest but wide. Every OOD number is exploratory.
- IndEgo = 150 windows / 7 clips / ~3 wearers, single labeller.
- z-score combos fit the per-feature **sign** on the full IndEgo set (mild optimism);
  the magnitude fit is unsupervised (per-domain z-score).
- Hand mask = YCrCb skin threshold, not a hand detector; `mo_hand_motion_excess`
  measures *skin-region* motion excess.
- Batch q1/q2 differ (see `Z3.03_160.json` batch splits in O4/PR #131); pooled 0.76
  is not a stable rate.
