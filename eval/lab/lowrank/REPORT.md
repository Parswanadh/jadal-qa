# Direction 1 — Low-rank + sparse decomposition (Robust-PCA-flavoured) prototype

## 1. Exact formulation

Per 10-s gold window (or full short clip for constructed dups):

1. **Decode (read-only, deterministic):** `cv2.VideoCapture`, seek to `start_s`,
   read all frames through `end_s` (~100 frames at 10 fps), uniform `linspace`
   sample of **n = 50 frames** (stride ≈ 2). Each frame → luma (BGR2GRAY),
   `INTER_AREA` resize to **64×64**, float32/255. Full-clip mode (constructed
   dups, which are short files, not windows) samples 50 frames uniformly across
   the whole file. Short reads are padded by repeating the last frame
   (deterministic; never happened — 0 decode failures).
2. **Temporal matrix:** `M ∈ R^(p×n)`, p = 4096 pixels (columns = frames).
   Per-pixel temporal mean removed: `M0 = M − μ·1ᵀ`. **No global-motion
   compensation** (deliberate: at 64 px, egocentric shake is broadband and would
   be absorbed into the low-rank part anyway; phase-correlation warping adds
   cost plus border artefacts; frozen-vs-active contrast is preserved in raw M).
3. **Rank profile (economy SVD of M0, float64, `np.linalg.svd`; matrix is only
   4096×50 so no randomized SVD was needed — see §6):**
   - `stable_rank = ‖M0‖_F² / σ₁²` (low ≈ rank collapse / frozen).
   - `σ₁_share = σ₁² / Σσ²` (= 1/stable_rank; reported for readability).
   - **MP signal rank:** standardise globally `Z = M0/std(M0)`, aspect
     `c = p/n = 81.92`, noise edge `T = √n·(1+√c)` for singular values;
     `mp_rank = #{σ(Z) > T}`. Median 7–9 of 50.
   - No periodicity analysis of right singular vectors was run (loop hypothesis
     untested — stated as gap, not claimed).
4. **Sparse proxy (no PCP solver):** rank-2 truncation `L₂`, then
   `sparse_frac = ‖M0 − L₂‖₁ / ‖M0‖₁` at k = 2. Intended as "activity" score
   (moving hands/tools = high sparse energy).
5. **Duplicate cue:** top-5 right-singular subspace `Vk` (n×5); distance between
   windows = `Σ sin²θᵢ` (chordal, from SVD of `QaᵀQb`); lower = more similar.

Classifier for LOCO: `StandardScaler + LogisticRegression(C=1.0)` on the 3
features `[stable_rank, sparse_frac_k2, mp_rank]`, fit on train clips, scored on
the held-out clip. Failed decodes (none) would score 0.

## 2. Headline evaluation (leave-one-clip-out over 7 clips)

| metric | value |
|---|---|
| pooled LOCO AUROC (usable=1, concat of 7 held-out folds) | **0.312** |
| clip-bootstrap 95% CI (1000 reps over 7 clips, 1000/1000 valid) | **[0.234, 0.487]** |
| per-clip AUROC desk / ecu_full / ecu_late / logistics4 / organise / pallet / pallet2 | 0.486 / 0.247 / 1.000 / 0.667 / 0.422 / 0.470 / 0.000 |

Descriptive (pooled, non-LOCO, fixed sign) single-feature AUCs: sparse_frac
0.628, mp_rank 0.616, stable_rank 0.585, σ₁_share 0.585 — i.e. weak signal with
the *opposite* sign to the hypothesis (bad windows have slightly *higher*
rank/sparse energy), and LOCO folds flip sign (ecu_late 1.0 vs pallet2 0.0),
so the pooled LOCO score lands below chance. **Verdict: no usable-vs-bad
signal from these features under LOCO.**

## 3. Per-reason notes (medians; n small — descriptive only)

| group (n) | stable_rank | sparse_frac | mp_rank |
|---|---|---|---|
| good (100) | 3.16 | 0.662 | 7.0 |
| NO_ACTIVE_WORK (23) | 3.25 | 0.699 | 8.0 |
| HANDS_NOT_VISIBLE (14) | 3.99 | 0.737 | 8.5 |
| DUPLICATE (6) | 3.56 | 0.758 | 9.0 |
| TOO_DARK (6) | **1.78** | **0.411** | **5.0** |
| UNDECODABLE (1) | 4.19 | 0.765 | 9.0 |

- **TOO_DARK is the one real find:** rank collapse exactly as hypothesised
  (σ₁_share AUC 0.811 vs rest; Youden cut 0.458 → recall 4/6 at FPR 0.139).
  Dark/static windows genuinely are ≈ rank-1.
- NO_ACTIVE_WORK / HANDS_NOT_VISIBLE are *indistinguishable to slightly
  higher-rank* than good windows — plausibly egocentric background motion
  dominates the spectrum and hands add a static-ish foreground; rank statistics
  of raw luma cannot see "hands doing work".
- DUPLICATE (n=6): nothing special in rank features (they are duplicates of
  *other* content, not of each other — see §4).

## 4. Duplicate cue (subspace distance)

- Gold pairs (150 choose 2 = 11175): 15 dup–dup pairs vs rest → AUC **0.177**
  (dup–dup median dist 3.46 vs rest 2.80 — *farther apart*, as expected since
  the 6 gold DUPLICATEs are not copies of each other). Gold dups cannot
  validate a similarity cue; honest negative control.
- Constructed pairs (128, `manifest_dups.json`): dup-vs-all-neg AUC **0.845**.
  By kind vs easy-negatives: reencode 1.000 (med dist 0.001), crop5 1.000
  (0.002), bright 1.000 (0.0004), speed11 0.778 (1.86), **offset1 0.511
  (2.31)** — i.e. the cue is a near-perfect *temporal-alignment* detector and
  collapses under a 1-frame shift (right-singular vectors are frame-indexed).
  hard-neg vs easy-neg 0.556 (correctly ~chance).

## 5. Cost / determinism / streaming / hashability

| property | result |
|---|---|
| decode | **711 ms/window** mean (single core, OMP=2; dominates) |
| SVD + features (4096×50) | **51 ms/window** |
| total | **762 ms/window**; full run 150 + 48 files ≈ 2.5 min CPU |
| deterministic | **yes** — bit-identical features on re-run (verified); fixed linspace sampling, no randomness |
| streaming (2-s chunks, past-only) | **no, as-is** — SVD needs the full window; path forward is incremental SVD (Brand-2002 rank-1 updates) or fixed reference basis, not implemented |
| hashable | **yes** — 3 floats/window; sha256 of rounded feature vector stable (e.g. `cfa51fee0c5223ee` for g000 config); quantise to float16/4 dp for cross-machine hashes |

## 6. Literature (all fetched live 2026-10-10 except MP-Wikipedia excerpt via search; none invented)

1. Candès, Li, Ma, Wright — *Robust Principal Component Analysis?*, JACM 58(3),
   2011. PCP: exact recovery of L₀+S₀ via nuclear+ℓ₁ minimisation; video
   surveillance (static background = low-rank) is the motivating example.
   https://arxiv.org/abs/0912.3599 / https://doi.org/10.1145/1970392.1970395 [fetched]
2. Zhou, Li, Wright, Candès, Ma — *Stable Principal Component Pursuit*, ISIT
   2010. Relaxed PCP stable to entrywise noise + gross errors; justifies our
   "noisy small matrix, skip the solver" stance. https://arxiv.org/abs/1001.2363 [fetched]
3. Halko, Martinsson, Tropp — *Finding Structure with Randomness*, SIAM Rev
   53(2), 2011. Randomized SVD beats classical at scale; **not used here
   because 4096×50 exact SVD costs 51 ms** — randomised methods only pay off
   far beyond this regime. https://arxiv.org/abs/0909.4061 [fetched]
4. Marčenko & Pastur (1967); MP law: sample-covariance bulk edge
   `(1±√c)²`, i.e. singular-value edge `√n·(1+√c)` used for mp_rank.
   https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution [fetched]

## 7. Would-this-help-us verdict (10 lines)

1. As a usable-vs-bad window gate, no: LOCO AUROC 0.31 (CI entirely below 0.5)
   kills it; bad windows are not lower-rank than good ones except dark ones.
2. The single transferable finding is TOO_DARK ≈ rank collapse (AUC 0.81,
   recall 4/6) — but a mean-brightness check does that job for microseconds.
3. Sparse-energy fraction, the hypothesised "activity" score, points the wrong
   way (bad ≥ good); egocentric background motion swamps hand motion in raw luma.
4. The subspace duplicate cue is excellent at exact/near-exact dup detection
   (AUC 0.85–1.0) but is really an alignment detector: 1-frame offset kills it.
5. Genuine gaps left open: right-singular-vector periodicity (loop hypothesis)
   was never tested; no GMC tried; no ALM even on tiny matrices.
6. Cost profile is fine (51 ms features) but decode (711 ms) dominates any
   deployment math, same as every other pixel method.
7. Deterministic and hashable — the one engineering virtue worth reusing.
8. Not streaming as built; incremental-SVD path exists but unimplemented.
9. Do not promote any of these 3 features to the funnel; do not tune further
   on this n=150 sample.
10. If dups matter later, pair this cue with a shift-invariant alignment step
    (or perceptual hash) rather than using it standalone.

## 8. Honest caveats

n=150 windows, 7 clips, ~3 wearers; per-reason cells as small as n=1–6, so all
of §3 is descriptive. LOCO folds range from n=6 (pallet2) to n=48 (ecu_full);
clip-bootstrap over 7 clips is coarse by construction. Single resolution
(64×64), single frame budget (50), single k=2/k=5 — no sensitivity sweep was
run. H.264 decode + `VideoCapture` seeking is treated as deterministic on this
machine (verified by re-run) but byte-identical seeking across
OpenCV/ffmpeg builds is not guaranteed. One early arXiv-ID recall was wrong
(0910.0651/0909.4067 are unrelated papers) and corrected via search — only
fetched sources are cited above.

## Reproduce

`OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python proto_lowrank.py`
→ `results/{features,loco,dups,cost,descriptive,summary}.json`. No GPU, no
network, no writes outside this dir. Decode is the full 150 windows (no
subset; ~2.5 min total including 48 constructed-dup files).
