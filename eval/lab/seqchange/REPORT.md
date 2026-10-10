# Direction 3 — Streaming early-reject sequential tests (REPORT)

Subagent output. Worktree `/home/parshu/projects/jadal-qa-wt/breakthrough-lab`,
branch `feat/breakthrough-lab`. Code: `eval/lab/seqchange/proto_seqchange.py`,
machine tables: `eval/lab/seqchange/results/*.json`. Nothing committed/pushed.

## 1. Exact formulation

**Sampling (deterministic, read-only).** Each 10-s gold window decoded with cv2
(10 fps native), every 2nd frame kept → 5 fps, ≤50 frames/window, resized to
64×64 luma. Synth files decoded whole-file, same stride. No RNG anywhere.

**Per-frame statistics** (frame `t`, `Y` = 64×64 luma):

- `MAD_t = mean|Y_t − Y_{t−1}|` — motion energy; frozen/dark streams collapse it.
- `BRI_t = mean(Y_t)`, `DARK_t = frac(Y_t<16)`, `SHARP_t = mean(|Sx|+|Sy|)`
  (Sobel ksize=3) — computed and stored, but the three sequential tests below
  all run on the **MAD series** (brightness/Sobel kept as logged features;
  MAD alone carries the frozen/dark signal — see §4).

**Normalisation (documented choice).** Fixed robust scale, *not* a
Poisson-Gaussian flat-block fit: on TRAIN-good windows only,
`mu0 = median(MAD)`, `sig0 = 1.4826·MAD(MAD)` (Gaussian-consistency constant).
Standardised collapse score `z_t = (mu0 − x_t)/sig0` (positive = degradation).
Global fit: `mu0=10.20, sig0=7.54`; LOCO folds range `mu0 ∈ [10.81,13.89]`,
`sig0 ∈ [8.28,8.79]` (fold table in `loco_predictions.json`).

**Streaming protocol.** `CUSUM/SPRT/EBCS` classes expose `update(v)`; the driver
feeds frames in **2-s chunks (10 frames @5 fps)** in order and records the
first-firing frame → `fire_s = fire_idx/5`. `fire = reject`, no-fire at end of
window = accept (truncated rule). Each verdict carries a sha256
`hash(id|method|fire|mu0|sig0|params)` in `loco_predictions.json`
(deterministic / streaming / hashable ✔).

**Boundaries (fixed a priori, never tuned on test; scale params from TRAIN only):**

| Method | Rule | Parameters |
|---|---|---|
| M1 CUSUM (one-sided) | `S_n=max(0,S_{n−1}+z_n−k)`, fire if `S_n>h` | `k=0.5` (tuned to 1σ shift), `h=5.0` |
| M2 SPRT (Gaussian mean-shift) | per-obs LLR `δ·z_n−δ²/2`, fire if cumsum `≥ B` | `δ=1.0, α=0.05, β=0.10`; `A=log(β/(1−α))=−2.251`, `B=log((1−β)/α)=+2.891` |
| M3 mixture-CS (Howard et al. style) | `w_n=σ√(2(nρ²+1)/(n²ρ²)·log(√(nρ²+1)/α))`, fire if `mu0−x̄_n>w_n`, `n≥5` | `α=0.05, ρ=1.0` (σ-known Gaussian-mixture boundary, Thm 1 of Howard et al.) |

Page–Hinkley is not implemented separately: one-sided CUSUM with restart-at-zero
is its functional equivalent for downward shifts (noted, not double-counted).

## 2. What each method guarantees BY CONSTRUCTION (nominal) vs observed

| Method | Guarantee by construction | Assumptions | Nominal err. rate | Observed (LOCO) |
|---|---|---|---|---|
| CUSUM | ARL₀ ≈ 285 frames ≈ 57 s @5 fps (Siegmund approx `(e^{2kh}−2kh−1)/2k²`, k=0.5,h=5) → ≈0.16 false-fire prob per 50-frame window | i.i.d. sub-Gaussian increments, known pre-change mean/scale | FA≈0.16 | **FA 0.45** |
| SPRT | `P(fire\|H0)≤α=0.05`, `P(no-fire\|H1)≤β=0.10` at exactly-δ shift (Wald) | i.i.d. Gaussian, **known** variance, exact H1 mean | FA 0.05 / power 0.90 | **FA 0.38 / det 0.20** |
| mixture-CS | `P(∀n: μ∈CI_n)≥1−α=0.95` uniformly over time (Howard et al. Thm 1) → one-sided false-fire ≤0.05 | indep. sub-Gaussian/known σ, **true** μ₀/σ (ours are train-estimated → coverage is approximate, not exact) | FA 0.05 | **FA 0.37** |

**Do video MAD statistics satisfy the assumptions? No.** Frames are strongly
autocorrelated, heavy-tailed (scene cuts, camera moves), and the "known"
`mu0/sig0` are estimated from other clips whose motion regime differs
(fold `mu0` varies 10.8→13.9). The 3–9× gap between nominal and observed FA is
the honest measure of assumption violation. Guarantees hold for the *procedure*
under its model, not for these data.

## 3. LOCO evaluation (leave-one-clip-out, 7 clips, fire=reject)

Pooled over 150 windows (100 good / 50 bad); 95% CIs from 2000 clip-bootstraps
(resample 7 clips with replacement, recompute pooled rates).

| Method | Detection rate (bad) | 95% CI | False-alarm rate (good) | 95% CI |
|---|---|---|---|---|
| CUSUM | **0.24** | [0.09, 0.40] | **0.45** | [0.04, 0.71] |
| SPRT | **0.20** | [0.09, 0.33] | **0.38** | [0.04, 0.60] |
| mixture-CS | **0.22** | [0.07, 0.37] | **0.37** | [0.04, 0.58] |

(`results/loco_summary.json`; per-window fires+hashes in `loco_predictions.json`.)

**Detection by gold reason** (why the pooled rate is low — the tests measure
motion collapse, most bad labels are semantic):

| Reason | n | CUSUM | SPRT | EB-CS |
|---|---|---|---|---|
| TOO_DARK | 6 | **6/6** | 4/6 | **6/6** |
| DUPLICATE | 6 | 1/6 | 1/6 | 1/6 |
| HANDS_NOT_VISIBLE | 14 | 1/14 | 1/14 | 1/14 |
| NO_ACTIVE_WORK | 23 | 4/23 | 4/23 | 3/23 |
| UNDECODABLE | 1 | 0/1 | 0/1 | 0/1 |

The single UNDECODABLE window (g082) decodes without error — its pixels are
ordinary, so a pixel-statistic test *cannot* see it (fail-open-at-decode-failure
path never triggered). Static-camera good windows also trip the tests
(motion absence ≠ unusability) — hence FA≈0.4.

**Early-fire fraction among the 50 bad windows:** CUSUM 12% ≤2 s / 16% ≤4 s;
SPRT 16% / 18%; EB-CS 12% / 14%.

## 4. Time-to-detection vs severity (synthetic, global calibration)

60 synth clips (6 sources × 10 degradations). Fire time in seconds (`None` = no fire):

| Degradation (n=6) | CUSUM | SPRT | EB-CS |
|---|---|---|---|
| black | 6/6, all 1.2 s | 6/6, all 0.8 s | 6/6, all 1.2 s |
| frozen (first-frame loop) | 6/6, all 1.2 s | 6/6, all 0.8 s | 6/6, all 1.2 s |
| darken (eq −0.7) | 6/6, 1.2–1.4 s | 6/6, all 0.8 s | 6/6, 1.2–1.4 s |
| blur (σ=8) | 3/6 (1.4–8.8 s) | 1/6 | 1/6 |
| overexpose | 4/6 | 2/6 | 3/6 |
| noise (σ=35) | 1/6 | 1/6 | 1/6 |
| lowbitrate 70k | 2/6 | 1/6 | 1/6 |
| shake | 0/6 | 0/6 | 0/6 |
| static (uniform RGB) | 0/6 | 0/6 | 0/6 |
| clean | 2/6 FA (1.8 s, 9.6 s) | 1/6 FA | 1/6 FA |

(`results/ttd.json`.) Physical motion-collapse degradations are caught in
**≤1.4 s (≤7 frames)** by all three methods — the early-reject use-case works
exactly where the hypothesis matches. `shake`/`static` correctly never fire
(one-sided: motion *increase* is not the degradation direction). Clean-clip FAs
come from one static source scene (fires even without degradation), same story
as gold FA. Real TOO_DARK windows: CUSUM fires at 1.4–4.4 s (6/6), SPRT 0.8–1.6 s
(4/6, two `None`), EB-CS 1.4–9.6 s (6/6).

## 5. Cost (single core, OMP_NUM_THREADS=2, CPU only)

Feature extraction (decode+resize+stats): **mean 801 ms / p95 1264 ms per
10-s window**. Sequential update: **mean 0.11 ms per window** (all three
detectors, O(1) state each). Synth decode+features mean 649 ms. Total run:
≈120 s (gold) + ≈39 s (synth) + bootstrap ≈ **under 4 min**. (`results/cost.json`.)

## 6. Honest caveats

1. `n=150` windows from only 7 clips; clip-bootstrap CIs are extremely wide
   (FA CI spans ~0.04–0.71) — LOCO numbers are indicative, not precise.
2. Frames are non-i.i.d. (autocorrelated, scene-dependent); all three nominal
   guarantees are violated in practice (observed FA 3–9× nominal).
3. Mixture-CS uses train-estimated `mu0/sig0` and a Gaussian-mixture boundary,
   not the fully empirical-Bernstein variance-adaptive form — its 95% coverage
   claim is approximate here.
4. Tests detect *motion/exposure collapse* only; semantic bad reasons
   (HANDS_NOT_VISIBLE, NO_ACTIVE_WORK ≈ 74% of bad) are near-invisible to them.
   A production early-reject gate needs this as one arm (fast physical check),
   not a standalone QA verdict.
5. Brightness/Sobel series logged but unused by detectors; combining them
   (e.g. dark-clip fraction for TOO_DARK) is the obvious next step.
6. Full 150 windows + 60 synth evaluated; no subsetting. Seeds: bootstrap
   seed=0 (`loco_summary.json`).

## 7. Literature (3–6, fetched vs recalled marked)

- [fetched] Page, E.S. (1954). Continuous Inspection Schemes. *Biometrika*
  41(1–2), 100–115. https://doi.org/10.1093/biomet/41.1-2.100 (verified via
  Oxford Academic/JSTOR search).
- [recalled — not fetched] Wald, A. (1947). *Sequential Analysis*. Wiley.
  (SPRT boundaries A/B; standard textbook result, Eq. 3.2.)
- [fetched] Howard, S.R., Ramdas, A., McAuliffe, J., Sekhon, J. (2021).
  Time-uniform, nonparametric, nonasymptotic confidence sequences.
  *Ann. Statist.* 49(2), 1055–1080. https://arxiv.org/abs/1810.08240
  (verified via arXiv search; mixture boundary §3 / Thm 1).
- [recalled — not fetched] Hinkley, D.V. (1971). Inference about the
  change-point from cumulative sum tests. *Biometrika* 58(3), 509–523.
  (Page–Hinkley stopping rule; cited for the CUSUM equivalence note only.)
- [recalled — not fetched] Basseville, M., Nikiforov, I.V. (1993).
  *Detection of Abrupt Changes*. Prentice Hall, Ch. 2.
  (Siegmund ARL approximation used for the CUSUM ARL₀≈285 figure.)
