# Direction 2 — Koopman / exact-DMD for egocentric activity — REPORT

Subagent prototype. Worktree `/home/parshu/projects/jadal-qa-wt/breakthrough-lab`,
branch `feat/breakthrough-lab` (not switched; nothing committed/pushed).
Code: `eval/lab/koopman/proto_koopman.py`; data: `eval/lab/koopman/results/*.json`.

## 1. Method

**Exact DMD** (Schmid 2010; Tu et al. 2014): snapshot matrices `X1=[x1..x_{m-1}]`,
`X2=[x2..xm]`, fit `X2 ≈ A X1` via truncated SVD `X1 = U S V*` (rank 15),
reduced operator `Ã = Uᵀ X2 V S⁻¹`, eigendecomposition `ÃW = WΛ`,
exact modes `Φ = X2 V S⁻¹ W`. Continuous-time `ω = log(λ)/dt` gives
frequency `|Im ω|/2π`, damping ratio `-Re ω/|ω|`; amplitudes `b = Φ⁺x1`,
mode energies `|b|²`. Reconstruction quality = one-step relative residual
`‖X2 − A_approx X1‖/‖X2‖`.

**Windowed/streaming variant** (in the spirit of Hemati et al. 2014): independent
rank-8 exact DMD on sliding 2-s windows (12 frames @ 6 fps, 1-s stride, ~9
subwindows per 10-s clip); aggregates: mean/std of rhythmic fraction, fraction
of active subwindows, std of dominant frequency. Genuinely streaming-capable:
only a 2-s buffer is ever needed.

**Input choice (deliberate): FRAME DIFFERENCES, not raw luma.** Sibling
direction-1 found raw-luma rank features *anti-correlate* with usable labels
(ego-motion swamps hands). DMD here runs on `d_t = f_{t+1} − f_t`, grayscale
32×32, 60 frames/10 s (6 fps effective), flattened to 1024-dim snapshots.
Differencing is a cheap motion-compensation: static background and constant
ego-translation cancel to first order, leaving transients (hands/tools). No
optical-flow warping (CPU budget).

**Primary score:** `win_mean_rhythmic_frac` = mean over subwindows of mode-energy
fraction in the rhythmic band (0.3–8 Hz, 0.80 ≤ |λ| ≤ 1.20, near unit circle).
Secondary: 8-feature logistic regression (rhythmic frac, log rhythmic energy,
dominant freq/damping, residual, windowed stats, static fraction).

**Hypothesis:** rhythmic manual work → oscillatory modes near unit circle (high
score); frozen → eigenvalue ≈1 at zero freq (high `static_frac`, low score);
dark → near-zero total energy (low score).

## 2. Evaluation (as required: leave-one-clip-out, 7 clips)

Threshold (Youden-J) and logistic models **trained on train clips only**,
applied to the held-out clip; pooled out-of-fold AUROC; **clip-bootstrap
(1000 resamples) 95% CI**.

| model | pooled AUROC | 95% CI |
|---|---|---|
| rhythmic-fraction score | **0.41** | [0.34, 0.55] |
| 8-feat logistic | **0.29** | [0.24, 0.47] |

Per-clip AUROC is unstable and mostly < 0.5 (only `ecu_late` score 0.78 and
`pallet` logistic 0.82 exceed chance, on n=14/17 — noise). No single feature
separates even pooled/in-sample (best: `win_std_domfreq` 0.60, `full_mean_mag`
0.54; primary score 0.41 pooled; `full_total_energy` 0.39).

**Per-reason descriptives (mean primary score):** usable (`null`, n=100) 0.080;
NO_ACTIVE_WORK (23) 0.080; DUPLICATE (6) 0.085; HANDS_NOT_VISIBLE (14) 0.101;
**TOO_DARK (6) 0.138 — highest.** Dark sensor noise produces spurious
near-unit-circle eigenvalues, exactly inverting the "dark = no energy"
hypothesis. Frozen/static windows also reconstruct almost perfectly through the
λ≈1 mode, so the residual carries no signal either (pooled AUC 0.46).

## 3. Verdicts

- **Deterministic? YES** — rerun abs diff 0.0 (uniform sampling, fixed ranks,
  numpy SVD; `determinism.deterministic=true` in summary.json).
- **Streaming? YES** — windowed variant needs only a 2-s buffer; per-subwindow
  cost ≈ 5 ms; no full-window state.
- **Hashable? YES** — `round(featvec,6) → json(sort_keys) → sha256`
  (example `e387e6ff27551a02`).

## 4. Cost per window (single core, OMP=2)

- Decode (amortized, sequential single-pass per clip): **≈265 ms**
- DMD (full rank-15 + ~9 rank-8 subwindows): **≈45 ms** (p50 48 ms)
- Total incl. decode ≈ 310 ms; **excl. decode ≈ 45 ms**.

Note: an earlier per-window random-seek decoder exceeded the 14-min budget
(H.264 seeks); replaced by one sequential pass per clip (7 clips). Decode cost
above is the amortized sequential figure.

## 5. Comparison — an honest negative result

| approach | AUROC |
|---|---|
| task baseline | 0.80 |
| direction-1 (raw-luma rank) | 0.31 |
| **direction-2 DMD rhythmic (this)** | **0.41 / 0.29** |

DMD rhythmic energy does **not** beat the baseline and is no better than
direction-1's negative control. Failure analysis: (a) at 6 fps / 32×32,
hand-work transients are broadband, not narrowband oscillation — energy spreads
across the spectrum instead of concentrating in 0.3–8 Hz unit-circle modes;
(b) ego-motion residuals after differencing still dominate hand motion;
(c) dark-clip sensor noise creates fake oscillatory modes (TOO_DARK scores
highest). Frame-differencing fixed direction-1's raw-luma confound but the
DMD spectral signature itself carries no usable-vs-bad signal here.
**Recommendation: do not pursue DMD-band-energy for this task**; if Koopman
ideas are revisited, try observables richer than pixels (e.g. DMD on tracked
keypoint trajectories) rather than pixel-DMD.

## 6. Literature (all 4 URLs fetched via web search 2026-10-10; content recalled)

1. P. J. Schmid, "Dynamic mode decomposition of numerical and experimental
   data," *J. Fluid Mech.* 656:5–28, 2010. doi:10.1017/S0022112010001217 —
   https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/dynamic-mode-decomposition-of-numerical-and-experimental-data/AA4C763B525515AD4521A6CC5E10DBD4 (fetched URL, recalled content)
2. J. H. Tu, C. W. Rowley, D. M. Luchtenburg, S. L. Brunton, J. N. Kutz, "On
   dynamic mode decomposition: theory and applications," *J. Comput. Dyn.*
   1(2):391–421, 2014. doi:10.3934/jcd.2014.1.391 —
   https://cwrowley.princeton.edu/papers/Tu-DMD.pdf (fetched URL, recalled content)
3. M. S. Hemati, M. O. Williams, C. W. Rowley, "Dynamic mode decomposition for
   large and streaming datasets," *Phys. Fluids* 26:111701, 2014. arXiv —
   https://arxiv.org/pdf/1406.7187 (fetched URL, recalled content)
4. I. Mezić, "Analysis of fluid flows via spectral properties of the Koopman
   operator," *Annu. Rev. Fluid Mech.* 45:357–378, 2013.
   doi:10.1146/annurev-fluid-011212-140652 —
   https://www.annualreviews.org/content/journals/10.1146/annurev-fluid-011212-140652 (fetched URL, recalled content)

No citations invented; DOIs/venues as listed by publishers.

## 7. Reproducibility

```
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /home/parshu/envs/jadal/bin/python \
  eval/lab/koopman/proto_koopman.py \
  --manifest /home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json \
  --outdir eval/lab/koopman
```

Full 150 windows, 7 clips, CPU-only, numpy+cv2+sklearn. Runtime ≈ 5–8 min
single core. Only `eval/lab/koopman/` touched.
