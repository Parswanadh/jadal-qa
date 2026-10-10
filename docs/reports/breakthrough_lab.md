# Breakthrough lab — ranked results (2026-10-10, evening)

Issue #65. All numbers: leave-one-clip-out over the 150 gold windows
(100 usable / 50 bad; 7 clips; reasons NO_ACTIVE_WORK 23,
HANDS_NOT_VISIBLE 14, DUPLICATE 6, TOO_DARK 6, UNDECODABLE 1),
clip-bootstrap 95% CIs. Baseline to beat: 27 per-pixel features +
logistic, LOCO AUROC **0.80 (0.71-0.87)**. Evidence per direction in
`eval/lab/<topic>/{REPORT.md, proto_*.py, results/*.json}`.
Prototypes: `proto_*.py` (CPU, OMP=2, deterministic sampling —
see each REPORT for exact protocol). Adopted modules: `jadal/lab/`
+ `tests/test_lab_*.py` (19 tests, synthetic, <1 s).

## Ranked table

| idea | theory / guarantee | LOCO result (CI) | cost / window (1 core) | determ? | stream? | hash? | verdict |
|---|---|---|---|---|---|---|---|
| 3. Sequential collapse tests (CUSUM/SPRT/mixture-CS on MAD) | Wald/SPRT + Siegmund ARL + Howard et al. time-uniform CS — nominal only; observed FA 3-9x nominal (frames violate i.i.d.) | det 0.24 / FA 0.45 pooled (semantic bad near-invisible); **physical collapse: synth black/frozen/darken 6/6 ≤1.4 s, real TOO_DARK 6/6** | 801 ms feats + 0.11 ms update | yes | yes | yes | **ADOPT narrowed**: stage-0 dark/frozen gate only → `jadal/lab/early_reject.py` |
| 4. Conformal / distribution-free risk control | CRC/LTT + Clopper-Pearson + Kish DEFF; 86-vs-84 arithmetic verified (m=4); calibration promised ≤5%, **measured 46% on held-out clips**; ICC=0.58, eff. n≈11/100 | no tighter bound exists at n=150; honest dashboard paragraph + labelling plan (new clips, ≥12-15 before clip-generalised claims) | CPU seconds (offline) | yes | n/a | yes | **ADOPT as methodology** → `jadal/lab/honest_bounds.py` |
| 6b. Permutation entropy (m=4) of MAD series | Bandt-Pompe PE; looped → low PE weakly consistent (DUPLICATE median lowest, n=6) | **0.62 (0.56-0.72)** best single info-theoretic cue; combined 0.76 vs pixel-only 0.67, gap CI [-0.02,+0.21] incl. 0 | sub-ms on saved series | yes | partial | yes | **ADOPT as feature** (never a gate) → `jadal/lab/perm_entropy.py` |
| 5. Scattering/wavelet + phase-corr + OT dedup | Mallat-style Haar modulus (no training); Kuglin-Hines phase corr; 1-D Wasserstein | SHARP_H 0.67 (0.46-0.77); PC_SHIFT 0.80 pooled but folds 0.45-0.86 (motion confound, not a gate); OT dup F1 0.73 held-out, **0 recall on brightness** unless mean-centred | 161 ms mean / 290 p95 | yes | partial | yes | **MAYBE**: OT+mean-centre paired with dHash for dedup; PC_SHIFT needs LOCO-stable validation |
| 6a. H.264 bits/frame (no pixel decode) | rate-distortion proxy; per-stream normalised | all bit cues ≤0.47 (anti-generalise across clips — rate-control confound); I-size-as-sharpness REJECTED | 324 ms (one-time ffprobe, ~0 marginal) | yes | yes | yes | **DROP** as gate/score (bits add nothing significant) |
| 1. Low-rank + sparse (SVD rank profile, MP edge, sparse frac) | PCP / randomised SVD / Marchenko-Pastur | **0.31 (0.23-0.49)** — anti-correlates; folds flip sign | 711 ms decode + 51 ms | yes | no | yes | **DROP** for activity; note: TOO_DARK rank collapse real (AUC 0.81) but brightness does it cheaper |
| 2. Koopman / exact-DMD rhythmic modes | Schmid/Tu exact DMD + windowed streaming DMD on frame-differences | rhythmic 0.41 (0.34-0.55); 8-feat 0.29; dark noise → spurious unit-circle modes | 45 ms + 265 ms decode | yes | yes | yes | **DROP** (hand work is broadband, not narrowband oscillation) |
| 7. Spectral-graph loop/dup + skin-locus hand proxy | von Luxburg graphs; YCrCb skin locus | graph null (dense-graph degeneracy: n_near_zero ≡ 1); hand proxy 0.58 null; skin fires without hands (wood confound); constructed dup_cos 1.0 but **gold DUPLICATEs are not copies** (0.46-0.72, in negative band) | ~2.9 s (decode-dominated) | yes | graph no / hand maybe | partial | **DROP** (redesign as sparse k-NN graph if revisited; gloves unmeasured) |
| Lit: NR-VQA for UGC/egocentric | NIQE/BRISQUE fail on UGC (SRCC ≤0.54); TLVQM/VIDEVAL-class needed for 0.80 — heavy/offline | sets ceiling + design rule (motion-aware features required; stage-0 = triage, not regression) | VIDEVAL 60-350 s/video (offline teacher only) | feats yes | no | no | **ADOPT direction**: NSS-naturalness reject gate + TLVQM-LCF fixed thresholds (sketched, not yet built) |
| Lit: near-dup detection | dHash/temporal fingerprints; 1M-scale Hamming studies | adoption sketch: interframe-maxima + dHash dedup/loop gate | <1 ms/pair | yes | yes | yes | **MAYBE** (pairs with OT variant above; existing pHash path already F1 0.947 on constructed) |
| Lit: ego activity w/o DL + data-centric + edge | hand-crafted motion/skin baselines; Cleanlab audit; Reducto/NoScope cascades | adoption sketches: deterministic L0→L3 cascade; Cleanlab audit of 150 labels; margin+diversity next-label picker | cascade L1/L2 integer ops | yes | yes | yes | **ADOPT direction**: cascade architecture + label audit (next) |

## Adopted modules (this PR)

- `jadal/lab/early_reject.py` — `CollapseGate` (CUSUM + dark fast-path) and
  `SprtGate` (Wald boundaries) over per-frame MAD/dark-frac. Contract: rejects
  only on collapse evidence; no-fire = undecided, never good. Defaults fitted
  on IndEgo — MUST recalibrate per deployment (see `honest_bounds`).
- `jadal/lab/perm_entropy.py` — deterministic Bandt-Pompe PE (+ `pe_loop_score`
  descriptive flag) for the stage-1 feature set.
- `jadal/lab/honest_bounds.py` — `cp_interval`, `kish_deff`/`effective_n`,
  `required_n_zero_failure` (the 86 arithmetic), `required_n_rate`,
  `dashboard_line`. Reporting discipline: observed rate + exact CI +
  clustering caveat on every number.

## What is genuinely new here (one page, honest)

To our knowledge — after two literature subagents searched NR-VQA,
near-duplicate detection, non-deep egocentric activity, data-centric QA and
edge video analytics (27 sources, fetched-vs-recalled marked in
`eval/lab/lit_*/REPORT.md`) — the following is new *as an assembled,
measured result on egocentric factory QA*, not as individual mathematics
(CUSUM, DMD, PE, CRC are all textbook; we claim no novelty for them):

1. **A failure map with numbers.** Seven deterministic directions were
   prototyped under one LOCO protocol on the same 150 windows and mostly
   failed for activity judgement (0.29-0.62 vs 0.80 baseline). Negative
   results with CIs are the contribution: they tell the next worker what
   not to build. Three independent replications (low-rank, DMD,
   residual-motion) agree on the mechanism: **egocentric background motion
   dominates raw motion spectra**, so any uncompensated motion-energy score
   anti-correlates with usable.
2. **Conformal transfer measured, not assumed.** Calibration-to-held-out-clips
   (2.7% → 46% false-reject) with ICC=0.58 is, to our knowledge, the first
   published clip-grouped risk-control stress test on egocentric QA labels;
   it converts "no certified bounds at n=150" from an arithmetic worry into
   a measured fact plus a labelling plan (new clips/wearers, ≥12-15 clips).
3. **Gold DUPLICATEs are procedural, not copies** — confirmed three ways
   (subspace dup-dup AUC 0.18, OT 0.55, signature 0.46-0.72 in the negative
   band). This reframes the dedup task: copy-detection (solved, F1 ~0.95)
   is the wrong target; procedural-repeat detection is open.
4. **A narrowed, shippable gate.** The collapse-gate keeps only the claim
   that survived (dark/frozen ≤1.4 s, hashed evidence, O(1) streaming
   state) and refuses the claim that did not (semantic QA). The honest-bounds
   module makes the reporting discipline importable.

Limits (same as each REPORT): n=150, 7 clips, 2 wearers in paths, one
dataset, no deployment footage; CIs wide; thresholds not transferable
across clips. Nothing here is "state of the art" or "first" beyond the
assembly above.
