# Direction 5 — Scattering/wavelet-modulus + phase-correlation + optimal-transport cues (REPORT)

Subagent output. Worktree `/home/parshu/projects/jadal-qa-wt/breakthrough-lab`,
branch `feat/breakthrough-lab`. Code: `eval/lab/scatter/proto_scatter.py`,
machine tables: `eval/lab/scatter/results/*.json`. Nothing committed/pushed.

## 1. Exact formulation

**Sampling (deterministic, read-only, CPU only, OMP_NUM_THREADS=2).**
Each 10-s gold window: cv2 seek to `start_s`, keep every 5th decoded frame,
cap 12 kept frames, gray, width-160 (wavelet/histogram) + 64×64 copy (FFT).
Dup files: same from 0 s. No RNG in features (bootstrap seed=0 only).
Full 150 gold windows + 48 dup files evaluated; no subsetting.

**(a) Wavelet-modulus blur/texture (training-free, no learned weights).**
1-level Haar DWT (`pywt.dwt2`) per kept frame on 160-wide luma — a 1-scale
scattering-style modulus mean:
`SHARP_H = mean_t mean(|cH|+|cV|+|cD|)/3`, `SHARP_R = mean_t high/mean|cA|`,
plus classical `LAPL = mean_t Var(Laplacian)`. Hypothesis: higher = sharper =
more usable (label 1).

**(b) Phase correlation (Kuglin & Hines) between consecutive kept frames.**
`R = F1·conj(F2)/(|F1||F2|+eps)`, `pc = |ifft2(R)|` on 64×64 luma.
`PC_PEAK` = mean peak, `PC_FROZ` = fraction with peak>0.4 and shift≤1 px
(frozen/duplicate cue), `PC_SHIFT` = median shift magnitude (ego-motion estimator).

**(c) Optimal-transport deduplication.** Per-window pooled 32-bin luma
histogram; pair distance = exact 1-D Wasserstein
(`scipy.stats.wasserstein_distance` on bin centers — the closed form behind
each sliced-Wasserstein projection). Lower = more duplicate. Secondary:
1-D Wasserstein between frame-mean series (logged, not thresholded).
Brightness-invariant variant (§4): recenter each histogram at its own mean
before OT (post-hoc, same protocol, no re-decode).

## 2. Quality/activity: LOCO AUROC + clip-bootstrap 95% CI

7 clips; pooled AUROC over 150 windows (sign fixed a priori: sharp high =
usable; phase-corr peak/frozen/shift tested in the stated direction);
CI = 1000 clip-bootstraps (resample 7 clips w/ replacement, seed 0).
`results/loco_quality.json`.

| Score | Pooled AUROC | 95% CI | Mean per-fold | Fold range |
|---|---|---|---|---|
| SHARP_H (wavelet) | **0.670** | [0.46, 0.77] | 0.624 | 0.35–0.87 |
| SHARP_R | 0.587 | [0.45, 0.70] | 0.591 | 0.35–1.00 |
| LAPL | **0.664** | [0.46, 0.77] | 0.645 | 0.38–0.93 |
| PC_PEAK (frozen cue) | 0.246* | [0.19, 0.40] | 0.286 | 0.00–0.62 |
| PC_FROZ | 0.305* | [0.23, 0.46] | 0.240 | 0.00–0.39 |
| PC_SHIFT (ego-motion) | **0.800** | [0.67, 0.84] | 0.704 | 0.45–0.86 |

\* scored "higher peak = usable"; AUROC<0.5 means the effect points the other
way (bad windows have *higher* peaks, i.e. are more static — consistent with
NO_ACTIVE_WORK/HANDS_NOT_VISIBLE bad reasons). Flipped, PC_PEAK ≈ 0.75 pooled
for the bad direction.

**Reading.** Wavelet blur statistics are weak quality gates here (0.59–0.67):
most bad labels are semantic (hands not visible, no active work), not optical
blur — same lesson as sibling directions. The surprise is PC_SHIFT at 0.80
pooled: inter-frame ego-motion magnitude separates usable from bad windows,
but folds vary widely (pallet 0.45 vs logistics4 0.86) and the CI is wide —
a motion *confound*, not a gate. Phase-correlation peak as a frozen cue is
directionally consistent (bad = more static) but too weak/volatile alone
(pallet2 fold 0.00 — degenerate single-class-ish fold).

## 3. Dedup: constructed pairs (primary) + 6 real DUPLICATEs (descriptive)

128 constructed pairs (40 dup / 88 non-dup). Threshold = max-F1 on even-indexed
pairs, evaluated on odd-indexed pairs. `results/dup_eval.json`.

| Variant | Thr | Train P/R/F1 (n=64) | Test P/R/F1 (n=64) | Test recall by kind |
|---|---|---|---|---|
| Raw luma-hist OT | 5.29 | 0.889/0.800/**0.842** | 0.667/0.800/**0.727** | reencode/crop5/speed11/offset1 = 1.00; **bright = 0.00**; FP 0.18/0.18 (hard/easy neg) |
| Mean-centered OT | 5.91 | 0.769/1.000/0.870 | 0.633/0.950/0.760 | bright = **0.75**; FP 0.32/0.18 |

**Honest verdict.** OT-on-luminance-histograms deduplicates re-encode, 5%-crop,
speed and offset variants near-perfectly (F1 ≈ 0.73–0.76 held-out) but is
**blind to the brightness perturbation by construction** (a global histogram
shift is exactly what Wasserstein penalises); mean-centering recovers bright
recall 0→0.75 at the cost of more hard-negative FPs. Luminance-OT is therefore
a *partial* dup cue — it must be paired with a brightness-invariant hash
(e.g. gradient/dHash per sibling lit review), never used alone.

**Real gold DUPLICATEs (descriptive, n=6 → 15 dup-dup pairs vs 200 background
pairs):** dup-dup median 15.2 vs background median 16.6 (bg 5th pct 5.77);
dup-dup AUC vs background = **0.55** ≈ chance. Consistent with the sibling
finding (their dup-dup AUC 0.18 negative control): the 6 gold DUPLICATEs are
not copies of each other, so no dup detector should fire on them — and ours
doesn't. Constructed-pair metrics above remain the primary evidence.

## 4. Cost (single core, OMP_NUM_THREADS=2, CPU only)

Decode + all features: **mean 161 ms / p50 142 ms / p95 290 ms / max 433 ms
per 10-s window** (n=198 incl. dup files). OT distances are sub-ms per pair
(32-bin 1-D closed form). Total run ≈ under 6 min. `results/cost.json`.

## 5. Deterministic? / Streaming? / Hashable?

- **Deterministic: yes.** Fixed seek/stride/cap/resize, closed-form
  statistics, no RNG (bootstrap seed fixed). Audit sha in driver stdout.
- **Streaming: partial.** Wavelet/phase-corr/histogram features are per-frame
  and poolable over 2-s chunks (histograms additive, means running); the OT
  *distance* needs two complete window histograms, so the verdict is
  window-boundary, not per-chunk.
- **Hashable: yes with quantization.** Scores + quantized histograms hash
  cleanly (sha over `id|sharp|peak|hist` demonstrated in driver).

## 6. Honest caveats

1. n=150 windows from 7 clips; clip-bootstrap CIs span ~0.3 — LOCO numbers are
   indicative, not precise.
2. Wavelet blur scores (0.59–0.67) cannot carry a quality gate: bad labels are
   ~74% semantic (HANDS_NOT_VISIBLE, NO_ACTIVE_WORK), invisible to texture stats.
3. PC_SHIFT 0.80 is a motion confound with fold range 0.45–0.86 — not shippable.
4. Luminance-OT is brightness-blind by construction (0.00 bright recall raw);
   centered variant trades precision for recall. Pair with gradient hashing.
5. Only 1 Haar scale (scattering depth 1, no 2nd-order coefficients, no
   Morlet bank): a deliberately minimal "fast" scattering proxy; 2nd-order
   texture discrimination (Mallat's higher-moment argument) is untested here.
6. Threshold protocol (even/odd pair split) mixes sources across splits —
   optimistic vs a leave-one-source-out scheme; F1 ≈ 0.73 should be read as an
   upper-bound-ish estimate.

## 7. Literature (fetched vs recalled marked; nothing invented)

- [fetched — search excerpt] Bruna, J. & Mallat, S. (2012). Invariant
  Scattering Convolution Networks. https://arxiv.org/abs/1203.1513
  (translation-invariant wavelet-modulus cascade; texture discrimination via
  higher-order moments — our 1-scale Haar-modulus mean is its minimal prefix).
- [fetched — search excerpt] Mallat, S. (2012). Group Invariant Scattering.
  *Comm. Pure Appl. Math.* 65, 1331–1398. https://arxiv.org/abs/1101.2286
  (Lipschitz stability to diffeomorphisms; justifies modulus over raw wavelet).
- [fetched — search excerpt] Bonneel, N., Rabin, J., Peyré, G. & Pfister, H.
  (2015). Sliced and Radon Wasserstein Barycenters of Measures. *J. Math.
  Imaging Vis.* 51, 22–45. https://perso.liris.cnrs.fr/nicolas.bonneel/WassersteinSliced-JMIV.pdf
  (1-D projections make OT tractable; our 1-D histogram Wasserstein is the
  single-projection closed form).
- [fetched — search excerpt] Kuglin, C.D. & Hines, D.C. (1975). The Phase
  Correlation Image Alignment Method. *Proc. IEEE Int. Conf. Cybernetics and
  Society*, 163–165. (FFT normalized cross-spectrum peak = translation;
  peak height = frozen-cue basis. Bibliographic details corroborated by
  Keller 2007 review and PMC11937376.)
- [fetched — search excerpt] Tong, H., Li, M., Zhang, H. & Zhang, C. (2004).
  Blur detection for digital images using wavelet transform. *Proc. IEEE ICME*,
  17–20. https://doi.org/10.1109/ICME.2004.1394114
  (Haar-wavelet edge-type/sharpness blur rule — direct precedent for SHARP_H/R).
- [recalled — standard library, verified by local `scipy` docstring, not a
  paper claim] `scipy.stats.wasserstein_distance` = exact 1-D Earth Mover's
  Distance via CDF integration (Vallender 1974 identity, standard textbook).
