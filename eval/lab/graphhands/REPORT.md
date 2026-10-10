# Direction 7 — Spectral-graph loop/duplicate detection + learning-free hand-activity proxy

Prototype: `proto_graphhands.py`. Results: `results/{window_features.json,loco.json,dup_pairs.json,gold_dup_cross.json}`.

## Method (deterministic, CPU-only, read-only sampling)

Per 10 s gold window, **N=20 frames** at fixed timestamps (0.5 s spacing, 96×96 working
resolution, `OMP_NUM_THREADS=2`):

- **(a) Frame-similarity graph.** Nodes = frames; edge weights from cosine similarity of
  zero-mean 16×16 luma vectors (plus a parallel dHash–Hamming `1−ham/64` matrix).
  Features: unnormalized-Laplacian eigenvalues (`lam1`, `spectral_gap=lam1/lam_max`,
  `n_near_zero` = #{lam < 1e−3·lam_max}), `mean/std_offdiag`, `frozen_frac` (adjacent
  cosine > 0.995), loop profile `prof[k]=mean diag(S,k)`: `loop_strength = max_{k≥3}prof − median(prof)`,
  `loop_period_lag`, `loop_peak`.
- **(b) Hand-activity proxy (no MediaPipe).** Phase-correlation (`cv2.phaseCorrelate`, Hann
  window) aligns consecutive frames → global-motion-removed residual `|cur − warp(prev)|`.
  YCrCb skin locus mask (Y>40, 133≤Cr≤173, 77≤Cb≤127) on the lower-frame ROI (y∈[0.5,1]).
  `hand_activity` = skin-union-weighted mean residual; also `resid_raw`, `skin_coverage`,
  `mean_shift_px`.

**Evaluation.** Unsupervised scores under LOCO over the 7 clips: per-held-out-clip AUROC
(no fitting, fixed orientation) + pooled AUROC with **clip-bootstrap 95% CI (B=2000, seed 0)**.
Dup scores (`dup_cos` = mean row/col-max cross-cosine, `dup_dhash`) on all **128 constructed
pairs** (40 dup / 88 neg) with pair-bootstrap CI. All AUROCs reported **usable-positive**
(higher = usable); bad-positive features (frozen/offdiag/loop) are negated, flagged `(neg)`.

## Results — gold windows (LOCO)

| score | pooled AUROC | 95% CI | per-clip spread | verdict |
|---|---|---|---|---|
| `skin_coverage` | 0.662 | [0.573, 0.807] | 0.474 (desk) – 1.000 (pallet2, n=5) | weak; see caveat |
| `hand_activity` | 0.575 | [0.467, 0.739] | 0.400–0.856, CI covers 0.5 | **null** |
| `loop_strength` | 0.531 | [0.418, 0.639] | inconsistent sign | **null** |
| `frozen_frac` (neg) | 0.386 | [0.328, 0.516] | ≈0.4–0.6 everywhere | **null** |
| `resid_raw` | 0.396 | [0.184, 0.631] | ecu_full 0.169, ecu_late 0.100 | **anti-correlates** (replicates sibling ego-motion finding) |
| `mean_offdiag` (neg) | 0.292 (= 0.708 bad-positive) | [0.174, 0.453] | driven by ecu_full (4 bad windows); desk 0.469, pallet 0.500 | **not LOCO-stable, reject** |
| `loop_peak` (neg) | 0.278 | [0.157, 0.466] | same ecu_full artefact | **reject** |
| `n_near_zero` | 0.500 | [0.500, 0.500] | constant = 1.0 on all 144 windows | **degenerate by construction** |
| `spectral_gap` | 0.395 | [0.222, 0.593] | 0.041–0.833 across clips | **null/unstable** |

Key points:

1. **Clean negatives with reasons.** `n_near_zero` is exactly 1.0 everywhere because the graph
   `(S+1)/2` is dense (all weights > 0) — a fully connected graph has one zero eigenvalue
   *by construction*. Component counting needs a sparsified (k-NN/ε) graph; as built the
   feature cannot work. `loop_strength`/`frozen_frac` carry no usable-vs-bad signal on this
   footage (no loops/frozen segments exist in gold; badness is semantic, not structural).
   The pooled 0.708 bad-positive `mean_offdiag` is an ecu_full small-sample artefact
   (4 bad windows, all below the good mean 0.756±0.106) that vanishes on desk/pallet —
   it fails the clip-generalization bar, so we reject it.
2. **Raw motion anti-correlates** (`resid_raw` 0.396; 0.313 excluding TOO_DARK): bad windows
   move *more* — ego-motion dominates, confirming the sibling result. Global-motion removal
   via phase correlation does not rescue it (`hand_activity` null).
3. **Glove/skin caveat — quantified.** `skin_coverage` in lower ROI: GOOD 0.178 vs
   HANDS_NOT_VISIBLE 0.151 vs NO_ACTIVE_WORK 0.133. The locus fires nearly as much when
   hands are *absent* (warm wood/desks are classic YCrCb false positives; desk-clip AUROC =
   0.474), and TOO_DARK suppresses it trivially (0.043, Y>40 fails). So `skin_coverage`
   (pooled 0.662) is better read as a "bright warm static material" proxy than a hand
   detector, and under gloves the true-hand recall is unmeasured-but-presumed-poor. Any
   deployment needs glove-aware calibration, not this locus.

## Results — duplicates

- **Constructed pairs (128): `dup_cos` AUROC = 1.000 [1.000, 1.000]; `dup_dhash` ≈ 1.000.**
  Per-kind means (`dup_cos`): reencode 1.000, bright 1.000, offset1 0.977, crop5 0.969,
  speed11 0.958 vs hard-neg 0.683, easy-neg 0.569. Near-copy detection with compact
  signatures is solved — but only for *copies*.
- **Gold DUPLICATEs are not copies** (`results/gold_dup_cross.json`, same `xsim` metric):
  dup–dup pairs 0.462–0.724 (inside the hard-negative band, far below the 0.958
  constructed-dup floor), overlapping dup–good matched pairs (0.354–0.632). Signature
  similarity **cannot** catch the gold DUPLICATE class — clean negative, reason identified
  (procedural/semantic duplicates, consistent with sibling note).

## Cost / runtime / verdicts

- Cost: **mean 2897 ms / median 2698 ms per window** (20-frame seeks+decodes dominate;
  eigendecomposition of 20×20 is negligible). Total run **≈9.0 min CPU** (< 15 min budget).
- Deterministic subset: **144/150 windows** decoded. 6 windows fail deterministically
  (content-independent): g047, g061, g090, g096, g113, g149 — window end at/past EOF
  duration, so end-of-window seeks return no frame (loses 2 of 6 gold DUPLICATEs).
  Determinism self-check (recompute first window): max_abs_diff = 0.0 → **PASS**;
  bootstrap seeded (0); no RNG in features → **deterministic: YES**.
- Streaming: graph features **NO** (need full-window N×N matrix + eigendecomposition,
  batch-only). Hand proxy **YES-able** (per-frame skin mask + adjacent-frame phase
  correlation + running means, O(1) frame state).
- Hashable: frame signatures **YES** (64-bit dHash + 256-D luma vectors are compact,
  storable, indexable — the dup detector already operates on them). Hand-activity
  **NO** (flow/state-dependent, not a stable hash).

## Verdict

Both hypotheses fail as usable/bad classifiers on gold, for identified reasons (dense-graph
degeneracy; no loops in gold; ego-motion dominance; skin locus confounded by materials and
blind under gloves). The only keeper: cheap near-copy signatures (`dup_cos`/`dup_dhash`)
for constructed-duplicate retrieval — disjoint from the gold DUPLICATE problem.
**Recommendation: do not pursue either as a quality gate without (i) sparse-graph
reformulation, (ii) glove-aware hand features, (iii) ego-motion-robust normalization.**

## Literature

1. U. von Luxburg, "A Tutorial on Spectral Clustering", *Statistics and Computing* 17(4),
   2007. https://arxiv.org/abs/0711.0189 — **[fetched 2026-10-10]** (graph Laplacians,
   spectra vs connectivity; underpins §(a)).
2. P. Kakumanu, S. Makrogiannis, N. Bourbakis, "A survey of skin-color modeling and
   detection methods", *Pattern Recognition* 40(3):1106–1122, 2007.
   https://doi.org/10.1016/j.patcog.2006.06.010 — **[fetched via search 2026-10-10]**
   (YCrCb locus used in §(b); documents wood-like false positives).
3. B. S. Reddy & B. N. Chatterji (and Foroosh–Zerubia–Berthod extension), phase-correlation
   image alignment; H. Foroosh, J. B. Zerubia, M. Berthod, "Extension of phase correlation
   to subpixel registration", *IEEE Trans. Image Processing* 11(3), 2002.
   https://doi.org/10.1109/41.924928 — **[recalled, not fetched]** (global-motion removal).
4. N. Krawetz, "Looks Like It" (dHash perceptual hash), Hacker Factor blog.
   http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html —
   **[recalled, not fetched]** (difference-hash used for `dup_dhash`/graph variants).
5. T. Fawcett, "An introduction to ROC analysis", *Pattern Recognition Letters* 27(8), 2006.
   https://doi.org/10.1016/j.patrec.2005.10.010 — **[recalled, not fetched]** (AUROC
   evaluation; Mann–Whitney interpretation).
