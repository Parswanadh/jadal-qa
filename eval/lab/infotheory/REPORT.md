# Direction 6 — Information-theoretic cues: REPORT

Worktree `feat/breakthrough-lab`. Gold manifest: 150 ten-second windows, 7 clips
(100 usable / 50 bad). Reasons: GOOD 100, NO_ACTIVE_WORK 23,
HANDS_NOT_VISIBLE 14, DUPLICATE 6, TOO_DARK 6, UNDECODABLE 1. CPU-only,
`OMP_NUM_THREADS=2`, runtime **194 s** total. Prototype:
`eval/lab/infotheory/proto_infotheory.py`; numbers: `results/00_manifest.json`
… `results/06_cost.json` (+ `04_scores.json` LOCO scores per cue).

## Method

**(a) Compressed-domain (no pixel output).** One full-clip
`ffprobe -show_frames` pass per clip (7 passes, 48.4 s total), parsed to
`(timestamp, pkt_size, pict_type)`; windows sliced by timestamp in Python (no
per-window seeking — seek was verified inaccurate: a `-read_intervals 14.6%+11`
probe landed at 4.6 s). Per-window: P-mean/median/p10, I-mean, P-byte-fraction,
P-count-fraction, bytes/frame. **Per-stream normalisation** (sibling-doc
rate-control warning): `p_mean_z`, `i_mean_z`, `bytes_pf_z` are z-scores vs the
clip-global median/IQR computed over *all* frames of that clip — label-free, no
leakage. Caveat honestly noted: `ffprobe -show_frames` runs the decoder
internally; "no pixel decode" means no pixel buffers are produced or touched by
us — the information used (type + packet size) is demux/parse-level.

**(b) Permutation entropy.** Pixel path decodes each window (`cv2`, 64×64 gray,
~5 fps → 50 frames), builds the frame-difference MAD series (49 diffs), and
computes normalised Bandt–Pompe PE (÷ ln(m!)), m = 3 and m = 4, τ = 1, ties
broken by position order (deterministic). Pixel baselines on the same frames:
`mad_mean`, `sharp_mean` (Laplacian variance).

**(c) Evaluation.** Identical LOCO protocol to sibling dirs (L2-logistic C=1,
train-standardised, cross-fitted P(usable) scores) for every cue, univariate and
grouped; clip-bootstrap (resample 7 clips w/ replacement, B=2000, seed 6) 95%
CIs; bootstrap CI on AUROC gaps with shared resamples.

## Results (LOCO AUROC [clip-bootstrap 95% CI])

| cue | family | AUROC | 95% CI |
|---|---|---|---|
| `p_mean_z` | compressed | 0.469 | [0.345, 0.708] |
| `i_mean_z` | compressed | 0.333 | [0.286, 0.593] |
| `p_byte_frac` | compressed | 0.269 | [0.227, 0.454] |
| `bytes_pf_z` | compressed | 0.262 | [0.180, 0.492] |
| `compressed_all` (4 feats) | compressed | 0.385 | [0.289, 0.633] |
| `pe_m3` | PE | 0.562 | [0.510, 0.683] |
| `pe_m4` | PE | **0.618** | [0.555, 0.723] |
| `pe_all` | PE | 0.620 | [0.561, 0.723] |
| `mad_mean` | pixel | 0.573 | [0.437, 0.739] |
| `sharp_mean` | pixel | 0.545 | [0.385, 0.735] |
| `pixel_all` | pixel | 0.668 | [0.502, 0.784] |
| `compressed+pixel` | combined | 0.727 | [0.603, 0.820] |
| `all` (+PE) | combined | **0.761** | [0.656, 0.829] |

Gaps (shared-resample bootstrap): `all − pixel_all` = +0.094 [−0.018, +0.210];
`compressed_all − pixel_all` = −0.231 [−0.445, +0.071].

Per-reason medians (`p_mean_z / i_mean_z / mad / pe3`):
GOOD .16/.00/14.0/.97 · NO_ACTIVE_WORK .05/.00/17.4/.95 ·
HANDS_NOT_VISIBLE .23/.02/21.9/.94 · DUPLICATE .04/.10/23.0/.93 ·
TOO_DARK **−0.73**/.00/7.1/.96 · UNDECODABLE(n=1) −.51/−.74/18.4/.98.

## Hypothesis verdicts — mostly honest negatives

- **Frozen → P≈0: UNTESTABLE on gold.** No FROZEN reason exists; the closest,
  NO_ACTIVE_WORK, has median `p_mean_z` ≈ +0.05, indistinguishable from GOOD.
  Static-but-valid scenes appear to keep P-frames non-trivial (encoder noise
  floor), so a "P≈0 ⇒ frozen" rule has no support here.
- **Black/blocked → I tiny: REJECTED.** `i_mean_z` median is ≈0 for every
  reason including TOO_DARK; univariate AUROC 0.33 (anti-predictive). At these
  bitrates I-frame size tracks rate control, not scene content. I-size is not a
  sharpness proxy on this data.
- **Activity → P large: WEAK/CONFOUNDED.** Raw direction reverses across clips
  (univariate compressed AUROCs < 0.5 even after per-stream normalisation), the
  predicted rate-control confound. One real crumb: TOO_DARK windows have
  markedly small P-frames (`p_mean_z` −0.73) and low MAD (7.1) — darkness reads
  as *absence* of bits, but it does not separate the other bad reasons, whose
  P-stats overlap GOOD.
- **Looped/periodic → low PE: WEAKLY CONSISTENT, underpowered.** DUPLICATE
  (n=6) has the lowest median PE (m3 0.93 vs GOOD 0.97); `pe_m4` is the best
  single cue at 0.62 [0.56, 0.72]. Suggestive, not conclusive.
- **Does bits/frame add anything over MAD/sharpness? BARELY, NOT
  SIGNIFICANTLY.** Combined 0.76 vs pixel-only 0.67; gap CI [−0.02, +0.21]
  includes 0, and compressed-alone (0.39) is *worse* than pixel-alone. The lift
  comes from ensembling weak diverse cues, not from bits carrying independent
  signal. Do not ship a bits-only gate.

## MDL-flavoured summary ('useful bits' — analysis, not a feature)

Through an MDL lens, a window is usable when its frame sequence is cheap to
describe *given a work-activity model* but expensive given a noise-only model:
useful-bits ≈ bits(actual encoding) − bits(same frames under i.i.d.-noise
prediction). Our measurements are a crude observable of this: TOO_DARK windows
are cheap under *both* models (tiny P-frames, MAD 7.1 — a static dark frame is
nearly free to encode and nearly free to predict), i.e. ~zero useful bits,
which is why darkness is the one failure mode bits detect. Conversely the
failure of I-size and raw P-size shows the confound MDL warns about: the
encoder's rate controller is itself a (mismatched) model, so raw codelength
confounds scene compressibility with the controller's bit budget — codelength
is only meaningful relative to a fixed reference model, which per-clip
normalisation approximates but does not achieve. A principled follow-up would
fix the model (re-encode window slices at fixed QP and compare residual
codelengths) rather than reading off the shipped encode's packet sizes.

## Cost

Compressed path: 48.6 s / 150 = **324 ms/window** (all ffprobe; parsing ~0.2 s;
amortised over windows in a clip — marginal cost per extra window ≈ 0).
Pixel path: 58.0 s / 150 = **386 ms/window** (decode 64×64 + MAD + PE).
On this box the "no-decode" saving is real but modest (~16%) because one
full-clip ffprobe pass still decodes headers; per-window marginal cost strongly
favours the compressed path once the clip table is cached.

## Verdicts

- **Deterministic? YES**, given identical ffprobe/cv2 builds: features use no
  RNG (PE ties → position order; bootstrap seed 6 fixed). Re-run variance
  expected only from `POS_MSEC` seek rounding across OpenCV builds.
- **Streaming? COMPRESSED YES** (per-frame type+size rows arrive in order; P/I
  running stats + frozen/dark flags are O(1) state), **PE PARTIAL** (needs the
  ~50-sample MAD buffer per window; m≤4 histograms are tiny).
- **Hashable? YES** (fixed float vector per window id; round to 1e-6 for stable
  hashes).

## Literature (all three fetched live 2026-10-10; none invented)

1. C. Bandt & B. Pompe, "Permutation entropy: a natural complexity measure for
   time series", *Phys. Rev. Lett.* 88:174102 (2002). — **fetched** (doi.org
   resolver returned citation). https://doi.org/10.1103/PhysRevLett.88.174102
2. R. Cilibrasi & P. Vitányi, "Clustering by compression", *IEEE Trans. Inf.
   Theory* 51(4) (2005) — compression-as-complexity (NCD) theoretical backing
   for codelength-based similarity. — **fetched** (arxiv abs page).
   https://arxiv.org/abs/cs/0312044
3. FFmpeg project, `ffprobe` documentation (`-show_frames`, `-read_intervals`
   seek-inaccuracy semantics we hit empirically). — **fetched**.
   https://ffmpeg.org/ffprobe.html

## Limits / next steps

Single-label gold (n=150, 7 clips); CIs are wide and all "wins" are
non-significant. Synthetic manifest (`manifest_synth.json`) not used — LOCO
protocol kept gold-identical to siblings. Not tried: fixed-QP re-encode
residuals (the MDL-honest bit measure), PE on brightness/phase series, P-size
autocorrelation spectra for loop detection.
