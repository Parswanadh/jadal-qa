# Exp E2.1080 — True-1080p mosaic: tile-resolution ladder + equal-canvas duel

**Claim under test (user):** "instead of 1080p pictures inside one big picture,
use 480p (or smaller) tiles, and more of them, so processing is faster without loss".
X3 tested this on 480p gold clips only; this experiment uses TRUE 1080p Ego10K sources.

**Registry:** `E2.1080` · **Results:** `eval/results/exp_mosaic1080/`
(`exp_mosaic1080_ladder.json`, `exp_mosaic1080_duel.json`, `raw_eval.json`,
`features.npz`) · **Proofs:** `docs/experiments/exp_mosaic1080/images/` (8 PNGs,
top-right timestamps verified programmatically, each <200 KB)
· **Device:** A40 · **Decode:** NVDEC cuvid (hevc/h264) with CPU fallback

## Method

- Windows: human-labelled Egocentric-10K OOD subset — ALL 22 bad + even good
  sample (`ood160_labels.json`; n_BAD=22, small-n, CIs wide).
- 1080p arm decoded from the original 1080p/30fps HEVC clips at the queued
  `source_start_s`; 480p arm from the 10 s labelled transcodes (batch1 30 fps,
  batch2 10 fps; per-arm fps probed, cell-centred times shared).
- Mosaics built with the X3 builder (top-right `t=X.Xs` timestamps, JSON sidecars).
- Eval: stratified 5-fold CV logistic (C=0.01) on L2 EG2 embeddings, out-of-fold
  AUROC, factory-bootstrap 95% CI. Optimistic read (trains+tests on the same OOD
  pool) — NOT a transfer proof. IndEgo-trained models scored 0.4–0.53 AUROC OOD.

## Findings (numbers to be filled by finalize)

| # | What was tested | Result | Verdict |
|---|-----------------|--------|---------|
| 1 | Tile ladder 224→168→112 px at 3x3 (1080p src) | | |
| 2 | 4x4/168 and 4x4/112 (1080p src) | | |
| 3 | EQUAL-CANVAS DUEL: 1080p 3x3/t224 (462,400 px) vs 480p 4x4/t168 (465,124 px) | | |
| 4 | Source ablation 480p vs 1080p at 3x3/t224 | | |
| 5 | Best mosaic vs mean-of-frames control | | |
| 6 | Build+embed ms/window and canvas px | | |

## Verdict (adopt | maybe | drop) + WHY

TBD after numbers.
