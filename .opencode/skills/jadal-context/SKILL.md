---
name: jadal-context
description: Project background for Jadal (deterministic QA of head-camera factory video): goal, pipeline stages, what is measured so far, what failed, current numbers and open questions. Load first in any new session.
license: Apache-2.0
metadata:
  audience: all-agents
---
# Jadal in one page
**Goal.** Judge each 10-second window of crowd-captured egocentric factory video: ACCEPT / REJECT (evidence: feature, value, op, threshold) / REVIEW (uncertain), plus a content-hashed receipt. Deterministic, auditable, cheap enough for edge devices, every relevant frame analysed per pixel (no "8 sampled frames"), no generic VLM (rejected as probabilistic), RGB only (no hyperspectral claims).
**Pipeline (design).** Stage 0: first 1-2 s, exact rules (dark/blocked/frozen/over-exposed) -> immediate reject with evidence. Stage 1: whole-window per-pixel/spectral features + score. Stage 2: only uncertain windows get heavier tools (EmbeddingGemma-2, MediaPipe). Receipt = SHA-256 of canonical JSON.
**Builder's current experiments.** (E1) frames per second 1-10 from 30; (E2) mosaic: N timestamped frames tiled into ONE image (timestamp top-right of each tile) judged in one evaluation; (E3) skip/sparse temporal sampling (do not process irrelevant stretches); (E4) data regimes (blurry vs clear, resolution/fps) -> selectable profiles/models; (E5) domain adaptation; (E6) what the motion cue measures; (E8) re-score all windows. Every experiment logged with its reason, data type and outputs.
**Measured so far (read `$JADAL_BUNDLE/docs/FACTS.md` and `experiments/EXPERIMENTS.md` for current).** In-domain IndEgo (150 human windows, 7 clips, leave-one-clip-out): EmbeddingGemma-2 window embedding logistic ~0.85 AUROC (nested 0.846), 26 per-pixel features ~0.79-0.81, two-view average 0.83-0.84; physics quality checks 0.85-0.90 for dark/blur/duplicate. **Out-of-domain (Egocentric-10K, human-labelled): every IndEgo-trained model failed (0.4-0.53 AUROC).** Closed lines (do not redo): spatial localisation of motion (pooled control 0.790 beats spatial grid 0.753), temporal rhythm, synthetic negatives (collapse real AUROC to 0.351), ego-motion compensation (not load-bearing), generic VLM, hyperspectral reconstruction from RGB. The residual-motion magnitude cue is INVERTED (good windows have less residual flow): it behaves like "wearer is not walking".
**Honesty ceiling.** One human labeller, 7 clips, ~3 wearers; certified error bounds are not achievable at this n (clip correlation); accuracy is not yet high. Say so.
