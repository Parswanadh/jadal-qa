# E1.04 — Frame-selection strategies at an equal 20-frame budget (IndEgo LOCO)

**Why:** Uniform sampling may waste the 20-frame budget on static stretches; motion-adaptive or keyframe-only selection could give the same accuracy for less decode/compute, or better accuracy at equal cost.

**Hypothesis:** motion-adaptive >= uniform20; keyframe-only is worse than uniform (I-frame placement is arbitrary for content) and degenerates to uniform when a 10s window has fewer than 20 I-frames; seeded random sits between uniform and motion.

**Data / label source:** 150 IndEgo gold windows (human labels, 100 good / 50 bad) from labels/manifest_gold.json; 480p ~10fps H.264 in-domain.

**Data regime:** in-domain IndEgo 480p 10fps H.264, 7 clips, ~3 wearers, one labeller

**Method:** equal budget 20 frames/window (~2fps): uniform (idx=arange(0,T,5)) vs motion-adaptive (top-19 inter-frame |diff| + frame0 anchor) vs keyframe-only (I-frame pts_time via ffprobe mapped to the 10fps grid, padded with uniform when <20 I-frames) vs seeded random (seed 0); same LOCO logistic C=0.3 + clip-bootstrap CI95 B=1000; per-window frame overlap (Jaccard) and score rank-correlation vs uniform20.

**Owner:** experimenter  **Started:** 2026-10-10 15:17

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
