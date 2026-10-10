# E1.04 — Frame-selection strategies at an equal 20-frame budget (IndEgo LOCO)

**Why:** Uniform sampling may waste the 20-frame budget on static stretches; motion-adaptive or keyframe-only selection could give the same accuracy for less decode/compute, or better accuracy at equal cost.

**Hypothesis:** motion-adaptive >= uniform20; keyframe-only is worse than uniform (I-frame placement is arbitrary for content) and degenerates to uniform when a 10s window has fewer than 20 I-frames; seeded random sits between uniform and motion.

**Data / label source:** 150 IndEgo gold windows (human labels, 100 good / 50 bad) from labels/manifest_gold.json; 480p ~10fps H.264 in-domain.

**Data regime:** in-domain IndEgo 480p 10fps H.264, 7 clips, ~3 wearers, one labeller

**Method:** equal budget 20 frames/window (~2fps): uniform (idx=arange(0,T,5)) vs motion-adaptive (top-19 inter-frame |diff| + frame0 anchor) vs keyframe-only (I-frame pts_time via ffprobe mapped to the 10fps grid, padded with uniform when <20 I-frames) vs seeded random (seed 0); same LOCO logistic C=0.3 + clip-bootstrap CI95 B=1000; per-window frame overlap (Jaccard) and score rank-correlation vs uniform20.

**Owner:** experimenter  **Started:** 2026-10-10 15:17

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 15:49): drop.** n=150 IndEgo human windows, LOCO logistic C=0.3, equal 20-frame budget: uniform-20 0.7866 [0.650,0.866] bal 0.725 vs motion-adaptive 0.7620 [0.675,0.802], keyframe-only 0.7798 [0.662,0.835], seeded random 0.7386 [0.586,0.805]; all CI95 overlap, no arm beats uniform at equal cost. Motion picks different frames (mean Jaccard vs uniform 0.1347, score rho 0.733) and is worse; keyframe has mean 1.23 I-frames/window and 150/150 windows below the 20-frame budget, so it is always padded back to uniform (Jaccard 0.9244, 63/150 identical). Corrected the bundle's partial e1_04.json, whose keyframe arm silently equalled uniform (ffprobe 6.x: pkt_pts_time is empty).

Results file: eval/results/exp_fps/E1.04.json

Decision: Keep uniform sampling (arange(0,T,5)) at the equal 20-frame budget; do not implement motion-adaptive, keyframe-only or random selection. No arm reduces cost: the full native window is decoded in every arm (decode 129 ms/window clean), so selection only changes which 20 frames the features see.

Cost: feature 53 ms/window clean (220 ms/window in the loaded run) at 20 frames/window, identical for all four arms; decode 129 ms/window clean; 0 GPU
