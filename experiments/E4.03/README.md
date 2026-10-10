# E4.03 — Domain-gap control: re-encode to the IndEgo spec (480x480/10fps)

**Why:** Test whether matching the input container spec fixes OOD transfer or whether the gap is content

**Hypothesis:** Re-encoding shrinks container descriptors but not content descriptors, and transfer stays poor

**Data / label source:** 60 OOD queue-1 windows (human labels, 9 bad); paired source 1080p/30 vs 480p/30 transcode vs new 480x480/10 re-encode; frozen ensemble ood_scores.json baseline

**Data regime:** 480x480/10 H264 vs 1080p/30 HEVC vs 854x480/30 H264

**Method:** re-encode with ffmpeg; paired descriptors; IndEgo-vs-OOD Cohen's d before/after; frozen-ensemble score audit

**Owner:** experimenter-a3  **Started:** 2026-10-10 15:18

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 15:30): drop.** Re-encoding 60 OOD sources to the IndEgo spec (480x480/10fps H.264) shrinks the IndEgo content gap for only 6/18 content descriptors; several grow (luma_p05 d from -1.34 to +5.02). Frozen-ensemble OOD AUROC 0.415 (n=60, n_bad=9, exploratory) on the 480p/30 transcode was not improved (re-encode not re-scored; frozen pipeline unavailable). Matching the container does not fix the content gap.

Results file: eval/results/exp_regimes/E4.03.json

Decision: Do not re-encode as a transfer fix; keep container/spec as a router input only

Cost: 60 windows ffmpeg re-encode + descriptors, CPU, ~12 min
