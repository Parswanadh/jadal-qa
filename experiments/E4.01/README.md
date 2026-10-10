# E4.01 — Regime descriptors: IndEgo vs Egocentric-10K domain-shift diagnosis (re-derived)

**Why:** Every IndEgo-trained scorer failed on real Egocentric-10K windows (0.4-0.53); need deterministic descriptors showing which content descriptors separate the datasets beyond container spec

**Hypothesis:** resolution/fps/codec separate by construction; content descriptors (sharpness, exposure, ego-motion) also separate but with smaller effects

**Data / label source:** 150 IndEgo gold windows (human) + 60 OOD labelled (human, 9 bad) on both 480p/30 transcode and 1080p/30 source + 100 queue-2 480p/10 transcodes (unlabelled) + 300 seed-fixed Ego10K sample (unlabelled)

**Data regime:** 480x480/10fps H264 (IndEgo) vs 1080p/30 HEVC (OOD src) vs 854x480/30 (OOD tc) vs 854x480/10 (queue2)

**Method:** deterministic per-window descriptors (ffprobe + 8 sampled-frame stats, no GPU); Cohen's d ranking

**Owner:** experimenter-a3  **Started:** 2026-10-10 15:18

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 15:30): adopt.** 730 deterministic descriptor records: 150 IndEgo gold + 120 OOD-60 (480p/30 transcode + 1080p/30 source) + 100 OOD-2 transcodes + 300 seed-7 Ego10K + 60 IndEgo-spec re-encode. Container separators are exact by construction (width/height/fps d=inf); strongest CONTENT separators IndEgo-vs-OOD-source are chroma_spread d=-3.19, bright_frac -2.65, luma_mean -2.57, luma_p95 -2.54, temp_energy -2.37; sharpness only d=0.61 (overlap). ci95 n/a (deterministic).

Results file: eval/results/exp_regimes/E4.01.json

Decision: Use content separators to justify per-regime profiles; do not treat resolution/fps as findings

Cost: 730 windows x 8 frames, CPU only, no GPU, ~10 min
