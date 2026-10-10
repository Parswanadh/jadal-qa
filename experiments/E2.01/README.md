# E2.01 — Mosaic embedding (adopt candidate)

**Why:** Test if ONE tiled timestamped image can replace N sequential frame evaluations, same accuracy at <=1/4 cost.

**Hypothesis:** Mosaic embedding is within 0.02 AUROC of mean-of-frames at N frames with 1/N the embedding passes.

**Data / label source:** IndEgo gold 150 human windows (LOCO, 7 clips) + OOD 60 human windows (51 good/9 bad); label_source human

**Data regime:** 480p@10fps IndEgo, 854x480@30fps Egocentric-10K; 10-s windows

**Method:** EmbeddingGemma-2 image embedding of mosaic, logistic C=0.01 LOCO, clip-bootstrap CI; N in {4,6,9,12,16} tile 224

**Owner:** experimenter  **Started:** 2026-10-10 15:20

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
