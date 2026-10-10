# E2.01 — Mosaic embedding (adopt candidate)

**Why:** Test if ONE tiled timestamped image can replace N sequential frame evaluations, same accuracy at <=1/4 cost.

**Hypothesis:** Mosaic embedding is within 0.02 AUROC of mean-of-frames at N frames with 1/N the embedding passes.

**Data / label source:** IndEgo gold 150 human windows (LOCO, 7 clips) + OOD 60 human windows (51 good/9 bad); label_source human

**Data regime:** 480p@10fps IndEgo, 854x480@30fps Egocentric-10K; 10-s windows

**Method:** EmbeddingGemma-2 image embedding of mosaic, logistic C=0.01 LOCO, clip-bootstrap CI; N in {4,6,9,12,16} tile 224

**Owner:** experimenter  **Started:** 2026-10-10 15:20

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:07): drop.** IndEgo LOCO (n=150, 7 clips) mosaic AUROC 0.757-0.842 over N=4..100, always below matched mean-of-frames (0.822-0.860); deltas -0.001..-0.214 (only N30 within 0.02 but OOD 0.155). OOD (n=60, n_bad=9): mosaic 0.155-0.638 vs mean 0.420-0.627, inconsistent. Fails the within-0.02-on-BOTH-regimes bar -> NOT adopted as model input; adopted ONLY as the human-evidence artefact (dashboard why-view). 95% CIs clip-bootstrap, optimistic.

Results file: eval/results/exp_mosaic/E2.01.json

Decision: Do not use mosaic as model input; ship mosaic PNG+sidecar as the reviewer 'why' artefact.

Cost: 1 embed pass/window vs N for mean-of-frames (<=1/4 for N>=4); 224px tiles; build 1.6s/window, embed 1 pass; A40 VRAM peak 3967 MB
