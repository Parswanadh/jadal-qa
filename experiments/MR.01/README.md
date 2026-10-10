# MR.01 — Low-res mosaic: tile-resolution ladder (3x3, 224->64px)

**Why:** E2 dropped mosaic as model input; test whether lower-res tiles change the verdict and cut cost.

**Hypothesis:** Smaller tiles (<=112px) at fixed grid keep AUROC within 0.02 of the 224px mosaic at lower canvas cost.

**Data / label source:** 150 IndEgo gold windows (manifest_gold.json), human, LOCO over 7 clips

**Data regime:** in-domain IndEgo 480p

**Method:** EG2 mosaic embedding, logistic C=0.01, clip-bootstrap CI B=1000

**Owner:** monitor  **Started:** 2026-10-10 16:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:50): adopt.** Tiles 112-160px at 3x3 match/better 224px (0.803-0.804 vs 0.785) at -36% ms/win and -74% canvas px; smallest within 0.02 of best = 112px. Still below mean-of-frames N9 0.851. Adopt small tiles for a mosaic pipeline.

Results file: eval/results/exp_mosaic_res/MR.01.json

Decision: adopt 112-160px tiles at 3x3

Cost: 43.7 ms/win, 118k canvas px @112px vs 68.5 ms/win, 462k px @224px
