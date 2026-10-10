# MR.01 — Low-res mosaic: tile-resolution ladder (3x3, 224->64px)

**Why:** E2 dropped mosaic as model input; test whether lower-res tiles change the verdict and cut cost.

**Hypothesis:** Smaller tiles (<=112px) at fixed grid keep AUROC within 0.02 of the 224px mosaic at lower canvas cost.

**Data / label source:** 150 IndEgo gold windows (manifest_gold.json), human, LOCO over 7 clips

**Data regime:** in-domain IndEgo 480p

**Method:** EG2 mosaic embedding, logistic C=0.01, clip-bootstrap CI B=1000

**Owner:** monitor  **Started:** 2026-10-10 16:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
