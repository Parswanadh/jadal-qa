# E2.1080 — True-1080p mosaic tile-resolution ladder + equal-canvas duel

**Why:** User claim: 480p-or-smaller tiles with more of them process faster without accuracy loss; X3 tested this on 480p gold clips only, need TRUE 1080p source

**Hypothesis:** At equal canvas pixels, 1080p-source 3x3/t224 ties 480p-source 4x4/t168 within 0.02 AUROC; smaller tiles cut canvas px and ms

**Data / label source:** Egocentric-10K human-labelled OOD windows ood160_labels.json (160; 138 good/22 bad), 1080p HEVC source + 480p H264 transcodes

**Data regime:** 1080p/30fps HEVC factory egocentric + 480p/30fps transcodes; mixed lighting

**Method:** mosaic EG2 embeddings, stratified 5-fold CV logistic C=0.01, factory-bootstrap CI

**Owner:** H2  **Started:** 2026-10-10 17:43

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
