# F50.01 — Per-pixel features 26 -> 50: in-domain LOCO

**Why:** Does expanding the per-pixel feature vector from 26 to 50 improve IndEgo LOCO accuracy?

**Hypothesis:** Added features raise in-domain AUROC above 0.810

**Data / label source:** IndEgo gold 150 windows (7 clips), manifest_gold.json + caches/gpu/features.json

**Data regime:** in-domain, leave-one-clip-out

**Method:** logistic C=0.3 balanced, clip-bootstrap CI

**Owner:** F50  **Started:** 2026-10-10 21:22

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
