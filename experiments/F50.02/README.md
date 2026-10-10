# F50.02 — Per-pixel features 26 -> 50: OOD adapted

**Why:** Does the 50-feature set improve OOD-160 adapted z-score over 0.781?

**Hypothesis:** Additional transferable features raise OOD AUROC

**Data / label source:** OOD-160 human labels (138/22, 47 factories); adapted per-domain z-score

**Data regime:** out-of-domain, factory-grouped CI

**Method:** per-domain z-score, IndEgo sign, equal-weight mean; factory-grouped bootstrap

**Owner:** F50  **Started:** 2026-10-10 21:22

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
