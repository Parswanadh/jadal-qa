# F50.02 — Per-pixel features 26 -> 50: OOD adapted

**Why:** Does the 50-feature set improve OOD-160 adapted z-score over 0.781?

**Hypothesis:** Additional transferable features raise OOD AUROC

**Data / label source:** OOD-160 human labels (138/22, 47 factories); adapted per-domain z-score

**Data regime:** out-of-domain, factory-grouped CI

**Method:** per-domain z-score, IndEgo sign, equal-weight mean; factory-grouped bootstrap

**Owner:** F50  **Started:** 2026-10-10 21:22

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 21:34): drop.** OOD-160 adapted: documented 14-feat 0.781; +18 new 0.657 (-0.123 [-0.224,-0.035], sig. worse); all-50 0.512. No transferable gain.

Results file: eval/results/exp_feat50/ood.json

Decision: Do not adopt the 50-feature set.

Cost: CPU, cached features
