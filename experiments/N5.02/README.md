# N5.02 — OOD validation of the recalibrated sparse heads (160 windows)

**Why:** The refit heads must not degrade out-of-domain error vs the frozen heads.

**Hypothesis:** Sparse-refit heads preserve the frozen OOD AUROC within the grouped-CI.

**Data / label source:** 160 human-labelled Egocentric-10K windows (ood160_labels.json, 138 good/22 bad, 47 factories)

**Data regime:** out-of-domain

**Method:** apply frozen vs sparse-refit heads to sparse features; factory-grouped bootstrap CI

**Owner:** N5  **Started:** 2026-10-10 19:27

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
