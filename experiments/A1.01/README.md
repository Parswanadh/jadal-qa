# A1.01 — Capacity sweep: logistic/GBT/MLP/SVM on cached feature views

**Why:** Find the highest achievable accuracy with compute-free capacity+ensembling and state the gap to 95%.

**Hypothesis:** Higher-capacity models on the full cached feature set beat the linear two-view baseline in-domain and transfer to OOD grouped CV.

**Data / label source:** IndEgo gold 150 (LOCO); OOD-160 (grouped); confirmatory 108 (evaluate only); cached features

**Data regime:** CPU numpy/sklearn; cached features

**Method:** Nested grouped CV (LOCO gold, leave-one-factory-out OOD): standardise+impute, inner-CV hyperparameter selection, AUROC + BA + grouped bootstrap CI.

**Owner:** A1  **Started:** 2026-10-10 20:52

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
