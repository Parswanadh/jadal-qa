# A2.03 — Best combined signal set vs the ensemble baseline

**Why:** Pick the best combined new-signal configuration and report the gap to 95%.

**Hypothesis:** flow+hand+full-res together beat every subset and the baseline.

**Data / label source:** IndEgo gold 150 (LOCO) + OOD-160 (factory-grouped)

**Method:** ens+{flow,hand,fullres,flow+hand,flow+hand+fullres} logistic; AUROC + paired gains.

**Owner:** A2  **Started:** 2026-10-10 20:50

## Results
See `eval/results/exp_acc/A2.03.json`.

**Verdict:** pending
