# A2.02 — Incremental value on top of the 26-feature + EG2 ensemble

**Why:** Measure whether the new signals add accuracy to the current best frozen ensemble.

**Hypothesis:** Adding flow + hand features to the ensemble lifts in-domain LOCO and OOD AUROC.

**Data / label source:** IndEgo gold 150 (LOCO) + OOD-160 (factory-grouped)

**Method:** Logistic on [ensemble good-score] + group features; LOCO on gold, train-all-gold
-> OOD. Paired grouped bootstrap gain vs baseline ensemble.

**Owner:** A2  **Started:** 2026-10-10 20:50

## Results
See `eval/results/exp_acc/A2.02.json`.

**Verdict:** pending
