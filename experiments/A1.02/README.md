# A1.02 — Ensembling/stacking: two views x adapted x rules x new features

**Why:** Find the highest achievable accuracy with compute-free capacity+ensembling and state the gap to 95%.

**Hypothesis:** Nested stacking of pixel/EG2/adapted/rules base scores clears 0.95 in-domain and improves OOD over the best single scorer.

**Data / label source:** IndEgo gold 150 (LOCO); OOD-160 (grouped); confirmatory 108 (evaluate only); cached features

**Data regime:** CPU numpy/sklearn; cached features

**Method:** Rank-average and nested logistic meta-learner over OOF base scores; paired grouped bootstrap gains.

**Owner:** A1  **Started:** 2026-10-10 20:52

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 21:14): mixed.** Nested stacking helps OOD (+0.157 [+0.044,+0.280] vs frozen two-view) but is a tie in-domain (0.836 vs 0.846 EG2). Neither domain clears 0.95.

Results file: eval/results/exp_acc/A1.02.json

Decision: ensembling is a tie in-domain, modest OOD gain

Cost: CPU, 113s
