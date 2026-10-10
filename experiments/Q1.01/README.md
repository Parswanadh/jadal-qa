# Q1.01 — Complete deterministic physics rule-list model (in-domain LOCO)

**Why:** Spec demands deterministic, mathematics-first decisions; the earlier naive rule list was degenerate. Build the real rule list with exact thresholds and evaluate honestly against the learned baselines.

**Hypothesis:** A fixed rule list with training-clip quantile thresholds and an integer score can match the logistic baseline (AUROC ~0.81) without any learned weights.

**Data / label source:** 150 IndEgo human-gold windows (100 good / 50 bad), 7 clips; cached 26 per-pixel + 16 physics features

**Data regime:** 480p/10fps IndEgo, in-domain

**Method:** leave-one-clip-out; thresholds = p01/p05/p10/p25 of training clips quantised to 4dp; integer score = count of crossed thresholds; deterministic ACCEPT/REJECT/REVIEW with evidence; clip-bootstrap CI; per-rule precision/recall + leave-one-rule-out

**Owner:** Q1  **Started:** 2026-10-10 18:53

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 18:53): adopt.** Deterministic rule list AUROC 0.8224 (95% CI 0.715-0.861) vs logistic 0.8016 and EG2 0.8458 under the same LOCO. Determinism cost vs logistic is -0.021 (rules better); vs EG2 +0.023. Conservative 5%-FR operating point: BA 0.808, REVIEW 0.60, false-reject 0.10, bad recall 0.52. Per-fold feature selection fails (0.762); the fixed spec candidate list transfers. Signal carried by NO_ACTIVE_WORK/BLUR; OVEREXPOSED/FROZEN_STATIC add slight noise in-domain.

Results file: eval/results/exp_phys/Q1.01.json

Decision: Ship the deterministic rule list as the primary explainable scorer; keep EG2 as the accuracy baseline. Thresholds in eval/results/exp_phys/rule_table.json.

Cost: cached features only; <5 s CPU; no GPU
