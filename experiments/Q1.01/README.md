# Q1.01 — Complete deterministic physics rule-list model (in-domain LOCO)

**Why:** Spec demands deterministic, mathematics-first decisions; the earlier naive rule list was degenerate. Build the real rule list with exact thresholds and evaluate honestly against the learned baselines.

**Hypothesis:** A fixed rule list with training-clip quantile thresholds and an integer score can match the logistic baseline (AUROC ~0.81) without any learned weights.

**Data / label source:** 150 IndEgo human-gold windows (100 good / 50 bad), 7 clips; cached 26 per-pixel + 16 physics features

**Data regime:** 480p/10fps IndEgo, in-domain

**Method:** leave-one-clip-out; thresholds = p01/p05/p10/p25 of training clips quantised to 4dp; integer score = count of crossed thresholds; deterministic ACCEPT/REJECT/REVIEW with evidence; clip-bootstrap CI; per-rule precision/recall + leave-one-rule-out; per-window scores + paired clip-bootstrap deltas committed

**Owner:** Q1  **Started:** 2026-10-10 18:53

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 18:53): adopt.** Deterministic rule list AUROC 0.8224 (95% CI 0.715-0.861; OPTIMISTIC: candidate list, features and signs chosen with all 150 labels) vs logistic 26-feat 0.8016, like-for-like logistic 42-feat 0.7870, EG2 0.8458. Paired clip-bootstrap deltas all contain 0 => no difference established at n=150. Honest full-population BA 0.710 (review-as-accept) / 0.590 (review-as-reject) at 40% coverage; 0.808 is decided-only. Fully nested per-fold feature+sign selection 0.762. Per-window scores committed for traceability.

Results file: eval/results/exp_phys/Q1.01.json

Decision: Ship the deterministic rule list as the primary explainable scorer (in-domain, IndEgo-comparable footage only); quote the full-population BA 0.710/0.590, NOT the decided-only 0.808. Keep EG2 as the accuracy baseline. Thresholds in eval/results/exp_phys/rule_table.json; per-window scores in Q1.01.json.

Cost: cached features only; <5 s CPU; no GPU
