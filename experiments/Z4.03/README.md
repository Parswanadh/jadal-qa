# Z4.03 — Best sequential cell + next-100 active labels

**Why:** Goal (a)+(b): pick cheapest tolerated policy and rank the 1,546 unlabelled Egocentric-10K windows by expected information for OOD error

**Hypothesis:** Diversified uncertainty sampling surfaces the most valuable next labels

**Data / label source:** artifacts/windows_rescored.jsonl (1,546 unlabelled); OOD-60 labels for EVALUATION ONLY

**Data regime:** 1080p/30fps Egocentric-10K, out-of-domain

**Method:** tolerance-based best-cell selection; greedy 4p(1-p)+view-gap with factory/clip redundancy penalty; OOD-60 eval

**Owner:** Z4  **Started:** 2026-10-10 17:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 17:30): maybe.** Best tolerated cell = A=4.9,B=3.7 hard (5/5 correct, 2/7 abstain, 66.5% windows). Next-100 active labels written (47/47 factories). OOD-60 eval: uncertainty U AUROC 0.349 (anti-correlated) -> uncertainty sampling does not find confidently-wrong domain-shift errors; list is coverage/audit, not guaranteed error reduction.

Results file: eval/results/exp_clip/Z4.03.json

Decision: Label the top-100 for factory coverage/audit; do not expect them to be the OOD errors. Add domain-shift-aware acquisition before relying on active learning.

Cost: CPU-only, cached features, no GPU
