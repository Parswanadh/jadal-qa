# N3.03 — Rank-average fusion: adapted x frozen (x rules)

**Why:** Does fusion reduce the ambiguous fraction at equal/better error?

**Hypothesis:** Rank-averaging the adapted score with the frozen ensemble improves grouped AUROC and cuts the review band

**Data / label source:** IndEgo gold 150 (LOCO) + OOD-160 (factory-grouped)

**Data regime:** n/a

**Method:** Rank-average of per-domain rank01 good-scores; AUROC grouped bootstrap; calibrated [0.4,0.6] review proxy

**Owner:** N3  **Started:** 2026-10-10 19:47

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
