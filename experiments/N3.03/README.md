# N3.03 — Rank-average fusion: adapted x frozen (x rules)

**Why:** Does fusion reduce the ambiguous fraction at equal/better error?

**Hypothesis:** Rank-averaging the adapted score with the frozen ensemble improves grouped AUROC and cuts the review band

**Data / label source:** IndEgo gold 150 (LOCO) + OOD-160 (factory-grouped)

**Data regime:** n/a

**Method:** Rank-average of per-domain rank01 good-scores; AUROC grouped bootstrap; calibrated [0.4,0.6] review proxy

**Owner:** N3  **Started:** 2026-10-10 19:47

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 19:59): negative.** Rank-average adapted+frozen is a compromise not a win: in-domain 0.834->0.784, OOD 0.759->0.686 (paired OOD gain vs frozen +0.120 [0.04,0.20], vs adapted -0.073 [-0.16,+0.00]). Mean-domain AUROC 0.735 beats singles but < router 0.791. Does NOT cut ambiguity: OOD review 0.31->0.53->0.86.

Results file: eval/results/exp_acc/N3.03.json

Decision: Do not ship fusion as the primary; ship the router (rules in-domain, adapted OOD). Fusion only if no regime signal

Cost: CPU, cached features, ~50s
