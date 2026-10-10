# A1.03 — Regime-specific heads, hold-out 108, compute cost, verdict

**Why:** Find the highest achievable accuracy with compute-free capacity+ensembling and state the gap to 95%.

**Hypothesis:** Per-regime heads + routing beat a single global model across domains; the confirmatory 108 quantifies the honest gap to 95%.

**Data / label source:** IndEgo gold 150 (LOCO); OOD-160 (grouped); confirmatory 108 (evaluate only); cached features

**Data regime:** CPU numpy/sklearn; cached features

**Method:** E4-descriptor regimes + routing; frozen recipes applied once to the 108 hold-out; measured compute cost; verdict.

**Owner:** A1  **Started:** 2026-10-10 20:52

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 21:14): mixed.** Aspect router only re-selects the domain-appropriate single scorer; quality routing negative. Clean 108 hold-out best 0.757 [0.613,0.873] (rank pixel+eg2+z3six); OOD-trained recipes leaked by 44/44 shared factories. Verdict: in-domain 0.846, OOD 0.769; gap to 0.95 = 0.104 / 0.181; reason = labels/domain, not capacity.

Results file: eval/results/exp_acc/A1.03.json

Decision: data/label ceiling, not model capacity

Cost: CPU, 67s; EG2 feature ~7-9s/window GPU dominates
