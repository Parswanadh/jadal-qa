# A1.03 — Regime-specific heads, hold-out 108, compute cost, verdict

**Why:** Find the highest achievable accuracy with compute-free capacity+ensembling and state the gap to 95%.

**Hypothesis:** Per-regime heads + routing beat a single global model across domains; the confirmatory 108 quantifies the honest gap to 95%.

**Data / label source:** IndEgo gold 150 (LOCO); OOD-160 (grouped); confirmatory 108 (evaluate only); cached features

**Data regime:** CPU numpy/sklearn; cached features

**Method:** E4-descriptor regimes + routing; frozen recipes applied once to the 108 hold-out; measured compute cost; verdict.

**Owner:** A1  **Started:** 2026-10-10 20:52

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
