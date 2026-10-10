# SKIP.03 — Best-cell selection: cheapest policy within verdict/error tolerance

**Why:** Q2 deliverable: pick the cheapest policy keeping clip-verdict agreement >=90% and bad-fraction error <=0.05

**Hypothesis:** A section-skip policy is the cheapest cell; otherwise E3 every-2 stands

**Data / label source:** SKIP.01+SKIP.02 result JSONs

**Data regime:** IndEgo

**Method:** tolerance-based selection on the cost-error frontier; per-clip detail at 0.30 and 0.50 thresholds

**Owner:** X2  **Started:** 2026-10-10 16:27

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:27): adopt.** Best cell = uniform-every-2: 51.5% processed, error 0.049 [0.019,0.087], 7/7 verdicts @0.30. Only policy meeting agreement>=0.90 AND error<=0.05; still wins when relaxing error to 0.07/0.10. No section-skip policy is cheaper within tolerance.

Results file: eval/results/exp_skip/SKIP.03.json

Decision: Re-affirm E3.01 uniform-every-2 (optionally + E3.03 4s early exit). No skip-anywhere rule adopted.

Cost: CPU-only synthesis, no GPU
