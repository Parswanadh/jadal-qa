# Y3.01 — Cached-view inventory + two-view baseline reproduction

**Why:** Anchor the review-reduction study: which cached views exist and does the frozen two-view ensemble reproduce AUROC 0.8534 OOF?

**Hypothesis:** The frozen pixel-26+EG2@2fps probability average reproduces AUROC 0.8534; extra cached views exist (per-pixel series/physics/theory).

**Data / label source:** 150-window IndEgo human gold (LOCO, 7 clips)

**Data regime:** in-domain LOCO

**Method:** nested LOCO logistic per view; fixed-C frozen anchor

**Owner:** Y3  **Started:** 2026-10-10 16:58

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:58): adopt.** Baseline reproduced: fixed-C two-view prob-average AUROC 0.8534, nested-LOCO 0.8336. Per-view nested AUROC: eg2 0.8458, perpixel121 0.8140, pixel26 0.7916, series16 0.7912, physics16 0.7608, theory18 0.6698. EG2@1fps/@0.5fps caches absent (only @2fps).

Results file: eval/results/exp_review/Y3.01.json

Decision: Use as reference; combo sweep in Y3.02

Cost: CPU <1 min
