# GRID.V06 — Role in the funnel: review-band resolver and veto on high-confidence accepts

**Why:** A model that is only a ranker is still useful if it resolves the review band or vetoes confident accepts.

**Hypothesis:** As a resolver it lowers error at 100 % coverage versus AUTO.00's forced 0.5; as a veto it catches false accepts without destroying the accept share.

**Data / label source:** IndEgo-150 and OOD-160 with frozen model outputs from V02 plus fresh calls on the IndEgo review band

**Data regime:** Frozen prompt and model id; (a) resolver for the band, (b) veto on windows the shipped head accepts at p>=0.75; error at 100 % coverage and the resulting review load

**Method:** 100 %-coverage balanced accuracy and review share for each role, with clip- and factory-grouped intervals, against AUTO.00's forced decision.

**Owner:** dsh  **Started:** 2026-10-18 08:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
