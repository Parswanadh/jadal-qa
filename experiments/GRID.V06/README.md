# GRID.V06 — Role in the funnel: review-band resolver and veto on high-confidence accepts

**Why:** A model that is only a ranker is still useful if it resolves the review band or vetoes confident accepts.

**Hypothesis:** As a resolver it lowers error at 100 % coverage versus AUTO.00's forced 0.5; as a veto it catches false accepts without destroying the accept share.

**Data / label source:** IndEgo-150 and OOD-160 with frozen model outputs from V02 plus fresh calls on the IndEgo review band

**Data regime:** Frozen prompt and model id; (a) resolver for the band, (b) veto on windows the shipped head accepts at p>=0.75; error at 100 % coverage and the resulting review load

**Method:** 100 %-coverage balanced accuracy and review share for each role, with clip- and factory-grouped intervals, against AUTO.00's forced decision.

**Owner:** dsh  **Started:** 2026-10-18 08:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 09:12): maybe.** ROLE (a) OOD-160 VETO, zero new calls (reuses V02): the shipped head accepts 71 windows (63 good, 8 bad). The VLM vetoes 4 of those 8 false accepts (50%) at a cost of 3 good windows (4.76% of the good the head accepted). Balanced accuracy 0.5464 -> 0.6265, false-accept 0.3636 -> 0.1818, false-reject 0.5435 -> 0.5652. Paired factory-clustered difference vs the head alone +0.08, CI [0.0000, 0.1591] - DOES NOT EXCLUDE ZERO, so the gain is not established at n_bad=22. ROLE (b) IndEgo RESOLVER, 115 new image calls of a 120 cap (band 72 + high-confidence accepts 43): forced-0.5 baseline reproduces AUTO.00 exactly (BA 0.740). Throwing the band away 0.640; resolving the band with the VLM 0.755; adding the veto on high-confidence accepts also 0.755. band_resolve - thrown_away = +0.115 [0.0361, 0.1794] EXCLUDES ZERO; band_resolve - forced05 = +0.015 [-0.0609, 0.1089] NOT established; full_veto - forced05 = +0.015 [-0.0448, 0.0876] NOT established. The VLM said usable for 65/72 band windows (90.3%) and 105/115 targets (91.3%), so the success bias reappears in-domain: FAR rises 0.26 -> 0.38 even as FRR falls 0.26 -> 0.11. V04 already showed a single call is not a measurement (kappa 0.52, AUROC spread 0.071), so every number here is repeat-needed.

Results file: eval/results/exp_vlm/GRID.V06.json

Decision: The VLM as a review-band resolver is significantly better than DISCARDING the band (+0.115 BA, CI excludes zero) but NOT significantly better than the free forced-0.5 rule in-domain (+0.015, CI includes zero), so it does not justify its API cost as an accuracy device there. Out of domain the veto halves the head's false accepts and lifts BA by 0.08 but the interval touches zero, so it is a promising, unestablished signal that needs round-4 labels (and repeated calls) before it can be recommended. Treat VLM output as a ranker feeding a human, never as an automatic verdict.

Cost: 115 image calls for V06 of a 120 cap (all tag=v06 in the ledger; 0 failures, 0 retries); role (a) 0 calls; 11.3 s mean per call; no GPU
