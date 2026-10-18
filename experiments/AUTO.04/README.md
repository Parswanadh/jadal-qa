# AUTO.04 — Cost-sensitive thresholds at false-accept:false-reject cost ratios 1:1, 3:1, 1:3

**Why:** Awign's cost of accepting a bad window is not the cost of rejecting a good one; the threshold must follow the ratio and be chosen on training folds only.

**Hypothesis:** 3:1 makes the policy accept more (lower threshold) and raises false accepts; on OOD-160 the IndEgo-selected threshold does not track the OOD optimum.

**Data / label source:** IndEgo 150 (clip-grouped LOCO); OOD-160 frozen scores (factory-grouped)

**Data regime:** cached scores, CPU

**Method:** empirical cost minimiser on the training clips' inner-LOCO probabilities per fold; rates/CIs as AUTO.00; analytic Bayes thresholds reported as arithmetic for contrast

**Owner:** dsh-s4-auto  **Started:** 2026-10-18 06:42

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 06:58): maybe.** Cost-sensitive thresholds chosen on training folds only (false accept : false reject). IndEgo-150 held-out clips, cost per 1000 windows in each row's own units: 1:1 threshold 0.186/0.417/0.501 (min/median/max) cost 267 vs 260 at fixed 0.5; 3:1 threshold 0.500/0.725/0.768 cost 507 vs 433 at fixed 0.5; 1:3 threshold 0.009/0.133/0.219 cost 373 vs 607 at fixed 0.5. The direction is right (3:1 raises the threshold and cuts FAR to 0.12; 1:3 lowers it and cuts FRR to 0.04) but the tuned threshold only pays for itself at 1:3. OOD-160 with IndEgo-selected thresholds: 1:1 t=0.396 bal-acc 0.552, 3:1 t=0.642 bal-acc 0.565, 1:3 t=0.118 bal-acc 0.562 - all near chance.

Results file: eval/results/auto/AUTO.04.json

Decision: Keep a cost-sensitive threshold only for a strongly false-reject-averse (1:3) setting; at 1:1 and 3:1 the fixed 0.5 rule is cheaper on held-out clips.

Cost: 0 GPU, ~2 s CPU
