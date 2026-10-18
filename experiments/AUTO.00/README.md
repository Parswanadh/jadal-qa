# AUTO.00 — Forced-decision baseline: full-coverage error with the review band removed

**Why:** The shipped funnel sends |p-0.5|<0.25 to review, so its error at 100% coverage has never been measured; the brief's decision rule needs that number with an interval before 'fully automated' can be considered.

**Hypothesis:** Forced accept-at-0.5 gives balanced accuracy near the LOCO AUROC's chance-corrected level on IndEgo; on OOD-160 the lower bound is well below 0.70 because 19/22 bad windows are NO_ACTIVE_WORK.

**Data / label source:** IndEgo 150 windows (100 good/50 bad, ONE human labeller) 26-feature cache; OOD-160 frozen shipped-head scores (138 good/22 bad, 19 NO_ACTIVE_WORK, 47 factories, ONE human labeller); round-3 108 read-only

**Data regime:** cached features, CPU, no GPU

**Method:** StandardScaler+LogisticRegression(C=0.3,balanced) leave-one-clip-out for (a) accept iff p>=0.5 and (b) Youden threshold chosen on the 6 training clips' inner-LOCO probs per fold; clip-bootstrap CI; OOD-160 frozen p26 with factory-grouped CI and window bootstrap marked optimistic

**Owner:** dsh-s4-auto  **Started:** 2026-10-18 06:42

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 06:58): adopt.** Forced-decision baseline at 100 % coverage. IndEgo-150 (n=150, n_bad=50, 7 clips, ONE labeller, LOCO, clip-bootstrap): P1 accept iff p>=0.5 bal-acc 0.740 [0.631,0.789], FAR 0.260, FRR 0.260, accept 0.580; P2 Youden chosen on the 6 training clips' inner-LOCO probs per fold bal-acc 0.755 [0.646,0.797], thresholds 0.359/0.501/0.558 min/median/max. OOD-160 frozen shipped 26-number head (n=160, n_bad=22, 19 of 22 NO_ACTIVE_WORK, 47 factories): AUROC 0.588 [0.450,0.728] factory-grouped (window bootstrap 0.470-0.711 OPTIMISTIC); NO_ACTIVE_WORK-only AUROC 0.576 [0.427,0.739]; forced 0.5 bal-acc 0.556 [0.448,0.673], FAR 0.409, FRR 0.478; IndEgo-Youden transferred (t=0.396) bal-acc 0.552 [0.426,0.684]. Band share 0.480 in-domain, 0.138 OOD. Round-3 108 for the 26-number single head SKIPPED: no per-window score file exists without re-running a model; a read-only DIFFERENT-model arm (frozen two-view ensemble read from the C1 PR, nothing scored here) gives AUROC 0.690 [0.534,0.823], forced-0.5 bal-acc 0.640.

Results file: eval/results/auto/AUTO.00.json

Decision: Use as the reference 100 %-coverage number: OOD-160 lower bound 0.448 < 0.70, so the brief's decision rule requires tiered automation with audit, not fully automated.

Cost: 0 GPU, ~3.5 s CPU (7 outer + 42 inner logistic fits, 2000-draw bootstraps)
