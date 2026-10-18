# AUTO.03 — Risk-coverage frontier and a conformal-style bound selected on training folds only

**Why:** A deployment decision needs the error-vs-coverage curve and, for a target error, the maximum coverage the training folds can support - and the repo already showed nominal conformal calibration breaking on held-out clips.

**Hypothesis:** Forcing 100% coverage is expensive on OOD; a conformal-risk-control-style lambda chosen on training folds reaches 5-20% error only at much lower coverage, and on OOD the bound does not transfer because exchangeability fails.

**Data / label source:** IndEgo 150 (LOCO, clip-grouped); OOD-160 frozen scores (factory-grouped); second view from segments.npz

**Data regime:** cached scores, CPU

**Method:** coverage sweep on a confidence ranking for three policies (S1@0.5, per-fold Youden, band+second view) with grouped CIs; CRC-style lambda (n/(n+1))*risk_hat+1/(n+1)<=alpha selected on training folds only for alpha in {0.05,0.10,0.20}

**Owner:** dsh-s4-auto  **Started:** 2026-10-18 06:42

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 06:58): maybe.** Risk-coverage frontier measured for three policies with clip-grouped (IndEgo) and factory-grouped (OOD) bootstraps that RE-SELECT inside each resample. IndEgo balanced accuracy at coverage 100/80/60/40/20 %: P1 0.740/0.844/0.826/0.771/0.796, P2 0.755/0.821/0.823/0.804/0.768, P3 band+EG2 0.740/0.763/0.808/0.800/0.812. P3 is unavailable on OOD-160 (no cached second-view embeddings for those windows; none were created). Conformal-style bound under exchangeability, assumptions listed: lambda chosen on training folds only by (n/(n+1))*risk_hat+1/(n+1)<=alpha. IndEgo max coverage supported: 5 % -> 7.3 % [1.1,16.9] (P3, realized risk 18.2 %), 10 % -> 13.3 % [4.0,27.5] (realized 25.0 %), 20 % -> 46.0 % [23.2,80.8] (realized 30.4 %). The bound FAILED on held-out clips at every target, reproducing the repo's 2.7 %->46 % finding. OOD with lambda from IndEgo: infeasible at 5 % and 10 %, and at 20 % it selects 95.0 % coverage [92.1,97.9] with a realized OOD risk of 46.1 % -> descriptive only. The only construction whose realized risk tracked its target is a factory-level split inside OOD-160 (P2: 13.6 % [3.5,27.2] at alpha=0.05, 42.0 % [27.4,58.8] at 0.10, on 47 factories / 22 bad windows).

Results file: eval/results/auto/AUTO.03.json

Decision: Report the frontier as measurement; do NOT deploy the conformal-style bound as a guarantee anywhere, and especially not out of domain.

Cost: 0 GPU, ~23 s CPU
