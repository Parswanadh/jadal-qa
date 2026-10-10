# Z2.01 — Split-conformal decision-level review/error frontier (IndEgo LOCO)

**Why:** Drive human review down with a stated error bound instead of a vibe; need a review%-vs-error curve a reviewer can trust.

**Hypothesis:** Mapping the ensemble score to conformal sets {ACCEPT}/{REJECT}/{ACCEPT,REJECT} yields a monotone review-vs-error frontier; clip-grouped splits expose the exchangeability break from intra-clip correlation.

**Data / label source:** IndEgo human gold 150 windows/7 clips (n_good=100,n_bad=50) OOF p_ens (human labels, nested LOCO) + E8 1,546 unlabelled windows (descriptive only)

**Data regime:** 480p/10fps IndEgo assembly+logistics; E8 1080p/30 Egocentric-10K

**Method:** Split conformal (LAC) on p_ens=P(good); clip-grouped LOCO; sweep coverage target; review=non-singleton share; clip-bootstrap CI

**Owner:** Z2  **Started:** 2026-10-10 17:15

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 17:33): adopt.** Split-conformal decision sets give a monotone U-shaped review/error frontier under clip-grouped LOCO; empirical coverage tracks nominal. Min review 3.3% at coverage 0.80 (marginal FA 6%); coverage 0.85 gives review 12% with marginal FA 4%<=5% and stated marginal false-decision bound 15%.

Results file: eval/results/exp_conformal/Z2.01.json

Decision: Adopt coverage-0.85 operating point (review 12% vs 75.5% today)

Cost: CPU only, <1 min, seed 7
