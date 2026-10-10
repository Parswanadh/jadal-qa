# N4.02 — Certification budget: labelled clusters for a false-accept bound

**Why:** State the exact label budget that makes the review/error claim certifiable, not descriptive.

**Hypothesis:** A cluster-Hoeffding/LTT bound at 95% confidence needs ~86 factories for r=0.20, ~342 for r=0.10, ~1365 for r=0.05; 47 today certifies only r~0.27.

**Data / label source:** 47 factories / 58 video clips / 1,546 unlabelled windows; Z2 conformal machinery

**Data regime:** cluster-level (factory) exchangeability

**Method:** invert UCB = Rhat + sqrt(ln(1/delta')/(2K)) <= r, delta'=delta or delta/grid over the LTT grid

**Owner:** N4  **Started:** 2026-10-10 19:30

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 19:30): adopt.** At 1-delta=0.95, certifying a marginal false-accept bound needs K factories: r=0.20 -> 86 (grid-union) / 38 (single threshold); r=0.10 -> 342 / 150; r=0.05 -> 1365 / 600. With 47 factories today the tightest certifiable r is ~0.27 (grid-union) / ~0.18 (single): a 5-10% bound is NOT certifiable. The current 'review 12%, FA 4%' point is descriptive. All 1,546 pool windows are in the same 47 factories, so the pool adds zero clusters; certification needs NEW factories.

Results file: eval/results/exp_acc/N4.02.json

Decision: Open a data request for >=39 new factories (r=0.20); label the 200-window list for model error in the meantime.

Cost: closed form, no model run
