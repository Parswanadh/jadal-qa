# Z2.02 — Group/Mondrian-conditional + block-cluster conformal under dependence

**Why:** Intra-clip correlation (~0.5 ICC, n_eff ~11-14) breaks plain conformal exchangeability; need conditional and cluster-robust variants with measured coverage.

**Hypothesis:** Mondrian-by-group and design-effect / cluster-bootstrap conformal restore nominal coverage on held-out clips where window-level split conformal is anti-conservative.

**Data / label source:** IndEgo human gold 150 windows/7 clips OOF p_ens + clip labels (human)

**Data regime:** IndEgo 480p/10fps, 7 clips

**Method:** Mondrian (score-strata + within-clip cross-fit), design-effect n_eff quantile, cluster-bootstrap conformal; nominal-vs-empirical coverage, per-clip

**Owner:** Z2  **Started:** 2026-10-10 17:15

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 17:33): adopt.** Marginal split-conformal coverage holds (0.80->0.800, 0.90->0.893) because ICC of the conformity score is ~0.02 despite raw-score ICC 0.46; but clip-conditional coverage fails (worst clip 0.647 at nominal 0.90). Mondrian-by-clip lifts clip-conditional coverage to 0.873; Mondrian-by-score-strata near-nominal; cluster-bootstrap over-corrects at K=7.

Results file: eval/results/exp_conformal/Z2.02.json

Decision: Report marginal coverage; use Mondrian-by-clip when per-clip fairness matters

Cost: CPU only, <1 min
