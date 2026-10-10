# Z2.02 — Group/Mondrian-conditional + block-cluster conformal under dependence

**Why:** Intra-clip correlation (~0.5 ICC, n_eff ~11-14) breaks plain conformal exchangeability; need conditional and cluster-robust variants with measured coverage.

**Hypothesis:** Mondrian-by-group and design-effect / cluster-bootstrap conformal restore nominal coverage on held-out clips where window-level split conformal is anti-conservative.

**Data / label source:** IndEgo human gold 150 windows/7 clips OOF p_ens + clip labels (human)

**Data regime:** IndEgo 480p/10fps, 7 clips

**Method:** Mondrian (score-strata + within-clip cross-fit), design-effect n_eff quantile, cluster-bootstrap conformal; nominal-vs-empirical coverage, per-clip

**Owner:** Z2  **Started:** 2026-10-10 17:15

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
