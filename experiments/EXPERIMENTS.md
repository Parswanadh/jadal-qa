# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| Z2.01 | Z2 | done | adopt | Split-conformal decision-level review/error frontier (IndEgo LOCO) | IndEgo human gold 150 windows/7 clips (n_good=100,n_bad=50)  / 480p/10fps IndEgo assembly+logistics; E8 1080p/30 Egocentric-10K |
| Z2.02 | Z2 | done | adopt | Group/Mondrian-conditional + block-cluster conformal under dependence | IndEgo human gold 150 windows/7 clips OOF p_ens + clip label / IndEgo 480p/10fps, 7 clips |
| Z2.03 | Z2 | done | drop | Risk-controlling prediction (CRC/LTT) with cluster-aware UCB | IndEgo human gold 150 windows/7 clips OOF p_ens (human label / IndEgo 480p/10fps |

## Z2.01 — Split-conformal decision-level review/error frontier (IndEgo LOCO)
- **Why:** Drive human review down with a stated error bound instead of a vibe; need a review%-vs-error curve a reviewer can trust.
- **Hypothesis:** Mapping the ensemble score to conformal sets {ACCEPT}/{REJECT}/{ACCEPT,REJECT} yields a monotone review-vs-error frontier; clip-grouped splits expose the exchangeability break from intra-clip correlation.
- **Data / label source:** IndEgo human gold 150 windows/7 clips (n_good=100,n_bad=50) OOF p_ens (human labels, nested LOCO) + E8 1,546 unlabelled windows (descriptive only)
- **Regime:** 480p/10fps IndEgo assembly+logistics; E8 1080p/30 Egocentric-10K
- **Method:** Split conformal (LAC) on p_ens=P(good); clip-grouped LOCO; sweep coverage target; review=non-singleton share; clip-bootstrap CI
- **Owner / started / finished:** Z2 / 2026-10-10 17:15 / 2026-10-10 17:33
- **Verdict:** adopt — Split-conformal decision sets give a monotone U-shaped review/error frontier under clip-grouped LOCO; empirical coverage tracks nominal. Min review 3.3% at coverage 0.80 (marginal FA 6%); coverage 0.85 gives review 12% with marginal FA 4%<=5% and stated marginal false-decision bound 15%.
- **Decision:** Adopt coverage-0.85 operating point (review 12% vs 75.5% today)
- **Cost:** CPU only, <1 min, seed 7
- **Results:** eval/results/exp_conformal/Z2.01.json

## Z2.02 — Group/Mondrian-conditional + block-cluster conformal under dependence
- **Why:** Intra-clip correlation (~0.5 ICC, n_eff ~11-14) breaks plain conformal exchangeability; need conditional and cluster-robust variants with measured coverage.
- **Hypothesis:** Mondrian-by-group and design-effect / cluster-bootstrap conformal restore nominal coverage on held-out clips where window-level split conformal is anti-conservative.
- **Data / label source:** IndEgo human gold 150 windows/7 clips OOF p_ens + clip labels (human)
- **Regime:** IndEgo 480p/10fps, 7 clips
- **Method:** Mondrian (score-strata + within-clip cross-fit), design-effect n_eff quantile, cluster-bootstrap conformal; nominal-vs-empirical coverage, per-clip
- **Owner / started / finished:** Z2 / 2026-10-10 17:15 / 2026-10-10 17:33
- **Verdict:** adopt — Marginal split-conformal coverage holds (0.80->0.800, 0.90->0.893) because ICC of the conformity score is ~0.02 despite raw-score ICC 0.46; but clip-conditional coverage fails (worst clip 0.647 at nominal 0.90). Mondrian-by-clip lifts clip-conditional coverage to 0.873; Mondrian-by-score-strata near-nominal; cluster-bootstrap over-corrects at K=7.
- **Decision:** Report marginal coverage; use Mondrian-by-clip when per-clip fairness matters
- **Cost:** CPU only, <1 min
- **Results:** eval/results/exp_conformal/Z2.02.json

## Z2.03 — Risk-controlling prediction (CRC/LTT) with cluster-aware UCB
- **Why:** Need P(false-decision rate <= r) >= 1-delta at minimum review; window-level concentration is anti-conservative under clip dependence.
- **Hypothesis:** Cluster-level Hoeffding / empirical-Bernstein UCB (K clips) gives an honest guarantee; the binding constraint is dependence, not the tail bound.
- **Data / label source:** IndEgo human gold 150 windows/7 clips OOF p_ens (human labels)
- **Regime:** IndEgo 480p/10fps
- **Method:** Learn-then-Test / CRC: sweep symmetric accept/reject threshold; cluster-Hoeffding + Maurer-Pontil EB UCB; selective-risk ratio bound; compare window-level binomial/CP
- **Owner / started / finished:** Z2 / 2026-10-10 17:15 / 2026-10-10 17:33
- **Verdict:** drop — With K=7 exchangeable clips a nonparametric cluster UCB cannot certify a 5% false-accept bound at 90% confidence (floor UCB 0.66; EB 1.0; LOCO-honest no fold). Window-Bentkus reaches 5% at 88.7% review but assumes window exchangeability, which intra-clip correlation violates. The guarantee is data-limited by the number of clips, not the tail bound.
- **Decision:** Do NOT ship a 5% CRC bound on 7 clips; recommend the conformal 15% bound and collect more labelled clips for a rigorous r-bound
- **Cost:** CPU only, <1 min
- **Results:** eval/results/exp_conformal/Z2.03.json
