# Z2.03 — Risk-controlling prediction (CRC/LTT) with cluster-aware UCB

**Why:** Need P(false-decision rate <= r) >= 1-delta at minimum review; window-level concentration is anti-conservative under clip dependence.

**Hypothesis:** Cluster-level Hoeffding / empirical-Bernstein UCB (K clips) gives an honest guarantee; the binding constraint is dependence, not the tail bound.

**Data / label source:** IndEgo human gold 150 windows/7 clips OOF p_ens (human labels)

**Data regime:** IndEgo 480p/10fps

**Method:** Learn-then-Test / CRC: sweep symmetric accept/reject threshold; cluster-Hoeffding + Maurer-Pontil EB UCB; selective-risk ratio bound; compare window-level binomial/CP

**Owner:** Z2  **Started:** 2026-10-10 17:15

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
