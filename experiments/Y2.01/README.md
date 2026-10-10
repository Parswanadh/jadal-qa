# Y2.01 — Decision-theoretic gating: review-vs-error sweep

**Why:** The frozen symmetric band leaves 75.5% of the 1,546 Egocentric-10K windows in REVIEW; the PoC needs <60%.

**Hypothesis:** Deciding the most cost-confident windows and reviewing the rest cuts REVIEW below 60% at a measurable balanced-accuracy cost on gold LOCO, with asymmetric costs shifting the boundary.

**Data / label source:** gold LOCO-OOF caches/ensemble/oof_probs.npz (n=150, human); OOD-60 caches/ensemble/ood_scores.json (n=60, human, eval-only); 1,546 unlabelled /workspace/artifacts/windows_rescored.jsonl

**Data regime:** IndEgo 480p/10 in-domain + Egocentric-10K 1080p/30 OOD

**Method:** Selective classification by cost-margin |p-p*|, p*=C_fa/(C_fa+C_fr); coverage 40-100%; costs 1:1/2:1/5:1; clip-bootstrap CI; LOCO temperature variant

**Owner:** monitor  **Started:** 2026-10-10 16:47

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:48): adopt.** Decision-theoretic gating: gating at coverage 45% (cost 1:1) cuts the 1,546-window REVIEW share 75.5%->55.0%; gold-LOCO BA 0.837 [0.645,0.937] (n=150); OOD-60 eval-only BA 0.496 (near chance, AUROC 0.415, n_bad=9); asymmetric costs 2:1/5:1 hurt (weighted err 0.309/0.382 vs 0.162).

Results file: eval/results/exp_review/Y2.01.json

Decision: adopt coverage 0.45, cost 1:1

Cost: CPU, <5 s, cached only
