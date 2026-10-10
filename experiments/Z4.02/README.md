# Z4.02 — Sequential SPRT clip decision frontier

**Why:** Goal (a): process windows one at a time and stop when evidence crosses accept/reject/abstain bounds; sweep bounds to get clip error vs windows processed vs abstain

**Hypothesis:** A Wald SPRT on calibrated log-odds reaches the tolerated clip-verdict accuracy with fewer reviewed windows than a fixed grid

**Data / label source:** IndEgo gold + cached oof_probs.npz; OOD-60 never used for fitting

**Data regime:** 480p/10fps IndEgo

**Method:** soft Bernoulli SPRT on 1-p; hard (binary at 0.5), soft (p_platt) and label-oracle modes; (A,B) sweep; error/cost/abstain frontier

**Owner:** Z4  **Started:** 2026-10-10 17:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 17:30): drop.** Best tolerated SPRT cell (A=4.9,B=3.7, hard) decides 5/5 correct with 2/7 abstained but at 66.5% of windows -- MORE than every-2's 51.5%. Label-oracle SPRT also 65.5% at 7/7. Soft expected-LLR never accepts (clip-level miscalibration). Sequential early stop does NOT drop review at equal error.

Results file: eval/results/exp_clip/Z4.02.json

Decision: Keep fixed every-2 grid; use SPRT only as a clip-level abstain gate. Negative result kept.

Cost: CPU-only, cached scores, no GPU
