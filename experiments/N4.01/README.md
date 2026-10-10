# N4.01 — Label-budget curves under acquisition strategies (what to label next)

**Why:** Tell the human exactly which labels cut OOD error fastest, quantified.

**Hypothesis:** Diversity/coverage and new-factory breadth beat random; model-uncertainty sampling is anti-correlated with true OOD error (Z4).

**Data / label source:** 160 human OOD labels (138 good/22 bad, 47 factories) + frozen 26 per-pixel features; 1,546 unlabelled windows (same 47 factories)

**Data regime:** out-of-domain Egocentric-10K; factory-grouped CV

**Method:** 26-feature logistic, StratifiedGroupKFold(5, groups=factory), 5 acquisition strategies x k in {10,20,40,80,120}, paired vs random

**Owner:** N4  **Started:** 2026-10-10 19:30

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 19:30): adopt.** Diversity/maximin is the best strategy at mid budget (k=80: error 0.257 vs random 0.338, -0.081 paired, better in 4/5 folds); breadth (one window per new factory) wins at low budget (k=20: 0.384 vs 0.421); depth (more windows per existing factory) is harmful early (+0.182 at k=40); uncertainty sampling is worst mid-budget (+0.063 at k=80, worse in 4/5 folds), confirming Z4 AUROC 0.349. n=160, n_bad=22 -> differences are suggestive, not significant.

Results file: eval/results/exp_acc/N4.01.json

Decision: Label the top-200 diversity/balance list (eval/results/exp_acc/labels_next.json); do NOT label by model uncertainty.

Cost: CPU only, ~2 min, no GPU, 160+1546 cached windows
