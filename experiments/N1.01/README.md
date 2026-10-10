# N1.01 — OOD model shape + grouped-CV feature selection + domain adaptation on 160 labels

**Why:** Can a fitted model or in-fold selection beat the fixed equal-weight transferable score (0.781 AUROC) under factory-grouped CV?

**Hypothesis:** No: fitted classifiers overfit 22 bad windows; selection is neutral on AUROC, marginally better on balanced error

**Data / label source:** labels/ood160_labels.json (160, 138 good/22 bad, 47 factories) + manifest_gold.json (150 IndEgo); cached Z3-6 + stable-8 features

**Data regime:** OOD Egocentric-10K 160 windows; 47 factories; n_bad=22

**Method:** factory-grouped StratifiedGroupKFold(5) OOF; per-domain z-score; IndEgo sign; grouped bootstrap CI

**Owner:** N1  **Started:** 2026-10-10 19:33

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 19:33): maybe.** No fitted model beats the equal-weight mean (0.781). Best point estimate median 0.785 (+0.004, ns). Top-12 grouped-CV selection gives best balanced error 0.2592 (BA 0.7408). Features NOT domain-invariant (0.999); per-factory test-time normalization hurts (0.383).

Results file: eval/results/exp_acc/N1.01.json

Decision: Keep the equal-weight/median transferable score; do not ship fitted classifiers.

Cost: CPU only, ~4 min
