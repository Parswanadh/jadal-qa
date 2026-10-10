# N1.03 — Best OOD score with grouped CI and error-vs-labels curve

**Why:** What is the best honest OOD score and does accuracy improve with more human OOD labels?

**Hypothesis:** Best point estimate is median aggregation 0.785 AUROC (+0.004, ns); label-using recipes plateau below the zero-shot transferable score

**Data / label source:** 160 OOD labels + IndEgo gold; cached features

**Data regime:** OOD Egocentric-10K 160 windows; 47 factories; n_bad=22

**Method:** grouped CV + repeated grouped subsampling k=10..160; grouped bootstrap CI

**Owner:** N1  **Started:** 2026-10-10 19:33

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 19:33): maybe.** Best OOD AUROC 0.7852 [0.649,0.897] (median), +0.004 vs 0.781, not significant; best error 0.2592 (top-12 selection) vs 0.2675. Error-vs-labels curve: label-free score flat; label-using recipes plateau below it.

Results file: eval/results/exp_acc/N1.03.json

Decision: Report marginal gain honestly; the zero-shot transferable score is near the ceiling for these features.

Cost: CPU only
