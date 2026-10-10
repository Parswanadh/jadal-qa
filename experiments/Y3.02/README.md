# Y3.02 — Cheap view-ensembling combo sweep for score separability

**Why:** Does adding cached views or switching prob-avg->rank-avg cut the REVIEW (ambiguous) share at equal/better AUROC?

**Hypothesis:** More views => more separable OOF scores at equal or better AUROC.

**Data / label source:** 150-window IndEgo human gold (LOCO); 60-window OOD evaluation only

**Data regime:** in-domain LOCO + OOD descriptive

**Method:** nested-LOCO prob/rank fusion of 6 cached views; ambiguous fraction in [0.4,0.6] and profile band [0.25,0.75]

**Owner:** Y3  **Started:** 2026-10-10 16:58

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:58): drop.** NEGATIVE: adding cached views does not cut ambiguity at equal/better AUROC; every extra view raises the review-band mass and AUROC is flat/worse. EG2-alone dominates the two-view baseline (AUROC .834->.846, profile-band .653->.420). Rank-average is a free improvement (.653->.607). All AUROC diffs within CI.

Results file: eval/results/exp_review/Y3.02.json

Decision: Adopt rank-average (free); consider EG2-only; reject adding per-pixel/physics/series/theory views

Cost: CPU ~3 min
