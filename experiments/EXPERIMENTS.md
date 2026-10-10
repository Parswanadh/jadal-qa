# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| Y3.01 | Y3 | done | adopt | Cached-view inventory + two-view baseline reproduction | 150-window IndEgo human gold (LOCO, 7 clips) / in-domain LOCO |
| Y3.02 | Y3 | done | drop | Cheap view-ensembling combo sweep for score separability | 150-window IndEgo human gold (LOCO); 60-window OOD evaluatio / in-domain LOCO + OOD descriptive |

## Y3.01 — Cached-view inventory + two-view baseline reproduction
- **Why:** Anchor the review-reduction study: which cached views exist and does the frozen two-view ensemble reproduce AUROC 0.8534 OOF?
- **Hypothesis:** The frozen pixel-26+EG2@2fps probability average reproduces AUROC 0.8534; extra cached views exist (per-pixel series/physics/theory).
- **Data / label source:** 150-window IndEgo human gold (LOCO, 7 clips)
- **Regime:** in-domain LOCO
- **Method:** nested LOCO logistic per view; fixed-C frozen anchor
- **Owner / started / finished:** Y3 / 2026-10-10 16:58 / 2026-10-10 16:58
- **Verdict:** adopt — Baseline reproduced: fixed-C two-view prob-average AUROC 0.8534, nested-LOCO 0.8336. Per-view nested AUROC: eg2 0.8458, perpixel121 0.8140, pixel26 0.7916, series16 0.7912, physics16 0.7608, theory18 0.6698. EG2@1fps/@0.5fps caches absent (only @2fps).
- **Decision:** Use as reference; combo sweep in Y3.02
- **Cost:** CPU <1 min
- **Results:** eval/results/exp_review/Y3.01.json

## Y3.02 — Cheap view-ensembling combo sweep for score separability
- **Why:** Does adding cached views or switching prob-avg->rank-avg cut the REVIEW (ambiguous) share at equal/better AUROC?
- **Hypothesis:** More views => more separable OOF scores at equal or better AUROC.
- **Data / label source:** 150-window IndEgo human gold (LOCO); 60-window OOD evaluation only
- **Regime:** in-domain LOCO + OOD descriptive
- **Method:** nested-LOCO prob/rank fusion of 6 cached views; ambiguous fraction in [0.4,0.6] and profile band [0.25,0.75]
- **Owner / started / finished:** Y3 / 2026-10-10 16:58 / 2026-10-10 16:58
- **Verdict:** drop — NEGATIVE: adding cached views does not cut ambiguity at equal/better AUROC; every extra view raises the review-band mass and AUROC is flat/worse. EG2-alone dominates the two-view baseline (AUROC .834->.846, profile-band .653->.420). Rank-average is a free improvement (.653->.607). All AUROC diffs within CI.
- **Decision:** Adopt rank-average (free); consider EG2-only; reject adding per-pixel/physics/series/theory views
- **Cost:** CPU ~3 min
- **Results:** eval/results/exp_review/Y3.02.json
