# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| N5.01 | N5 | done | adopt | Refit/recalibrate decision heads for the optimized sparse pipeline (gold) | IndEgo gold 150 windows (manifest_gold.json, human, 100 good / in-domain LOCO |
| N5.02 | N5 | done | adopt | OOD validation of the recalibrated sparse heads (160 windows) | 160 human-labelled Egocentric-10K windows (ood160_labels.jso / out-of-domain |

## N5.01 — Refit/recalibrate decision heads for the optimized sparse pipeline (gold)
- **Why:** L3 (PR #148) found the sparse sampling shifts the frozen-head score (+0.20) so the optimized run agrees with the frozen verdicts only 54%.
- **Hypothesis:** Refitting the EG2/pixel logistic heads on sparse features and re-deriving the threshold/abstain band restores the frozen operating point (review share + confident balanced accuracy) at no AUROC cost.
- **Data / label source:** IndEgo gold 150 windows (manifest_gold.json, human, 100 good/50 bad), 7 clips
- **Regime:** in-domain LOCO
- **Method:** re-extract pixel 3.33fps + EG2 1fps + 4s prefix via the optimized runner code path; nested-LOCO refit logistic (balanced, max_iter=5000); threshold/band re-derivation
- **Owner / started / finished:** N5 / 2026-10-10 19:27 / 2026-10-10 19:41
- **Verdict:** adopt — Sparse-feature refit of both heads restores the frozen gold operating point (review share 0.64 vs 0.653 at profile 0.75/0.25; confident BA 0.848 vs 0.840) at no ensemble AUROC cost (nested-C LOCO 0.840 vs 0.834). Ship refit heads + band [0.4193,0.6112].
- **Decision:** ship configs/ensemble_v1_sparse.json (refit heads, frozen C); profile thresholds unchanged
- **Cost:** GPU ~3 min (A40), CPU ~30 s
- **Results:** eval/results/exp_recal/N5.01.json

## N5.02 — OOD validation of the recalibrated sparse heads (160 windows)
- **Why:** The refit heads must not degrade out-of-domain error vs the frozen heads.
- **Hypothesis:** Sparse-refit heads preserve the frozen OOD AUROC within the grouped-CI.
- **Data / label source:** 160 human-labelled Egocentric-10K windows (ood160_labels.json, 138 good/22 bad, 47 factories)
- **Regime:** out-of-domain
- **Method:** apply frozen vs sparse-refit heads to sparse features; factory-grouped bootstrap CI
- **Owner / started / finished:** N5 / 2026-10-10 19:27 / 2026-10-10 19:41
- **Verdict:** adopt — Sparse-refit OOD AUROC 0.5231 vs frozen 0.5660 (delta -0.043), inside the frozen factory-grouped CI; both near chance (no transfer, n_bad=22). OOD error preserved.
- **Decision:** no OOD regression; ship
- **Cost:** GPU ~2 min (A40), CPU ~15 s
- **Results:** eval/results/exp_recal/N5.02.json
