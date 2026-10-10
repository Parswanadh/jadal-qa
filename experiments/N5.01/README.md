# N5.01 — Refit/recalibrate decision heads for the optimized sparse pipeline (gold)

**Why:** L3 (PR #148) found the sparse sampling shifts the frozen-head score (+0.20) so the optimized run agrees with the frozen verdicts only 54%.

**Hypothesis:** Refitting the EG2/pixel logistic heads on sparse features and re-deriving the threshold/abstain band restores the frozen operating point (review share + confident balanced accuracy) at no AUROC cost.

**Data / label source:** IndEgo gold 150 windows (manifest_gold.json, human, 100 good/50 bad), 7 clips

**Data regime:** in-domain LOCO

**Method:** re-extract pixel 3.33fps + EG2 1fps + 4s prefix via the optimized runner code path; nested-LOCO refit logistic (balanced, max_iter=5000); threshold/band re-derivation

**Owner:** N5  **Started:** 2026-10-10 19:27

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 19:41): adopt.** Sparse-feature refit of both heads restores the frozen gold operating point (review share 0.64 vs 0.653 at profile 0.75/0.25; confident BA 0.848 vs 0.840) at no ensemble AUROC cost (nested-C LOCO 0.840 vs 0.834). Ship refit heads + band [0.4193,0.6112].

Results file: eval/results/exp_recal/N5.01.json

Decision: ship configs/ensemble_v1_sparse.json (refit heads, frozen C); profile thresholds unchanged

Cost: GPU ~3 min (A40), CPU ~30 s
