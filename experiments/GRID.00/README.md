# GRID.00 — Baseline gate: 26-feature logistic LOCO on IndEgo-150 must reproduce AUROC 0.810

**Why:** Pre-registered gate from the DSH brief: no GRID experiment starts until the documented baseline is reproduced on this machine from the pod-bundle cache, proving data paths, features and folds are the same as the pod.

**Hypothesis:** Pooled leave-one-clip-out AUROC on the 150 human-labelled IndEgo windows = 0.810 +/- 0.01; semantic-bad and quality-bad breakdown as documented (pallet/pallet2 at chance).

**Data / label source:** IndEgo 150 windows (100 good / 50 bad), ONE human labeller; features from pod bundle cache caches/gpu/features.json (26 per-pixel features); labels labels/manifest_gold.json

**Data regime:** 480p, ~10 fps, IndEgo assembly+logistics, 7 clips

**Method:** tools/port/baseline_logistic_loco.py unchanged: StandardScaler + LogisticRegression(C=0.3, class_weight=balanced), leave-one-clip-out over 7 clips, pooled AUROC; clip-bootstrap 95% CI

**Owner:** dsh  **Started:** 2026-10-18 06:30

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 06:32): adopt.** PASS. 26-number logistic LOCO AUROC 0.810 [0.683,0.892] clip-bootstrap (window-bootstrap 0.722-0.889, optimistic), n=150 (100 good/50 bad, ONE human labeller), 7 clips. good-vs-semantic-bad 0.797, good-vs-quality-bad 0.846. pallet/pallet2 at chance (0.50) as documented. Gate expected 0.810 +/-0.01: observed 0.810.

Results file: eval/results/grid/GRID.00.json

Decision: Gate passed: GRID.01-09 and AUTO.00+ may start. Cached pod-bundle 26 features reproduce exactly on this laptop; use jadal-qa-wt/handover/jadal_pod_bundle/pod_bundle as JADAL_BUNDLE.

Cost: 0 GPU, ~2 s CPU
