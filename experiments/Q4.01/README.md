# Q4.01 — Deterministic physics model: runnable package + pure-vs-hybrid comparison

**Why:** Make the deterministic physics model ready to ship as a runnable, documented entry point, and quantify what determinism costs vs a rules+logistic-fallback hybrid.

**Hypothesis:** A packaged integer-only rule scorer reproduces Q1's LOCO decisions byte-for-byte; a deterministic abstain to the frozen ensemble only for REVIEW recovers the review band without changing ACCEPT/REJECT.

**Data / label source:** 150 IndEgo human-gold windows (100/50, 7 clips, LOCO) + 160 Egocentric-10K human windows (138/22, 47 factories, OOD); cached features + freshly extracted OOD full/phys features

**Data regime:** in-domain IndEgo 480p/10fps; OOD Egocentric-10K downscaled to 160x90

**Method:** tools/phys integer-quantised scorer (Q=4, sorted canonical receipt); gold LOCO with fold-local thresholds + 5% FR reject point; OOD frozen table; hybrid REVIEW->ensemble_v1 accept iff prob>=0.5; clip / factory-grouped bootstrap CI (2000)

**Owner:** Q4  **Started:** 2026-10-10 19:04

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 19:04): adopt.** Package tools/phys (rules.json, model.py, score_window.py, extract_features.py, tests, README) reproduces Q1 LOCO (AUROC 0.8222 [0.715,0.860]; OPTIMISTIC: rules/features/signs chosen with all 150 labels). Gold full-population BA 0.710 (review-as-accept) / 0.590 (review-as-reject) at 40% coverage; 0.808 is decided-only. Paired clip-bootstrap rules-minus-ensemble AUROC -0.0114 CI [-0.136,+0.060] (contains 0 => not established); fallback BA gain +0.080 CI [+0.016,+0.131]. Hybrid BA 0.790 [0.691,0.827]. OOD-160: pure AUROC 0.4272 [0.275,0.561] (CI contains 0.5, not distinguishable from chance), 0% ACCEPT, 35.6% REVIEW; hybrid BA 0.468 [0.384,0.544]. Fallback source sha256 bound into each receipt (fallback.source_sha256). Per-window scores committed for traceability.

Results file: eval/results/exp_phys/Q4.01.json

Decision: Ship the pure deterministic rules on IndEgo-comparable footage only (reject_threshold=14, 5%-FR point) as the deterministic product mode; quote full-population BA 0.710/0.590, NOT decided-only 0.808. Keep the ensemble fallback as an explicit assisted mode (needs a hashed source). Do not ship the table on OOD (not distinguishable from chance, flags ~100%). No replacement OOD scorer is established by this PR.

Cost: CPU only; 13 unit tests <1s; OOD feature extraction 160 windows/2 workers/99s; hybrid eval 16s
