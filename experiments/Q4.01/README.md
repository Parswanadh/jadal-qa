# Q4.01 — Deterministic physics model: runnable package + pure-vs-hybrid comparison

**Why:** Make the deterministic physics model ready to ship as a runnable, documented entry point, and quantify what determinism costs vs a rules+logistic-fallback hybrid.

**Hypothesis:** A packaged integer-only rule scorer reproduces Q1's LOCO decisions byte-for-byte; a deterministic abstain to the frozen ensemble only for REVIEW recovers the review band without changing ACCEPT/REJECT.

**Data / label source:** 150 IndEgo human-gold windows (100/50, 7 clips, LOCO) + 160 Egocentric-10K human windows (138/22, 47 factories, OOD); cached features + freshly extracted OOD full/phys features

**Data regime:** in-domain IndEgo 480p/10fps; OOD Egocentric-10K downscaled to 160x90

**Method:** tools/phys integer-quantised scorer (Q=4, sorted canonical receipt); gold LOCO with fold-local thresholds + 5% FR reject point; OOD frozen table; hybrid REVIEW->ensemble_v1 accept iff prob>=0.5; clip / factory-grouped bootstrap CI (2000)

**Owner:** Q4  **Started:** 2026-10-10 19:04

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 19:04): adopt.** Package tools/phys (rules.json, model.py, score_window.py, extract_features.py, tests, README) reproduces Q1 LOCO exactly (AUROC 0.8222, BA 0.808 decided-only, REVIEW 0.60) and is integer-only/canonical. Gold: pure AUROC 0.8222 [0.713,0.855], BA(review->accept) 0.710, 60% REVIEW, 40% decided deterministically; hybrid resolves REVIEW to BA 0.790 [0.692,0.828] (ensemble OOF AUROC 0.8336; determinism cost +0.0114). OOD-160: pure AUROC 0.4272 [0.269,0.562] below chance, 0% ACCEPT, 35.6% REVIEW; hybrid BA 0.468 (ensemble 0.566; determinism cost +0.1388). Fallback buys +0.080 BA gold / +0.078 OOD but needs the learned ensemble.

Results file: eval/results/exp_phys/Q4.01.json

Decision: Ship the pure deterministic rules (reject_threshold=14, 5%-FR point) as the deterministic product mode; keep the ensemble fallback as an explicit assisted mode only. Do not ship the table on OOD (below chance, flags 100%).

Cost: CPU only; 12 unit tests <1s; OOD feature extraction 160 windows/2 workers/99s; hybrid eval 16s
