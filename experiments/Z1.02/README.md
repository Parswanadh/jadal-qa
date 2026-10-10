# Z1.02 — Unsupervised alignment + per-clip normalisation for zero-shot OOD transfer

**Why:** Maybe source/target feature distributions are merely shifted; unlabelled alignment could fix it without labels

**Hypothesis:** Per-domain standardisation/whitening, CORAL and per-clip normalisation improve OOD AUROC over raw features

**Data / label source:** IndEgo 150 + OOD-60 26 feats + unlabelled Ego10K E8 corpus 1546 windows (no labels)

**Data regime:** IndEgo ~10fps vs Egocentric-10K ~30fps

**Method:** per-domain z-score/ZCA, CORAL source<->target, per-clip z/rank on IndEgo; zero-shot OOD eval

**Owner:** Z1  **Started:** 2026-10-10 17:25

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 17:25): drop.** No unsupervised alignment beat raw: coral_t2s 0.547, stable8+coral_s2t 0.678, domain_whiten worse (0.33-0.37), perclip_z 0.283; Ego10K-corpus CORAL no gain (fps confound).

Results file: eval/results/exp_adapt/Z1.02.json

Decision: Do not ship alignment; report negative.

Cost: CPU ~2 min
