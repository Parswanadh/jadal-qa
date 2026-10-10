# Z1.01 — Stable-feature selection for zero-shot OOD transfer

**Why:** IndEgo-trained pixel heads score near chance OOD (0.41-0.53 AUROC); the 26 features include sign-flipping motion/temporal features (E5.03)

**Hypothesis:** Restricting the head to the 8 sign-stable features (and the 3 sharpness features) cuts OOD error vs all-26

**Data / label source:** IndEgo gold 150 windows (human) + OOD-60 (human, 51 good/9 bad)

**Data regime:** IndEgo ~10fps 480p vs Egocentric-10K transcodes

**Method:** LOCO logistic on IndEgo, zero-shot scored on OOD-60; C by inner LOCO

**Owner:** Z1  **Started:** 2026-10-10 17:25

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 17:25): maybe.** Sharpness-3 head (radial_slope,hf_ratio,dct_energy) cuts OOD AUROC 0.534->0.706 [0.518,0.867] (frozen 0.415); stable-8 only 0.551; in-domain LOCO drops 0.774->0.578. n_bad=9 exploratory.

Results file: eval/results/exp_adapt/Z1.01.json

Decision: Adopt sharpness-3 as the no-label OOD view; keep full-26 for in-domain.

Cost: CPU ~seconds
