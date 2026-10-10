# E4.02 — Regime-specific vs global models within IndEgo (nested LOCO)

**Why:** Decide whether blurry/low-light/walking terciles deserve their own models or thresholds

**Hypothesis:** Regime-specific models beat global only where regimes have both classes and enough clips

**Data / label source:** 150 IndEgo gold windows, human labels, leave-one-clip-out over 7 clips

**Data regime:** terciles of sharpness/lighting/motion

**Method:** logistic (C=0.01, balanced) on 26 per-pixel feats and EG2 embedding; global vs regime-only; 2000x clip-bootstrap CI

**Owner:** experimenter-a3  **Started:** 2026-10-10 15:18

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 15:30): drop.** No regime split adopted. Under identical LOCO folds regime-only logistic never beat global by >0.03 with a CI supporting it: best feat26 cell lighting_lo +0.125 [-0.10,0.35]; EG2 regime models were worse (delta -0.01..-0.18). Global AUROC feat26 0.712 [0.634,0.811], EG2 0.848 [0.719,0.912] (n=150, 7 clips, 2000x clip bootstrap). Motion terciles are thin (motion_lo n_bad=4).

Results file: eval/results/exp_regimes/E4.02.json

Decision: Keep one global model per view; no per-regime model; report thin regimes

Cost: 150 windows, CPU, ~10 min
