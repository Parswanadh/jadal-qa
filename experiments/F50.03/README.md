# F50.03 — Ablation of the 24 added features

**Why:** Which added feature families carry the gain; is there saturation?

**Hypothesis:** A few families dominate; saturation after ~6-10 features

**Data / label source:** same as F50.02

**Data regime:** OOD-160

**Method:** leave-one-family-out + family-only + cumulative

**Owner:** F50  **Started:** 2026-10-10 21:22

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 21:34): drop.** Ablation: only dct_bands (0.728 alone) and edge_lap (0.680) carry OOD signal; saturation immediate (0.781->0.779->...->0.657).

Results file: eval/results/exp_feat50/ablation.json

Decision: No family worth adding.

Cost: n/a
