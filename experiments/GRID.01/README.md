# GRID.01 — Mosaic-native spatial features + logistic on N=9 3x3 t224 (IndEgo-150, LOCO)

**Why:** The builder's headline question: can ONE composite image of N frames be scored by OUR OWN extractors and heads instead of the full-window pipeline? Only EmbeddingGemma-2 was ever tested on mosaics; the 26-number head cannot run on a still mosaic because 10 of its 26 features are measured along time at each pixel.

**Hypothesis:** Per-tile spatial statistics pooled over the 9 tiles carry usable quality signal: nested-LOCO AUROC within 0.05 of the 26-feature baseline (0.810) but below it, because the temporal features that carry much of the baseline are absent; the mean-pooled 15-feature variant will not be worse than the 60-feature pooled variant by more than 0.02 (capacity control).

**Data / label source:** IndEgo 150 windows (100 good / 50 bad), ONE human labeller, 7 clips; mosaic built from the 480p clips with jadal/mosaic/build.py (mosaic-1.0.0, N=9 3x3 224px, timestamps on, row-major time order); features from jadal/mosaic/features.py mosaic-features-1.0.0

**Data regime:** 480p ~10 fps square 480x480 IndEgo assembly+logistics, CPU only

**Method:** StandardScaler + LogisticRegression(C=0.3, class_weight=balanced), leave-one-clip-out over 7 clips, pooled AUROC + clip-bootstrap 95% CI (2000 draws, seed 0); paired clip-bootstrap vs the 26-feature baseline; negative controls: shuffled labels, random Gaussian features, flat-grey null mosaic

**Owner:** dsh  **Started:** 2026-10-18 06:38

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 06:43): drop.** Mosaic-native SPATIAL features only, N=9 3x3 t224, LOCO n=150 (100 good/50 bad, ONE labeller, 7 clips): AUROC 0.6864 [0.522,0.772] clip-bootstrap vs the 26-number baseline 0.810 [0.683,0.892] on the SAME folds. Paired clip-bootstrap difference -0.124 [-0.202,-0.071], EXCLUDES ZERO: the still picture is significantly WORSE than the full-window pipeline, not within 0.02. Mean-only pooling (15 feats) 0.7122 [0.504,0.788], delta vs 60-feature pooling -0.026 [-0.093,+0.072] (not established). Negative controls pass: shuffled labels 0.504, random Gaussian 0.470, null grey mosaic 0.500.

Results file: eval/results/exp_grid/GRID.01.json

Decision: DROP as a decision input on in-domain evidence (decision rule (a) fails by 0.10). Keep the mosaic as an evidence picture. GRID.02/03 still run because they were pre-registered; the kill-gate at GRID.02 governs whether the line stops.

Cost: 195 ms/window for build+features (CPU), 0 GPU; 150 windows x 3 arms = ~90 s
