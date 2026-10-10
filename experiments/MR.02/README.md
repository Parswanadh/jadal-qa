# MR.02 — Low-res mosaic: density ladder (tile112, 2x2->6x6) vs mean-of-frames

**Why:** Does packing more lower-res frames recover the E2 gap vs mean-of-frame embeddings?

**Hypothesis:** Higher density at small tiles narrows but does not close the mean-of-frames gap.

**Data / label source:** 150 IndEgo gold windows, human, LOCO over 7 clips

**Data regime:** in-domain IndEgo 480p

**Method:** EG2 mosaic embedding vs mean-of-frame embeddings, LOCO

**Owner:** monitor  **Started:** 2026-10-10 16:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:50): drop.** At matched frame count mosaic loses to mean-of-frames at EVERY density (-0.048 N9, -0.063 N16, -0.113 N25, -0.081 N36). Best density 3x3 (0.803). E2 verdict stands; density does not recover the gap.

Results file: eval/results/exp_mosaic_res/MR.02.json

Decision: keep as evidence artefact only

Cost: 43.7 ms/win @3x3/112 to 107.4 ms/win @6x6/112
