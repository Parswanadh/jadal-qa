# E2.03 — Shuffled tile order

**Why:** Does temporal/spatial layout carry signal, or is it only the bag of tiles?

**Hypothesis:** Shuffling tile placement costs little if content dominates; a drop means layout matters.

**Data / label source:** same windows/folds; label_source human

**Data regime:** same as E2.01

**Method:** mosaic N=9 tile 224 with deterministic placement permutation (seed 1), same eval

**Owner:** experimenter  **Started:** 2026-10-10 15:20

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:07): drop.** shuffled tile placement IndEgo 0.801 [0.64,0.89] vs ordered mosaic N9 0.785 [0.64,0.86]: delta +0.016, inside the CI; OOD 0.423. Tile order does not carry load-bearing signal -> no reason to shuffle.

Results file: eval/results/exp_mosaic/E2.03.json

Decision: Drop shuffled-order input.

Cost: 1 embed pass/window; A40
