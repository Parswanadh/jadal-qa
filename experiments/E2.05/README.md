# E2.05 — Single middle frame

**Why:** Lower bound: one frame instead of N (cheapest input).

**Hypothesis:** Single frame is clearly worse than N-tile mosaic and mean-of-frames.

**Data / label source:** same windows/folds; label_source human

**Data regime:** same as E2.01

**Method:** 1x1 mosaic of the middle frame (t=5s), same eval

**Owner:** experimenter  **Started:** 2026-10-10 15:20

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:07): maybe.** single middle frame (1x1, t=5s) IndEgo 0.784 [0.69,0.85] ~= mosaic N9 0.785 [0.64,0.86] (1/9 the embedding cost) but 0.068 below mean-of-frames N9 0.852; OOD 0.514. Cheapest input but below the 0.02 bar to the baseline.

Results file: eval/results/exp_mosaic/E2.05.json

Decision: Keep as a cost floor / dashboard thumbnail; not a model input.

Cost: 1 embed pass/window; A40
