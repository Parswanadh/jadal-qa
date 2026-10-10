# F50.05 — Cost of the 24 added features

**Why:** Edge budget: ms/window of the added kernels

**Hypothesis:** Added kernels < 250 ms/window CPU

**Data / label source:** one real 10-s window + synthetic frames

**Data regime:** CPU

**Method:** wall-clock repeats

**Owner:** F50  **Started:** 2026-10-10 21:22

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 21:34): adopt.** Added cost 148 ms/window (18 kernels 43 ms + ffprobe 104 ms); decode 157 ms shared.

Results file: eval/results/exp_feat50/cost.json

Decision: Documented for the edge budget.

Cost: 148 ms/window CPU
