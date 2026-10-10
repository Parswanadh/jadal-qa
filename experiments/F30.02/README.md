# F30.02 — F30 cost: ms/window of the 4 added invariants

**Why:** Quantify the extraction cost of the new features

**Hypothesis:** The 4 invariants add <30% over the 26-feature cost

**Data / label source:** 16 OOD windows, 160x90, ~10 fps

**Data regime:** CPU timing

**Method:** time decode / 26 / 4 / 30 per window

**Owner:** F30  **Started:** 2026-10-10 21:32

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 21:32): adopt.** Decode 637 ms, 26 feats 229 ms, added 4 = 57.4 ms/window (25.1%), 30 total 284 ms (median, 16 windows).

Results file: eval/results/exp_feat30/F30.02_cost.json

Decision: Cost is acceptable.

Cost: n/a
