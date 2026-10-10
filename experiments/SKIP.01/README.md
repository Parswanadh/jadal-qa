# SKIP.01 — Section-drop rules anywhere (start/middle/end/evenly) on IndEgo gold

**Why:** Q2: E3 only tested skipping the START and it failed; test whether dropping 1-2 of 5 equal sections ANYWHERE keeps the clip verdict at lower cost

**Hypothesis:** Dropping middle/end/evenly biases the clip bad-fraction less than dropping the start, and may match the verdict at lower cost

**Data / label source:** IndEgo human gold 150 windows/7 clips (labels/manifest_gold.json) + cached oof_probs.npz

**Data regime:** 480p/10fps IndEgo assembly+logistics

**Method:** 5 equal sections per clip; drop 1/2 under START/MIDDLE/END/EVENLY (seeds 0-2); error vs full coverage + verdict agreement @0.30; clip bootstrap CI

**Owner:** X2  **Started:** 2026-10-10 16:27

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:27): maybe.** drop1-start err 0.110 (worst, confirms E3) vs drop1-end 0.057 / drop1-evenly 0.033-0.041; drop2-start 0.189. START-drop biases high, END-drop low (bad rate sec1 0.09 -> sec5 0.58). END/MIDDLE/EVENLY keep 7/7 verdicts but every variant is dominated by uniform-every-2 (0.049 @ 51%). No section drop adopted.

Results file: eval/results/exp_skip/SKIP.01.json

Decision: Do not adopt any section-drop rule; keep every-2. Direction (start worst, end/even safest) is evidence for time-spread sampling.

Cost: CPU-only simulation on cached labels+scores, no GPU
