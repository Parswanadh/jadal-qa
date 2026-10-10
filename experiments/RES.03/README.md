# RES.03 — Cross cells 360p@CRF23/28, 240p@CRF28

**Why:** combine best of both axes

**Hypothesis:** 360p@CRF28 matches 480p accuracy at 40% bytes

**Data / label source:** IndEgo gold 150 windows human labels

**Data regime:** 480p 10fps H264 in-domain

**Method:** re-encode -> 160x90 -> 26 feats -> LOCO

**Owner:** monitor  **Started:** 2026-10-10 16:48

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:49): adopt.** 360p@CRF28=0.8036 at 257 KiB (best bytes/accuracy, 40% of 480p@CRF23 bytes); 240p@CRF28=0.7934 at 169 KiB inside budget.

Results file: eval/results/exp_res/RES.03.json

Decision: n/a

Cost: 257 KiB/window @ ~3.0 s/window
