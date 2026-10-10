# RES.02 — Compression ladder: H.264 CRF 18/23/28/33 at 480p

**Why:** CRF28 gives ~50% bytes with dAUROC<0.02

**Hypothesis:** CRF28 within 0.02 and ~57% bytes; CRF33 36%

**Data / label source:** IndEgo gold 150 windows human labels

**Data regime:** 480p 10fps H264 in-domain

**Method:** re-encode crf ladder -> 160x90 -> 26 feats -> LOCO

**Owner:** monitor  **Started:** 2026-10-10 16:48

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:49): adopt.** Target met: CRF28=0.7960 (dAUROC -0.014) at 57% of CRF23 bytes; CRF33=0.7946 (-0.015) at 36%; no CRF broke 0.02.

Results file: eval/results/exp_res/RES.02.json

Decision: n/a

Cost: flat ms; bytes 1.14MB->230KB
