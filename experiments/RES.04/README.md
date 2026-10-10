# RES.04 — OOD resolution sensitivity: frozen pixel-26 on Ego10K 1080p/480p/360p

**Why:** resolution reduction is OOD-neutral

**Hypothesis:** delta within +/-0.013 of 1080p

**Data / label source:** Egocentric-10K OOD 60 windows (51 good/9 bad) human labels

**Data regime:** 1080p30 HEVC OOD

**Method:** NVDEC decode -> one-pass downscale -> frozen pixel-26 head -> AUROC + bootstrap CI

**Owner:** monitor  **Started:** 2026-10-10 16:48

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:49): maybe.** Pixel-26 OOD AUROC~0.43 (IndEgo head does not transfer) and is resolution-neutral: 480p d-0.002 [-0.026,+0.019], 360p d+0.013 [-0.034,+0.059]. EG2 view NOT recomputed.

Results file: eval/results/exp_res/RES.04.json

Decision: n/a

Cost: NVDEC; 60 windows
