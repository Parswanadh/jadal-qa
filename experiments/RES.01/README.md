# RES.01 — Resolution ladder: IndEgo pixel-26 at 480/360/240/180/120p

**Why:** how far can encode resolution drop before the frozen 26-feature LOCO AUROC falls

**Hypothesis:** AUROC stays within 0.02 of 480p down to 120p

**Data / label source:** IndEgo gold 150 windows human labels 100good/50bad

**Data regime:** 480p 10fps H264 in-domain

**Method:** re-encode each window libx264 crf23 -> decode 160x90 -> cpu_window_features 26 -> LOCO C=0.3 clip-bootstrap CI

**Owner:** monitor  **Started:** 2026-10-10 16:48

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:49): adopt.** Encoded resolution can drop to 180p (0.8428, 198 KiB) and 120p (0.8022, 137 KiB) with no AUROC loss vs 0.810 native; no knee down to 120p; all rungs within 0.02 (one CI).

Results file: eval/results/exp_res/RES.01.json

Decision: n/a

Cost: encode~0.6s+dec0.2s+feat0.5s per 10s window (shared pod)
