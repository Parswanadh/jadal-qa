# P1.01 — Calibrate adapted OOD score on 160 labels + joint review/error frontier

**Why:** O4 adapted score is an uncalibrated signed z-score so frozen band cannot be applied; need calibrated P(bad) and operating point

**Hypothesis:** Platt/isotonic/beta calibration under leave-factory-out CV yields usable P(bad); band sweep on calibrated prob gives review/error frontier competitive with frozen O3 point

**Data / label source:** human Egocentric-10K 160 windows (138 good/22 bad, 47 factories)

**Data regime:** factory-mixed egocentric 1080p/30s rescores; RGB

**Method:** LFO grouped CV; Platt vs isotonic vs beta calibration; symmetric band sweep on calibrated P(bad)

**Owner:** P1  **Started:** 2026-10-10 17:55

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 17:59): drop.** Beta/Platt LFO calibration Brier 0.112 vs 0.119 base rate (n=160,22 bad,47 factories, grouped CI); symmetric 0.5-band frontier degenerate (BA 0.5, false-accept 1.0 everywhere, no operating point); frozen same-160 band op w=0.19 BA 0.634

Results file: eval/results/exp_calib160/P1.01.json

Decision: Do not ship 0.5-band on calibrated adapted score; cost-aware threshold (miss cost 3-4x) on separate data is the follow-up

Cost: CPU only, OMP_NUM_THREADS=2, seconds
