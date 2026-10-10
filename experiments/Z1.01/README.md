# Z1.01 — Mosaic stage ON vs OFF: wall-time delta + verdict identity

**Why:** Confirm the mosaic is an evidence artefact only: it should add wall time but not change any verdict/score/features_digest

**Hypothesis:** Mosaic ON adds its build cost to the per-window wall but verdicts, scores and features_digest are byte-identical to mosaic OFF

**Data / label source:** 30-window sample spread over 30 factories, queues/manifest_scale.json, 1080p30 HEVC; label_source none

**Data regime:** factory Egocentric-10K, 1080p, 30fps, HEVC

**Method:** repo E8.01 pipeline (eval/exp_rescore/run_e8.py) run twice per window with only jadal.mosaic.build enabled/disabled; perf_counter per stage; verdict/score/features_digest compared

**Owner:** l4-zerocopy  **Started:** 2026-10-10 18:47

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 21:14): adopt.** Mosaic ON vs OFF on 30 windows (30 factories): the mosaic build is the only difference and adds 233 ms/window (p50 227); verdicts, scores and features_digest are IDENTICAL (0 disagreements). Mosaic is an evidence artefact only.

Results file: eval/results/exp_zerocopy/Z1.01.json

Decision: Keep mosaic as evidence, never as a model input.

Cost: 30 windows x 2 full pipeline passes on A40; mosaic ~233 ms/window (CPU PIL).
