# L3.01 — L3.01 full-corpus pipeline at full resources (7 workers) + adopted optimizations

**Why:** The frozen L2.01 baseline ran deliberately slow (4 workers, 300 pixel-frames/window, 2 fps EG2, mosaic on). Measure the same corpus with the pod's adopted optimizations and full cgroup resources to quantify the speedup.

**Hypothesis:** E1.01 3.33 fps pixel + E1.02 1 fps EG2 + E3.03 4 s early exit + E2 skip mosaic cut feature/EG2/mosaic cost several-fold; E3.01 every-2nd halves windows; throughput rises well above the 0.84 win/s baseline while the verdict distribution stays close.

**Data / label source:** queues/manifest_scale.json: 1546 windows / 58 clips / 4.404 h, 1080p30 HEVC; label_source none (descriptive verdicts)

**Data regime:** factory Egocentric-10K, 1080p, 30 fps, HEVC

**Method:** same harness as L2.01 (repo E8.01 pipeline, perf_counter per stage); adopted optimizations as CLI/env flags; scoring code unchanged; stride-1 full-corpus run + stride-2 every-2nd run; A40, 7 workers, JADAL_GPU_SLOTS=6

**Owner:** l3-latency  **Started:** 2026-10-10 18:41

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
