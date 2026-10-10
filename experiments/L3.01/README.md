# L3.01 — L3.01 full-corpus pipeline at full resources (7 workers) + adopted optimizations

**Why:** The frozen L2.01 baseline ran deliberately slow (4 workers, 300 pixel-frames/window, 2 fps EG2, mosaic on). Measure the same corpus with the pod's adopted optimizations and full cgroup resources to quantify the speedup.

**Hypothesis:** E1.01 3.33 fps pixel + E1.02 1 fps EG2 + E3.03 4 s early exit + E2 skip mosaic cut feature/EG2/mosaic cost several-fold; E3.01 every-2nd halves windows; throughput rises well above the 0.84 win/s baseline while the verdict distribution stays close.

**Data / label source:** queues/manifest_scale.json: 1546 windows / 58 clips / 4.404 h, 1080p30 HEVC; label_source none (descriptive verdicts)

**Data regime:** factory Egocentric-10K, 1080p, 30 fps, HEVC

**Method:** same harness as L2.01 (repo E8.01 pipeline, perf_counter per stage); adopted optimizations as CLI/env flags; scoring code unchanged; stride-1 full-corpus run + stride-2 every-2nd run; A40, 7 workers, JADAL_GPU_SLOTS=6

**Owner:** l3-latency  **Started:** 2026-10-10 18:41

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 19:20): maybe.** Full corpus 1546 windows/4.404 h at 7 workers + JADAL_GPU_SLOTS=6 with adopted opts (E1.01 pixel 3.33 fps, E1.02 EG2 1 fps, E3.03 4 s exit, E2 skip mosaic). Wall 801 s, 1.93 win/s, 19.8 footage-h/wall-h = 2.30x baseline throughput (0.84 win/s). Clean 1-worker per-window total p50 1483 ms = 2.99x baseline; per-stage p50 speedups pixel 13.2x, EG2 6.6x, descriptors 2.1x, mosaic skipped. BUT the frozen-head verdict distribution is NOT close: score mean +0.20 (0.486->0.687), accept 296->876, agreement 54% — driven by pixel-fps (+0.15) and early-exit (+0.12), EG2 neutral; needs head refit/recalibration. Decode is the bottleneck (81% of per-window) and 7 workers serialise on the single A40 NVDEC (per-window decode 2780 ms at 7 workers vs 943 ms at 1).

Results file: eval/results/exp_latency_fast/L3.01.json

Decision: Adopt the resource/feature optimizations for throughput; do NOT ship sparse sampling with the frozen head until it is refit or the profile thresholds recalibrated. Every-2nd (E3.01) gives 2x footage coverage on top (39.6 footage-h/wall-h derived; 32.9 measured, partly contended).

Cost: 7 workers x 801 s + 482 s (stride-2) + 1-worker probes on a shared A40; 1546 windows x (13 pixel frames @160x90 + 4 EG2 frames @512x288 + 3 native frames)
