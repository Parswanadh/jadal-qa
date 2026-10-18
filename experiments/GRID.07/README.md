# GRID.07 — Cost of the shipped full-window path vs the mosaic path, CPU-only at 480p: decode/build/features split

**Why:** Footage-hours per wall-hour on the actual deployment CPU decides whether the mosaic path is affordable; the S1 stream measured build+features (~195 ms/window) excluding decode, so decode must be measured separately.

**Hypothesis:** At N=9 3x3 224 px tiles the mosaic path is dominated by decode on this laptop; the shipped 160x90 feature path is cheaper per window than the mosaic path.

**Data / label source:** IndEgo 480p clip windows (real footage, read-only); no frames committed

**Data regime:** CPU-only on this laptop (22 cores / 15 GB / no GPU used), 480p source, 10 s windows, 2 worker processes max, nice -n 10, OMP_NUM_THREADS=2

**Method:** Per-stage wall timing in the same process: cv2 decode of a 10 s window, mosaic build, mosaic features, and the shipped path (ffmpeg 160x90 decode + luma + s0_from_luma + boxcar + cpu_window_features); median and IQR over repeated windows; footage-hours/wall-hour = 3600 / (ms per window) * (10 s per window) / 3600

**Owner:** dsh  **Started:** 2026-10-18 06:50

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 07:30): adopt.** CPU-only, 480p, 10 s windows, one clip (ecu_full), 12 windows x 3 repeats, every stage timed in-process, plus measured 2-worker runs. Stage medians (ms): shipped arm on the 30 fps-normalised stream decode 130.3 + S0 61.2 + features 151.1 = 348.4 (IQR 336-453); shipped arm decoding the raw 10 fps source and frame-repeating decode 104.3 + repeat 1.0 + S0 61.2 + features 151.1 = 330.3; mosaic arm cv2 480p decode (9 kept frames) 58.4 + build 18.6 + feature_vector 112.0 = 190.5 (IQR 182-197), with the GRID.05 canvas S0 probe at 63.3 ms reported separately and EXCLUDED from the total. Footage-hours per wall-hour = 10000/median_total_ms [DERIVED]: shipped-normalised 28.7, shipped-raw 30.3, mosaic 52.5, mosaic decode+build+features only 75.7. Measured 2-worker: mixed shipped+mosaic 52.6 (72 jobs, 13.7 s) and mosaic-only 66.8 (36 jobs, 5.4 s) => the second worker buys 1.27x, because the per-job median rises 190.5 -> 223.5 ms; the laptop is the bottleneck. The 30 fps normalisation pre-pass costs 49.7 ms per source second (497 ms per 10 s window amortised, one-off per clip) and is NOT included in any per-window number. The S1 stream reported ~195 ms for mosaic build+features excluding decode; this session measures 131-132 ms for the same configuration - both are single-session medians under different desktop load, and the difference was not investigated. CAVEATS: medians under desktop load with no cpu pinning (one shipped outlier at 976 ms), all windows from ONE clip so no clip-to-clip variation, no GPU/cross-device/Raspberry-Pi claim, no service overhead, nothing about 1080p.

Results file: eval/results/exp_grid/GRID.07.json

Decision: Cost baseline for the memo: the mosaic path (480p decode + 3x3/224px build + full feature vector) costs 190.5 ms/window and yields 52.5 footage-hours per wall-hour per worker on this laptop, versus 348.4 ms and 28.7 footage-h/wall-h for the shipped full-window path on the same footage; build+features alone are 131-132 ms (75.7 footage-h/wall-h). Two workers give a MEASURED 66.8 footage-h/wall-h for mosaic-only (1.27x, not 2x).

Cost: 36 passes per arm single-process + 72 mixed and 36 mosaic-only 2-worker jobs + one 30fps normalisation pre-pass; CPU only, nice -n 10, OMP_NUM_THREADS=2, ~3 min wall
