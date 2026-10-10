# L1.01 — End-to-end pipeline latency: full-corpus run + atomic per-stage analysis

**Why:** Need per-stage timings and footage/wall-hour to size the PoC and find the bottleneck

**Hypothesis:** Decode or EG2-embed dominates wall time; NVDEC beats CPU decode; sparse-probe profile roughly halves cost

**Data / label source:** Egocentric-10K 1080p/30 HEVC 58 clips/1546 windows (unlabelled) + subset timed atomically; IndEgo 7 clips for reference

**Data regime:** 1080p30 HEVC native + 480p transcodes; A40

**Method:** perf_counter around io_probe/decode_cpu/decode_nvdec/features_pixel26/eg2_embed/mosaic/decision/write; OMP_NUM_THREADS=1; nvidia-smi sampling

**Owner:** L1  **Started:** 2026-10-10 17:58

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
