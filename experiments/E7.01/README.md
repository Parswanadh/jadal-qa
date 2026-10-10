# E7.01 — Cost model: hours of video per wall-hour on the pod A40, per profile

**Why:** The product needs one honest cost table (fps x window sparsity x early exit x mosaic) computed only from measured numbers on THIS pod, so the demo/dashboard does not quote laptop or guessed throughput.

**Hypothesis:** Sparse probing (every-2nd) and a 4 s early exit multiply, and with NVDEC decode the A40 stays feature/embedding-bound; hours-of-video per wall-hour should rise >= 4x vs the full-sampling profile.

**Data / label source:** n/a (cost model): measured timings on the pod A40 from this run + source JSONs E1.01, E1.02, E1.04, E2.01, E3.04; profiles/*.json (5 profiles). No labels used.

**Data regime:** 480p/10fps H.264 IndEgo + 1080p/30fps HEVC Egocentric-10K

**Method:** timed decode (CPU ffmpeg vs NVDEC cuvid), pixel26 CPU features, EmbeddingGemma-2 batch-1 image embeddings on real mosaic PNGs; arithmetic cost model per profile with stated assumptions

**Owner:** experimenter  **Started:** 2026-10-10 15:39

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
