# Z1.02 — Zero-copy GPU path attempt (NVDEC->VRAM features/embed) vs hwdownload

**Why:** Measure the mosaic/round-trip hypothesis: can decoded frames stay GPU-resident from NVDEC through pixel-26 features and the EG2 embedder, avoiding hwdownload+re-upload

**Hypothesis:** A GPU-resident path removes the VRAM<->RAM round trips for the pixel+EG2 branches and speeds up decode/pixel_features/eg2_embed; if the repo cannot consume CUDA frames the step is reported blocked

**Data / label source:** same 30-window sample; label_source none

**Data regime:** factory Egocentric-10K, 1080p, 30fps, HEVC

**Method:** baseline repo hwdownload path vs GPU path: in-process NVDEC (torchcodec) -> jadal.gpu.features.gpu_window_features_batch on CUDA tensors -> EmbeddingGemma-2 GPU preprocess+forward directly on device tensors; per-stage perf_counter; feature deltas + EG2 cosine + verdict agreement + PCIe bytes

**Owner:** l4-zerocopy  **Started:** 2026-10-10 18:47

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 21:14): maybe.** True zero-copy (NVDEC->VRAM features/embed) is BLOCKED: the repo decodes out-of-process via a host-memory ffmpeg pipe, and no in-process NVDEC binding works here (torchcodec 0.6 rejects cuda; 0.10/0.12 fail to load; 0.17 needs absent CUDA-13 runtime; PyNvVideoCodec/cv2.cudacodec/decord absent). The partial GPU path (hwdownload then upload pix+EG2) is measured: pixel-26 822->18 ms (44x), EG2 581->393 ms (1.48x), total 3402->2356 ms (1.44x), EG2 cosine 0.9999, verdict agreement 29/30; PCIe bytes saved -13 MB/window (the partial path adds bytes; only true zero-copy would save ~80 MB).

Results file: eval/results/exp_zerocopy/Z1.02.json

Decision: Adopt the GPU pixel-26 + GPU EG2 compute path; do NOT claim zero-copy until an in-process NVDEC binding is available.

Cost: 30 windows, A40, single process.
