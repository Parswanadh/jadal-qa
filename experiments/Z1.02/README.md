# Z1.02 — Zero-copy GPU path attempt (NVDEC->VRAM features/embed) vs hwdownload

**Why:** Measure the mosaic/round-trip hypothesis: can decoded frames stay GPU-resident from NVDEC through pixel-26 features and the EG2 embedder, avoiding hwdownload+re-upload

**Hypothesis:** A GPU-resident path removes the VRAM<->RAM round trips for the pixel+EG2 branches and speeds up decode/pixel_features/eg2_embed; if the repo cannot consume CUDA frames the step is reported blocked

**Data / label source:** same 30-window sample; label_source none

**Data regime:** factory Egocentric-10K, 1080p, 30fps, HEVC

**Method:** baseline repo hwdownload path vs GPU path: in-process NVDEC (torchcodec) -> jadal.gpu.features.gpu_window_features_batch on CUDA tensors -> EmbeddingGemma-2 GPU preprocess+forward directly on device tensors; per-stage perf_counter; feature deltas + EG2 cosine + verdict agreement + PCIe bytes

**Owner:** l4-zerocopy  **Started:** 2026-10-10 18:47

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
