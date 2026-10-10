# L2.01 — Full-corpus PoC pipeline run: outcomes + atomic per-stage latency (L2)

**Why:** The PoC needs one honest end-to-end measurement of the repo pipeline over the whole 4.404 h corpus: does it reproduce E8.01 verdicts, what is the true per-stage cost, and what is the bottleneck

**Hypothesis:** The repo E8.01 pipeline (exp_rescore + ensemble + router) reproduces E8.01 verdicts exactly over all 1,546 windows; EG2 embedding is the per-window bottleneck; NVDEC is much faster than CPU decode

**Data / label source:** queues/manifest_scale.json: 1546 windows / 58 clips / 4.404 h, 1080p30 HEVC; label_source none

**Data regime:** factory Egocentric-10K, 1080p, 30 fps, HEVC

**Method:** per window perf_counter around io_probe | decode(3-branch NVDEC) | pixel26 | descriptors | eg2_embed(2fps) | mosaic | decision | write; one JSONL row/window; verdicts/score/features_digest compared to E8.01; decode comparison CPU vs -c:v hevc_cuvid on first window/clip; GPU util sampled 0.5s

**Owner:** l2-latency  **Started:** 2026-10-10 18:14

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
