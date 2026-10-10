# L2.01 — Full-corpus PoC pipeline run: outcomes + atomic per-stage latency (L2)

**Why:** The PoC needs one honest end-to-end measurement of the repo pipeline over the whole 4.404 h corpus: does it reproduce E8.01 verdicts, what is the true per-stage cost, and what is the bottleneck

**Hypothesis:** The repo E8.01 pipeline (exp_rescore + ensemble + router) reproduces E8.01 verdicts exactly over all 1,546 windows; EG2 embedding is the per-window bottleneck; NVDEC is much faster than CPU decode

**Data / label source:** queues/manifest_scale.json: 1546 windows / 58 clips / 4.404 h, 1080p30 HEVC; label_source none

**Data regime:** factory Egocentric-10K, 1080p, 30 fps, HEVC

**Method:** per window perf_counter around io_probe | decode(3-branch NVDEC) | pixel26 | descriptors | eg2_embed(2fps) | mosaic | decision | write; one JSONL row/window; verdicts/score/features_digest compared to E8.01; decode comparison CPU vs -c:v hevc_cuvid on first window/clip; GPU util sampled 0.5s

**Owner:** l2-latency  **Started:** 2026-10-10 18:14

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 18:43): maybe.** PARTIAL baseline: 190 of 1546 windows (contiguous prefix, first 2 clips) of the repo E8.01 pipeline with atomic perf_counter per stage; run STOPPED and superseded by L3/L4. Verdicts/score/features_digest reproduce E8.01 EXACTLY on all 190 compared windows (0 mismatches). Verdicts (descriptive): accept 27 (14.21%), reject 2 (1.05%), review 161 (84.74%) - same as E8.01 on these windows. Throughput 0.80 windows/s, 8.0 footage-h/wall-h, 4 workers OMP=1. Per-stage mean: decode 1780 ms (36.7%), EG2 1235 ms (25.4%), pixel26 894 ms (18.4%), descriptors 735 ms (15.1%), mosaic 208 ms (4.3%); bottleneck = NVDEC decode. CPU vs NVDEC cuvid 4.1x (n=2). write stage not captured (stopped before the flush).

Results file: eval/results/exp_latency_full/L2.01.json

Decision: Partial-but-honest; full-corpus baseline superseded by L3 (optimized full-potential) and L4 (zero-copy)

Cost: 4 workers x 237 s for 190 windows (0.80 win/s); full corpus not completed
