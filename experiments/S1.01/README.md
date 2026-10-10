# S1.01 — Sharded isolated workers: parallel scaling ceiling + RAM/VRAM config

**Why:** The 4.404 h corpus is processed single-pod at ~19.8 footage-h/wall-hour; to reach 50-1000 footage-h/wall-hour we must shard the pipeline into isolated parallel workers and know the per-pod ceiling to size a multi-pod fleet

**Hypothesis:** The pod is CPU-quota-bound (7.65 cores) and/or NVDEC-engine-bound (one A40); isolated shard workers scale linearly up to K=7 then plateau; VRAM headroom (46 GB vs ~15 GB used) is not the limit

**Data / label source:** manifest_scale.json: 58 clips / 1546 windows / 4.404 h (unlabelled); scaling subset = 6 smallest clips / 102 windows

**Data regime:** 1080p/30fps HEVC Egocentric-10K, adopted optimizations (pixel 3.33fps, EG2 1fps, 4s early exit, mosaic off)

**Method:** deterministic clip->shard LPT plan; K isolated single-core shard processes; nvidia-smi CPU/NVDEC/SM/VRAM sampling; shard-invariance digest

**Owner:** S1  **Started:** 2026-10-10 19:17

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 19:32): adopt.** Sharded pipeline into K isolated workers (own process/JSONL/state) with a deterministic partition-independent combined digest; shard-invariance proven (K=1,2,4,7 identical digest). On the pod ceiling = ~1.1 win/s / 11 h/h (K>=4) on a 6-clip subset and ~19.8 h/h at corpus-scale shards; GPU-decode-contention-bound, not CPU-quota (4.06/7.65 cores) or VRAM (16.9/46 GB). VRAM: 1.7 GB/worker, embed batch has no effect, VRAM not the limit. Multi-pod: ~20 h/h/pod => 3 pods @50, 6 @100, 51 @1000 h/h.

Results file: eval/results/exp_shard/S1.01.json

Decision: Adopt K=4-7 isolated shard workers/pod (8-9 clips/shard); scale horizontally by adding pods (containers on one pod share quota+GPU => reproducibility not throughput)

Cost: ~15 min GPU on a shared A40
