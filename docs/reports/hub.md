# Hub report — video hub over EmbeddingGemma 2 (issue #27)

Branch `feat/hub`. All numbers below were measured by this worker; every number
states N and what data it was measured on. Files: `eval/results/hub_*.json`.

## What was built (owned files only)

- `jadal/hub_embed.py` — `HubEmbedder`: `embed_segment(path, start_s, end_s)`
  -> float32 unit vector (truncation dim configurable, default 256);
  `embed_text(query)` with the `SearchQuery` prompt; `embed_image(path)`.
  Video route passes **fixed OpenCV-sampled PIL frames** (`N` at fractions
  `(i+0.5)/N` of the segment, `N` configurable, default 8) as
  `{"text": "<|video|>", "video": [frames]}` — verified working on the real
  model (sentence-transformers 6.1, bfloat16, NEVER fp16, batch size 1,
  `torch.use_deterministic_algorithms(True, warn_only=True)`, fixed seeds).
  Cache key = sha256(frame bytes + interval + N + dim + backbone weights hash).
  No model/GPU load at import (lazy).
- `jadal/hub.py` — `Index`: float16 `(N, D)` matrix + metadata, `.npz` +
  `.jsonl` under `data_runtime/hub/` (git-ignored); canonical build order
  (sorted by id); `index_hash` = sha256(matrix bytes + canonical metadata);
  `add/save/load`, `search_text/search_by_vector/search_similar`
  (filters: site, worker, verdict, reason, clip, step, t0/t1 overlap;
  ties by segment_id; `results_hash` = sha256 of canonical ids+scores@4dp);
  `dedup_clusters(threshold)` = connected components over exact blocked cosine,
  medoid representative (ties by id), `dup_fraction` reported.
  `search_similar` **excludes the query segment itself** from hits.
- `jadal/hub_api.py` — `create_hub_router(index, embed_fn, thumb_cache_dir)` ->
  FastAPI `APIRouter(prefix="/hub")`. No GPU at import; index + embedder are
  injected so tests use fakes.
- `eval/hub_eval.py` — subcommands `throughput|sizelat|recall|determinism|cascade|all`.
- `tests/test_hub.py` — 10 hardware-free tests, all passing
  (`PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_hub`).

## API JSON shapes

- `GET /hub/search?q=&k=5&site=&worker=&verdict=&reason=&clip=&step=&t0=&t1=`
  -> `{"hits": [{"segment_id","score","clip","start_s","end_s","worker_id",`
  `"site_id","verdict","reasons","step"}], "results_hash": hex}`
- `GET /hub/similar?segment_id=&k=5` -> same shape (query excluded).
- `GET /hub/clusters?threshold=0.95`
  -> `{"clusters": [{"representative","members","size"}], "n_segments",`
  `"n_clusters", "dup_fraction"}`.
- `GET /hub/thumb/{segment_id}.jpg` -> JPEG mid-frame (cached under
  `data_runtime/hub/thumbs/`; id allow-listed + must exist in index; missing
  clip -> 404, never leaks paths).
- `GET /hub/stats` -> `{"n_segments","dim","index_hash"}`.

## Integration (for the mounting PR — hub touches nothing else)

Hub never imports `jadal.pipeline/service/decision/config`. To mount, integration adds
(~3 lines in `jadal/service.py`, owned by integration, NOT this branch):

```python
from jadal.hub_api import create_hub_router
# after building Index + HubEmbedder once at startup:
app.include_router(create_hub_router(index, embed_fn=hub_embedder.embed_text))
```

Suggested wiring: segmenter emits segments -> pipeline verdicts -> hub `add()`
(segment_id, embedder.embed_segment(...), {clip, start_s, end_s, worker_id,
site_id, verdict, reasons, step, path}) -> `save("data_runtime/hub")`.
Thumbs read the stored `path`, so keep clip files (or update `path` on move).

## Measured numbers (all by this worker, 2026-10-10)

Python: `PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python` (torch
2.14.1+cu130, sentence-transformers 6.1.0). GPU jobs wrapped in
`flock -w 1800 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock`. Clips read
in place from `/home/parshu/projects/awing/data/indego` (never copied).

**(a) Embedding throughput** (`hub_throughput.json`, N=2 reps per cell, real
IndEgo clip `User_1_412_2010_3_480.mp4`, 480x480 10fps, device cuda, Median GPU
s/segment, peak VRAM, segments/hour):

| seg_len | N=4 frames | N=8 frames |
|---|---|---|
| 5 s | 0.447 s, 1.66 GB, 8054/h | 0.591 s, 1.81 GB, 6089/h |
| 10 s | 0.295 s, 1.66 GB, 12184/h | 0.605 s, 1.81 GB, 5946/h |
| 20 s | 0.299 s, 1.66 GB, 12043/h | 0.666 s, 1.81 GB, 5408/h |

GPU time depends on **frame count, not segment length** (fixed sampling) —
the 5 s vs 10 s spread at N=4 is first-call warmup noise (reps=2, warmup not
excluded). Peak VRAM ~1.7–1.8 GB: fits a laptop GPU.

**(b) Index size + exact-search latency** (`hub_sizelat.json`):
segments/1000 h at 10 s segments = 360,000 exactly.
bytes = segments * dim * 2 → **dim256: 184,320,000 B (~176 MiB)**,
**dim128: 92,160,000 B (~88 MiB)** per 1000 h.
Exact-search latency (CPU float32 matmul, dim 256, 20 repeats, **vectors
SYNTHETIC random unit**: median/p95 ms): N=1e4: 3.01/5.78; N=1e5: 3.04/6.01;
N=1e6: 17.6/21.52. Conclusion: **no ANN library needed** — exact search over
1M vectors is ~20 ms on CPU; an ANN is NOT proposed.

**(c) Retrieval quality** (`hub_recall.json`): index over **91 fixed 10 s
segments** of the two real ECU-assembly IndEgo clips; ground truth = **20 VIA
step intervals** (15 + 5); query = step name as text ("attach hard disk",
...); hit = segment overlaps the step interval. Truncation = leading dims of
the same measured 768-d vectors, renormalised:

| dim | recall@1 | recall@5 | recall@10 | N |
|---|---|---|---|---|
| 768 | 0.25 | 0.65 | 0.75 | 20 intervals / 91 segs |
| 256 | 0.35 | 0.55 | 0.70 | same |
| 128 | 0.20 | 0.50 | 0.65 | same |

N is small — do not generalise. 256 beating 768 at @1 is small-N noise, not a
claim. Google's "~95% at 256" is NOT confirmed nor refuted by this N=20 probe.
Egocentric-10K frames skipped (not a video test) per issue.

**(d) Determinism**: same segment embedded twice → max abs diff **0.0**
(real model, GPU); same query twice → diff **0.0**; same query twice →
identical `results_hash` (N=91 measured 768-d vectors: `78ba0a…`);
rebuild-twice → identical `index_hash` (`aad210…`). CPU/GPU same-machine
rebuild check: NOT DONE (no second device Fibonacci; hashes are pure functions
of bytes so cross-device risk is confined to the embedding step, which is
bit-identical on repeat runs).

**(e) Triage cascade** (`hub_cascade.json`, parametric): L0 proxy (Laplacian
variance ≥ 30 + brightness, 4 frames, CPU 0.1594 s/seg, N=91 real segments)
**survivor fraction = 1.0** — the proxy does not discriminate on these
well-lit ECU clips (honest null result; real funnel fractions need
`jadal.signals`, not on main yet). Embed cost measured 0.605 s/seg (10 s segs,
N=8). Hours of video/day on one machine: **L0-only ≈ 1505 h/day**,
**L0 + hub-on-survivors ≈ 314 h/day** (at survivor fraction 1.0; scales down
linearly with fewer survivors). No Awign numbers used or invented.

**Dedup on real data** (91 measured 768-d segment vectors, extra probe, same
code path): threshold 0.99 → 91 clusters (dup 0.0); 0.95 → 11 clusters (dup
0.879); 0.90 → 1 cluster (dup 0.989). Threshold must be calibrated against QA
near-duplicate labels — NOT chosen here.

## NOT DONE / honesty list

- Rebuild-across-devices (CPU vs GPU) `index_hash` comparison: NOT DONE.
- ANN: not needed (measured ~20 ms @ 1M exact, CPU); no library proposed.
- `similar-by-upload` (query clip bytes) in the API: NOT DONE (out of scope).
- Cache: file cache implemented but throughput measured cache-bypassed (stated).
- Recall N=20 is a smoke probe, not a quality claim; threshold calibration needs
  QA labels.
- GPU lock discipline kept: 4 short locked jobs (smoke, throughput, recall,
  bit-identical check); no CUDA setup touched; nothing pip-installed.
