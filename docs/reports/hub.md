# Hub report — video hub over EmbeddingGemma 2 (issue #27)

Branch `feat/hub`. All numbers below were measured by this worker; every number
states N and what data it was measured on. Files: `eval/results/hub_*.json`
(round 1; superseded claims from the original report are kept only in git
history — every claim below matches its file).

## What was built (owned files only)

- `jadal/hub_embed.py` — `HubEmbedder`: `embed_segment(path, start_s, end_s)`
  -> float32 unit vector (truncation dim configurable, default 256);
  `embed_text(query)` with the `SearchQuery` prompt; `embed_image(path)`.
  Video route passes **fixed OpenCV-sampled PIL frames** (`N` at fractions
  `(i+0.5)/N` of the segment, `N` configurable, default 8) as
  `{"text": "<|video|>", "video": [frames]}` — verified working on the real
  model (sentence-transformers 6.1, bfloat16 ACTUAL runtime dtype recorded in
  every GPU output, NEVER fp16, batch size 1,
  `torch.use_deterministic_algorithms(True, warn_only=True)`, fixed seeds).
  NO fallback: a video-route error raises (nothing cached); the route string
  is part of the cache key and recorded as `route` in eval outputs.
  A failed frame read raises IOError (never substitutes frame 0); end_s is
  checked against clip duration. `device` is a top-level SentenceTransformer
  argument (tested with a fake model class for 'cpu'/'cuda').
  Cache key = sha256(frame bytes + interval + N + dim + route + PARTIAL
  backbone hash — labelled partial: config + head/tail 8 MB + size).
  No model/GPU load at import (lazy).
- `jadal/hub.py` — `Index`: float16 `(N, D)` matrix + metadata, `.npz` (plain
  unicode/float16 arrays, `allow_pickle=False`) + `.jsonl` under
  `data_runtime/hub/` (git-ignored); `load()` recomputes and verifies
  `index_hash` (mismatch => ValueError); canonical build order (sorted by id);
  ordered ids + CONTIGUOUS float16/float32 matrices cached and rebuilt only on
  `add()` — search/hash never re-sort per query; exact top-k via threshold +
  id tie-break (no full sort; boundary ties resolve by id); `results_hash`
  as before; `dedup_clusters` refuses N > `max_clusters_n` (default 20000,
  `ClusterLimitError`), upper-triangle blocks, blockwise medoid (O(m*cb)).
  Public `get_meta`/`get_vector` accessors (no private poking needed).
  `search_similar` **excludes the query segment itself** from hits.
  Optional `use_torch=True` backend (same deterministic post-processing).
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
  injected so tests use fakes. ONE id grammar
  `^[A-Za-z0-9][A-Za-z0-9_.#\-]*$` (covers eval ids like
  `User_1_412_2010_3_480#0003`; `#` must be `%23` in URLs); `/`,`..`,`%`
  rejected post-decode; id must exist in the index. 500s are generic (no
  exception text). `/hub/clusters` maps over-limit to HTTP 413. Thumb cache
  filenames carry a sha1 tag (no collisions); thumbs dir honours
  `JADAL_HUB_DIR`.
- `eval/hub_eval.py` — subcommands
  `throughput|sizelat [--backend numpy|torch]|recall|determinism [--gpu-repeat]|dedup|cascade|all`.
- `tests/test_hub.py` — 21 tests (1 GPU-gated), all passing, CPU and GPU:
  `JADAL_TEST_GPU=1 ... -m unittest discover -s tests -p 'test_hub*'` -> OK.

## API JSON shapes

- `GET /hub/search?q=&k=5&site=&worker=&verdict=&reason=&clip=&step=&t0=&t1=`
  -> `{"hits": [{"segment_id","score","clip","start_s","end_s","worker_id",`
  `"site_id","verdict","reasons","step"}], "results_hash": hex}`
- `GET /hub/similar?segment_id=&k=5` -> same shape (query excluded).
- `GET /hub/clusters?threshold=0.95`
  -> `{"clusters": [{"representative","members","size"}], "n_segments",`
  `"n_clusters", "dup_fraction"}`; 413 when N > max_clusters_n (default 20000).
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

## Measured numbers (all by this worker; round-1 values, 2026-10-10/11)

Python: `PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python` (torch
2.14.1+cu130, sentence-transformers 6.1.0). GPU jobs wrapped in
`flock -w 1800 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock`. Clips read
in place from `/home/parshu/projects/awing/data/indego` (never copied).

**(a) Embedding throughput** (`hub_throughput.json`, N=3 reps per cell +
warmup excluded, real IndEgo clip `User_1_412_2010_3_480.mp4`, 480x480 10fps,
device cuda, runtime_dtype bfloat16; decode = OpenCV sample_frames, model =
one GPU encode call):

| seg_len | N=4: decode / model / wall | N=8: decode / model / wall | peak VRAM |
|---|---|---|---|
| 5 s | 0.067 / 0.157 / 0.224 s | 0.119 / 0.401 / 0.520 s | 1.66 / 1.81 GB |
| 10 s | 0.131 / 0.157 / 0.288 s | 0.229 / 0.398 / 0.627 s | 1.66 / 1.81 GB |
| 20 s | 0.140 / 0.163 / 0.303 s | 0.285 / 0.397 / 0.682 s | 1.66 / 1.81 GB |

Model time depends on **frame count, not segment length** (fixed sampling);
decode grows with N (seeks) and slightly with span. VRAM peak measured
1.66 GB (N=4) / 1.81 GB (N=8) on this machine's GPU — the "fits a laptop"
line from round 0 is withdrawn (not measured on a laptop).

**(b) Index size + exact-search latency** (`hub_sizelat.json`, real
`Index.search_by_vector` path, k=5, numpy-CPU backend 20 repeats, torch-GPU
backend 10 repeats; **vectors SYNTHETIC random unit**, labelled in file):

segments/1000 h at 10 s segments = 360,000 exactly.
MATRIX bytes = segments * dim * 2 → **dim256: 184,320,000 B (~176 MiB)**,
**dim128: 92,160,000 B (~88 MiB)** per 1000 h (float16 matrix only;
excludes .npz compression and meta.jsonl).

| N | dim256 numpy (med/p95 ms) | dim256 torch-GPU | dim128 numpy | dim128 torch-GPU |
|---|---|---|---|---|
| 1e4 | 5.6 / 7.3 | 3.5 / 5.0 | 6.0 / 6.1 | 3.3 / 3.7 |
| 1e5 | 72.5 / 85.2 | 62.6 / 66.8 | 30.1 / 33.5 | 26.0 / 33.2 |
| 1e6 | 656.1 / 722.9 | 615.7 / 621.5 | 405.4 / 450.4 | 393.3 / 488.5 |

(numpy: 20 repeats; torch: 10 repeats; k=5; exact cells in
`hub_sizelat_{numpy,torch}.json`, merged in `hub_sizelat.json`. An earlier
numpy run on a loaded machine read 12.0/14.9 @1e4 and 80.8/88.7 @1e5 —
same order, reported here as machine noise.)
The torch-GPU matmul path does NOT help (PCIe transfer dominates) — numpy CPU
stays the recommended path. Corrected conclusion: exact search is interactive
(<100 ms p95) **up to N=1e5** at dim 256 (1e5 = ~280 h of video at 10 s
segments); at **N=1e6 p95 is ~0.4–0.8 s** (fine for QA triage, not realtime).
The round-0 sentence "exact search over 1M vectors is ~20 ms ... no ANN
needed" is withdrawn — it timed the wrong code path. No ANN is proposed for
the current scale (single-site QA, N ≲ 1e5); an ANN would need its own
recall-vs-latency measurement first.

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
Egocentric-10K frames skipped (not a video test) per issue. `hub_recall.json`
records the video `route` and `runtime_dtype` (bfloat16) used.

**(d) Determinism** (`hub_determinism.json` + `hub_batch_invariance.json`):
same segment embedded twice → max abs diff **0.0** (real model, GPU,
`User_1_412_2010_3_480.mp4 [10,20]s N=8`, bfloat16,
route `{"text": "<|video|>", ...}`); same query twice → diff **0.0**;
SPEC batch-invariance (segment alone vs inside a batch of 5, real model,
`[70,80]s N=4`): diff **0.0**, index hashes identical (`019092…` both);
same query twice → identical `results_hash` (N=91 measured 768-d vectors:
`78ba0a…`); rebuild-twice → identical `index_hash` (`aad210…`).
CPU/GPU same-machine rebuild comparison: NOT DONE (hashes are pure functions
of bytes; cross-device risk is confined to the embedding step, bit-identical
on repeat runs).

**(e) Triage cascade** (`hub_cascade.json`, parametric): L0 proxy (Laplacian
variance ≥ 30 + brightness, 4 frames, CPU 0.2204 s/seg, N=91 real segments)
**survivor fraction = 1.0** — the proxy does not discriminate on these
well-lit ECU clips (honest null result; real funnel fractions need
`jadal.signals`, not on main yet). Hub cost now measured WALL time
(decode+model) 0.627 s/seg (10 s segs, N=8). Hours of video/day on one
machine: **L0-only ≈ 1089 h/day**, **L0 + hub-on-survivors ≈ 283 h/day**
(at survivor fraction 1.0; scales down linearly with fewer survivors).
No Awign numbers used or invented.

**Dedup** (`hub_dedup.json`, `dedup` subcommand): on the 91 measured 768-d
IndEgo segment vectors — threshold 0.99 → 91 clusters (dup 0.0); 0.95 → 11
clusters (dup 0.8791); 0.90 → 1 cluster (dup 0.989); wall ~1 ms.
Scale timing (SYNTHETIC random unit vectors, threshold 0.95): N=1e4 → 0.44 s
(10000 clusters); N=1e5 → 26.6 s (100000 clusters, all singletons).
Threshold must be calibrated against QA near-duplicate labels — NOT chosen
here; the endpoint refuses N > 20000 with 413 (measured 26.6 s @1e5 is why).

## NOT DONE / honesty list

- Rebuild-across-devices (CPU vs GPU) `index_hash` comparison: NOT DONE.
- ANN: no claim beyond the table above (interactive <100 ms p95 to N=1e5
  @dim256; ~0.4–0.7 s @1e6). No library proposed.
- `similar-by-upload` (query clip bytes) in the API: NOT DONE (out of scope).
- Cache: file cache implemented but all evals measured cache-bypassed (stated).
- Recall N=20 is a smoke probe, not a quality claim; threshold calibration needs
  QA labels.
- GPU lock discipline kept: short flocked jobs only (smoke, throughput,
  recall, repeat-diff, batch-invariance, torch-latency); no CUDA setup touched;
  nothing pip-installed.

## Fix round 1 (review of PR #28 -> this push)

Rulings addressed, each with its evidence file or test:

- CRITICAL 1 (SPEC batch-invariance): `TestBatchInvariance` (fake, always) +
  `TestBatchInvarianceGPU` (JADAL_TEST_GPU=1, writes
  `eval/results/hub_batch_invariance.json`: diff 0.0, hashes `019092…`=`019092…`).
- CRITICAL 2 (real-path latency): `Index` keeps cached contiguous matrices;
  vectorised mask + exact threshold top-k (boundary ties by id — tested with
  100 identical vectors, k=5); `hub_sizelat_{numpy,torch}.json` time
  `search_by_vector` at 1e4/1e5/1e6 x dim 256/128; torch-GPU path measured and
  NOT recommended (transfer-dominated); §(b) claims rewritten, round-0 "20 ms
  @1M / no ANN" withdrawn.
- CRITICAL 3 (every number filed): `dedup` subcommand -> `hub_dedup.json`;
  `determinism --gpu-repeat` persists diffs/route/dtype in
  `hub_determinism.json`; VRAM peak in every `hub_throughput.json` row;
  laptop-fit sentence withdrawn.
- IMPORTANT: no silent fallback (raises, route in cache key + outputs);
  frame-read failure raises IOError (no frame-0 substitution); `device`
  top-level (fake-class test for cpu/cuda); `#`-ids in one grammar, `%23`
  tested for similar+thumb (real tiny mp4), traversal still 404;
  `allow_pickle=False` + load verifies `index_hash.txt` (tamper tests);
  clusters capped (unit 413 test + 26.6 s @1e5 evidence); decode vs model
  times split in §(a); video route recorded (`route`, `runtime_dtype`).
- MINOR: partial-hash labelling, end_s-vs-duration check, generic 500s,
  `JADAL_HUB_DIR` thumbs default, `get_meta`/`get_vector` accessors, sha1-tagged
  thumb filenames, dead code removed, warmup excluded (reps=3), "matrix bytes"
  labelling, repo-local test scratch.

Real outputs:
```
CPU:  Ran 21 tests in 7.915s / OK (skipped=1)
GPU (JADAL_TEST_GPU=1, flock): Ran 21 tests in 35.444s / OK
```
