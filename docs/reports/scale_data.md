# Scale-data run (issue #53): real Egocentric-10K subset + local capacity

Worker: SCALE-DATA, branch `feat/scale-data`. Data (outside git):
`/home/parshu/projects/awing/data/ego10k_scale/` (selection 7.93 GB; ~15 GB with
extracted mp4s + label windows; disk 97 GB free, was ~115 GB).

## Corpus facts (`eval/results/scale/repo_listing.json`, public repo metadata)
- HF `builddotai/Egocentric-10K`: **19,495 webdataset tars, ~18.0 TB**,
  **85 factories, 2,144 workers** (2..142 workers/factory). Publisher declares:
  10,000 h, 192,900 clips, median 180 s, H.265/MP4 1080p30, fisheye intrinsics.
- ffprobe on real downloads CONFIRMS: HEVC, 1920x1080, 30 fps (3/3 files);
  sidecar JSON durations 180 s / 1200 s match `duration_sec`.
- Auth: original token 403'd on gated content; human replaced token ~16:28, verified OK.

## Acquired subset (COMPLETE: 47/47 tars OK, 0 failures)
- Rule SELECT-8G (seed 8): seeded factory shuffle, smallest-first round-robin, 8 GB cap.
- **47 tars, 7.93 GB, 47 factories, 47 workers** (`selection_small.json`).
- Extracted: **58 clips, 4.404 h total** (`eval/scale/manifest_scale.json`:
  1,546 non-overlapping 10-s windows, ids `s_<sha1[:12]>`, factory/worker/res/fps).
- Downloader `eval/scale/dl_small.py`: resume + 6-attempt backoff + size/tar/ffprobe
  verification; log `download.log`, per-shard record `download_verify_small.json`.
  No `.incomplete` files in the final set (partials stay in HF cache).
- Superseded plan SELECT-53 (31 tars/29.78 GB, `selection.json`) kept for later.

## Capacity @1080p, REAL files (NOT a Pi: laptop x86-64, RTX 4070 8 GB)
`eval/scale/capacity_real.py` -> `capacity_real1080p.json` (120-s samples, 3 files,
3 factories; RTF = video_s/wall_s). Single-core CPU (taskset -c 0, -threads 1):

| file (factory) | full 1080p CPU | 854x480 CPU | 160x90 CPU | NVDEC hevc_cuvid |
|---|---|---|---|---|
| factory001_worker006_00008 (f001) | 2.31x | 1.77x | 1.65x | 50.38x |
| factory026_worker005_00111 (f026) | 2.10x | 2.02x | 2.01x | 51.29x |
| factory046_worker008_00618 (f046) | 1.15x | 0.80x | 0.94x | 57.94x |

- Real H.265 decodes at **~1.2-2.3x single-core** — ~2x SLOWER than my libx265-veryfast
  proxies (~5x in `capacity_1080p.json`); proxies underestimated encode complexity.
  Scaled decode is SLOWER than full decode on 1 core (the scale filter is serial cost).
- NVDEC (hevc_cuvid, shared lock held once, timed inside, lock-wait excluded):
  **50-58x real-time on real files** (proxy had shown ~43x). NVDEC is ~25-50x faster
  than single-core CPU decode — the decode tier belongs on GPU.
- Disk read: **115 MB/s** buffered; transcode 1080p->480p H.264 single core: **0.98x**
  (~1x, i.e. ~1 core-hour per video-hour for a 480p copy).
- Throughput implication (honest, decode only): at ~2x single-core, 10,000 h in 2 h
  needs ~2,500 cores for full-res decode; the 160x90 cheap tier does NOT help on CPU
  (still ~2x: decode dominates, scale adds cost). NVDEC (~50x) would need ~100 GPU-
  equivalents. Full funnel math belongs to the harness/PoC, not this worker.

## Labelling queue (`eval/results/scale/label_queue_random.json`)
- 60 windows, seed 60, stratified round-robin over all **47 factories**
  (50 `random` + 10 `uncertainty_placeholder` for later model-uncertainty refill).
- Each window transcoded to 10-s 480p H.264 (`label_windows/qNNN_*.mp4`, ffprobe-verified
  854x480/H.264/10.0 s); queue items use the segments.json schema
  (segment_id, clip, path, start_s=0, end_s=10 + source_* traceability).
- PRE-REGISTERED ONLY — worker did no labelling; human labels for the out-of-domain test.
- Untouched: `funnel/eval/gold/labels.json`, ports 8800/8812.

## Honesty ledger
- On disk and verified: 47 tars / 58 clips / 4.404 h real Ego10K + 60 label windows.
- NOT done at push: per-tar durations beyond sidecars; SELECT-53 30 GB expansion
  (needs human go-ahead on disk).
- Proxy capacity file `capacity_1080p.json` kept but its NVDEC entries are INVALID
  (included lock-wait); CPU entries are proxy-only and superseded by real numbers above.
- Tests: `tests/test_scale_data.py` (7 tests, <1 s, no GPU/network). Full suite run
  before PR per house rules.
