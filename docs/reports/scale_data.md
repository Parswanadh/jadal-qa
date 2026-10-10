# Scale-data run (issue #53): real Egocentric-10K subset + local capacity — INTERIM

Worker: SCALE-DATA, branch `feat/scale-data`. Data (outside git):
`/home/parshu/projects/awing/data/ego10k_scale/` (8 GB cap; disk ~115 GB free).

## Corpus facts (from public repo metadata, `eval/results/scale/repo_listing.json`)
- HF `builddotai/Egocentric-10K`: **19,495 webdataset tars, ~18.0 TB**, **85 factories,
  2,144 workers** (1..142 workers/factory). Publisher declares: 10,000 h, 192,900 clips,
  median 180 s, **H.265/MP4 1080p30**, fisheye intrinsics per worker.
- Per-tar/per-clip durations NOT verifiable from metadata (inside-tar JSON sidecars);
  README median used for estimates only. Resolution/codec publisher-declared until
  ffprobe runs on real downloads.
- Auth: original token got 403 on gated content; human replaced token ~16:28, verified OK.

## Subset (downloading in background, `eval/scale/dl_small.py` -> `download.log`)
- Rule SELECT-8G (seed 8): seeded factory shuffle, smallest-first round-robin, 8 GB cap.
- **47 tars, 7.93 GB, 47 factories, 47 workers** (`selection_small.json`).
- Resumable + 6-attempt backoff + size/tar/ffprobe verification; only verified files
  land in `clips/`. Progress: `download_status.json`.
- Superseded plan SELECT-53 (31 tars/29.78 GB, `selection.json`) kept for when budget allows.

## Capacity @1080p (PROXY files so far — real-file numbers follow on first shards)
Proxies: 60 s 1080p30 H.265 @3.5 Mb/s (publisher bitrate); 1 synthetic + 2 IndEgo upscales.
`eval/results/scale/capacity_1080p.json`. Single-core CPU full-decode RTF ~5-5.8x;
scaled-decode (854x480 / 160x90) RTF ~3.9-5x (scale filter costs on 1 core).
NVDEC in that file is INVALID (included GPU-lock wait); true NVDEC ~43x measured
separately (lock held once, timed inside). NOT a Pi: laptop x86-64 + RTX 4070.
Real-file rerun: `eval/scale/capacity_real.py` -> `capacity_real1080p.json` (pending).

## Labelling queue (pending >= 8 factories on disk)
`eval/scale/make_label_queue.py`: 60 windows stratified by factory (seed 60;
50 random + 10 uncertainty placeholders), 10-s 480p H.264 transcodes to
`.../ego10k_scale/label_windows/`, queue in segments.json schema. Human labels;
worker does not label, does not touch `funnel/eval/gold/labels.json` or port 8812.

## On disk right now
Download in progress (see `download.log`); manifest/label-queue/capacity-real run
as soon as shards land. Honest state: **0 GB real Ego10K verified at time of push.**
