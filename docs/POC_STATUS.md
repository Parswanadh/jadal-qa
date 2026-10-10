# PoC status — Egocentric-10K factory video through the per-pixel QA funnel

> **Honesty banner (also shown on every dashboard screen):** descriptive rates
> on **unlabelled** video — **not accuracy**. Accuracy comes only from the 150
> human-labelled IndEgo windows (leave-one-clip-out AUROC ~0.7–0.8, not yet
> high). Thresholds are not certified. Not measured on a Raspberry Pi.
> Receipts are content-hashed, not signed.

## What runs (integration/poc, all merged)

- **Batch run** (`scripts/poc_run.py`): real Egocentric-10K 1080p/30fps HEVC
  clips → 10-s windows → S0/S1 cascade → `poc_data/windows.jsonl` + 3 thumbs
  per window + `summary.json` + content-hashed receipt per window.
- **Dashboard** (`scripts/poc_server.py`, stdlib only, offline, no CDN):
  Overview tiles + reason bars + per-factory stacks + time-to-verdict,
  Gallery with filters + pitch examples, Detail drawer with evidence table
  and **Re-verify** button (`POST /api/reverify/<id>` → `poc_run.py --verify`),
  Live replay panel, About claims table.
- **Live replay** (`scripts/poc_stream_replay.py` + `jadal/poc_stream/`):
  2-second chunks, scripted blackout/freeze/blur, S0 early reject, events to
  `poc_data/stream_events.jsonl`.

## How to start the demo (one command)

```bash
./scripts/poc_up.sh --replay
# dashboard: http://localhost:8840/  (overview, gallery, live, about)
```

- Serves the **real** `poc_data` (never `--demo-fixture`). Port 8840 is
  currently served by pid 3824977 (poc-dashboard worktree, identical server
  code, real data) — `poc_up.sh` detects and reuses it, kills nothing.
- `--replay` replays a real clip in 2-s chunks with a blackout at t=12 s;
  the Live panel then shows the early reject. Cleans up only PIDs it started.

## Measured numbers (from files, not claims)

Batch run (`poc_data/summary.json` + `windows.clean.jsonl` snapshot, partial run):

- 1587 windows · 4.404 h video · accept 709 (44.7%) · reject 702 (44.2%) ·
  review/S2 176 (11.1%) · stages S1 1411 / S2 176 / S0 0
- 0 duplicate window_ids, 0 malformed lines, 0 missing thumbs (4761 checked)
- per-window sequential latency p50 548 ms / 10-s window (RTF ≈ 18×);
  dashboard throughput tile 17.91 h video per wall-h; decode `cpu-sequential`
- re-verify: **5/5** random windows reproduce stored `receipt_sha256`
  (`f029…`, `f009…`, `f033…`, `f058…`, `f003…`, CPU sequential recompute)
- full suite: **662 tests green** (4 skipped)

Replay (measured 2026-10-10): 16 chunks, blackout at t=12 s → S0 `black`
rule fires (`p95 0.0 < 0.03`), time-to-reject 0.0 s of video, mean chunk
latency 533.5 ms, 9.3 s wall, ~171 MB RAM, CPU decode.

Accuracy (only labelled data, `eval/results/perpixel/compare.json`):
nested leave-one-clip-out on 7 IndEgo clips / 150 windows — conv1d AUROC
**0.782** (95% CI 0.63–0.86), quality-stage sens 0.714 / bal_acc 0.792.

## Not claimed

- **No Egocentric-10K accuracy**: the video is unlabelled; 44.2% reject is a
  descriptive share, probably inflated by the 480p/10fps → 1080p/30fps domain
  shift (thresholds were set on IndEgo working resolution).
- Not measured on a Pi (RTX 4070 Laptop, not datacenter, not a Pi).
- Receipts content-hashed (sha256), not signed; no certificate chain.
- RGB only — no hyperspectral. Certified bounds not achievable at n=150.
- Reject reasons are generic codes (`S1_UNUSABLE`/`S1_USABLE`/`UNCERTAIN`);
  per-reject evidence is the deciding score vs threshold (the actual rule),
  not a certified diagnosis. Per-feature deviation explanations were skipped
  to preserve receipt-hash integrity.
- `stream_events.jsonl` is a shared live file (demo worker also appends);
  Live panel content varies with the latest replay.
