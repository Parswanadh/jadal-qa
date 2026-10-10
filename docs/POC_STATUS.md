# PoC status — Egocentric-10K factory video through the per-pixel QA funnel

> **Finding (also shown on every dashboard screen):** on 60 human-labelled
> Egocentric-10K windows the shipped cascade is **at chance** (score AUROC 0.529;
> 21 of 51 good windows falsely rejected). The ~40% reject share on this footage
> is mostly false rejects from domain shift. Verdict rates here are descriptive,
> not quality. Accuracy on the 150 IndEgo windows (leave-one-clip-out AUROC
> ~0.7–0.8, not yet high) is a different data set. Thresholds are not certified.
> Not measured on a Raspberry Pi. Receipts are content-hashed, not signed.

## What runs (integration/poc)

- **Batch run** (`scripts/poc_run.py`): real Egocentric-10K 1080p/30fps HEVC
  clips -> 10-s windows -> S0/S1 cascade -> `poc_data/windows.jsonl` + 3 thumbs
  per window + content-hashed receipt per window.
- **Dashboard** (`scripts/poc_server.py`, stdlib only, offline, no CDN):
  Overview, Gallery (filters, pitch selection), Detail drawer with Re-verify
  (`POST /api/reverify/<id>` -> `poc_run.py --verify`, Host/Origin checked),
  Live replay panel (scripted blackout, labelled as such), About claims table.
- **Live replay** (`scripts/poc_stream_replay.py` + `jadal/poc_stream/`):
  2-second chunks of a real clip with a **scripted** blackout injected at t=12 s
  (not from the footage) -> S0 early reject -> `poc_data/stream_events.jsonl`.
- **Evidence and summaries** (committed, regenerable):
  - `eval/poc/summarize.py` -> `eval/results/poc/run_summary.json` (counts, hours,
    factories, latency percentiles, computed from `windows.jsonl` with dedupe).
  - `eval/poc/ood_check.py` -> `eval/results/poc/ood_check_60.json` (the 60-window
    human-labelled check below).

## How to start the demo (one command)

```bash
./scripts/poc_up.sh --replay
# dashboard: http://localhost:8840/  (overview, gallery, live, about)
```

- Serves the **real** `poc_data` (never `--demo-fixture`). If port 8840 is already
  served, `poc_up.sh` reports the owning PID and reuses it; it kills nothing it
  did not start.
- Paths are set by env vars with defaults: `JADAL_PY` (python3 on PATH),
  `JADAL_POC_DATA` (`../poc_data`), `JADAL_CLIP_ROOT` (`../../awing/data/ego10k_scale/clips`).

## Measured numbers (from files, not claims)

Batch run, from `eval/results/poc/run_summary.json` (`partial: false`):

- 1587 unique windows (0 duplicate, 0 malformed lines) · 4.404 h video · 58 clips · 47 factories
- accept 709 (44.7%) · reject 702 (44.2%) · review/S2 176 (11.1%)
- stages S0 0 / S1 1411 / S2 176
- per-window sequential latency p50 548.1 ms, p90 699.2 ms (RTF about 18x; the
  dashboard tile shows 17.91 video-hours per compute-hour, concurrency excluded)
- Re-verify: 5 sampled windows reproduced their stored receipt in the run log
  (not committed as a file; use `poc_run.py --verify <id>` to repeat one).

Replay (measured 2026-10-10): 16 chunks, scripted blackout at t=12 s -> S0 `black`
rule fires (`p95 0.0 < 0.03`), time-to-reject 0.0 s of video, mean chunk
latency 533.5 ms, 9.3 s wall, ~171 MB RAM, CPU decode. The blackout is injected
by the script; it is not an event in the footage.

Accuracy on labelled IndEgo data (`eval/results/gpu/cascade.json`, the shipped
S1 logistic head): leave-one-clip-out AUROC **0.80** (95% CI 0.71-0.87), n = 150,
threshold 0.6. Not certified; n is small. A different model, the 1D-conv
(`eval/results/perpixel/compare.json`, AUROC 0.782, CI 0.63-0.86), is not shipped.

## Human-labelled out-of-domain check

`eval/results/poc/ood_check_60.json` (from `eval/poc/ood_check.py`): 60 random
real Egocentric-10K windows, labelled by a human: 51 good, 9 bad. The labels are
joined to the PoC verdicts by source clip and nearest window start. Exploratory:
n_bad = 9 is below the pre-registered minimum of 10, and the window-level bootstrap
CI is optimistic.

- PoC cascade as shipped (IndEgo-calibrated): score AUROC **0.529**
  (window-bootstrap CI 0.32-0.73); balanced accuracy 0.51.
- Rejected **21 of 51 good windows (41%)**; 24 accepted, 6 review. Bad windows:
  5 rejected, 4 accepted.
- Two other scorers, **reported, not recomputed here** (they live on other
  branches): frozen two-view ensemble `ensemble_v1` AUROC 0.415
  (`handover/ood_eval_ensemble_v1_exploratory.txt`); frozen per-pixel scorer
  AUROC 0.515 (`jadal-ops/explainer/FACTS.md`). The FACTS lines are copied into
  the JSON with their source line numbers.
- The ~40% reject share shown on Egocentric-10K is therefore mostly false
  rejects, caused by the 480p/10 fps (IndEgo) -> 1080p/30 fps HEVC
  (Egocentric-10K) domain shift.

Verdict rates on this footage are descriptive, not quality.

## Licensed frames in this repo (read before any public snapshot)

- `docs/screens/poc_*.png` and `docs/screens/poc_demo_backup.mp4` show real
  Egocentric-10K frames. The dataset is licensed. This repo is private; do not
  make it public, and do not copy these files into a public snapshot.
- `docs/PUBLIC_SNAPSHOT_EXCLUDE.txt` lists every path pattern to exclude from a
  public snapshot. `docs/screens/README.md` explains the screenshots.
- `tests/fixtures/poc/thumbs/` is git-ignored. Its JPEGs are generated by
  `tests/make_poc_fixture.py` and are synthetic colour boxes; they are still
  listed in the exclude file.
- The per-pixel `.npz` files under `eval/results/perpixel/` are features derived
  from the licensed footage (no pixels), and are listed too.

## Not claimed

- **No Egocentric-10K quality claim**: the batch run is unlabelled. The 44.2%
  reject share is descriptive. The 60-window check measured AUROC 0.529 (chance),
  21 of 51 good windows falsely rejected (41%), balanced accuracy 0.51
  (`eval/results/poc/ood_check_60.json`); frozen two-view 0.415 and per-pixel
  0.515 are reported from other branches.
- Not measured on a Pi (RTX 4070 Laptop, not datacenter, not a Pi).
- Receipts content-hashed (sha256), not signed; no certificate chain.
- RGB only — no hyperspectral. Certified bounds not achievable at n=150.
- Reject reasons are generic codes (`S1_UNUSABLE`/`S1_USABLE`/`UNCERTAIN`);
  per-reject evidence is the deciding score vs threshold (the actual rule),
  not a certified diagnosis. Per-feature deviation explanations were skipped
  to preserve receipt-hash integrity.
- `stream_events.jsonl` is a shared live file (demo worker also appends);
  Live panel content varies with the latest replay.
