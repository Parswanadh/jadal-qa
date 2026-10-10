# PoC stream-replay report (feat/poc-demo, issue #64)

Date: 2026-10-10. Clip: `factory_069/worker_004/factory069_worker004_00077.mp4`
(1920x1080 HEVC 30 fps, 180 s; one of the real Egocentric-10K scale clips).
All numbers below are **descriptive behaviour on unlabelled video, NOT
accuracy** (no human labels exist for this video).

## How it runs

`scripts/poc_stream_replay.py` delivers the clip in 2 s chunks (`ffmpeg
-ss/-t`, 160 px wide @ 5 fps ⇒ ~10 frames/chunk, O(window) memory), evaluates
the S0 rules per chunk (same rule/threshold shape as
`eval/gpu/cascade.py::S0_RULES` on origin/feat/gpu-pipeline, loaded read-only
by path where available — here the local identical copy was used), feeds the
frames through the real `SpectralStream` from
`jadal/spectral/stream.py` (origin/feat/spectral-features, loaded read-only by
path; its `brightness/motion_energy/temp_frozen_frac/spec_hfratio` are stored
per event), and appends
`{"t","window_id","chunk","running_score","decision","reason"}` (+
`latency_ms`, `ram_mb`, `decode`, `engine`) to
`/home/parshu/projects/jadal-qa-wt/poc_data/stream_events.jsonl` (flushed per
chunk). `--pico` sends accept→green / continue→amber / reject→red via
`jadal/pico_link.py` under `flock tools/pico.lock`; headless it logs
`pico: not connected` (verified with `FakePicoSerial`: `LED R` sent).

## Measured results

| Run (10–16 chunks, degrade at t=6/12 s video) | time-to-reject (video s) | evidence |
|---|---|---|
| clean (no degrade) | n/a — `continue` ×4 then `accept`, score ~0.78 | window complete, score ≥ 0.5 |
| `--degrade blackout` | **0.0** (fires on the chunk the blackout starts) | `black: p95=0.0 < 0.03` (scope=chunk) |
| `--degrade freeze` | **0.0** | `frozen: temp_e < 5e-4` (scope=chunk) |
| `--degrade blur` (31×31) | **no reject in 10 chunks** — score dips 0.77→0.51 then ~0.55 | no S0 rule covers blur; handed on (honest non-result) |

Chunk-level S0 scope matters: evaluating the same rules on the running
window alone delayed blackout rejection to 4 s and freeze to 6 s of video
(window dilution); the current-chunk check keeps both at 0.0 s.

Cost per chunk (this machine, saturated, CPU decode — the ffmpeg build has no
CUDA hwaccel): wall 0.6–6.0 s/chunk (1080p HEVC decode dominates; recorded per
event, never averaged away), Python RSS 78–122 MB (far under the 1.5 GB
budget), 16-chunk demo wall 41–56 s for 32 s of video. Decode backend
`"cpu"` is recorded in every event.

## Artefacts

- `poc_data/stream_events.jsonl` — 16 demo events (blackout at t=12 s).
- `docs/screens/poc_demo_backup.mp4` — 80 s backup render (PIL cards +
  ffmpeg, no screen capture): sparkline, timeline, banner, evidence.
- `scripts/poc_demo.sh` — one command: dashboard `:8840` if
  `scripts/poc_server.py` exists (it does not on any branch yet → stub line),
  scripted blackout replay, prints URLs, kills only PIDs it started.
- `tests/test_poc_demo.py` — 15 tests, <30 s, no GPU/network/real video.

## Caveats

S0 thresholds are the shared a-priori constants (frozen 5e-4 justified by the
codec-noise floor in cascade.py), not certified on this video. The running
score is a fixed-weight descriptive composite, not a validated classifier.
Blur has no S0 rule by design — blurry-but-bright footage is an S1/S2 job.

## Fix round 1 (review of PR #66 — all Critical + Important + Minors fixed)

- C1 (module poisoning): `make_spectral_stream()` no longer touches
  `sys.path`/`sys.modules` at all. origin/main merged spectral-features
  in-tree, so it is a plain `from jadal.spectral.stream import
  SpectralStream`. Probe: `jadal.__file__` identical before/after, and
  `jadal.pico_link` still resolves to poc-demo — SAME, no poisoning.
  Engine tag is now `jadal.spectral.stream.SpectralStream (in-tree)`.
- I1 (append forever): `replay(..., append=False)` truncates `--out` at
  start; CLI gains `--append`. Test `test_rerun_starts_fresh_unless_append`:
  4 lines after run 1, still 4 after run 2, 8 with `append=True`.
- I2 (window cap): cap is now `round(per_window*chunk_s*fps)+8` = 58 frames
  for the 10 s window (was 33 ≈ 6.6 s). The "10 s window" label is now true.
  TTR re-measured after the fix: blackout 0.0 s, freeze 0.0 s, blur no
  reject — unchanged (chunk-scope check fires first).
- I3 (wrong-rule test): `test_frozen_rule_fires` uses one fixed noise
  pattern ×6 (high spatial_var, zero temp_e) and asserts
  `rule=="frozen"`; uniform-grey case kept separately as
  `test_lens_blocked_rule_fires` asserting `rule=="lens_blocked"`.
- I4 (stub leak + unclosed file): `_run` saves/restores
  `make_spectral_stream` in `finally`; events read via `with open(...)`.
  No ResourceWarning.
- I5 (non-16:9 garble): `-vf scale=160:90` exact + hard error on raw-size
  mismatch or zero frames (was: silent mis-slice).
- I6 (fresh-clone crash): `poc_record.py` creates `.scratch/` if missing.
- M1: demo default clip pinned to the measured file, find-fallback kept.
- M2: DEMO.md AUROC now cites `gpu-pipeline/eval/results/gpu/cascade.json`
  (`loco`: auc 0.8016, CI95 0.7096–0.8697, n=150) and
  `spectral_data/GOLD_FINAL_150*`.
- M3: single-open lock fd + non-blocking flock (`pico: busy` instead of a
  hang); headless test mocks `PicoLink.connect` so it is deterministic even
  with /dev/ttyACM0 present (it is, on this machine).
- M4: card footer prints the event's `engine` field. M5: freeze
  comprehension simplified.
- M6 (re-checkable artefact): `poc_data/stream_events.jsonl` =
  16 lines, sha256
  `f5f9a97af6dd615ab8e9a43df09cfbfc85f3b61441b4224f4a19e0a23a8d17f4`,
  decisions continue×4, accept, continue, early_reject×10 (blackout t=12 s),
  latency 345.7–783.2 ms, RAM 79.4–171.1 MB this run. Backup mp4 re-rendered
  (80 s, 16 cards).

Real command outputs:
- `python -m unittest tests.test_poc_demo` → `Ran 14 tests in 0.329s` / `OK`
- `PYTHONNOUSERSITE=1 timeout 250 python -m unittest discover -s tests`
  → `Ran 596 tests in 28.036s` / `OK (skipped=8)`
- Receipts/contract/probes (review §11–12): receipts remain poc-run-owned
  (windows.jsonl `receipt_sha256`); stream chunk events carry evidence, not
  receipts — unchanged by design, documented in DEMO.md.

## Fix round 2 (round-2 review of PR #66 — both Important + all Minors fixed)

- I1 (window_id never advances): `replay()` now stamps
  `window_id_for(clip, (n // per_window) * window_s)` per event instead of one
  id from t=0. Regenerated deliverable: 16 events span
  `f69_w4_t0000/t0010/t0020/t0030`. New test
  `test_window_id_advances_per_window` (5× one id, 6th chunk `_t0010`).
- I2 (sha not re-checkable): the full-file sha256 mixed in wall-clock
  `latency_ms`/`ram_mb`, so no re-run could reproduce it (reviewer's probe
  confirmed; not disputed). Fix: publish the hash of deterministic fields
  only. Re-check command and output:
  `python3 -c "import json,hashlib; proj=[dict(sorted({k:e[k] for k in
  ('t','window_id','chunk','running_score','decision','reason')}.items()))
  for e in map(json.loads,open('poc_data/stream_events.jsonl'))];
  print(hashlib.sha256(json.dumps(proj).encode()).hexdigest())"`
  → deliverable file: `edc3abe97f38c245d4aa9709257fcc61a5706d0db2efe73871fc63cc249360d6`.
  Stability proven on real video: two identical 6-chunk clean re-runs hash
  `3b70b8bc...a02bd801f` twice (latency/RAM differ, digest identical).
- M1: "12 tests" → corrected (15 then; 16 with the new window test).
- M2: `poc_record.py` reads events via `with open(...)`.
- M3: demo fallback `find ... | sort | head -n 1`.
- M4: `send_pico`/`replay` lock default is now
  `<repo-root>/tools/pico.lock` derived from `__file__` (CWD-independent);
  explicit paths still honoured (tests pass absolute tmp paths).
- M5: DEMO.md states chunk quantization explicitly (up to one 2 s chunk of
  understatement for off-grid degrade_at).

Real command outputs:
- `python -m unittest tests.test_poc_demo` → `Ran 15 tests in 0.302s` / `OK`
  (includes the new window-id test)
- `PYTHONNOUSERSITE=1 timeout 250 python -m unittest discover -s tests`
  → `Ran 597 tests in 36.383s` / `OK (skipped=8)`
- TTR after round-2 changes (16-chunk blackout@12 run): `time_to_reject_s:
  0.0`, `mean_latency_ms: 534.9`, wall 8.6 s. Backup mp4 re-rendered
  (80 s, 16 cards).
