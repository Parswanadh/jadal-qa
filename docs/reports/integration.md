# integration — module report (issue #17)

Owner of `jadal/service.py`, `jadal/pipeline.py`, `jadal/store.py`,
`jadal/stubs.py`, `sim/dock_sim.py`, `scripts/`, `Makefile`, `docs/API.md`,
`tests/test_service.py`, `tests/test_integration.py` after the first merges.

## What exists

- **docs/API.md** — exact JSON shape (field names, types, one real example
  each) of every endpoint, produced by running the service with stubs:
  `POST /ingest/event`, `GET /health`, `/batches`, `/batches/{id}`
  (+`receipt_ids`), `/receipts/{id}`, `POST /receipts/{id}/reverify`
  (+`version_match`/`version_diff`), `/stats/risk_coverage`,
  `/stats/sop_gaps`, `/sop/search`, `/voice/{code}/{lang}`, `/pico/status`,
  `WS /ws`, `GET /ingest/errors`. Dashboard-v2 builds against it. The
  "Missing" section lists the dashboard API needs not yet exposed
  (per-verdict `receipt_id/source_clip/worker_id/site_id/latency_ms`,
  risk bound metadata, `blurred_sha256`).
- **Real-module wiring** (`pipeline._load_stage`, unchanged design):
  `jadal.signals.compute_signals`, `jadal.decision.Decider`,
  `jadal.segmenter.split_clip` are preferred when importable; stubs only
  under `JADAL_USE_STUBS=1`, else a clear `RuntimeError`. Every model call is
  per segment (batch size 1); `ensure_determinism()` pins seeds +
  torch deterministic kernels (all guarded).
- **Version provenance** — every receipt `config.versions` now carries the
  L0 backend report (`signals.backend_versions()`, else `backend_info()`,
  JSON-encoded as `signals_backend`) and the face/hand weight-file hashes
  (`face_detector_task`/`hand_landmarker_task` as `name:sha256:<hex>`,
  hashed once and cached), plus `pipeline_version` (was already there).
- **Real-data stats** — `/stats/risk_coverage`: SQLite calibration rows
  first, else validated `eval/results/calibration.json`, else `[]` (the
  honest "no measured data" state — never fabricated). `/stats/sop_gaps`:
  real reason-code counts over stored verdicts
  (`[{reason_code, count, share}]`, count desc); the `sop/` module takes over
  when it merges.
- **Deferred minors fixed** — failed batch requeues its segments at the
  front of the site buffer (never dropped; caller still sees the 500);
  watcher retries bad files with exponential backoff (2^attempts s, cap
  30 s, so a poison file cannot spin); errors exposed at
  `GET /ingest/errors`; reverify compares stored vs running versions
  (`version_match`, `version_diff`).
- **`make demo`** (`Makefile` + `scripts/demo.sh`) — free port, fresh
  `data_runtime/`, 7 catalog IndEgo clips symlinked and docked through
  `sim/dock_sim.py`, verdict table, reverify of every receipt, 10x same-hash
  assert + alone-vs-batch-of-5 assert, batch latency print.
- **`tests/test_integration.py`** — 12 tests: version provenance + weight
  tokens, `compare_versions`, 10x determinism, solo-vs-5, failed-batch
  retry, watcher backoff + `/ingest/errors`, empty-risk honesty, eval-file
  parsing (malformed row dropped), sop-gap counts, reverify version keys.

## How to run

```sh
make test   # full suite (stubs; no hardware, no network)
make demo   # end-to-end on the IndEgo clips
make service
```

## Tests — actually run

```sh
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests
# Ran 225 tests in 7.713s — OK
# (38 service + 12 integration + certify/receipts/pico suites from main)
```

`make demo` real output tail (2026-10-10, service on :48619, stubs):

```
batches closed: 2 (7 clips -> batch of 5 + timeout-flushed 2)
batch b2a9096c13dd07127 site=site-demo decision=REJECT rule='reject if rejected_count > k' k=1 n=5
  seg-675a30d26e33-0000 reject  conf=0.9300 reasons=[HANDS_NOT_VISIBLE]
  ...
batch bbf76e31013a6558d site=site-demo decision=ACCEPT rule='reject if rejected_count > k' k=1 n=2
  ...
== reverify (7 receipts) ==
  11bad92f6c2d1b70 MATCH versions_match=True
  ... (7/7 MATCH, versions_match=True)
all receipts reverify MATCH
determinism: same clip 10x -> identical a333ed9af8f9142f...
determinism: alone vs batch-of-5 -> IDENTICAL receipt_sha256
batch latency: dock-to-batches wall time = 10.9s (7 clips, 2 batches, includes 5s timeout flush)
demo OK
```

Real-module interop (unmerged `feat/signals` code, scratch sandbox in
`/tmp/opencode/sigtest`, nothing written to this worktree): `impl_versions`
detects `jadal.signals`/`jadal.segmenter` as real, records `signals_backend`
from `backend_info()`; real segmenter split a 95 s IndEgo clip into 5
segments; real L0 returned the exact SPEC signals key set; the SPEC hash rule
recomputed clean. `jadal.decision` does not exist yet on `feat/decider`, so
L1 stays on the stub fallback by design.

## Known gaps / NOT DONE

- L0/L1 values in the demo are hash-derived stub fakes (whole clip = one
  segment); the verdicts above (HANDS_NOT_VISIBLE/SHAKY) reflect stub bytes,
  not clip content. Real values arrive when `feat/signals` merges to main
  (verified compatible, see above) — no pipeline change needed.
- `jadal.decision.Decider` does not exist yet (`feat/decider` = main); L1
  stays on stubs until it lands. Built against the SPEC interface.
- `eval/results/` is absent in this worktree, so `/stats/risk_coverage`
  returns `[]` (no measured data). Note: the evalharness calibration file is
  built from synthetic demo predictions (auto-labels), so even after it
  merges, its points are eval-module output, not pipeline measurements.
- Dashboard "API needs" enrichment (§Missing in docs/API.md) is documented,
  not implemented: per-verdict `receipt_id/source_clip/worker_id/site_id/
  latency_ms` (proposed: `GET /segments/{id}`), risk bound metadata,
  `blurred_sha256`. No receipt hashing rule was changed.
- `sop/` module absent: `/sop/search` returns `{"results": []}`.
- No GPU used; no secrets in code; no network at import.
- Proposals for other owners: none (no config/SPEC changes needed).

## Fix round 1 (review of PR #19)

Both Important items fixed in owned files; minors left untouched.

1. **`flush_site` save inside the requeue guard** (`jadal/service.py`):
   `store.save_batch` moved into the same try/except as `run_batch`, so a
   SQLite write failure requeues the segments at the front of the site buffer
   instead of dropping the batch (the save is INSERT OR REPLACE over one
   transaction, so the retry converges). `_emit_batch` stays outside the
   guard — a WS push failure must not requeue already-saved segments. New
   test `test_failed_save_requeues_segments_for_retry` injects a once-failing
   save and asserts: 500 on the closing ingest, 5 segments still pending, no
   batch stored, then a retry closes the batch with the ORIGINAL five clips
   in ingest order and the receipt re-verifies MATCH.
2. **`reverify_receipt` seeds determinism** (`jadal/pipeline.py`):
   `ensure_determinism(config)` at the top (exactly like `run_batch`) plus
   `ensure_determinism(replay_config)` once the stored config is rebuilt, so
   a torch-backed L0/L1 replays with the seed that produced the receipt. New
   test `test_reverify_calls_ensure_determinism` patches
   `pipeline.ensure_determinism`, runs a reverify, and asserts it was called
   (and the receipt still matches).

Also merged `origin/main` first (only `.env.example` + `.gitignore` changed).

Commands and REAL output tails:

```sh
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_integration tests.test_service
# Ran 52 tests in 4.092s
# OK

PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests
# ERROR: test_run_without_client_and_without_sdk_returns_2 (test_voicebank.RunEndToEndTests.test_run_without_client_and_without_sdk_returns_2)
# FAIL: test_no_sarvamai_imported (test_voicebank.SynthesisTests.test_no_sarvamai_imported)
# Ran 227 tests in 8.069s
# FAILED (failures=1, errors=1)
```

The 1 failure + 1 error are both in `tests/test_voicebank.py`, which this
branch does not touch; they are pre-existing on `origin/main` (confirmed in
the review) and are being fixed separately by another worker. Owned suites:
52/52 OK (38 service + 14 integration).

## Integration round 2 (issue #17 follow-up, branch `feat/integration2`)

Hazard (found by a trial merge of the real `jadal.signals`): with the real
decoder importable, `test_websocket_pushes_batch_event` hung forever -- fake
mp4 bytes hit the real decoder (`moov atom not found`), no batch was emitted
and no wait timed out. Fixed (1)-(7) in owned files only (`jadal/service.py`, `jadal/pipeline.py`,
`Makefile`, `docs/API.md`, both test files, this report; `jadal/store.py`,
`jadal/stubs.py`, `sim/dock_sim.py`, `scripts/**` needed no changes; NOT
touched: `jadal/config.py`, `jadal/__init__.py`, `docs/SPEC.md`).

1. **Forced stubs in tests** -- `BucketCase`/`IntegrationCase` now inject
   stub callables (the three pipeline resolvers are mock-patched to
   `stubs.*`) on top of `JADAL_USE_STUBS=1`, so results never depend on which
   modules are importable. New `TestStubIndependence` installs a fake
   `jadal.signals`/`decision`/`segmenter` stack whose entry points RAISE on
   any call and asserts batch verdicts are byte-identical to the no-fake run
   with zero fake calls.
2. **Bounded waits** -- every wait has a timeout: `/ws` server receives use
   `asyncio.wait_for(..., JADAL_WS_RECV_TIMEOUT_S=60 s)` (idle clients wait
   again, never block); tests read through `recv_json_with_timeout` (thread
   + join, raises `TimeoutError` with a clear message -- starlette 1.7's
   TestClient offers no timeout knob); `TestWsTimeout` proves the fast
   failure. No queue/thread waits exist besides the worker `_stop.wait(poll)`.
3. **UNDECODABLE, never a silent loop** -- `process_clip` returns a
   deterministic whole-clip placeholder when the segmenter fails on the
   bytes (or yields zero segments); only `OSError` propagates as transient.
   `run_batch`/`reverify_receipt` decode via `_compute_signals_safe`: capped
   retries (`_SIGNAL_IO_RETRIES=3`) for `OSError`, then raise (existing
   requeue + `/ingest/errors` path); any other decoder failure synthesises
   `decodable=False` signals so the segment closes with a reject verdict
   with reason UNDECODABLE and a normal batch event. The watcher still backs
   off, but parks a poison file after `WATCH_MAX_ATTEMPTS=5` with a terminal
   "gave up after N attempts" error (`TestWatcherCap`).
4. **Per-head `/stats/risk_coverage`** -- `pipeline.risk_coverage_report`
   returns `[{head, points, chosen_threshold, upper_bound, target_err, delta,
   n_calibration, n_test, label_source}]` from
   `eval/results/decider_baseline.json` (per L1 head) and
   `eval/results/funnel_calibration.json` (`accept` + `reject:<rule>`),
   then SQLite rows (head `"answer"`), then legacy `calibration.json`; `[]`
   when nothing was measured. `chosen_threshold: null` = measured but
   explicitly NOT certified; the key is absent only when nothing about a
   threshold was measured. `label_source` is `gemini|human_gold|synthetic`
   (an explicit labeller mention wins: "labels by gemini-2.5-flash, not
   human-checked" -> `gemini`) and is omitted when provenance is unstated.
   `docs/API.md` shows the real service output.
5. **Funnel seam** -- `pipeline.route_verdict` resolves `jadal.funnel.route`
   at runtime on every call (never cached) with the SPEC signature
   `route(signals, decisions, thresholds)`; absent -> inline `make_verdict`
   (unchanged). The funnel path carries optional `verdict.evidence`, which
   the receipt hash includes only when present (proven by hash comparison in
   `TestFunnelSeam`); funnel logic itself is NOT implemented here.
6. **Hub mount** -- `create_app` includes `jadal.hub_api.router` under `/hub`
   when importable, guarded (absent -> zero `/hub` routes, verified on the
   running service); `TestHubMount` covers both sides with a fake router.
7. **Suite under 120 s** -- `make test` runs both suites with `timeout 120`.

Commands and REAL output tails (this branch, stubs):

```sh
$ make test
# Ran 271 tests in 14.675s -- OK (skipped=2: the real-signals trial tests)
# Ran 19 tests in 0.227s -- OK   (web suite via `discover -s web`;
#   the `-t .` form in web/test_dashboard.py's docstring fails with
#   "Start directory is not importable" -- dashboard worker's doc issue)

$ timeout 300 make demo   # IndEgo clips, stubs
# all receipts reverify MATCH (7/7, versions_match=True)
# determinism: same clip 10x -> identical a333ed9af8f9142f...
# determinism: alone vs batch-of-5 -> IDENTICAL receipt_sha256
# batch latency: dock-to-batches wall time = 10.2s
# demo OK
```

Trial merge proof (throwaway worktree `/tmp/jadal_trial_signals`, removed
after; `git merge origin/feat/signals` into round-2 HEAD, real decoder live):

```sh
$ python -m unittest tests.test_integration.TestRealSignalsTrial -v
# test_garbage_file_yields_undecodable_reject ... ok
# test_real_mp4_decodes_and_batches ... ok   (tiny mp4 via cv2.VideoWriter)
# Ran 2 tests in 2.061s -- OK

$ python -m unittest tests.test_service tests.test_integration
# Ran 69 tests in 5.598s -- OK (skipped=1: test_stubs_require_flag,
#   correctly skipped with real modules present)

$ python -m unittest discover -s tests   # x4 consecutive runs
# Ran 295 tests in 26-40s -- OK (skipped=1) each time
```

Honesty note: one earlier trial `discover -s tests` run reported
`FAILED (errors=20)`; the next four consecutive runs were fully green and it
did not reproduce (suspected transient mediapipe/resource flake in the
signals worker's own 24 slow tests, which this branch does not own). The
owned 69/69 pass deterministically in both trees.

NOT DONE / gaps: funnel thresholds/logic belong to the funnel worker (seam
only here); `jadal.decision.Decider` still absent so L1 stays on stubs;
`points` curves stay `[]` until a measurement file carries per-sample
conf/correct arrays; web `-t .` docstring untouched (not owned).

## Fix round 1 (review of PR #34)

Started by merging `origin/main` (funnel, certify, eval harness now in).
The merged real `jadal.funnel.route` decides all `run_batch` verdicts, which
changed four test expectations (L0 rules hard-reject, L1 outputs only demote
to review, uncalibrated placeholder thresholds route clean clips to review
with evidence). All five Important findings fixed in owned files:

1. **Hub prefix exactly once** (`jadal/service.py` `_mount_hub_router`): a
   router already carrying `/hub` is included as-is, any other router with
   `prefix="/hub"`. `TestHubMount` covers both fake-router cases (`/hub/ping`
   200, `/ping` 404) plus the absent case (no `/hub` routes).
2. **Capped batch-timeout flush** (`jadal/service.py` `_tick` /
   `_record_flush_failure`): per-site attempts with 1, 2, 4, 8, 16 s
   backoff (`FLUSH_MAX_ATTEMPTS=5`, injectable `svc._clock`), every failure
   recorded under `batch:<site>` in `GET /ingest/errors`; after the cap the
   segments park at the new `GET /batches/failed` (with `failed_segments` in
   `/health`) -- never dropped, never retried forever. `TestFlushCap` uses a
   failing save + fake clock (attempts, exact backoff schedule, error text,
   failed endpoint, no-op further ticks).
3. **Head-first `label_source`** (`jadal/pipeline.py` `_label_source`): the
   head's own provenance decides; a file-level disagreement reports
   `'conflict:<head>|<file>'` (the review's live probe -- human-gold head vs
   gemini file -- now yields `conflict:human_gold|gemini`, covered by test).
4. **No invented numbers** (`risk_coverage_report` + docs): the store-backed
   `answer` head omits `n_test` (was `0`); audit rule documented in the
   docstring and `docs/API.md` (`null` = measured-but-uncertified only where
   a measurement exists; absent key = nothing measured; `upper_bound` kept
   when measured even if uncertified).
5. **Exact docs** (`docs/API.md`): `target_err`/`delta` source per path
   (funnel file targets vs service config), uncertified-`upper_bound`
   semantics, regenerated examples. Committed fixtures
   `docs/api_examples/decider_baseline.json` +
   `funnel_calibration.json` with expected `risk_coverage.json`; the test
   `test_committed_examples_match_live_responses` fails on any drift.

Seam hardening alongside the fixes: `thresholds_snapshot` overlays the
service Config on the hash-verified funnel thresholds file (router keys
`accept_conf`/`motion_min`/`min_duration_s`/`under_exposed_frac_max`/
`calibration` now reach `route`; stored verbatim with
`funnel_thresholds_sha256`, so re-verify replays exactly); a raising
`funnel.route` propagates (never swallowed); signal I/O retries use doubling
backoff. Minors closed: temp-dir empty test, signals-blocked independence
test. `Makefile` uses `timeout 150` per suite.

Commands and REAL output tails:

```sh
$ timeout 150 make test
# Ran 332 tests in 11.644s -- OK (skipped=3)
# Ran 19 tests in 0.119s -- OK (web)

$ timeout 150 python -m unittest tests.test_service tests.test_integration
# Ran 80 tests in 5.336s -- OK (skipped=3)

$ timeout 300 make demo
# all receipts reverify MATCH (7/7, versions_match=True)
# determinism: same clip 10x -> identical 2052f35308948842f...
# determinism: alone vs batch-of-5 -> IDENTICAL receipt_sha256
# batch latency: dock-to-batches wall time = 10.1s (7 clips, 2 batches)
# demo OK   (both batches REJECT: stub bytes trip L0 hard rules via funnel)

$ # throwaway signals trial (merged origin/feat/signals, worktree removed after)
$ python -m unittest discover -s tests
# Ran 356 tests in 30.283s -- FAILED (failures=1, skipped=2)
# FAIL: test_funnel.TestGoldRecordsBuild.test_missing_signals_module_raises_clearly
```

The single trial failure is pre-existing and unowned: that funnel test
assumes `jadal.signals` is absent without blocking it, so it fails whenever
signals is importable. Proof of independence: neither
`tests/test_funnel.py` nor `eval/calibrate_funnel.py` imports
`jadal.pipeline`/`service`/`stubs` (grep-verified), and the test fails
identically as a single-test run. The owned 80/80 pass in both trees.

Hazard reproduced on pristine `origin/main` + signals (no round-2 code):
the suite spews `moov atom not found` and hangs until the 150 s timeout
kills it (exit 124) -- exactly the round-2 report's hazard; with this
branch's fixes the same tree finishes in ~30 s.
