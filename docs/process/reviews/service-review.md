SPEC: FAIL, QUALITY: CHANGES NEEDED

Scope: PR #13 (feat/service, commit ca235b6) against issue #3 and docs/SPEC.md. Diff: reviews/service.diff (1844 lines). Line numbers below are file lines in the feat/service tree (diff line minus the file header offset).

## Verification performed
- `unittest discover -s tests` on the worker tree: Ran 28 tests, OK (0.67 s). Matches docs/reports/service.md.
- Merged-tree check in a scratch copy (git archive of origin/feat/service plus origin/feat/pico pico_link.py plus origin/feat/certify certify.py and receipts.py; not written to the worker tree): 27 of 28 pass. Only test_pico_status_answers fails, with `pico_link unavailable: PicoLink.__init__() got an unexpected keyword argument 'fake'`.
- Voice traversal probe in the scratch copy: `GET /voice/secret/%2E%2E` returned 200 with the bytes of `secret.wav`, a file one directory above voicebank/.
- Not checked: the "live smoke test, batch b2ff74d85edb7c2e1" claim in docs/reports/service.md. I could not reproduce it from here.
- No edits to jadal/config.py, jadal/__init__.py or docs/SPEC.md. No network calls at import. No secrets found.

## SPEC compliance
1. Endpoints: PASS. All listed endpoints and WS /ws exist (service.py ~259-382). Static web/ mounts at / when the directory exists.
2. SQLite in data_runtime/jadal.db: PASS (store.py; db_path from config.py).
3. Batch trigger: PARTIAL. N from config.batch_size is applied (service.py:110). k from config is applied. T is an env var, not config (see I8). Batches can exceed N (see I4).
4. Stubs and interfaces: FAIL. Stubs live in jadal/stubs.py and the pipeline calls the SPEC interfaces through a resolver (pipeline.py:47-68). But jadal/receipts.py already exists on origin/feat/certify and the worker did not use it (see I6).
5. Reverify returns match/expected/actual/diff: PASS (pipeline.py:297-310; service.py:300-310). Caveat in I5.
6. sim/dock_sim.py copies and posts /ingest/event: PASS functionally. Race in I2.
7. Security (allowlists, bucket-restricted ingest): FAIL (C1, C2).
8. Event loop and SQLite thread safety: PASS. Handlers are sync def, WS handler only awaits sends, Store opens one connection per call, shared state is under an RLock.
Known integration issue (PicoLink fake= kwarg): FAIL (I1).

## Critical
C1. Ingest accepts any filesystem path. service.py:96-100 (handle_clip) and service.py:272 (ingest_event) never check the path is inside cfg.bucket_dir. Evidence: tests/test_service.py:93-98 write clips to tmp/clips, outside the bucket, and ingest returns 200 (test_service.py:1442-1446). Fix: in ingest_event, `clip = Path(event.path).resolve()`; if `not clip.is_relative_to(Path(cfg.bucket_dir).resolve())` raise HTTPException(400). Add a test that posts a path outside the bucket and expects 400.

C2. Voice path traversal. service.py:342-346 uses reason_code and lang unvalidated in `Path(cfg.voicebank_dir) / lang / f"{reason_code}.wav"`. Verified: `/voice/secret/%2E%2E` serves a file from the parent of voicebank/. Fix: `from jadal.config import LANGS, REASON_CODES`; return 404 unless `lang in LANGS and reason_code in REASON_CODES`. Add a test for `%2E%2E` and for an unknown code.

C3. Test-payload backdoor reachable in production. stubs.py:58-67 (`_read_payload`) parses any file starting with `JADALFAKE1\n` and merges its JSON into signals (stubs.py:131) and decisions (stubs.py:146-148). The pipeline uses these stubs whenever the real modules are absent, which is the case on this branch. Any uploaded clip can therefore set its own blur_var, hands_ok and answer_conf and force an accept. Fix: parse payloads only when an explicit env flag such as JADAL_STUB_PAYLOADS=1 is set, default off, and have tests set it in setUp. With the flag off, treat such files as ordinary bytes.

## Important
I1. /pico/status is broken against the real PicoLink. service.py:355 calls `pico_link.PicoLink(cfg.pico_port, cfg.pico_baud, fake=True)`. On origin/feat/pico the constructor is `PicoLink(serial=None, *, clock, open_serial, read_poll_s)` (pico_link.py:301-308), so this raises TypeError and the endpoint always answers with the error branch. Exact fix: `link = pico_link.PicoLink()` (the constructor does no I/O) and `return link.status()` (pico_link.py:439; keys connected, port, last_hb_age_s, last_hb_ticks_ms, last_hb_led). Better: keep one PicoLink on _Service, call `connect(cfg.pico_port)` once in lifespan inside try/except, and have /pico/status return its status(). Remove `fake=True` everywhere.

I2. Partial-copy race in the watcher. service.py:219-222 marks a clip name as seen before it is fully read, and sim/dock_sim.py:72 copies straight into the bucket with shutil.copy2. The watcher can ingest a half-written file and never retry it. The explicit POST from dock_sim (line 75) then hashes the full file, so the same clip is ingested twice under two sha256 values. Fix: dock_sim copies to `bucket/.<name>.part` and calls os.replace to the final name. The watcher should record (name, size, mtime_ns) and ingest only after the same tuple is seen on two consecutive polls.

I3. Failed clip is poisoned. service.py:99-101 adds the (path, sha) key to _ingested before process_clip runs. An exception in segmentation leaves the clip marked as ingested with no retry and no error reported. Fix: add the key only after processing succeeds.

I4. Batch can exceed N. service.py:110-111 flushes the whole buffer once len >= batch_size. A clip that yields more than one segment (the real segmenter, with segment_seconds) produces a batch with n > 5. Fix: take exactly batch_size segments per batch and keep the remainder pending.

I5. Reverify does not replay the stored config. pipeline.py:299-310 takes only the thresholds from the receipt. compute_signals and decide still receive the live Config (svc.cfg). SPEC requires re-running "with the stored config". Any config change then shows up as a mismatch that is not tampering. Fix: build the L0/L1 config from the receipt's stored config block and pass that, not the live object.

I6. Receipts claim is false and the real module is unused. docs/reports/service.md (lines 57-60) says origin/feat/certify has "no jadal/receipts.py yet". `git show origin/feat/certify:jadal/receipts.py` exists and exposes make_receipt(segment, signals, decisions, verdict, config, created_utc=None), receipt_hash and verify_receipt(receipt, recompute_fn). Merged in the scratch copy, the suite passes except the pico test (see Verification). Fix: merge origin/feat/certify (or cherry-pick receipts.py and certify.py) and correct the report. Note the hash differs by one detail: stubs.py:1177-1178 uses ensure_ascii default (True), the real module uses ensure_ascii=False. Align them before merge, or receipts created by stubs will not re-verify once the real module is in use, which is expected but must be documented.

I7. Tests codify the broken state and miss the security cases. test_service.py:469-473 asserts the "jadal.pico_link not installed" reason, which pins the import-failure path. Assert the shape instead (connected is a bool, plus the status keys). Missing tests: ingest outside bucket (C1), voice traversal (C2), stub payload gate (C3), partial-file watcher behaviour (I2), and dock_sim behaviour. TestDockSim (test_service.py:498-504) only checks that the module imports.

I8. Batch time trigger is not from config. service.py:237 reads JADAL_BATCH_MAX_AGE_S (default 30) from the environment. The issue says T comes from jadal/config.py, which has no such field. Proposal for the owner of config.py (read-only for this worker): add `batch_max_age_s: float = 30.0` to Config and read it here.

## Minor
M1. service.py:195-200 `_worker_loop` swallows every exception with `pass`, and service.py:224-225 `_watch_bucket_tick` does `continue` silently. Watcher and timeout-flush failures are invisible. Log them.
M2. store.py:75-78 `_connect` never closes its connection. `with conn:` commits but does not close. Use contextlib.closing.
M3. service.py:404 `app = create_app()` runs at import and creates data_runtime/jadal.db in the working tree. Tests that import the module touch that path. Move the store creation into lifespan or make it lazy.
M4. service.py:276-277 returns `detail=str(exc)` on unexpected errors, leaking internals. Log and return a generic 500.
M5. stubs.py:58 `_read_payload` reads the whole file as text on every call, up to three times per segment. For real video this loads the full clip into memory. Read only the first bytes and check the magic first.
M6. test_service.py:1765-1768 asserts /batches is empty immediately after ingest with a 50 ms age trigger. This is racy under load. Drop that assertion or use an injected clock.
M7. stubs.py:1177-1178 canonical JSON is ASCII-escaped (see I6).
M8. Module-level `list_batches(site_id)` filter is not exposed by the API. Dead parameter, low impact.

## Report accuracy (docs/reports/service.md)
- "Ran 28 tests ... OK": confirmed.
- "origin/feat/certify has no jadal/receipts.py yet": false (see I6).
- Live smoke test batch id and "decision accept, reverify match=true": not verified here.
- "No contract change proposed": accurate as far as the diff shows, but I8 is a proposed config change to pass to the config owner.
