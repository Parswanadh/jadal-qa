SPEC: PASS, QUALITY: CHANGES NEEDED

Scope: PR #19 (feat/integration, head a29bcd1, equal to origin/feat/integration) for issue #17. The diff touches only owned files: Makefile, docs/API.md, docs/reports/integration.md, jadal/pipeline.py, jadal/service.py, scripts/demo.sh, tests/test_integration.py. No changes to config.py, __init__.py, docs/SPEC.md, certify, receipts, pico_link, signals or decision.

## Checklist

1. ✅ docs/API.md matches the running service. Service run on stubs; every documented endpoint called with curl or a TestClient and fields diffed.
   - Matches: /health, POST /ingest/event (duplicate and batch-closing shapes; 400 outside bucket, 400 missing, traversal 400), GET /batches (9 keys), GET /batches/{id} (+receipt_ids), GET /receipts/{id}, POST reverify (+version_match, version_diff), 404s, /ingest/errors, /stats/risk_coverage ([]), /stats/sop_gaps, /sop/search, /pico/status, /ws (hello, pong, replay with receipt_ids), / (dashboard).
   - Not reproducible: the /voice 200 path. voicebank/ has no .wav files in the worktree, so every valid request returns 404. The doc's "real example" is unverified here.

2. ✅ Real modules are preferred, stubs are behind the flag. pipeline.py:61-89 falls back to stubs only when JADAL_USE_STUBS=1. With the flag unset and no real modules, process_clip and run_batch raise an explicit RuntimeError (verified). Ingest maps it to 500 "ingest unavailable" (service.py:405-408).

3. ✅ Versions and hashing. impl_versions (pipeline.py:92-141) writes config.versions. receipts.py is unchanged: HASHED_EXCLUDED = {receipt_sha256, receipt_id, created_utc} (receipts.py:24, 132); floats rounded to 4 dp (receipts.py:75-85).

4. ❌ Reverify replays the stored config and compares versions, but misses determinism setup.
   - Pass: the stored block is used verbatim (pipeline.py:402-416, stored_config at 268-280) and compare_versions runs (pipeline.py:438-439). Live reverify: match:true, version_match:true.
   - Fails: reverify_receipt (pipeline.py:395-440) never calls ensure_determinism(); run_batch does (pipeline.py:351). Once torch-backed L0/L1 land, a replay can run with unseeded RNG state and report a spurious MISMATCH.

5. ❌ Deferred minors: partially fixed.
   - Pass: the failed run_batch requeues segments at the front of the buffer (service.py:185-193). The watcher uses 2^n backoff capped at 30 s (service.py:314-347). GET /ingest/errors exists (service.py:413-417). The tests test_failed_batch_keeps_segments_for_retry and test_watcher_backoff_and_ingest_errors_exposed pass.
   - Fails: see Important #1 (save_batch is outside the requeue guard).

6. ✅ /stats/risk_coverage never fabricates. Order: SQLite calibration rows, then validated eval/results/calibration.json rows (pipeline.py:468-497, malformed rows dropped), then []. Live: []. Output keys threshold/coverage/error match certify.risk_coverage (certify.py:287-330).

7. ✅ Dry block of make demo verified by running; the full demo is not run.
   - Ran scripts/demo.sh lines 141-171 (the in-process determinism block) against the 7 docked clips: 10x identical hash and alone-vs-batch-of-5 identical both pass. All 7 IndEgo source clips exist.
   - Not run: the server, dock and reverify stages, because they wipe the worktree's data_runtime/. bash -n passes.
   - Minor gap: no count assertions. N_BATCHES (demo.sh:85-98) is printed but not asserted, and the reverify loop (demo.sh:127-137) passes vacuously with zero receipts.

8. ✅ Security intact. _clip_in_bucket uses realpath plus relative_to (service.py:111-122, 128-130). Live: /etc/passwd and ../ traversal both return 400. The /voice allowlists are checked before the path is built (service.py:489-493). Not changed by the diff.

9. ❌ Test claim not reproduced, but not a PR defect.
   - Full suite (unittest discover -s tests): Ran 225, FAILED (failures=1, errors=1). The report says "Ran 225 tests — OK".
   - Failures: tests/test_voicebank.py, which this PR does not touch. test_no_sarvamai_imported and test_run_without_client_and_without_sdk_returns_2. The same 1 failure and 1 error occur on a clean origin/main snapshot (31 tests), so they are pre-existing.
   - Owned suites pass: test_integration 12/12 OK, test_service 38/38 OK.

10. ✅ Files outside owned set: none. The diff is limited to the seven owned files.

## Findings

Critical: none.

Important:
1. service.py:185-196 (flush_site): only pipeline.run_batch is inside the requeue try. The segments are removed from _pending at line 184. store.save_batch (line 195) and _emit_batch (line 196) run outside the guard. A SQLite write failure silently drops the whole batch, which is the case requirement 5 says must not happen. Fix: move save_batch into the same try/except that requeues, and requeue only after the save succeeds or re-raise without losing segments.
2. pipeline.py:395-440 (reverify_receipt): no ensure_determinism() call. Fix: call ensure_determinism(replay_config) at the top of the function, before compute_signals and decide.

Minor:
3. service.py:288-297 (_tick): a failing timeout flush is only logged. Retries are spaced by batch_max_age_s, not spun, but the failure never reaches GET /ingest/errors. Fix: record per-site flush errors and show them there.
4. service.py:132-135 and 348-351: a stale _errors entry survives a successful duplicate retry, because neither path pops it. Fix: pop self._errors[str(resolved)] on those paths.
5. service.py:200-201: the recursive flush_site can raise after a batch was already saved, so the ingest caller gets a 500 for a batch that did close. Fix: catch and log the recursive error.
6. scripts/demo.sh:85-98 and 127-137: no assertion on batch count (expect 2) or receipt count (expect 7). Fix: assert both.
7. docs/API.md: the receipt example is stitched, not a byte-for-byte capture. The segment_id seg-b70ae1ab3f1e-0000 does not match segment_sha256 c11fecc12cc6..., since the stub derives the id from sha[:12]. receipt_id 4262773ed2ca299a is not among the documented batch's receipt_ids. The "…(64 hex)" placeholder and "ph_x" are not literal output. Fix: regenerate the example from one real run, or label it illustrative.
8. docs/reports/integration.md: "Ran 225 tests — OK" does not reproduce (see item 9). Fix: report the actual result, including the pre-existing test_voicebank failures on main.

## Environment side effects

Worktree data_runtime/ is git-ignored. The service was started on it, so the existing bucket clips were ingested into data_runtime/jadal.db. A temporary bucket/api_check/ folder was created and removed. The server was stopped. Scratch files are under reviews/ (server.log, server.pid, .port, ingest_*.json, demo_dry_check.py.txt, main_snapshot/). One curl output to /tmp/x.wav was removed.
