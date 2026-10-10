ALL ADDRESSED: yes, NEW BREAKAGE: none

Scope: fix round 1 diff f6b05a7..origin/feat/service (head 3e494c4), saved at reviews/service-fix1.diff. Certify/pico/receipts files excluded as instructed.

## Original findings
- C1 ADDRESSED. service.py `_clip_in_bucket` (realpath + relative_to on cfg.bucket_dir) gates handle_clip, used by /ingest/event (400) and the watcher. Probe: absolute outside path, symlink inside bucket pointing out, and `bucket/../outside/evil.mp4` all return 400.
- C2 ADDRESSED. service.py voice(): `lang not in LANGS or reason_code not in REASON_CODES` returns 404 before any path is built. Probe: `/voice/secret/%2E%2E`, `/voice/..%2Fsecret/hi-IN`, bad lang and bad code all 404. A valid code and lang returns 200.
- C3 ADDRESSED. `_read_payload` and the JADALFAKE1 parsing are removed from stubs.py. Signals and decisions derive from the SHA-256 only. Stubs load only with JADAL_USE_STUBS=1 (pipeline._load_stage); without it, ingest returns 500. Probe: a fake clip whose payload said blur_var=999 yields the hash-derived 63.0.
- I1 ADDRESSED. /pico/status returns svc.pico_status(), which uses PicoLink() plus connect(cfg.pico_port) in start(), inside try/except. No fake= kwarg remains. Probe: returns {"connected": false, "reason": "no pico on /dev/ttyACM0"}. I did not exercise status() against real hardware.
- I2 ADDRESSED. dock_sim.py stages `.<name>.part` then os.replace. The watcher skips dotfiles and .part and ingests only after the same (size, mtime_ns) on two polls (service.py `_watch_bucket_tick`).
- I3 ADDRESSED. handle_clip adds the key to _ingested only after process_clip succeeds. Test test_failed_clip_is_not_poisoned_and_can_retry exists.
- I4 ADDRESSED. flush_site takes exactly batch_size segments. Test test_batch_never_exceeds_n exists.
- I5 ADDRESSED, with a caveat. reverify_receipt rebuilds replay config from the stored block, uses the stored verdict threshold, and passes the stored block verbatim to make_receipt. Stored versions are not compared against live versions (see deferred).
- I6 ADDRESSED. pipeline uses jadal.certify and jadal.receipts unconditionally. The stub receipt code is removed.
- I7 PARTIAL. Tests added for outside-bucket, traversal, unknowns, stub gate, part files and failed-retry. I did not confirm the pico test asserts status shape, or that TestDockSim does more than import.
- I8 ADDRESSED as a documented env var (JADAL_BATCH_MAX_AGE_S, default 30.0). The config-owner proposal stays open.
- Minors: M1, M2, M3, M4 and M6 fixed. M5 and M7 are moot after the stub and receipt changes. M8 left as is.

## Tests (run now)
- `cd service && PYTHONNOUSERSITE=1 <env>/bin/python -m unittest discover -s tests`: Ran 127 tests, OK. Per file: certify 40, pico_link 22, receipts 27, service 38 (sums to 127).
- The report's named test lines and the 127 count match this run.
- `-s web`: no web/ directory or tests exist, so nothing to run.

## Report check (docs/reports/service.md, Fix round 1)
- Test count and named test lines: accurate.
- Old "Ran 28 tests" line in the pre-fix section is stale history, not a false claim about the fix.

## Deferred minors (not new breakage)
- flush_site clears the buffer before run_batch, so a failure drops those segments and the clip stays marked ingested. Same class as I3; pre-existing shape.
- The watcher retries a permanently failing file every two polls, logging a warning each time. No backoff.
- _errors is recorded but never exposed by any endpoint.
- Reverify does not compare stored `versions` against live implementations.
- Pre-fix ASCII-hashed receipts will not re-verify (documented in the report).
