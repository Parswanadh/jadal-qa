ALL ADDRESSED: yes, NEW BREAKAGE: none

Scope: fix round 1 diff for PR #14 (feat/certify), origin/feat/certify = worker HEAD c6f60f1. Diff saved at reviews/certify-fix1.diff (822 lines). Line refs below are diff line numbers.

Test run (worker tree, PYTHONNOUSERSITE=1, <env>/bin/python -m unittest discover -s tests): Ran 73 tests, OK. Per-module counts: 43 certify, 30 receipts (matches the report). No web/ directory exists, so the dashboard suite was not run.

## Original Critical / Important

- C1 pick_threshold fixed-sequence: ADDRESSED. Strictest-first loop, `break` on `accepted < min_accepted or upper_bound > target`, returns last certified (diff lines 244-248). Tests (a) stop-earlier, (b) exact known-counts bound, (c) None on strictest fail all present and pass.
- I1 tampered-only receipt_sha256: ADDRESSED. Diff lines 357-362 give `diff == ["receipt_sha256"]` when bodies match. Test test_tampered_hash_only_reports_the_hash_path passes.
- I2 NaN/inf rejection: ADDRESSED. _check_finite (diff lines 294-314) called from make_receipt (lines 327-330) with field-path errors. Test covers signals, decisions and verdict paths; -0.0 normalisation asserted as +0.0.
- I3 batch test: ADDRESSED per ruling. Batch helper with one decide_batch call (call log asserted == [5]), compared to single calls; limitation noted in report. Not a run_batch test, as the ruling allowed.
- I4 config versions scope: ADDRESSED in docs (Known gaps).

## Minors
M1-M6 addressed. M7 noted as out of scope in docs.

## Report check (docs/reports/certify.md)
- Counts 73 / 43 / 30 match the run.
- Doc defect: lines 60-66 keep the old "Ran 67 tests" block under the new "all 73 pass" header. Replace it with the 73-test output. The "Command (real output tail)" block at line 119 uses "..." elision; acceptable, but the stale block must go.

## Deferred minors (out of scope)
- Two blank lines before `def risk_coverage` were removed (PEP 8 spacing, certify.py diff ~line 265).
- No TestClient path probe was run; the unit tests cover the verify paths.
