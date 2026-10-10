SPEC: FAIL, QUALITY: CHANGES NEEDED

Scope: PR #14 (feat/certify) of Parswanadh/jadal-qa for issue #1. Diff saved to <worktrees>/reviews/certify.diff. Worker tree: <worktrees>/certify. No code was modified and nothing was pushed or commented on GitHub.

Test run (worker tree, <env>/bin/python, PYTHONNOUSERSITE=1):
  Ran 67 tests in 0.066s / OK
tests/test_receipts.py imports and runs (`import tests.test_receipts` raises nothing). The earlier test_receipts import error from the worker log is not reproducible on this branch head.

Numerical checks: exact_binomial_upper(0,100,0.1) = 0.022763 and exact_binomial_upper(0,200,0.1) = 0.011447. Both match scipy beta.ppf(0.9, 1, n) and the closed form 1 - 0.1^(1/n), and both match the SPEC values (0.0228, 0.0113).

## SPEC COMPLIANCE

1. certify.py interfaces: fit_temperature, ece, exact_binomial_upper, risk_coverage, batch_rule present with the specified signatures. PASS.
2. pick_threshold: FAIL. SPEC requires fixed-sequence testing from the strictest threshold downward, stopping at the first failure. jadal/certify.py:231-246 scans every distinct confidence and keeps the loosest admissible candidate; it never stops. This is a deliberate deviation (documented in docs/reports/certify.md:24-32 and the docstring), but it contradicts the SPEC.
3. receipts.py: canonical_json, make_receipt, receipt_hash, verify_receipt present; hash excludes receipt_sha256/receipt_id/created_utc; receipt_id = first 16 hex; floats rounded to 4 decimals with -0.0 normalized. PASS.
4. Required tests present: stable hash under key order (test_receipts.py, ReceiptHashTest), batch-of-5 equality, mismatch with changed field path, pick_threshold on synthetic data, temperature fit reduces ECE. PASS (with the caveat in Important #3).
5. Protected files (jadal/__init__.py, jadal/config.py, docs/SPEC.md) untouched in the diff. No secrets in the diff. PASS.

## CRITICAL

C1. Statistical validity of pick_threshold (jadal/certify.py:231-246). Scanning all candidates and returning the loosest one that passes the bound at delta=0.1 is a multiple-comparisons selection. The returned threshold does not carry the advertised 1-delta certificate, because the bound was checked on many candidates and the max-coverage one was picked. Fix: implement fixed-sequence testing as SPEC states:
    for t in np.unique(c)[::-1]:
        ... compute accepted, errors, upper
        if accepted < min_accepted or upper > target: break
        best = candidate
    return best (or threshold None if the first candidate fails)
Then update tests/test_certify.py PickThresholdTest (test_returns_the_loosest_threshold..., test_maximum_coverage_is_preferred...) and docs/reports/certify.md so they describe the stop-at-first-failure rule and no longer call the scan "fixed-sequence" while doing something else.

## IMPORTANT

I1. verify_receipt returns match=False with an empty diff when only the stored receipt_sha256 is wrong (jadal/receipts.py:192-194; confirmed by running it: stored hash 'f'*64, body unchanged, diff == []). SPEC says diff lists the differing field paths. Fix: when the stored hash differs from recomputed and the stripped bodies are equal, return diff ["receipt_sha256"]. Replace the current `if "receipt_sha256" not in receipt` check with an unconditional check that appends "receipt_sha256" when `receipt.get("receipt_sha256") != actual` and it is not already in diff. Add a test.

I2. make_receipt accepts NaN/inf (jadal/receipts.py:118-145, body built at ~134). A NaN signal produces a receipt whose canonical JSON contains the non-standard token NaN, so the receipt is not valid JSON for downstream consumers. Verified: a NaN blur_var is accepted and gets a receipt_id. Fix: in _round_floats (or make_receipt) raise ValueError for non-finite floats, and add a test.

I3. The batch-independence test does not exercise a batch code path (tests/test_receipts.py ~lines 252-264, _process). It loops over segments one at a time, and the receipt is a pure function of the segment, so the test passes by construction. It is a reasonable unit test of determinism, but it should not be presented as proof of the SPEC batch rule. Note this in docs/reports/certify.md and re-run the check once pipeline.run_batch exists.

I4. Receipt config versions only record python and config_sha256 for the dataclass path (jadal/receipts.py:~86-101). SPEC asks for torch, model names and weight hashes. Out of scope for this module, but the docs should say so explicitly.

## MINOR

M1. _round_floats (jadal/receipts.py:57-66) normalizes -0.0 correctly, but the test at tests/test_receipts.py only checks that "-0.0" does not appear in the output. Add a direct check that the output float is positive zero.

M2. risk_coverage (jadal/certify.py:~369-383) rounds threshold to 4 decimals, so distinct confidences that round equal produce duplicate rows. Document or de-duplicate.

M3. _prepare_conf_correct (jadal/certify.py:~278) uses ok.astype(bool), so string labels such as "False" silently become True. Validate that correct is boolean or 0/1 integers.

M4. fit_temperature objective clips probabilities at 1e-12 (jadal/certify.py:256), giving a flat gradient for badly misclassified rows at extreme T. Acceptable for a bounded search, but add a test with extreme logits.

M5. receipt_id is excluded from the hash, so a receipt whose receipt_id was edited still verifies. This is by design per SPEC, but nothing checks receipt_id == receipt_sha256[:16] during verification. Add that check.

M6. test_receipts.py uses an inline __import__("sys") (around line 1238 of the diff). Use a top-level import.

M7. SPEC says a receipt carries blurred_sha256 when faces are blurred. This module does not handle it, and test_segment_sha256_is_the_hash_of_the_original_segment_file asserts it is absent. Out of scope for this PR; say so in the report.

## DOCS

docs/reports/certify.md:24-32 describes pick_threshold as a fixed-sequence scan and says it "does not return the strictest passing threshold ... deliberate choice". The SPEC requires the opposite. Update the report once C1 is fixed. The rest of the report's test claims were checked against the run above (67 tests, OK) and match.
