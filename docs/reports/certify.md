# certify + receipts — module report

Worktree: `/home/parshu/projects/jadal-qa-wt/certify`, branch `feat/certify`.
Closes [Parswanadh/jadal-qa#1](https://github.com/Parswanadh/jadal-qa/issues/1).

## What exists

`jadal/certify.py` (pure python + numpy + scipy, no randomness anywhere):

- `exact_binomial_upper(errors, n, delta=0.1) -> float` — one-sided
  Clopper-Pearson upper bound (`scipy.stats.beta.ppf`; closed form
  `1 - delta**(1/n)` at zero errors, `1.0` at all errors).
- `ece(probs, y, bins=15) -> float` — equal-width-bin ECE over top-1
  predictions; accepts `(n, C)` probs or a 1-D binary confidence array.
- `fit_temperature(logits, y) -> float` — one scalar temperature minimizing the
  NLL of `softmax(logits / T)`, found with bounded deterministic Brent search
  over `log T` in `[log 0.01, log 100]`. 1-D binary logits supported.
- `pick_threshold(conf, correct, target_err=0.05, delta=0.1, min_accepted=30)`
  — fixed-sequence testing exactly as SPEC requires: the distinct confidence
  values are ordered strictest-first BEFORE looking at outcomes, each
  candidate (`conf >= t`) is tested once at level `delta`, the scan STOPS at
  the first candidate that fails (`accepted < min_accepted` or exact
  Clopper-Pearson bound `> target_err`) and returns the last certified one.
  Because the order is pre-fixed, no multiplicity correction is needed and
  the returned threshold keeps its `1 - delta` certificate (a scan-everything
  + pick-loosest selection would not). When even the strictest candidate
  fails, `threshold` is `None` and the other fields report the
  accept-everything point as an uncertified diagnostic summary.
  `correct` must be booleans or 0/1 numbers (strings rejected).
- `risk_coverage(conf, correct)` — per-threshold `{threshold, coverage,
  error}` rows ordered strictest-first (coverage grows along the curve),
  floats rounded to 4 decimals per the SPEC determinism rule.
- `batch_rule(verdicts, k) -> str` — the fixed rule `reject if rejected_count
  > k`; only `reject` counts, `review` does not sink a batch. Accepts a Batch
  mapping, a list of Verdict mappings, or plain verdict strings.

`jadal/receipts.py` (stdlib only):

- `canonical_json(obj) -> bytes` — sorted keys, `(",", ":")` separators,
  UTF-8, non-ASCII unescaped.
- `make_receipt(segment, signals, decisions, verdict, config, created_utc=None)`
  — floats rounded to 4 decimals recursively (`-0.0` normalized to `0.0`);
  the `Config` dataclass becomes `{thresholds, pipeline_version, versions}`
  with a `config_sha256` recorded under `versions`; a mapping must already
  have that shape. `receipt_sha256` hashes everything except `receipt_sha256`,
  `receipt_id`, `created_utc`; `receipt_id` is the first 16 hex chars.
- `receipt_hash(receipt)` / `verify_receipt(receipt, recompute_fn)` returning
  `{match, expected, actual, diff}` with `diff` as sorted dotted field paths
  (lists indexed). When the hashed bodies match but the stored hash was
  tampered, `diff == ["receipt_sha256"]`; an internally inconsistent
  `receipt_id` is reported as `["receipt_id"]`. `make_receipt` rejects
  NaN/inf with a `ValueError` naming the field path. "Signed" here means
  content-hashed, not private-key signed.

## How to run

`/home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v`
(python 3.12.15, numpy 2.4.4, scipy 1.17.1 — nothing installed, as required).

## Test output tail (real, all 73 pass: 43 certify + 30 receipts)

```text
Ran 67 tests in 0.061s

OK
```
Acceptance coverage from the issue:

- `exact_binomial_upper(0, 100, 0.1)` == 0.0228 to 4 places (0.022763), plus
  six more Clopper-Pearson known values, monotonicity checks, bad-input errors.
- Fixed-sequence logic on synthetic data: stop-at-first-failure verified
  against an independent re-derivation, including a case where a
  scan-everything approach would certify a looser threshold (0.5, coverage
  1.0) but the scan stops at 0.9/0.99; a deterministic known-counts case with
  the exact expected bound; `None` when even the strictest candidate fails.
- Temperature fitting: over-confident synthetic logits fit `T > 3`
  (true scale 6) and ECE after scaling is strictly lower; under-confident
  logits fit `T ≈ 1/3`.
- Canonical JSON stable under key order; UTF-8 non-ASCII unescaped.
- Batch determinism: a `FakeBatchDecider.decide_batch` processes all 5
  segments in ONE call (call log asserts `[5]`); those receipts are identical
  to five single-segment `decide` calls, and a batch-of-1 matches the
  batch-of-5 for the same segment. Note (review I3): this proves per-segment
  determinism through a one-call batch path, not a full batch rule — re-run
  the check once `pipeline.run_batch` exists.
- `verify_receipt` on a perturbed `verdict.answer_conf` returns
  `match=False` with `diff == ["verdict.answer_conf"]`, plus a multi-field
  case, a missing-hash case, a tampered-hash-only case with
  `diff == ["receipt_sha256"]`, a tampered-`receipt_id` case, and a
  `created_utc`-only-difference case that still matches.

## Fix round 1 (review of PR #14)

Ruling accepted: the SPEC governs — the scan-everything + keep-loosest
`pick_threshold` did not control the 1-delta error and is gone. Fixes, all in
owned files only:

- C1: `pick_threshold` is now fixed-sequence testing exactly per SPEC
  (pre-fixed strictest-first order, stop at first failure, last certified
  wins), with the FWER rationale in the docstring. Tests rewritten: (a)
  stop-earlier case where a loose scan would certify 0.5/coverage-1.0 but the
  scan stops at 0.99, (b) deterministic known-counts case asserting the exact
  bound `1 - 0.1 ** (1/50)`, (c) `None` when the strictest candidate fails.
- I1: tampered-only `receipt_sha256` now yields exactly
  `diff == ["receipt_sha256"]` (envelope paths are reported only when the
  hashed bodies are identical, so perturbed-body diffs stay exact).
- I2: `make_receipt` rejects NaN/inf with `ValueError` naming the field path
  (`signals.blur_var`, `decisions.step.probs.trim`, `verdict.answer_conf`
  all tested); `-0.0` normalization kept and now asserted as positive zero.
- I3: the batch test now goes through `FakeBatchDecider.decide_batch` (ONE
  call for all 5, asserted via call log) vs single `decide` calls, plus a
  batch-of-1 vs batch-of-5 comparison. Full `pipeline.run_batch` re-check
  still pending (noted above).
- Minors: M1 (+0.0 assert), M2 (duplicate-row doc note), M3 (bool/0-1
  validation of `correct`), M4 (extreme ±1000-logit temperature test), M5
  (`receipt_id` consistency check in verify), M6 (top-level `sys` import in
  tests), M7+I4 (out-of-scope notes in Known gaps).

Command (real output tail):

```text
$ cd /home/parshu/projects/jadal-qa-wt/certify && PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v 2>&1 | tail -30
...
test_tampered_hash_only_reports_the_hash_path (test_receipts.VerifyReceiptTest.test_tampered_hash_only_reports_the_hash_path) ... ok
test_tampered_receipt_id_is_reported (test_receipts.VerifyReceiptTest.test_tampered_receipt_id_is_reported) ... ok

----------------------------------------------------------------------
Ran 73 tests in 0.023s

OK
```

## Known gaps (also in the module docstrings)
- `pick_threshold` assumes comparable confidences and a labelled calibration
  set; no step-type grouping.
- Receipts are content-hashed, not private-key signed; no `receipt_id`
  persistence/lookup (service layer owns that).
- Out of scope for this module (review I4/M7, stated here explicitly): the
  receipt config `versions` records only `python` + `config_sha256` (no torch,
  model names or weight hashes — the pipeline/service owns those), and there
  is no `blurred_sha256` handling (faces-blurring output hashing belongs to
  the L0/pipeline layer).
- No contract changes needed: `docs/SPEC.md`, `jadal/config.py`,
  `jadal/__init__.py` untouched.
