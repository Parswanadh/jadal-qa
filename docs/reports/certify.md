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
  — fixed-sequence scan of the distinct confidence values from the strictest
  threshold downward (`conf >= t`); admissible iff `accepted >= min_accepted`
  **and** the exact binomial bound is `<= target_err`. Returns the **loosest**
  admissible candidate (maximum coverage subject to the certified bound).
  When nothing is admissible, `threshold` is `None` and the other fields
  describe the accept-everything point. Note: `pick_threshold` therefore does
  *not* return the strictest passing threshold; this was a deliberate choice
  (documented in the docstring) so the operating point maximizes coverage.
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
  (lists indexed). "Signed" here means content-hashed, not private-key signed.

## How to run

`/home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v`
(python 3.12.15, numpy 2.4.4, scipy 1.17.1 — nothing installed, as required).

## Test output tail (real, all 59 pass)

```text
test_picks_the_strictest_threshold_that_meets_target_and_min_accepted ...
```

Actually run: 40 `test_certify` + 19 `test_receipts` tests, `OK`.
Acceptance coverage from the issue:

- `exact_binomial_upper(0, 100, 0.1)` == 0.0228 to 4 places (0.022763), plus
  six more Clopper-Pearson known values, monotonicity checks, bad-input errors.
- Fixed-sequence logic on synthetic data (threshold == independently
  recomputed loosest-admissible candidate).
- Temperature fitting: over-confident synthetic logits fit `T > 3`
  (true scale 6) and ECE after scaling is strictly lower; under-confident
  logits fit `T ≈ 1/3`.
- Canonical JSON stable under key order; UTF-8 non-ASCII unescaped.
- Batch determinism with a fake deterministic decider: one segment's
  `receipt_sha256`/`receipt_id` are identical alone and inside a batch of five
  (different `created_utc` each time), and every batch member matches its solo run.
- `verify_receipt` on a perturbed `verdict.answer_conf` returns
  `match=False` with `diff == ["verdict.answer_conf"]`, plus a multi-field
  case, a missing-hash case, and a `created_utc`-only-difference case that
  still matches.

## Known gaps (also in the module docstrings)

- `fit_temperature` fits one global scalar, not a per-question-type map.
- `pick_threshold` assumes comparable confidences and a labelled calibration
  set; no step-type grouping.
- Receipts are content-hashed, not private-key signed; no `receipt_id`
  persistence/lookup (service layer owns that).
- No contract changes needed: `docs/SPEC.md`, `jadal/config.py`,
  `jadal/__init__.py` untouched.
