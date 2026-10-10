# props report — seeded property tests for certify and receipts (issue #36)

## What exists

- `tests/test_properties_certify.py` (10 tests) — seeded `random.Random` property
  tests over `jadal/certify.py`. Every test states its seed; loops are 200-300
  cases; file runs in ~0.3 s.
  - `exact_binomial_upper`: non-decreasing in errors (seed 3601, 300 neighbour
    checks), non-increasing in n for fixed errors (seed 3602, 300 checks),
    always `>= errors/n` and in `[0,1]` (seed 3603, 300 checks), anchor
    `0 errors, n=100, delta=0.1 -> 0.0228` exact to 10 places.
  - `pick_threshold` on random `(conf, correct)` data (quantized levels with
    repeats + a clean top block + adversarial pockets): a certified threshold's
    recomputed exact bound is always `<= target` and matches the reported bound
    (seed 3604, 300 datasets, 236/300 certified — non-vacuous); deterministic
    adversarial case (50 clean @0.99, 5 errors @0.9, 200 clean @0.5) proves the
    scan stops at the first failure and returns 0.99 while a loosest-admissible
    scan would return 0.5; `min_accepted` respected (seed 3605, 300 datasets);
    input-order invariance (seed 3606, 200 datasets).
  - `fit_temperature > 0` and finite, `ece` in `[0,1]` incl. binary 1-D spelling
    (seed 3607, 200 small 40x3 matrices).
  - `batch_rule` monotone in rejected count: accept->reject flips at most once
    (seed 3608, 300 random `(n, k)`).
- `tests/test_properties_receipts.py` (8 tests) — seeded property tests over
  `jadal/receipts.py`; file runs in ~0.2 s.
  - `canonical_json` invariant under recursive dict-key shuffle (seed 3611,
    300 random nested objects incl. unicode); stable across float spellings that
    round to the same 4-decimal value — identical `receipt_sha256` (seed 3612,
    300 cases).
  - `receipt_sha256` changes under EVERY one of 500 random single-leaf
    mutations of the hashed body (seed 3613; excluded keys never mutated);
    mutating `receipt_sha256`/`receipt_id`/`created_utc` does NOT change
    `receipt_hash` (seed 3614, 200 cases).
  - NaN/+inf/-inf at a random float leaf of signals/decisions/verdict raises
    `ValueError` (seed 3615, 200 cases); `-0.0`/tiny negatives become `+0.0`
    and no `-0.0` appears in canonical bytes (seed 3616, 200 cases).
  - Alone-vs-batch-of-5 with a fake deterministic decider (one `decide_batch`
    call): identical `receipt_sha256`/`receipt_id` (seed 3617, 40 batches x 5 =
    200 segments, differing `created_utc`).
  - `verify_receipt` reports `match=False` and names the mutated path in
    `diff` (seed 3618, 200 cases).

## How to run

```
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_properties_certify tests.test_properties_receipts
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests
```

## REAL test output tails

```
$ ... -m unittest tests.test_properties_certify tests.test_properties_receipts
..................
----------------------------------------------------------------------
Ran 18 tests in 0.494s
OK
```

```
$ ... -m unittest discover -s tests
Ran 322 tests in 9.322s
OK
```

(The dock-sim lines about `Connection refused` in the full-suite output are
pre-existing chatter from an unrelated service test, not a failure.)

## Gaps

- Property generators use small synthetic payloads, not real L0/L1 outputs;
  no real-clip coverage here (that belongs to the pipeline/determinism tests).
- `risk_coverage` has no property test (only deterministic unit tests in
  `tests/test_certify.py`); `freshness` receipt field is out of scope.
- Timing margin is large (new files ~0.5 s of a ~9.3 s suite), no hangs (no
  waits/retries anywhere in the new tests).

## BUGS FOUND

None. Every property held on the real code on this run, so no
`unittest.expectedFailure` markers were added.
