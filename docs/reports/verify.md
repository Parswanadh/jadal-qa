# verify — receipt verification CLI + optional HMAC (issue #35)

## What exists (new files only; `jadal/receipts.py` untouched)

- `jadal/receipt_mac.py` — `mac_for(receipt_sha256_hex, key)` returns hex
  HMAC-SHA256 over the lowercase hex digest's UTF-8 bytes (stdlib `hmac`
  only); `verify_mac(...)` compares with `hmac.compare_digest` (malformed
  MAC values return `False`, never raise). The key comes ONLY from the
  `JADAL_RECEIPT_KEY` env var (UTF-8) or an explicit argument — the module
  has no file-based key loading. The MAC is a sidecar (separate value /
  `.mac` file), never a field inside the hashed receipt.
- `scripts/verify_receipt.py` — `RECEIPT.json [--mac HEX | --mac-file F]
  [--clip PATH [--rerun]] [--json]`. Always recomputes `receipt_sha256`
  via `jadal.receipts.receipt_hash` and checks `receipt_id` (first 16 hex
  chars); names changed paths via `verify_receipt`'s diff against a
  re-derived copy (offline this can only name envelope paths —
  `receipt_sha256`/`receipt_id` — and says so). `--clip` checks the clip
  file's SHA-256 against `segment_sha256`. `--rerun` calls the real
  `jadal.pipeline.reverify_receipt(receipt, segment, Config())` and, in
  addition to its MATCH/MISMATCH, diffs the stored file body against the
  fresh pipeline output so edited fields are named even when the stored
  hash was left intact (in which case re-verify alone reports MATCH — see
  gap G2). Exit codes: 0 all requested checks pass; 1 any mismatch;
  2 usage (bad args, unreadable/malformed receipt, MAC requested but
  `JADAL_RECEIPT_KEY` unset); 3 re-run requested but the real pipeline is
  unavailable (fails clearly, never fakes). `--help` and this report state:
  a hash check proves the receipt was not edited, NOT that the verdict was
  right — re-verifying a VERDICT needs the clip, the pinned weights and
  the pinned stack, and even a MATCH only shows the receipt reproduces on
  this stack, not ground truth.
- `tests/test_receipt_mac.py`, `tests/test_verify_cli.py` — 34 tests:
  fixtures via `jadal.receipts.make_receipt` (offline) and via the stub
  pipeline (`JADAL_USE_STUBS=1`) for `--rerun`; tamper each of 6 content
  fields → exit 1 with the path named; wrong id; MAC good/bad/missing-key;
  stub `--rerun` MATCH; exit 3 on unavailable pipeline (mocked
  `RuntimeError`); unreadable/malformed JSON → exit 2.

## How to run

```
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_receipt_mac tests.test_verify_cli
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/verify_receipt.py RECEIPT.json [--mac-file R.mac] [--clip CLIP.mp4 [--rerun]] [--json]
```

## Real test output tails (ran 2026-10-10, this worktree)

New tests (all fixture clips/receipts are synthetic, built in tmp dirs):

```
Ran 34 tests in 2.713s
OK
```

Full suite:

```
Ran 338 tests in 9.764s
OK
```

CLI on a synthetic stub-pipeline receipt (valid + MAC sidecar + clip + rerun):

```
hash: OK (ea5f6464286396fb...)
id: OK (ea5f6464286396fb)
mac: OK (match)
clip: OK (sha256=e58d9c905087cb96...)
rerun: MATCH expected=ea5f... actual=ea5f...
RESULT: PASS
exit=0
```

Same receipt with `signals.blur_var` edited (hash left intact):

```
hash: MISMATCH stored=ea5f... recomputed=7694...
changed fields: receipt_id, receipt_sha256
rerun: MATCH expected=ea5f... actual=ea5f...
stored file differs from fresh pipeline output at: signals.blur_var
note: the stored hash still reproduces, so the pipeline output is unchanged — the listed fields were edited in the stored file after it was written.
RESULT: FAIL
exit=1
```

Unavailable pipeline / missing file / MAC-without-key:

```
verify_receipt: error: cannot re-run: no L0 signals implementation: jadal.signals.compute_signals is not importable and JADAL_USE_STUBS is not '1'
exit=3
verify_receipt: error: cannot read receipt file '/tmp/verifydemo/nope.json': [Errno 2] No such file or directory
exit=2
verify_receipt: error: MAC requested but no key is set: export JADAL_RECEIPT_KEY (refusing to report OK for a check that cannot run)
exit=2
```

## Gaps / NOT DONE

- G1: An edited `config.*` field is un-nameable by re-run: the pipeline
  replays the stored config block verbatim, so only the envelope
  (`receipt_sha256`) is reported. Covered by a test asserting exit 1.
- G2: By SPEC definition (`reverify_receipt` compares fresh hash vs stored
  hash), an edited file with the stored hash left intact re-verifies as
  MATCH. The CLI compensates with the offline body-hash check (exit 1) and
  the stored-vs-fresh field diff, and says so in the output — but callers
  using `reverify_receipt` directly should know.
- G3: Single active MAC key only; no rotation/multi-key ring.
- G4: `--rerun` smoke-tested on stubs only (`JADAL_USE_STUBS=1`); no
  real-weights re-run measured in this worktree (real L0/L1 not merged).
- G5: A receipt with a missing `receipt_id` is treated as id-OK (matches
  `verify_receipt` semantics); a missing `receipt_sha256` is a mismatch.

## Fix round 1 (review of PR #44; all Critical/Important + Minors fixed)

Review verdict was SPEC: FAIL / CHANGES NEEDED with 2 Important + 3 Minor
findings, all in owned files. No finding was rejected as wrong; the two
reviewer probes were reproduced by new regression tests and now pass.

- Important 1 (`--mac` + malformed stored hash → uncaught `ValueError`
  traceback, exit-1 conflation): `check_mac` now catches `ValueError` from
  `verify_mac` and reports `mac: MISMATCH (receipt hash malformed — cannot
  authenticate)`; overall stays exit 1 via the hash check, no traceback.
  Regression test `test_mac_with_malformed_stored_hash_is_clean_mismatch`.
- Important 2 (`--rerun` errors outside `(RuntimeError, OSError,
  ValueError)` escaped as traceback): `check_rerun` now wraps the
  `reverify_receipt` call in `except Exception → _CannotRerun` (exit 3).
  Regression test `test_unexpected_pipeline_error_is_exit_3_not_traceback`
  (`KeyError` from a mocked pipeline → exit 3, no traceback).
- Minor 1 (tautological key-isolation test): replaced the attribute-name
  check with an AST source assertion — no `open(`/file-read calls and
  imports ⊆ {hmac, os, __future__} in `jadal/receipt_mac.py`.
- Minor 2 (duplicated differ): deleted local `_diff_bodies`; stored-vs-fresh
  diff now reuses `jadal.receipts._diff_paths` (graceful [] fallback).
- Minor 3 (explicit-vs-env whitespace inconsistency): whitespace-only/empty
  explicit keys now count as missing like the env var; non-empty explicit
  keys are used byte-verbatim (documented in `load_key`); tests pin both.

Real outputs (ran 2026-10-10, this worktree, after `git merge origin/main`):

```
Ran 38 tests in 2.333s
OK
```

```
Ran 360 tests in 9.859s
OK
```

Changed files in this round (owned only, `git diff --stat HEAD`):
`jadal/receipt_mac.py`, `scripts/verify_receipt.py`,
`tests/test_receipt_mac.py`, `tests/test_verify_cli.py`. No other worker's
files, `config.py`, `__init__.py`, or `SPEC.md` touched.
