# service module report (issue #3)

Branch: `feat/service`. All work in files I own:
`jadal/service.py`, `jadal/pipeline.py`, `jadal/stubs.py`, `jadal/store.py`,
`sim/dock_sim.py`, `tests/test_service.py`.

## What exists

- **FastAPI app** (`jadal/service.py`, `app = create_app()`). Endpoints per
  docs/SPEC.md: `GET /health`, `POST /ingest/event`, `GET /batches`,
  `GET /batches/{id}`, `GET /receipts/{id}`, `POST /receipts/{id}/reverify`,
  `GET /stats/risk_coverage` (empty until eval data exists),
  `GET /stats/sop_gaps` + `GET /sop/search` (empty until sop/ merges),
  `GET /voice/{code}/{lang}` (serves wav, 404 if missing),
  `GET /pico/status` (offline until `jadal/pico_link.py` merges), `WS /ws`
  (hello + recent-event replay + live batch pushes). Serves `web/` at `/`
  if the directory exists.
- **Batch trigger**: N segments per site (from `jadal/config.py`) or T seconds
  (`JADAL_BATCH_MAX_AGE_S`, default 30) via a polling daemon thread.
- **Bucket watcher**: polls `data_runtime/bucket/`, dedupes by (path, sha256)
  so watcher + explicit `/ingest/event` for the same file ingest once.
- **Pipeline** (`jadal/pipeline.py`): segmenter→L0→L1→L2 verdict→batch rule→receipt,
  strictly per-segment so receipts hash identically solo or in a batch of 5.
  Resolves real `jadal.signals/decision/segmenter/certify/receipts` if present,
  else falls back to `jadal/stubs.py` (deterministic, `JADALFAKE1` clip payloads
  for tests). `batch_id` is a deterministic hash of site+segment ids+k.
- **SQLite** (`jadal/store.py`, `data_runtime/jadal.db`): segments, batches
  (verdicts embedded), receipts. `reverify_receipt` re-reads the stored clip and
  replays the receipt's stored config; returns `{match, expected, actual, diff}`.
- **Dock sim** (`sim/dock_sim.py`): copies clips into the bucket one by one
  with `--delay`, posts `/ingest/event` per clip. stdlib only.

## How to run

```
/home/parshu/envs/jadal/bin/python -m uvicorn jadal.service:app
/home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v
```

## Test output (real, from this env)

`python -m unittest discover -s tests` → `Ran 28 tests ... OK` (0.7s), incl.
ingest-5→batch-with-5-verdicts+5-receipts, reverify match=true (and mismatch on
tampered clip), batch rule k=1 (2 rejects→reject, 1 reject→accept), solo-vs-5
`receipt_sha256` equality, timeout trigger, /ws batch event, GET /batches and
/receipts. Live smoke test with real uvicorn + `sim/dock_sim.py` (5 clips):
batch `b2ff74d85edb7c2e1`, decision accept, reverify match=true.

## Known gaps (NOT DONE)

- L0/L1/segmenter/certify/receipts are stubs; `answer_threshold` is a fixed 0.9
  placeholder (no calibration data). `origin/feat/certify` has `certify.py` but
  **no `jadal/receipts.py` yet** → did not merge per the rules; pipeline will
  pick up the real modules automatically once merged (resolver in `pipeline.py`).
  Next worker step: `git merge origin/feat/certify` after receipts.py lands,
  confirm the 28 tests stay green.
- Verdict reason rules live in `pipeline.make_verdict` (certify API has no
  verdict function); flag if the certify owner wants them moved.
- No contract change proposed (no edits to config/SPEC needed).

## Fix round 1 (review of PR #13)

Merged `origin/feat/certify` (real `jadal/certify.py`, `jadal/receipts.py`) and
`origin/feat/pico` (real `jadal/pico_link.py`); both merged clean, no conflicts.
Correction to the earlier report: `jadal/receipts.py` DID exist on feat/certify.

- Pipeline now uses the real `jadal.certify` / `jadal.receipts` (SPEC hashing:
  canonical JSON with `ensure_ascii=False`, hash excludes
  receipt_sha256/receipt_id/created_utc). Receipts written by the old ASCII stub
  will NOT re-verify under the real module; only fresh DBs match.
- C1: ingest + watcher accept only realpath-inside-`cfg.bucket_dir` (400
  otherwise); tests ingest from the bucket; new outside-bucket 400 test.
- C2: `/voice` allowlists reason_code (`REASON_CODES`) and lang (`LANGS`),
  404 otherwise; traversal (`%2E%2E`, `..`) and unknown code/lang tests added.
- C3: `JADALFAKE1` content parsing deleted. Stubs are used only with
  `JADAL_USE_STUBS=1`, derive values from the file SHA-256 only; exact test
  values come from a programmatic, env-gated override registry. Resolver
  prefers real `jadal.signals/decision/segmenter`, else raises RuntimeError.
- I1: `/pico/status` keeps one real `PicoLink`, `connect(cfg.pico_port)` at
  startup inside try/except, returns `status()`; offline fallback kept.
- I2: dock_sim stages as `.<name>.part` + `os.replace`; watcher skips dotfiles
  / `.part` and ingests only after the same (size, mtime_ns) on two polls.
- I3: clips marked ingested only after success; failures logged, error
  recorded, retry allowed (test: fail once via mock → 500 without internals,
  then same path succeeds).
- I4: flush takes exactly `batch_size` segments; remainder stays buffered
  (test: 7 ingests → batch n=5 at the 5th, 2 pending).
- I5: reverify rebuilds a Config from the receipt's stored thresholds, takes
  the verdict threshold from the stored verdict, and passes the stored config
  block verbatim to `make_receipt` (test: drift live config → still match).
- I8: time trigger stays env-based because `jadal/config.py` is read-only:
  `JADAL_BATCH_MAX_AGE_S` (default 30.0 s). Proposal for the config owner: add
  `batch_max_age_s: float = 30.0` to `Config` (see review I8).
- Cheap minors: watcher/timeout errors logged (M1), SQLite connections closed
  via context manager (M2), lazy store creation (M3), generic 500s (M4),
  dropped the racy empty-batches assertion (M6). M8 (unused site filter) left.

Command: `cd /home/parshu/projects/jadal-qa-wt/service &&
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v 2>&1 | tail -30`
Real tail: `test_voice_rejects_traversal_and_unknowns ... ok`,
`test_watcher_ingests_dropped_files_without_posts ... ok`,
`test_watcher_ignores_part_files ... ok`, `Ran 127 tests in 3.451s`, `OK`
(127 = service suite + merged certify/receipts/pico suites, no regressions).
