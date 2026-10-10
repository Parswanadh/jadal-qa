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
