# integration — module report (issue #17)

Owner of `jadal/service.py`, `jadal/pipeline.py`, `jadal/store.py`,
`jadal/stubs.py`, `sim/dock_sim.py`, `scripts/`, `Makefile`, `docs/API.md`,
`tests/test_service.py`, `tests/test_integration.py` after the first merges.

## What exists

- **docs/API.md** — exact JSON shape (field names, types, one real example
  each) of every endpoint, produced by running the service with stubs:
  `POST /ingest/event`, `GET /health`, `/batches`, `/batches/{id}`
  (+`receipt_ids`), `/receipts/{id}`, `POST /receipts/{id}/reverify`
  (+`version_match`/`version_diff`), `/stats/risk_coverage`,
  `/stats/sop_gaps`, `/sop/search`, `/voice/{code}/{lang}`, `/pico/status`,
  `WS /ws`, `GET /ingest/errors`. Dashboard-v2 builds against it. The
  "Missing" section lists the dashboard API needs not yet exposed
  (per-verdict `receipt_id/source_clip/worker_id/site_id/latency_ms`,
  risk bound metadata, `blurred_sha256`).
- **Real-module wiring** (`pipeline._load_stage`, unchanged design):
  `jadal.signals.compute_signals`, `jadal.decision.Decider`,
  `jadal.segmenter.split_clip` are preferred when importable; stubs only
  under `JADAL_USE_STUBS=1`, else a clear `RuntimeError`. Every model call is
  per segment (batch size 1); `ensure_determinism()` pins seeds +
  torch deterministic kernels (all guarded).
- **Version provenance** — every receipt `config.versions` now carries the
  L0 backend report (`signals.backend_versions()`, else `backend_info()`,
  JSON-encoded as `signals_backend`) and the face/hand weight-file hashes
  (`face_detector_task`/`hand_landmarker_task` as `name:sha256:<hex>`,
  hashed once and cached), plus `pipeline_version` (was already there).
- **Real-data stats** — `/stats/risk_coverage`: SQLite calibration rows
  first, else validated `eval/results/calibration.json`, else `[]` (the
  honest "no measured data" state — never fabricated). `/stats/sop_gaps`:
  real reason-code counts over stored verdicts
  (`[{reason_code, count, share}]`, count desc); the `sop/` module takes over
  when it merges.
- **Deferred minors fixed** — failed batch requeues its segments at the
  front of the site buffer (never dropped; caller still sees the 500);
  watcher retries bad files with exponential backoff (2^attempts s, cap
  30 s, so a poison file cannot spin); errors exposed at
  `GET /ingest/errors`; reverify compares stored vs running versions
  (`version_match`, `version_diff`).
- **`make demo`** (`Makefile` + `scripts/demo.sh`) — free port, fresh
  `data_runtime/`, 7 catalog IndEgo clips symlinked and docked through
  `sim/dock_sim.py`, verdict table, reverify of every receipt, 10x same-hash
  assert + alone-vs-batch-of-5 assert, batch latency print.
- **`tests/test_integration.py`** — 12 tests: version provenance + weight
  tokens, `compare_versions`, 10x determinism, solo-vs-5, failed-batch
  retry, watcher backoff + `/ingest/errors`, empty-risk honesty, eval-file
  parsing (malformed row dropped), sop-gap counts, reverify version keys.

## How to run

```sh
make test   # full suite (stubs; no hardware, no network)
make demo   # end-to-end on the IndEgo clips
make service
```

## Tests — actually run

```sh
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests
# Ran 225 tests in 7.713s — OK
# (38 service + 12 integration + certify/receipts/pico suites from main)
```

`make demo` real output tail (2026-10-10, service on :48619, stubs):

```
batches closed: 2 (7 clips -> batch of 5 + timeout-flushed 2)
batch b2a9096c13dd07127 site=site-demo decision=REJECT rule='reject if rejected_count > k' k=1 n=5
  seg-675a30d26e33-0000 reject  conf=0.9300 reasons=[HANDS_NOT_VISIBLE]
  ...
batch bbf76e31013a6558d site=site-demo decision=ACCEPT rule='reject if rejected_count > k' k=1 n=2
  ...
== reverify (7 receipts) ==
  11bad92f6c2d1b70 MATCH versions_match=True
  ... (7/7 MATCH, versions_match=True)
all receipts reverify MATCH
determinism: same clip 10x -> identical a333ed9af8f9142f...
determinism: alone vs batch-of-5 -> IDENTICAL receipt_sha256
batch latency: dock-to-batches wall time = 10.9s (7 clips, 2 batches, includes 5s timeout flush)
demo OK
```

Real-module interop (unmerged `feat/signals` code, scratch sandbox in
`/tmp/opencode/sigtest`, nothing written to this worktree): `impl_versions`
detects `jadal.signals`/`jadal.segmenter` as real, records `signals_backend`
from `backend_info()`; real segmenter split a 95 s IndEgo clip into 5
segments; real L0 returned the exact SPEC signals key set; the SPEC hash rule
recomputed clean. `jadal.decision` does not exist yet on `feat/decider`, so
L1 stays on the stub fallback by design.

## Known gaps / NOT DONE

- L0/L1 values in the demo are hash-derived stub fakes (whole clip = one
  segment); the verdicts above (HANDS_NOT_VISIBLE/SHAKY) reflect stub bytes,
  not clip content. Real values arrive when `feat/signals` merges to main
  (verified compatible, see above) — no pipeline change needed.
- `jadal.decision.Decider` does not exist yet (`feat/decider` = main); L1
  stays on stubs until it lands. Built against the SPEC interface.
- `eval/results/` is absent in this worktree, so `/stats/risk_coverage`
  returns `[]` (no measured data). Note: the evalharness calibration file is
  built from synthetic demo predictions (auto-labels), so even after it
  merges, its points are eval-module output, not pipeline measurements.
- Dashboard "API needs" enrichment (§Missing in docs/API.md) is documented,
  not implemented: per-verdict `receipt_id/source_clip/worker_id/site_id/
  latency_ms` (proposed: `GET /segments/{id}`), risk bound metadata,
  `blurred_sha256`. No receipt hashing rule was changed.
- `sop/` module absent: `/sop/search` returns `{"results": []}`.
- No GPU used; no secrets in code; no network at import.
- Proposals for other owners: none (no config/SPEC changes needed).
