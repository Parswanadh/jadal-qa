# Jadal QA — service API (integration owner contract)

Source of truth: `jadal/service.py` (+ `pipeline.py`, `store.py`) on branch
`feat/integration`. Every shape below was produced by **actually running the
service with stubs** (`JADAL_USE_STUBS=1`, FastAPI TestClient, 5 clips
ingested → 1 batch of 5). Field names, types and nesting match the code
byte-for-byte; key order in JSON responses is not significant.

Base URL (demo): `http://127.0.0.1:8000`. Dashboard-v2 builds against this file.
Errors: `400` clip not found / outside bucket, `404` unknown id / no voice
message, `409` reverify with missing segment row, `500` ingest failure
(detail is always the generic string `"ingest failed"` / `"ingest
unavailable"` — no internals leak).

## POST /ingest/event

Request `{path: str, worker_id: str, site_id: str}` — `path` must resolve
inside the service bucket dir (`realpath` check), else `400`.

```json
{"duplicate": false, "segments": ["seg-c11fecc12cc6-0000"], "pending": 3, "batch": null}
```

- `duplicate: bool` — same (realpath, sha256) already ingested.
- `segments: string[]` — segment ids produced from this clip (stub: 1 per clip).
- `pending: int` — segments buffered across all sites.
- `batch: Batch|null` — the closed batch when this ingest hit `batch_size`,
  else `null`.

## GET /health

```json
{"status": "ok", "pipeline_version": "0.1.0", "pending_segments": 0, "started_utc": "2026-10-10T06:10:54.233520+00:00"}
```

## GET /batches → {"batches": BatchSummary[]}

`BatchSummary` (exactly the nine store keys, no enrichment):

```json
{
  "batch_id": "ba14b9f9a9a631df4",
  "site_id": "s-1",
  "segment_ids": ["seg-c11fecc12cc6-0000", "seg-6eec6ea4b6b3-0000"],
  "verdicts": [{"segment_id": "seg-c11fecc12cc6-0000", "verdict": "accept", "reasons": [], "answer_conf": 0.93, "threshold": 0.9}],
  "batch_decision": "accept",
  "rule": "reject if rejected_count > k",
  "k": 1,
  "n": 5,
  "created_utc": "2026-10-10T06:10:54.290767+00:00"
}
```

- `batch_id: str` — `"b" + sha256(site|segment_ids|k)[:16]`.
- `verdict: "accept"|"reject"|"review"`, `reasons: string[]` (SPEC reason
  codes, canonical order), `answer_conf/threshold: float` (4 dp).
- `batch_decision: "accept"|"reject"` — reject iff rejected_count > k.
  The rule is fixed; it carries no statistical guarantee for n = 5–10.

## GET /batches/{id} → BatchSummary + receipt_ids

Same nine keys as above plus:

```json
{"receipt_ids": ["349abfbc46285d60", "6a008b22959659df"]}
```

`receipt_ids` order follows `ORDER BY receipt_id`, **not** `segment_ids`
order — resolve segment → receipt via each receipt's `verdict.segment_id`.
`404 {"detail": "batch not found"}` when unknown.

## GET /receipts/{id} → Receipt

```json
{
  "segment_sha256": "c11fecc12cc61b2712775db713bd2d2a8291aeac00bf1535a4817dab303fa5ce",
  "signals": {"decodable": true, "blur_var": 120.0, "brightness_mean": 90.0, "under_exposed_frac": 0.0, "over_exposed_frac": 0.02, "motion_mean": 2.0, "hands_visible_frac": 0.8, "hand_count_mode": 2, "faces_found": 0, "faces_blurred": 0, "phash": "ph_x", "dup_of": null},
  "decisions": {"hands_ok": {"p_yes": 0.97}, "active_work": {"p_yes": 0.93}, "step": {"choice": "wash", "probs": {"wash": 0.93}, "answer_conf": 0.93}, "model_versions": {"stub": "0.1.0"}},
  "verdict": {"segment_id": "seg-b70ae1ab3f1e-0000", "verdict": "accept", "reasons": [], "answer_conf": 0.93, "threshold": 0.9},
  "config": {
    "thresholds": {"batch_size": 5, "batch_reject_k": 1, "segment_seconds": 20.0, "seed": 1234, "float_decimals": 4, "blur_var_min": 60.0, "brightness_min": 40.0, "over_exposed_frac_max": 0.35, "motion_max": 12.0, "hands_visible_frac_min": 0.5, "target_error": 0.05, "delta": 0.1, "min_accepted": 30, "answer_threshold": 0.9},
    "pipeline_version": "0.1.0",
    "versions": {"python": "3.12.15", "signals": "stubs.signals", "decision": "stubs.decision", "segmenter": "stubs.segmenter", "certify": "jadal.certify", "receipts": "jadal.receipts"}
  },
  "created_utc": "2026-10-10T06:11:08+00:00",
  "receipt_sha256": "4262773ed2ca299a…(64 hex)",
  "receipt_id": "4262773ed2ca299a"
}
```

- `receipt_sha256` = SHA-256 of canonical JSON (sorted keys, `(",", ":")`,
  UTF-8) of the receipt **without** `receipt_sha256`, `receipt_id`,
  `created_utc`. `receipt_id` = first 16 hex chars. Do not change this rule.
- `versions.*` names the implementation per stage (`stubs.*` until the real
  modules merge; then `jadal.signals` etc. plus backend/model versions).
- `404 {"detail": "receipt not found"}` when unknown.

## POST /receipts/{id}/reverify

Re-runs the pipeline on the stored segment with the receipt's **stored**
config (live-config drift does not affect it) and compares hashes.

```json
{"match": true, "expected": "349abfbc46285d600baaccc640f5a83bb3f8752b79c9f8364b8eee70b463fd00", "actual": "349abfbc46285d600baaccc640f5a83bb3f8752b79c9f8364b8eee70b463fd00", "diff": []}
```

- `match: bool`, `expected/actual: str|null` (hex; `actual` is `null` when
  the segment file is gone), `diff: string[]` (dotted field paths, lists
  indexed; `["<segment file missing>"]` when the file is gone).
- `version_match: bool` — stored receipt `config.versions` vs the running
  code. `version_diff: string[]` — version keys that differ (e.g. after a
  stubs → real-module upgrade, which also changes the hash). Real example:
  `{"match": true, …, "version_match": true, "version_diff": []}`.
- `404` unknown receipt, `409 {"detail": "segment row missing"}`.

## GET /ingest/errors → {"errors": {path: message}}

Last ingest error per clip path (watcher retries with exponential backoff:
2^attempts s, capped at 30 s — a poison file never spins the poll loop).
Real example: `{"errors": {"/…/data_runtime/bucket/poison.mp4":
"RuntimeError: cannot decode"}}`. Empty object when nothing has failed.

## GET /stats/risk_coverage → RiskPoint[]

Bare certify list, strictest threshold first. Sources in order: (1) `calibration`
rows in SQLite, (2) `eval/results/calibration.json` (`risk_coverage` key,
validated row-by-row) when the eval module has produced it. `[]` means **no
measured data** — never fabricated points.

```json
[{"threshold": 0.9, "coverage": 0.8, "error": 0.05}]
```

Shape per point: `{threshold: float, coverage: float, error: float}`.
No bound/threshold metadata is returned today (see "Missing" §2).

## GET /stats/sop_gaps → SopGap[]

Real reason-code counts over stored batch verdicts (until `sop/` merges, when
its module takes over). Sorted by count desc, then code.

```json
[{"reason_code": "BLUR", "count": 3, "share": 0.15}]
```

`reason_code: str` (SPEC code) + `count: int` required; `share: float` =
count / total stored verdicts. `[]` when no batches exist yet.

## GET /sop/search?q=&lang= → {"results": []}

`{"results": []}` until `sop/` merges.

## GET /voice/{code}/{lang} → audio/wav

- `{code}` ∈ the ten SPEC reason codes, `{lang}` ∈
  `hi-IN kn-IN ta-IN te-IN en-IN` (allowlisted; anything else → `404
  {"detail": "no voice message"}`).
- Success: `200`, `Content-Type: audio/wav`, body is the wav file
  (`voicebank/{lang}/{code}.wav`). Missing file → same `404`.
- Real example (header): `HTTP/1.1 200 OK / content-type: audio/wav`,
  body starts with `RIFF`.

## GET /pico/status

```json
{"connected": false, "reason": "no pico on /dev/ttyACM0"}
```

Offline shape (no hardware): `{connected: false, reason: str}`.
Online shape = `PicoLink.status()`: `{connected: true, port: str,
last_hb_age_s: float, last_hb_ticks_ms: int, last_hb_led: str}`.

## WS /ws (events)

Connect → server sends `{"type": "hello", "recent": 1}`, then replays recent
batch events, then pushes one event per closed batch:

```json
{"type": "batch", "batch_id": "ba14b9f9a9a631df4", "site_id": "s-1", "segment_ids": ["seg-…"], "verdicts": [ … ], "batch_decision": "accept", "rule": "reject if rejected_count > k", "k": 1, "n": 5, "created_utc": "…", "receipt_ids": ["349abfbc46285d60"]}
```

Full batch payload plus `receipt_ids`. Client `ping` → server
`{"type": "pong"}`. Anything else: ignore / soft-refresh.

## GET /voice note + GET / (dashboard)

`/` serves `web/` statically when present, else
`{"service": "jadal-qa", "docs": "/docs", "dashboard": "not built yet
(web/ missing)"}`.

## Missing (dashboard "API needs" — UI renders "—" until these land)

1. **Per-verdict enrichment** (`receipt_id`, `source_clip`, `worker_id`,
   `site_id`, `latency_ms`): NOT exposed today. Verdicts carry only
   `{segment_id, verdict, reasons, answer_conf, threshold}`; worker/clip/
   latency live in the internal `segments` table. Planned: `GET
   /segments/{segment_id}` returning the stored Segment row
   (`segment_id, source_clip, worker_id, site_id, start_s, end_s, path,
   sha256`).
2. **Risk metadata** (`upper_bound`, `chosen_threshold`, `n`, `errors`,
   `target_error`, `delta`) next to the points — not returned today.
3. **SOP gap rows** — endpoint returns `[]`; shape above is the contract.
4. **`blurred_sha256`** — not emitted on receipts yet (faces-blur artifact
   lands with the real signals module).
