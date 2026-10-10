# PoC dashboard report (feat/poc-dashboard)

Judge-ready, no-build-step dashboard at `web/poc/` (plain HTML/CSS/JS, inline
SVG, no CDNs, works offline), served by `scripts/poc_server.py` (stdlib only,
`127.0.0.1:8840`).

## How to run

```
python3 scripts/poc_server.py --demo-fixture   # fixture data (default: live poc_data)
python3 tests/test_poc_dashboard.py            # 9 tests, ~0.6 s
```

Server reads `poc_data/{summary.json,windows.jsonl,stream_events.jsonl,thumbs/}`
fresh on every request (3 s client poll, `partial run` badge while
`summary.partial` is true). Latest-first gallery page uses a tail read
(`collections.deque`); filtered views do one streaming pass. Thumbnails are
served with basename + realpath containment checks. `POST /api/reverify/<id>`
validates `^[A-Za-z0-9_.-]+$` and execs only
`["python3", "scripts/poc_run.py", "--verify", <id>]` (60 s timeout); with no
`poc_run.py` yet it returns 503 "not available yet".

## Screens (all verified by screenshot in `docs/screens/poc_*`)

1. **Overview** — tiles (windows, hours, accept/review/reject shares, ×/wall-hour,
   GPU label), verdict-by-reason bars, per-factory stacked bars, S0/S1/S2 stage
   split with early-reject share, honesty banner on every screen.
2. **Gallery** — 3-thumb contact sheets, verdict + reason chips (evidence
   `value op threshold` on hover), 5 dropdown filters + search + sort +
   pagination + "Show me examples" pitch set (12 accepted, 12 rejected/reason).
3. **Drawer** — big thumbs, evidence table, features, receipt hash, versions,
   Re-verify button.
4. **Live** — running-score sparkline per stream with the early-reject moment;
   honest placeholder when `stream_events.jsonl` is absent.
5. **About** — inline-SVG pipeline diagram + measured-vs-not-claimed table.

## Honesty

All rates are labelled descriptive on unlabelled video. Accuracy cited only as
AUROC ~0.7–0.8 on the 150 human IndEgo windows (not yet high). Not measured on
a Pi; receipts hashed, not signed; RGB only.

## Status / next

Developed against `tests/fixtures/poc/` (60 windows, contract-exact, generated
by `tests/make_poc_fixture.py`). Still to do: point at real `poc_data` once the
run worker lands (~15–20 min), re-screenshot with real thumbs, confirm Re-verify
against the real `poc_run.py`. No files outside `web/poc/`, `scripts/poc_server.py`,
`docs/reports/poc_dashboard.md`, `docs/screens/poc_*`, `tests/` were touched.

## Real command outputs

```
$ python3 tests/test_poc_dashboard.py
Ran 9 tests in 0.601s — OK
$ curl -s localhost:8840/api/summary | python3 -c "import json,sys; print(sorted(json.load(sys.stdin)))"
['by_factory', 'by_reason', 'by_verdict', 'generated_utc', 'hours', 'has_stream', 'live', 'n_windows', 'partial', 'source', 'throughput']
```
