# PoC dashboard report (feat/poc-dashboard)

Judge-ready, no-build-step dashboard at `web/poc/` (plain HTML/CSS/JS, inline
SVG, no CDNs, works offline), served by `scripts/poc_server.py` (stdlib only,
`127.0.0.1:8840`).

## How to run

```
python3 scripts/poc_server.py --port 8840                 # real poc_data (JADAL_POC_DATA)
python3 scripts/poc_server.py --demo-fixture --port 8849  # fixture data, for tests
$JADAL_PY tests/test_poc_dashboard.py                      # 19 tests, ~1 s
```

Server reads `poc_data/{summary.json,windows.jsonl,stream_events.jsonl,thumbs/}`
fresh on every request (3 s client poll, `partial run` badge while
`summary.partial` is true). Latest-first gallery page uses a tail read
(`collections.deque`); filtered views do one streaming pass. Thumbnails are
served with basename + realpath containment checks. `POST /api/reverify/<id>`
validates the id (`^[A-Za-z0-9_.][A-Za-z0-9_.-]*$`, so no leading `-`), checks
that Host is `127.0.0.1:<port>` or `localhost:<port>` and that Origin is absent
or the same origin (otherwise 403), and execs only
`[$JADAL_PY, "scripts/poc_run.py", "--verify", <id>]` as an argv list (60 s
timeout). `JADAL_PY` defaults to `python3` on PATH.

## Screens (retaken 2026-10-10 from the current build, port 8845, real poc_data)

Each PNG was captured with headless Chrome (`--headless=new --window-size=W,H
--virtual-time-budget=8000 --screenshot=...`) and viewed. Every 1280-wide screen
shows the out-of-domain banner at the top. Screens with real Egocentric-10K
frames are licensed; see `docs/screens/README.md`.

- `poc_overview_1280.png` (1280x1000), `poc_overview.png` (1280x1800),
  `poc_overview_390.png` (390x844, phone), `poc_overview_dark_1280.png`
  (dark scheme): `#/overview`
- `poc_gallery_1280.png` (`#/gallery`), `poc_gallery_rejects.png`
  (`?verdict=reject#/gallery`)
- `poc_detail_reverify.png` (`?wid=f069_w004_c00077_t0000#/gallery`, drawer open;
  the Re-verify button sits below the fold of this capture)
- `poc_live.png`, `poc_live_1280.png` (`#/live`): show the note "Replay: a real
  clip, with a scripted blackout injected at t=12 s (not from the footage)."
- `poc_about_1280.png` (1280x1500, `#/about`): the claims table and the evidence-file line, which point to the
  evidence files in `docs/POC_STATUS.md`.

Screens of the dashboard:

1. **Overview** — tiles (windows, hours, accept/review/reject shares, ×/wall-hour,
   GPU label), verdict-by-reason bars, per-factory stacked bars, S0/S1/S2 stage
   split with early-reject share, honesty banner on every screen.
2. **Gallery** — 3-thumb contact sheets, verdict + reason chips (evidence
   `value op threshold` on hover), 5 dropdown filters + search + sort +
   pagination + "Show me examples" pitch set. The pitch set is a SELECTION
   (top-12 accepts by score, 12 rejects per reason), not a sample.
3. **Drawer** — big thumbs, evidence table, features, receipt hash, versions,
   Re-verify button.
4. **Live** — running-score sparkline per stream with the early-reject moment.
   The blackout is scripted and labelled as such on screen; honest placeholder
   when `stream_events.jsonl` is absent.
5. **About** — inline-SVG pipeline diagram + measured-vs-not-claimed table.

## Honesty

All rates are labelled descriptive on unlabelled video. Accuracy cited only as
AUROC ~0.7–0.8 on the 150 human IndEgo windows (not yet high). On 60 human-labelled
Egocentric-10K windows the shipped cascade is at chance (AUROC 0.529, exploratory,
n_bad = 9). Not measured on a Pi; receipts hashed, not signed; RGB only.

Evidence files (About table links to these): `eval/results/poc/run_summary.json`
(batch counts), `eval/results/poc/ood_check_60.json` (out-of-domain check),
`eval/results/gpu/cascade.json` (IndEgo LOCO, shipped S1 head).

## Status / next

The dashboard runs on real `poc_data` (1587 windows). Header note: while
`poc_data/summary.json` still has `partial: true` (written by
`scripts/poc_summary.py` without `--complete`), the header shows "partial run".
`eval/results/poc/run_summary.json` is the committed source for the counts and
has `partial: false`. Re-verify runs the real `scripts/poc_run.py --verify`.
Fixture (`tests/fixtures/poc/`) is used only by the tests.

## Real command outputs

```
$ python3 -m unittest tests.test_poc_dashboard
Ran 19 tests — OK
$ curl -s 127.0.0.1:8840/api/summary | python3 -c "import json,sys; print(sorted(json.load(sys.stdin)))"
['by_factory', 'by_reason', 'by_verdict', 'generated_utc', 'hours', 'has_stream', 'live', 'n_windows', 'partial', 'source', 'throughput']
```
