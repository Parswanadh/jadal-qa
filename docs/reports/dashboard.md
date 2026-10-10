# dashboard — module report

Judge-facing surface for Jadal QA: three screens in one static page
(`web/index.html`, `web/app.js`, `web/style.css`). Vanilla JS, no build step,
no CDN, no remote assets. Light/dark comes from `prefers-color-scheme` only.
Works with `?mock=1` against `web/mock/*.json`, otherwise live against the
service in `docs/SPEC.md`.

## What exists

- **Batches screen** — live list: batch decision chip, site, created time,
  `rejected / n` count, the fixed reject rule, then per-segment rows with
  verdict chip, reason-code chips, worker, site and per-segment latency.
  Filter chips (All / Rejected / Accepted), newest first, newest…all rows
  shown, older batches collapsed after 5 rows with “Show all N”. WS `/ws`
  pushes merge into the list live (full-batch payload) or trigger a soft
  refresh; reconnects with a 1/2/4/8 s schedule and a “reconnecting…” pill.
- **Segment screen** — verdict + reasons, worker/site, per-segment meta;
  **Re-verify** button → `POST /receipts/{id}/reverify` with a prominent
  MATCH/MISMATCH banner (expected/actual hashes + differing field paths);
  voice player for `GET /voice/{code}/{lang}` with language selector
  (hi-IN kn-IN ta-IN te-IN en-IN) that reads the segment's reason code;
  receipt hashes panel; L0 signals grid; L1 decisions with step-probability
  bars and model versions; pretty receipt JSON with Copy button; an
  “open by receipt id” box for segments whose batch carries no receipt link.
- **Insights screen** — inline-SVG risk/coverage curve from
  `GET /stats/risk_coverage` (error-bound dashed red line, target-error line,
  operating point, stats row with n/errors/δ), and SOP-gap bars from
  `GET /stats/sop_gaps` (code + SOP section + count + share).
- **Pico strip** — `GET /pico/status` polled every 5 s: LED dot (G/A/R/off),
  label, heartbeat age. In mock mode the heartbeat advances each poll and the
  LED follows the newest batch decision.
- **Mock fixtures** — `web/mock/make_mock.py` (seed 1234) generates
  `batches.json` (3 batches, 20 segments), `receipts.json` (real SPEC
  canonical-JSON hashes, batch rule asserted), `risk_coverage.json`
  (error bound 0.0684 = real Clopper–Pearson on the fabricated n=240/errors=12),
  `sop_gaps.json`, `pico_status.json`. All numbers are **illustrative**, every
  mock file says so in a `_note` key, and the footer says so on screen.
- **Design** — calm editorial: paper background, serif display / sans UI /
  mono ids, hairline rules, uppercase micro-labels, generous spacing.
  Green/amber/red are used **only** for verdict chips, the bound line, and the
  Pico LED. Empty, loading-skeleton and error-with-retry states on all screens.

## How to run

```sh
cd web && python3 -m http.server 8123
# then open http://127.0.0.1:8123/?mock=1   (mock fixtures)
# live service: http://127.0.0.1:8123/      (expects jadal/service.py)
```

## Tests — actually run

```sh
/home/parshu/envs/jadal/bin/python web/test_dashboard.py -v
# Ran 11 tests in 0.035s — OK
```

Covers: files exist; no remote assets in HTML/CSS/JS; HTML wiring
(views, tabs, pico strip); dark-mode block + no fixed min-width > 0 that
could force sideways scroll; `node --check` on app.js; every endpoint from
SPEC is called (`/ws /batches /receipts/ /reverify /voice/ /stats/risk_coverage
/stats/sop_gaps /pico/status`); all 5 languages present; batch shapes +
batch-rule consistency; reason codes ⊆ SPEC set; receipt hashes recomputed by
the SPEC rule for all 20 receipts; floats rounded to 4 dp; risk/gaps/pico
shapes.

Visual verification (`docs/screens/take_shots.cjs`, Playwright +
chromium-1243, served locally, mock mode) — real output:

```
batches-1280.png:   1280x900        overflow=0px consoleErrors=0 failedReqs=0
segment-1280.png:   1280x900        overflow=0px consoleErrors=0 failedReqs=0
reverify-1280.png:  MATCH banner renders with expected == actual
insights-1280.png:  1280x900        overflow=0px consoleErrors=0 failedReqs=0
batches-390.png:    390x844         overflow=0px consoleErrors=0 failedReqs=0
segment-390.png:    390x844         overflow=0px consoleErrors=0 failedReqs=0
insights-390.png:   390x844         overflow=0px consoleErrors=0 failedReqs=0
batches-dark-1280.png, segment-dark-1280.png: dark mode, 0 errors
ALL SHOTS OK
```

Screenshots are in `docs/screens/` (9 PNGs + the runner script).

## Known gaps / NOT DONE

- Never exercised against a RUNNING service — `jadal/service.py` is not in
  this worktree. Live shapes were read from `origin/feat/service` and the
  mocks mirror them exactly (fix round 1), but no live HTTP call has been
  made yet.
- No `GET /voice/...` audio heard end-to-end (no voicebank files exist);
  mock mode plays a labelled placeholder tone.
- WS push shape read from the service source (`{type: batch, ...batch,
  receipt_ids}`); `hello`/`pong` ignored, anything else soft-refreshes.
- `?api=` base override exists for cross-origin demos but is untested.

## Fix round 1 (review of PR #15)

Read the real service on `origin/feat/service` (`jadal/service.py`,
`jadal/store.py`, `jadal/pipeline.py`, `docs/reports/service.md`) and the
pico status shape on `origin/feat/pico`. The service is the source of truth;
the mocks and the UI now follow it exactly.

Changed:

- **Mocks mirror live.** `batches.json` is now exactly `GET /batches`
  (`{"batches": [...]}` with the nine store keys only, `b`+16hex ids,
  isoformat `created_utc`; the invented `segments`/`rejected_count`
  enrichment is gone). `risk_coverage.json` is the bare certify list
  `[{threshold, coverage, error}]`, strictest first — no bound metadata, the
  service returns none. `sop_gaps.json` is a bare list
  (`reason_code`+`count` required; `sop_section`/`share` are illustrative
  extras the UI reads opportunistically). `pico_status.json` is the
  `PicoLink.status()` shape (`connected, port, last_hb_age_s,
  last_hb_ticks_ms, last_hb_led`).
- **Segment links use receipt ids only.** Rows show an “Open receipt” button
  that resolves segment → receipt through `GET /batches/{id}.receipt_ids` +
  each receipt's `verdict.segment_id` (cached), exactly as live requires.
  No `receipt_id = segment_id` fallback anywhere.
- **Missing fields render “—”.** Worker, clip path and per-segment latency
  are not exposed by the API; site comes from the batch. See “API needs”.
- **Field names:** `upper_bound` (certify's name) wins, `error_bound` kept as
  legacy fallback; `blurred_sha256` read from the receipt top level;
  no `source_clip` read from signals.
- **Re-verify race:** each request carries `state.rvSeq` + receipt id; stale
  responses are ignored. The button is disabled while busy and when the
  receipt has no `receipt_sha256`.
- **Colour:** green/amber/red now only on verdict chips, batch-decision
  badges and the Pico LED. Live/mock pills, the error-state border, the
  bound line (now accent) and both re-verify banners (✓/✕ + words, ink) are
  neutral. Footer text corrected.
- **390 px:** `overflow-x: hidden` removed from `body`; hashes wrap via
  `overflow-wrap: anywhere`. The 0 px overflow figures below are measured
  with no mask in place.
- **Insights:** “coverage” axis title sits below the tick row; one
  consistent threshold state — “not certified / no certified threshold
  yet”, no bound/target lines, no operating dot — until the payload carries
  `upper_bound`/`chosen_threshold`.
- **Cheap minors:** `aria-pressed` on filter chips; dead CSS removed
  (`select` rule, `.visually-hidden`); tie-safe `localeCompare` sorts;
  voice note says the first reason code; route refetch keyed on
  `requestedId`; WS `hello`/`pong` ignored.

Commands and REAL output tails:

```sh
cd /home/parshu/projects/jadal-qa-wt/dashboard
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s web
# .............
# ----------------------------------------------------------------------
# Ran 13 tests in 0.022s
# OK
node --check web/app.js   # no output = pass
node docs/screens/take_shots.cjs
# batches-1280.png:   1280x900  overflow=0px consoleErrors=0 failedReqs=0
# segment-1280.png:   1280x900  overflow=0px consoleErrors=0 failedReqs=0
# reverify-1280.png:  ✓ MATCH — receipt reproduces (expected == actual)
# reverify-mismatch-1280.png: ✕ MISMATCH — receipt does not reproduce + diff paths
# insights-1280.png:  1280x900  overflow=0px consoleErrors=0 failedReqs=0
# batches-390.png:    390x844   overflow=0px consoleErrors=0 failedReqs=0
# segment-390.png:    390x844   overflow=0px consoleErrors=0 failedReqs=0
# insights-390.png:   390x844   overflow=0px consoleErrors=0 failedReqs=0
# batches-dark-1280.png, segment-dark-1280.png: dark, 0 errors
# ALL SHOTS OK
```

## API needs (for the orchestrator → service worker)

Fields the UI wants that no endpoint provides today. In each case the UI
renders “—” and works without it; nothing is invented client-side.

1. **Per-segment worker, clip path, latency.** Wanted at JSON path
   `GET /batches/{id}.segments[]` with
   `{segment_id, receipt_id, worker_id, source_clip, latency_ms}` —
   or equivalently a `GET /segments/{id}` endpoint returning the stored
   Segment row (`segment_id, source_clip, worker_id, site_id, start_s,
   end_s, path, sha256`). Used by: batch rows (worker/latency) and the
   segment screen (worker/source clip/latency).
2. **Risk bound + threshold metadata.** Wanted alongside the points on
   `GET /stats/risk_coverage`: `upper_bound` (exact-binomial, certify's
   name), `chosen_threshold`, `n`, `errors`, `target_error`, `delta`
   (either top-level keys next to `points`, or a companion object). Used
   by: the error-bound line, the stats row, the operating-point dot.
   Until then the Insights screen shows “no certified threshold yet”.
3. **SOP gap field shapes.** Wanted: a documented row shape for
   `GET /stats/sop_gaps`, at minimum `{reason_code, count}`; `sop_section`
   (SOP reference for the code) would let the bars say where to look.
4. **Cheaper segment→receipt mapping.** Today opening one segment costs
   `GET /batches/{id}` + up to N `GET /receipts/{id}` (cached, fine for
   N=5–10). If the service ever includes `receipt_ids` aligned with
   `segment_ids` order in the batch detail, the UI can drop the N+1 walk.
   Not urgent; the current path is correct either way.
5. **`blurred_sha256` on receipts.** SPEC stores it on the receipt, but
   `receipts.make_receipt` does not emit it yet, so the Hashes card hides
   that row. Wanted: include it (or `null`) when faces were blurred.
