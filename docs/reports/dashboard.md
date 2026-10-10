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

- Never exercised against the real service — `jadal/service.py` does not
  exist yet in this worktree. Live-mode response shapes are guessed from SPEC
  with tolerant normalizers; see the contract proposals below.
- No `GET /voice/...` audio heard end-to-end (no voicebank files exist);
  mock mode plays a labelled placeholder tone.
- WS push payload shape is a guess (full Batch object or anything
  batch-like); anything else triggers a soft refresh, which is safe.
- `?api=` base override exists for cross-origin demos but is untested.

## Contract proposals for sibling modules (SPEC needs no change)

`docs/SPEC.md` can stay as-is, but the service agent should confirm these
response envelopes so the dashboard's normalizers can be tightened:

1. `GET /batches` → `{"batches": [Batch, …]}` (or a bare array — both work).
2. **Batch needs a `segments` enrichment**: SPEC's Batch has `segment_ids` +
   `verdicts` only, but the issue asks for worker/site/latency per row.
   Proposal: `segments: [{segment_id, receipt_id, worker_id, site_id,
   source_clip, start_s, end_s, latency_ms}]` alongside them (this is exactly
   what the mock fixtures carry; `latency_ms` = per-segment pipeline time).
3. `GET /stats/risk_coverage` → `{"points": [{threshold, coverage, error}],
   "error_bound", "n", "errors", "target_error", "delta", "min_accepted",
   "chosen_threshold"?}`.
4. `GET /stats/sop_gaps` → `{"gaps": [{reason_code, count, sop_section,
   "share"?}], "total_segments"?, "window_utc"?}`.
5. `GET /pico/status` → `{"connected", "led": "G|A|R|OFF",
   "last_heartbeat_utc", "led_meaning"?}` — the dashboard derives the age
   display from `last_heartbeat_utc`.
