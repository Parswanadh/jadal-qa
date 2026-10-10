# dashboard2 — module report (issue #18)

Judge-facing UX rewrite on top of the merged dashboard (`web/` on
`feat/dashboard2`, merged with `origin/main`). Vanilla JS, no build step,
no CDN, no remote assets. Light/dark via `prefers-color-scheme`. Live
against the service, or `?mock=1` against `web/mock/*.json`.

## What exists

- **`web/api.js` (new) — the single adapter.** ALL transport and ALL field
  access live here: `createClient` (`batches`, `batchDetail`, `receipt`,
  `reverify`, `risk`, `gaps`, `pico`, `voiceUrl`, `wsUrl`), normalizers
  (`normBatchList`, `normReceipt`, `normRisk`, `normGaps`, `normPico`,
  `normReverify` incl. `version_match`/`version_diff`), pure filter logic
  (`filterOptions`, `applyFilters`, `matchQuery`), hash router
  (`parseHash`/`buildHash`), `findReceiptForSegment`, `shortHash`.
  `app.js` reaches every operation through `client.*` / `A.*` (asserted by
  tests). Switching to real shapes = one-file change. Also loads under
  node so tests exercise it directly.
- **Global filter bar** — real `<select>`s (site, worker, batch, verdict,
  reason, language) + search box. State lives in the URL hash
  (`#/batches?site=…&verdict=reject&q=…`); bogus values are dropped by the
  parser. Search is debounced; selects apply instantly; Reset clears.
  Worker select is disabled with a tooltip until the API exposes workers.
- **Batches view** — compact cards with native `<details>` expandable rows
  (newest auto-expanded; all expand when filters are on) showing
  per-segment verdict + reason chips and an "Open receipt →" button.
  Meta line counts batches/segments/active filters.
- **Segment view** — verdict + reason chips; Segment/Re-verify grid;
  **collapsible receipt tree** (`<details>` per object, truncated hashes
  with full value on hover, per-hash copy buttons) instead of a JSON dump;
  **Re-verify** with unmistakable ✓ MATCH / ✕ MISMATCH banner (expected /
  actual + diff paths + code-version line when the payload carries it);
  **step timeline** (SOP steps left→right, detected step marked and tinted
  by segment verdict, dot filled when conf ≥ threshold; explicit "No step
  detected — top candidate …" note when `choice` is null); voice card with
  **language `<select>`** + ▶ Play message; Hashes card; compact signals
  grid; open-by-receipt-id box.
- **Insights** — risk/coverage SVG (bound/target lines + operating point
  when the payload carries them, honest "not certified" otherwise),
  **head/model `<select>`** (populates from opened receipts; selects an
  aggregate curve with a labelled note until per-head curves exist),
  SOP-gap bars.
- **System drawer** — `<details>` in the header, collapsed by default:
  Pico LED + heartbeat age + port + ticks, then model versions + pipeline
  version + python/torch/opencv once a segment is open.
- **Polish** — no walls of text anywhere; verdict green/amber/red only on
  verdict chips, the detected timeline step tint, and the Pico LED
  (timeline dots stay neutral ink); `:focus-visible` states; native
  keyboard-accessible `<details>`/`<select>`/tabs with arrow-key tab
  navigation; loading skeletons + empty + error-with-retry states;
  CSS-only transitions; 390 px clean (filter bar wraps 2-up).

## Built against

`docs/API.md` from `origin/feat/integration` (read in full, 204 lines —
it appeared mid-task; checked twice before that: no API doc, built
against `web/mock/*.json` + SPEC). The adapter already matches it:
`{batches: BatchSummary[]}` nine keys, detail `receipt_ids` ordered by
receipt_id (resolve via `verdict.segment_id`), receipt shape, reverify
`{match, expected, actual, diff}` + `version_match`/`version_diff`
(surfaced, added after reading the doc), bare risk/gap lists, pico
offline `{connected:false, reason}`, WS `hello`/`pong`/`batch`.
Its "Missing" §1–4 (per-verdict enrichment, risk metadata,
`blurred_sha256`) is exactly what the UI renders "—" for.

## How to run

```sh
cd web && python3 -m http.server 8123
# mock: http://127.0.0.1:8123/?mock=1
# live:  http://127.0.0.1:8123/   (expects jadal/service.py)
```

## Tests — actually run

```sh
PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python web/test_dashboard.py
# Ran 14 tests in 0.078s — OK
node --check web/api.js && node --check web/app.js   # pass
node docs/screens/take_shots.cjs
```

Real output tail:

```
batches-1280.png:   1280x900        overflow=0px consoleErrors=0 failedReqs=0
segment-1280.png:   1280x900        overflow=0px consoleErrors=0 failedReqs=0
reverify-1280.png:  ✓ MATCH — receipt reproduces (expected == actual)
reverify-mismatch-1280.png: ✕ MISMATCH + diff paths (decisions.step.answer_conf, _mock.forced_mismatch)
insights-1280.png:  1280x900        overflow=0px consoleErrors=0 failedReqs=0
batches-390.png:    390x844         overflow=0px consoleErrors=0 failedReqs=0
segment-390.png:    390x844         overflow=0px consoleErrors=0 failedReqs=0
insights-390.png:   390x844         overflow=0px consoleErrors=0 failedReqs=0
batches-dark-1280.png, segment-dark-1280.png: dark mode, 0 errors
ALL SHOTS OK
```

Plus two extra Playwright probes (not in the runner): accept receipt
`827335a0e262bed8` shows `tl-step is-detected verdict-accept` with flag
"detected · conf 0.91"; `#/batches?verdict=reject` renders from the hash
as "2 batches · 4 segments · 1 filter on" with only reject rows, 0
page errors. Screenshots in `docs/screens/` (11 PNGs + runner):
batches/timeline/filtered extras included.

Covers: files exist (incl. `api.js`); no remote assets across
HTML/CSS/both JS; HTML wiring (views, drawer, filterbar selects,
`<select>`/`<details>` present); dark-mode + no fixed min-width;
colour rule (verdict vars only on verdict/timeline/LED selectors);
endpoints + adapter helpers in `api.js`; `app.js` uses only
`client.*`/`A.*` (no own `fetch(API`/`fetch('mock`); v2 components
present; 5 languages; batch shapes + rule; receipt hashes recomputed
per SPEC; 4 dp floats; risk/gaps/pico shapes; **adapter node cases**
(normalizers, filterOptions, verdict/site/reason/search composition,
hash round-trip + bogus rejection, shortHash, pure receipt resolution).

## Known gaps / NOT DONE

- Never exercised against a RUNNING service — shapes follow the real
  `docs/API.md` but no live HTTP call has been made from this worktree.
- No `GET /voice/...` audio heard end-to-end (mock plays a labelled tone).
- Per-head risk curves need a service change (dropdown is label-only).
- Per-segment worker/clip/latency need `GET /segments/{id}` (renders "—").
- `?api=` cross-origin override untested; WS push against live untested.
- Proposals for other modules: none — no SPEC/config changes needed.

## Fix round 1 (PR #21 review)

All findings fixed in owned files only (`web/*`, `docs/screens/*`,
`docs/reports/dashboard2.md`):

- **CRITICAL `web/api.js` re-verify:** deleted the stray live-mode
  pre-flight GET (its 404 rejection was unhandled on every click).
  Live mode now goes straight to `POST /receipts/{id}/reverify`, and
  every request failure rejects to the caller, which surfaces it via
  `runReverify` ("Re-verify failed" banner).
- **`app.js` route params:** `routeFromHash` no longer interpolates the
  hash into `querySelector` (a value like `head=a"b` threw
  `SyntaxError` and killed routing). Head options are matched with
  `Array.prototype.some` over `sel.options`; unknown heads reset to
  the aggregate view.
- **Threshold absent vs null:** `normRisk` now uses a `present()`
  helper that preserves an explicit `null`. Absent
  `chosen_threshold` renders neutral "not reported / no certified
  threshold reported by the API" (never a verdict); explicit `null`
  renders "not certified"; a number renders the value.
- **Error bound:** the line is drawn only when the API provides
  `upper_bound`/`error_bound`; otherwise no line, no guess — the stats
  row says "— / bound not reported by the API", the chart area shows a
  small "bound not reported" caption, and the legend omits the entry.
- **Adapter-only reads:** `raw.signals` → `r.signals`,
  `segBatch.batch_id/site_id` → `A.first(...)`,
  `r.raw.decisions.hands_ok/active_work.p_yes` → new normalized
  `r.hands_p_yes` / `r.active_work_p_yes` from `normReceipt`. (`raw`
  is still handed to the generic receipt tree, which renders whatever
  keys exist without naming fields.)
- **Copy feedback:** `copyText` now takes the clicked button and shows
  an inline `copy-toast` ("copied", auto-removed after ~1.5 s) next to
  it instead of writing into the unrelated `#voice-note`.
- **UX 1 — Segment above the fold:** order is now verdict + reasons →
  Re-verify (result banner at the top) → step timeline → receipt tree
  → details (Segment facts, Voice, Hashes, Signals) below.
- **UX 2 — chart at 390 px:** 3 + 3 ticks (was 5 + 5), tick labels
  `tick-t` at 19 units with a `48rem` media query shrinking them to
  11/12 units on wide screens (the SVG rescales to ~0.45× on a phone,
  ~1.4× on desktop). No scroll container — the SVG keeps
  `width:100%`.
- **UX 3 — ID vs SHA-256:** labels are now "ID" vs "receipt/segment
  SHA-256"; the ID renders as a bold badge (`hash-id`), SHAs as plain
  mono; values show first-8…last-6 (ID) / 16-char truncate (SHA),
  full value on hover (`title`) and on click/Enter/Space toggle.
- **Screenshots:** re-took every view at 1280 / 390 / dark
  (`node docs/screens/take_shots.cjs`, added the missing
  `insights-dark-1280.png`) and inspected segment-1280,
  insights-1280, segment-390, insights-390: 0 overflow, 0 console
  errors, 0 failed requests on all shots.
- **Tests:** new `FixRound1` class (5 tests): no stray GET, route-param
  validation (source check + node `parseHash` with `head=a"b`),
  absent-vs-null threshold (source check + node `normRisk` cases),
  adapter-only reads, inline copy toast.

Commands + REAL output:

```sh
cd /home/parshu/projects/jadal-qa-wt/dashboard2 && PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s web 2>&1 | grep -E '^(Ran|OK|FAILED|ERROR|FAIL:)'
# Ran 19 tests in 0.122s
# OK
node --check web/app.js web/api.js
# (no output — both pass; checked individually: "api.js: syntax OK", "app.js: syntax OK")
node docs/screens/take_shots.cjs
# batches-1280.png:   1280x900 overflow=0px consoleErrors=0 failedReqs=0
# segment-1280.png:   1280x900 overflow=0px consoleErrors=0 failedReqs=0
# reverify-1280.png:  ✓ MATCH — receipt reproduces (expected == actual)
# reverify-mismatch-1280.png: ✕ MISMATCH + diff paths (decisions.step.answer_conf, _mock.forced_mismatch)
# insights-1280.png:  1280x900 overflow=0px consoleErrors=0 failedReqs=0
# batches-390.png:    390x844  overflow=0px consoleErrors=0 failedReqs=0
# segment-390.png:    390x844  overflow=0px consoleErrors=0 failedReqs=0
# insights-390.png:   390x844  overflow=0px consoleErrors=0 failedReqs=0
# batches-dark-1280.png, segment-dark-1280.png, insights-dark-1280.png (new): dark mode, 0 errors
# ALL SHOTS OK
```

## Files touched (all owned)

`web/api.js` (new), `web/app.js`, `web/index.html`, `web/style.css`,
`web/test_dashboard.py`, `docs/screens/*.png`, `docs/reports/dashboard2.md`.
