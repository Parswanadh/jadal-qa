# dashboard3 — DASHBOARD POLISH ROUND (follow-up to merged PR #21)

Branch: `feat/dashboard3` → PR against `main`. Owned files only:
`web/**`, `docs/screens/**`, `web/test_dashboard.py`, `docs/reports/dashboard3.md`.
No changes to `jadal/config.py`, `jadal/__init__.py`, `docs/SPEC.md`.

## What the re-review asked for, and what changed

### (1) Explicit-null error bound tri-state (the IMPORTANT bug)

`normRisk` already preserves explicit null via `present()`, but every
consumer tested `d.error_bound !== undefined && isFinite(d.error_bound)`.
`isFinite(null)` coerces to `0` → `true`, so `upper_bound: null` drew the
bound line at 0 and suppressed the caption.

Fix: one central gate in `web/api.js`:

```js
function isBound(v) {
  return typeof v === 'number' && isFinite(v);
}
```

All five consumers in `web/app.js` now go through `A.isBound`
(`riskChart` line, insights caption, `riskStatsRow`, `riskLegend`,
chart aria-label). Rendering:

- absent (`undefined`) → no line, caption `bound not reported`;
- explicit `null` → no line, caption `not certified (no threshold meets the target)`;
- finite number → line drawn.

New `Dashboard3` tests in `web/test_dashboard.py` pin the tri-state in
node (`BOUND-OK`) and fail on any bare-`isFinite(error_bound)` line.

### (2) Risk chart at 390 px

Old artboard was 720 units wide (≈0.45× shrink on a phone, 19-unit ticks
→ ~9 px effective) and the `error` axis title collided with the top tick.
New artboard is 480 wide (≈0.75× shrink), endpoints-only ticks (2 + 2),
16-unit type (≈12 px effective at 390), axis titles clear of ticks, and
`.chart svg` is capped at `max-width: 40rem` so desktop does not stretch
the art. Verified by reading `docs/screens/insights-390.png`.

### (3) Segment page order

New order: summary card (verdict chip + segment id + reason chips +
confidence/threshold/site/batch definition list) → Re-verify card (button
+ MATCH/MISMATCH banner, ids unchanged) → compact `Steps` strip (wrapping
chips, no horizontal scroll) → `Receipt detail` tree immediately below →
Segment facts, Voice message, Hashes, Signals each collapsed in
`<details class="more">`. Verified at 1280, 390 and dark.

### (4) No text dumps; funnel; QA queue; Video search

- Audit: no `JSON.stringify` / `<pre>` dumps existed in `app.js`
  (confirmed by grep — nothing found). `rv-detail`/`rv-diff` already
  render structured lists; hashes/signals were already stat grids and are
  now additionally collapsed behind `<details>`.
- Batches leads with a one-line funnel (`20 segments — 12 accept ·
  4 reject · 4 review`) plus a neutral-ink segmented bar (verdict colours
  stay on chips + Pico LED per the project colour rule). Counts come from
  `A.funnelCounts` over the currently filtered rows. Global dropdown
  filters unchanged.
- New `QA queue` view (`#/queue`): review-routed segments via
  `A.queueRows` (ranked by `queue_priority` desc; field absent in fixtures
  → `—` badge, fallback sort verdict → newest batch → id), with a
  reason-code dropdown and an `Open segment →` action reusing the
  segment→receipt resolver.
- New `Video search` view (`#/videos`): query box + site/verdict/reason
  dropdowns, results grid with thumbnails (`client.hubThumbUrl`) and
  on-demand `<video src="…#t=start,end">` (`client.hubVideoUrl`). Mock
  mode projects batch fixtures into labelled placeholders (no footage
  ships); live mode calls `GET /hub/search` through `client.hubSearch`
  (`A.normHub`); a 404 renders `Video hub not available in this
  deployment`. No hub URL literal appears in `app.js` (pinned by test).
- All field access stays in `web/api.js` (`isBound`, `funnelCounts`,
  `queueRows`, `queueReasons`, `normHub`, `hubSearch`, `hubThumbUrl`,
  `hubVideoUrl`, extended `parseHash` routes).

### (5) Screenshots

`docs/screens/take_shots.cjs` now covers all five views at 1280, 390 and
dark-1280 (plus the two re-verify banners), one fresh page per shot so
hash-only navigations cannot leak JS state between shots (this had left a
stale MISMATCH banner in segment shots). Every screenshot was re-taken and
read; fixes applied for what was seen (axis-label collision, stale
mismatch state, tab wrap at 390 handled with hint hiding).

## REAL command outputs

```
$ PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s web -t web
Ran 24 tests in 0.190s
OK
$ node --check web/api.js && node --check web/app.js
node --check: api.js + app.js OK
$ node -e <null-bound probe>
explicit null bound -> error_bound = null
riskChart hasBound (A.isBound) = false
caption = not certified (no threshold meets the target)
absent bound -> isBound = false caption = bound not reported
$ node docs/screens/take_shots.cjs
batches-1280.png: 1280x900 overflow=0px consoleErrors=0 failedReqs=0
queue-1280.png: 1280x900 overflow=0px consoleErrors=0 failedReqs=0
videos-1280.png: 1280x900 overflow=0px consoleErrors=0 failedReqs=0
segment-1280.png: 1280x900 overflow=0px consoleErrors=0 failedReqs=0
reverify-1280.png: ✓ MATCH — receipt reproduces
reverify-mismatch-1280.png: ✕ MISMATCH — receipt does not reproduce
insights-1280.png: 1280x900 overflow=0px consoleErrors=0 failedReqs=0
batches-390.png: 390x844 overflow=0px consoleErrors=0 failedReqs=0
queue-390.png: 390x844 overflow=0px consoleErrors=0 failedReqs=0
segment-390.png: 390x844 overflow=0px consoleErrors=0 failedReqs=0
videos-390.png: 390x844 overflow=0px consoleErrors=0 failedReqs=0
insights-390.png: 390x844 overflow=0px consoleErrors=0 failedReqs=0
batches-dark-1280.png, queue-dark-1280.png, videos-dark-1280.png,
segment-dark-1280.png, insights-dark-1280.png: overflow=0 consoleErrors=0 failedReqs=0
ALL SHOTS OK
```

NOT DONE:
- `queue_priority` is absent in fixtures, so queue order falls back to
  verdict → newest batch → id until the API sends the field.
- Video hub is mock placeholders only; live playback is untested against a
  real `/hub` (no such deployment here).
- `tests/` suite (`unittest discover -s tests`) was not run: outside owned
  files and untouched by this diff.
