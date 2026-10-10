SPEC: FAIL, QUALITY: CHANGES NEEDED

Review of PR #15 (feat/dashboard, Parswanadh/jadal-qa) against issue #4 and docs/SPEC.md. Worker tree: <worktrees>/dashboard/. Diff stat: <worktrees>/reviews/dashboard.stat. Nothing modified, merged, pushed or commented.

## Compliance (req 1-8)

1. Vanilla JS, no CDN, no build step: PASS. No remote src/href/@import/url() in index.html, style.css, app.js (grep). `node --check` OK.
2. Three screens: PASS in mock mode. Live path exists for all endpoints, but see Important 1 and 3.
3. Pico strip from GET /pico/status, polled every 5 s (app.js:1404-1405, 1185-1204): PASS.
4. ?mock=1 from web/mock/*.json (app.js:35, 308-318): PASS.
5. Field names: MISMATCHES, see Important 2 and 3.
6. Green/amber/red only for verdicts: FAIL. Used on non-verdict elements: `.pill-live` (style.css:306), `.pill-mock` (:307), Pico LED `.status-led[data-led]` (:211-213), error-state border (:489), chart `.bound` line (:644), re-verify MATCH/MISMATCH banner (:586-596), `.audio-note.is-bad` (:634). The footer text "reserved for verdicts" (index.html:110) is therefore inaccurate.
7. 390 px without horizontal scroll: FAIL in substance. `body { overflow-x: hidden }` (style.css:87) masks overflow, and the 390 screenshot segment-390.png shows the expected/actual SHA-256 in the re-verify banner clipped at the card edge. `.rv-detail` needs `word-break: break-all` or similar. The "overflow=0px" figure in the report is measured with the mask in place and is not evidence.
8. Screenshots: PASS (9 PNGs in dashboard/docs/screens/). Viewed batches-1280, segment-390, insights-1280, batches-390.

## Critical
- None that break safety. innerHTML use is limited to static markup and `esc()`-escaped interpolations (app.js:21-25, 1053); no unescaped API data found.

## Important
1. Live mode cannot work from SPEC data alone. Worker, site and latency come only from a non-SPEC `segments` enrichment (app.js:155-167). Without it every row shows "—" and links fall back to `receipt_id = segment_id` (app.js:157, 179-180), so `#/segment/<segment_id>` calls `GET /receipts/<segment_id>`, which 404s (receipt ids are 16-hex of the sha). Fix: add `segments` (or `receipt_id` per segment) to SPEC Batch, or resolve receipts via `GET /batches/{id}`.
2. Field-name mismatches with SPEC:
   - app.js:702 reads `source_clip` from signals; SPEC has it on the Segment object. Show it from the segment, or fetch it.
   - app.js:773 reads `blurred_sha256` from signals; SPEC says it is "stored in the receipt". Read `r.blurred_sha256` (or update SPEC).
   - app.js:213-233 `normRisk` expects `error_bound`; `certify.pick_threshold` returns `upper_bound`, and `risk_coverage` returns a bare list. Normalize `upper_bound` or fix the endpoint contract.
   - app.js:1057 `chosen_threshold` shows "—" while the chart labels "operating θ=0.00" (app.js:1144). The two disagree in the same screen (insights-1280.png).
3. Re-verify race: `runReverify` (app.js:420-435) writes `state.rv` after an await. If the user navigates to another receipt meanwhile, `loadReceipt` resets `rv`, then the old result lands and a MATCH/MISMATCH for receipt A is shown under receipt B. Fix: compare `id` with `state.receiptId` before assigning, or key `rv` by receipt id.
4. Insights 1280 chart: the x-axis title "coverage" overlaps the "50%" tick (app.js:1102 vs 1111; insights-1280.png). Move the title below the tick row.

## Minor
- app.js:1311 and 1297-ish route logic: when `receiptFallback` is set, `state.receiptId !== id`, so every route change refetches. Store the requested id separately.
- app.js:385-387 and 1229-1231: the sort comparator returns 1 on ties, which is inconsistent. Use `localeCompare` or return 0.
- app.js:420-429: Re-verify with missing `receipt_sha256` still POSTs and shows "—" as expected. Disable the button when the hash is absent.
- app.js:955-971: voice plays only `reasons[0]`; say so in the UI or loop through reasons.
- Filter chips (index.html:68-70) have no `aria-pressed`; the state is only shown by class.
- Dead CSS: `select` rule (style.css, the `select, .langbtn` block) and `.visually-hidden` (style.css:686), which nothing references.
- Tests: `web/test_dashboard.py` passes (11 tests). The `dashboard/tests` path from the review brief does not exist; the tests live under `web/`.

## Design verdict
The screens read as calm and editorial: batches and insights are clear and judge-ready at both widths, and the dark-mode shots look correct. The weak spots are the SHA clipping in the 390 re-verify banner, the "coverage" label collision on the insights chart, and the mock-only "all green" Pico and pill colors that dilute the verdict-only color rule.
