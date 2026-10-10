ALL ADDRESSED: yes, NEW BREAKAGE: none

# Re-review of PR #15 (feat/dashboard) after fix round 1

Scope: fix diff fc8415e..origin/feat/dashboard (= bdf2ea0), saved at reviews/dashboard-fix1.diff. Worker tree is clean at bdf2ea0. No worker files were edited, nothing was pushed or commented.

## Tests (run now)
- `unittest discover -s web`: Ran 13 tests, OK. Matches the report's "13 tests".
- `node --check web/app.js`: pass.

## Original findings
Important 1, live links and missing fields: ADDRESSED. The segment-to-receipt walk runs through GET /batches/{id}.receipt_ids plus each receipt's verdict.segment_id (app.js:362-385). The receipt_id=segment_id fallback is gone. Missing worker, site and latency render "—" (app.js:604, 741). Checked against feat/service: get_batch adds receipt_ids (service.py:388), and the batch row carries verdicts and segment_ids (store.py:127-143).

Important 2a, source_clip: ADDRESSED. No signals.source_clip read remains. Segment screen shows "—" (app.js:741-743).

Important 2b, blurred_sha256: ADDRESSED. Read from the receipt top level (app.js:803-813). The gap is documented in "API needs" item 5.

Important 2c, error_bound vs upper_bound: ADDRESSED. normRisk reads upper_bound first (app.js:198). Certify's upper_bound is not on the live /stats/risk_coverage (a bare list, service.py:410), so the UI shows "no certified threshold yet", which is documented.

Important 2d, chosen_threshold vs operating θ: ADDRESSED. One state: no chosen threshold means "not certified" in the stat, no operating dot, and no label (app.js:1089, 1172-1186).

Important 3, re-verify race: ADDRESSED. loadReceipt bumps rvSeq and resets rv (app.js:419-420). runReverify tags each request with my = ++rvSeq and discards the response unless my === rvSeq and id === state.receiptId (app.js:448-460).

Important 4, "coverage" label overlap: ADDRESSED by coordinates. Tick labels sit at y=T+ih+18 and the axis title at T+ih+40, with H=308 (app.js:1103-1111). I did not re-view the PNG.

Colour rule: ADDRESSED. The --accept/--review/--reject tokens appear only in .verdict-* and .status-led rules (style.css:210-212, 397-399). The pills, bound line and re-verify banner are neutral (style.css:307, 584-600, 647).

390 px overflow: ADDRESSED and verified. Probe at 390x844 with overflow-x:hidden removed: documentElement.scrollWidth = clientWidth = 390 on batches, insights and segment. Re-verify MATCH banner right edge is 357 px, the monospace hashes end at 335 px, and the hashes wrap via overflow-wrap:anywhere.

Minor items: aria-pressed on chips (index.html:68-69), dead select rule and .visually-hidden removed, localeCompare sorts, disabled re-verify button when there is no receipt_sha256 (app.js:756), route refetch keyed on requestedId, hello/pong ignored. All present.

## New Critical/Important breakage
None found.

## Deferred minors
- app.js:9 header comment says mock mode "fabricates only latency/labels", which is stale now.
- docs/screens/take_shots.cjs header says `node docs/screens/take_shots.mjs`; the file is .cjs. The report uses .cjs.
- resolveReceiptId walks receipts sequentially until it matches (N+1); batchDetailCache is never invalidated.
- normSegments pushes verdict-only ids into the raw response array (mutation, harmless).
- loadReceipt's live error path does not clear receiptFallback.
- The colour-rule test is a string match on CSS lines, so it is weak.
- The take_shots figures in the report were not re-run (that script writes into the worker tree). My own probe corroborates the overflow=0 figures.
