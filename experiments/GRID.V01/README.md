# GRID.V01 — Reading check: does the model read the sheet's tiles and timestamps correctly?

**Why:** The whole track depends on the model actually reading the labels; if it does not, every later verdict is uninterpretable.

**Hypothesis:** Per-tile text agrees with the tile on a human spot check of 20 windows; a timestamp-reading question returns the exact label.

**Data / label source:** 20 context sheets (OOD subset seed 0), one read_tiles call each + timestamp spot check

**Data regime:** 12-tile 4x3 426px sheets, 10 s windows, CPU sheet build + API call

**Method:** One read_tiles call per window; the builder spot-checks the tiles; exact-match rate on the 'tile 7' timestamp question.

**Owner:** dsh  **Started:** 2026-10-18 08:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 09:04): adopt.** Reading check PASSES at ceiling: 20/20 pre-registered windows, 240/240 sheet tiles returned a line (coverage 1.000) and 240/240 matched our own sidecar on BOTH tile number and timestamp (exact match 1.000 per sheet tile and per returned line), 0 failed calls, 0 unparsed, 0 out-of-range numbers, 0 right-number/wrong-timestamp lines; mean ~12 s/call; 20 image calls (PREREG cap 20). Independently re-derived by eval/exp_vlm/verify_v01.py (own regex, own sidecar re-read, no shared code): identical 240/240. Re-parsed with both the strict and the tolerant (post-fix) parse_tile_lines: identical, because the count is at ceiling. Negative control: a contentless flat-grey sheet through read_tiles produced 12 'uniform gray blank tile' lines and wrote every timestamp as --:--.- rather than inventing one.

Results file: eval/results/exp_vlm/GRID.V01.json

Decision: ADOPT the sheet FORMAT as readable (scope: readability only). This is a reading measurement and supports NO quality claim: a model that reads perfectly can still judge badly (that is V02/V03). Builder's human spot check of the 20 sheets is OUTSTANDING; every PNG sha256 and raw reply is in the results JSON for it.

Cost: 20 image API calls (V01 PREREG cap 20); ~12 s/call; 0 text calls; sheets built on CPU
