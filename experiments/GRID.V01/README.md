# GRID.V01 — Reading check: does the model read the sheet's tiles and timestamps correctly?

**Why:** The whole track depends on the model actually reading the labels; if it does not, every later verdict is uninterpretable.

**Hypothesis:** Per-tile text agrees with the tile on a human spot check of 20 windows; a timestamp-reading question returns the exact label.

**Data / label source:** 20 context sheets (OOD subset seed 0), one read_tiles call each + timestamp spot check

**Data regime:** 12-tile 4x3 426px sheets, 10 s windows, CPU sheet build + API call

**Method:** One read_tiles call per window; the builder spot-checks the tiles; exact-match rate on the 'tile 7' timestamp question.

**Owner:** dsh  **Started:** 2026-10-18 08:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
