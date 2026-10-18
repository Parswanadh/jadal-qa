# GRID.V03 — Ablations: timestamps, tile order, single frame, tile count, reasoning step, grey null

**Why:** V02 could be explained by any single frame, by the labels, or by the reasoning step rather than by the multi-frame sheet.

**Hypothesis:** Removing timestamps, shuffling tile order, or dropping to one frame changes the verdict materially; the grey null sheet is not read as usable.

**Data / label source:** The V03 stratified 20 windows (10 bad / 10 good, frozen ids in PREREG)

**Data regime:** Six arms x 20 windows: no timestamps, shuffled order, one middle frame, 6 tiles, 24 tiles, JSON-only prompt; plus a flat grey null sheet

**Method:** Per-arm AUROC and paired differences against the V02 primary arm on the same windows; the null must read cannot_tell or usable=false.

**Owner:** dsh  **Started:** 2026-10-18 08:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
