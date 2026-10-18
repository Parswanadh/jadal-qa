# GRID.V05 — Task level on IndEgo: 12-tile sheets vs video on the same 30 s slices, then L2

**Why:** L1 (is relevant work progressing) is answerable from video alone; L2 (was the task completed) needs the builder's definition. IndEgo's 4 annotated clips are free ground truth.

**Hypothesis:** Sheet L1 agrees with the step annotations at least as well as the video path, and the text-only L2 summariser beats the majority class.

**Data / label source:** 4 annotated IndEgo clips (ecu_full, ecu_late, desk, organise), 30 s slices, clip-absolute tile timestamps; L1 labels derived from the VIA step annotations; L2 from the builder's working definition (pick from one place, put in another, accurately)

**Data regime:** One slice_l1 call per 30 s slice as a sheet and as video via the existing analyze_video path; then ONE l2_text call over the six slice answers per clip

**Method:** L1 agreement vs annotations and vs the always-advancing baseline, kappa, seconds per call, bytes sent, image-sheet vs video; L2 accuracy vs the majority class. Reported as 'builder's definition, not Awign's'.

**Owner:** dsh  **Started:** 2026-10-18 08:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
