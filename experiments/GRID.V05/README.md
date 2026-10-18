# GRID.V05 — Task level on IndEgo: 12-tile sheets vs video on the same 30 s slices, then L2

**Why:** L1 (is relevant work progressing) is answerable from video alone; L2 (was the task completed) needs the builder's definition. IndEgo's 4 annotated clips are free ground truth.

**Hypothesis:** Sheet L1 agrees with the step annotations at least as well as the video path, and the text-only L2 summariser beats the majority class.

**Data / label source:** 4 annotated IndEgo clips (ecu_full, ecu_late, desk, organise), 30 s slices, clip-absolute tile timestamps; L1 labels derived from the VIA step annotations; L2 from the builder's working definition (pick from one place, put in another, accurately)

**Data regime:** One slice_l1 call per 30 s slice as a sheet and as video via the existing analyze_video path; then ONE l2_text call over the six slice answers per clip

**Method:** L1 agreement vs annotations and vs the always-advancing baseline, kappa, seconds per call, bytes sent, image-sheet vs video; L2 accuracy vs the majority class. Reported as 'builder's definition, not Awign's'.

**Owner:** dsh  **Started:** 2026-10-18 08:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 08:49): drop.** IndEgo 4 annotated clips, 6 evenly spaced 30 s slices each (24 slices, 2 of them no_relevant_work), 12-tile sheets with clip-absolute timestamps vs the same slices as VIDEO. L1 (is relevant work progressing): sheet agreement 0.7083 vs an always-advancing baseline of 0.9167, kappa -0.1351, gap recall 0/2 -- WORSE than a constant; video agreement 0.7917, also below the constant, and it also missed both gap slices. Reading is not the problem: V01 showed 240/240 exact tile+timestamp reads. L2 (was the task completed, BUILDER's definition): the free truth is 4/4 completed = ZERO variance, so no accuracy can be computed; the model answered not_completed once and cannot_tell three times, disagreeing with the annotations on all four clips. Cost: 24 image + 4 text + 24 video calls; sheet 10.04 s/call, video 9.63 s/call, video median 1.06 MB.

Results file: eval/results/exp_vlm/GRID.V05.json

Decision: DROP the VLM as an L1/L2 task judge on this evidence: on IndEgo it does not beat the always-advancing constant, and the L2 truth is too weak (zero variance) to support any claim. The 12-tile sheet and 30 s of video cost the same (~10 s) and perform the same, so the sheet is not a cheaper substitute for video on the task question either. Builder's definition of completion used for L2, NOT Awign's; Awign must confirm what counts as inaccurate.

Cost: 24 image + 4 text + 24 video API calls
