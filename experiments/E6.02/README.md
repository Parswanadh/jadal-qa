# E6.02 — Per-clip sign calibration from unlabelled windows only (LOCO)

**Why:** pallet is a per-clip sign inversion (raw 0.27) that puts some clips at chance. If the sign can be fixed at inference from the clip's own unlabelled windows, the cue becomes usable without labels.

**Hypothesis:** A label-free per-clip sign/magnitude calibration (majority-prior sign or within-clip |z|) raises the pooled LOCO AUROC above the raw oriented cue.

**Data / label source:** IndEgo human gold, 150 windows / 7 clips (one labeller) for SCORING only; calibration uses only the clip's own unlabelled cue values

**Data regime:** 480p/10fps IndEgo mp4_480

**Method:** leave-one-clip-out direction (6 training clips), per-clip label-free recalibration on the held-out clip's own window values (majority-prior sign; within-clip -|z|); pooled AUROC + clip-bootstrap CI

**Owner:** experimenter  **Started:** 2026-10-10 15:21

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
