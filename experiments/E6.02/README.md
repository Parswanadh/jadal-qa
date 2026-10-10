# E6.02 — Per-clip sign calibration from unlabelled windows only (LOCO)

**Why:** pallet is a per-clip sign inversion (raw 0.27) that puts some clips at chance. If the sign can be fixed at inference from the clip's own unlabelled windows, the cue becomes usable without labels.

**Hypothesis:** A label-free per-clip sign/magnitude calibration (majority-prior sign or within-clip |z|) raises the pooled LOCO AUROC above the raw oriented cue.

**Data / label source:** IndEgo human gold, 150 windows / 7 clips (one labeller) for SCORING only; calibration uses only the clip's own unlabelled cue values

**Data regime:** 480p/10fps IndEgo mp4_480

**Method:** leave-one-clip-out direction (6 training clips), per-clip label-free recalibration on the held-out clip's own window values (majority-prior sign; within-clip -|z|); pooled AUROC + clip-bootstrap CI

**Owner:** experimenter  **Started:** 2026-10-10 15:21

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 15:48): drop.** n=150 IndEgo human windows, 7 clips, one labeller. LOCO with per-clip sign from the clip's OWN unlabelled windows: raw pooled 0.692 [0.477,0.839]; within-clip -|z| 0.606 [0.525,0.708]; majority-prior sign 0.610 [0.449,0.767]. Mean per-clip AUROC raw 0.664 [0.486,0.809] vs majority-prior 0.691 [0.535,0.808] (CIs overlap); pallet 0.36->0.64 and organise 0.31->0.69 repaired but desk 0.73->0.27 broken. Oracle per-clip sign (uses test labels, UPPER BOUND only) mean 0.757.

Results file: eval/results/exp_motion_cue/E6.02.json

Decision: The pallet inversion cannot be fixed at inference without labels: the label-free sign rule trades one broken clip for two repaired and lowers the pooled AUROC. No per-clip sign calibration in the product.

Cost: no extra extraction (cached cue); 2000-draw clip bootstraps x 4 variants x 4 rules; device A40 (CPU).
