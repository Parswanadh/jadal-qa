# E6.01 — Motion cue conditioning on ego-motion magnitude (terciles + Spearman partial)

**Why:** The residual-motion cue is documented inverted (signed AUROC ~0.20). If it is only 'the wearer is walking', it is an ego-motion magnitude proxy and cannot separate usable windows within a motion level.

**Hypothesis:** The residual-motion cue survives within ego-motion terciles and its Spearman partial correlation with the label controlling for shift_px stays substantially negative.

**Data / label source:** IndEgo human gold, 150 windows / 7 clips (one labeller); cue recomputed from video, ego-motion from caches/perpixel/series_gold.npz shift_px

**Data regime:** 480p/10fps IndEgo mp4_480, ~100 frames per 10 s window

**Method:** per-window mean residual motion after translation-only phase-correlation compensation at 160x90; ego-motion = mean shift_px; within-tercile AUROC + Spearman partial correlation; clip-bootstrap CI (2000)

**Owner:** experimenter  **Started:** 2026-10-10 15:21

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
