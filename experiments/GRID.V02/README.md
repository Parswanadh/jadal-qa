# GRID.V02 — Quality verdict from a 12-tile sheet on OOD-160, factory-grouped

**Why:** The mosaic has never been shown to a vision-language model. Awign's description ('12 frames in one image') is this configuration. The key question is whether it catches NO_ACTIVE_WORK, which the pixel head misses.

**Hypothesis:** The model separates good from bad factory windows better than the frozen shipped head (AUROC 0.588) with a paired factory-clustered interval excluding zero, and catches a useful share of the 19 NO_ACTIVE_WORK windows.

**Data / label source:** OOD-160: 160 human-labelled Egocentric-10K windows, 138 good / 22 bad (19 NO_ACTIVE_WORK), 47 factories, ONE labeller

**Data regime:** One 10 s window = one 12-tile sheet; frozen quality prompt; factory-grouped bootstrap CI; paired comparison vs the frozen shipped head

**Method:** AUROC, recall of bad, false-reject on good, balanced accuracy, confusion by reason code, usable=true share, paired factory-clustered difference vs the shipped head.

**Owner:** dsh  **Started:** 2026-10-18 08:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
