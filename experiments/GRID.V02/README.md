# GRID.V02 — Quality verdict from a 12-tile sheet on OOD-160, factory-grouped

**Why:** The mosaic has never been shown to a vision-language model. Awign's description ('12 frames in one image') is this configuration. The key question is whether it catches NO_ACTIVE_WORK, which the pixel head misses.

**Hypothesis:** The model separates good from bad factory windows better than the frozen shipped head (AUROC 0.588) with a paired factory-clustered interval excluding zero, and catches a useful share of the 19 NO_ACTIVE_WORK windows.

**Data / label source:** OOD-160: 160 human-labelled Egocentric-10K windows, 138 good / 22 bad (19 NO_ACTIVE_WORK), 47 factories, ONE labeller

**Data regime:** One 10 s window = one 12-tile sheet; frozen quality prompt; factory-grouped bootstrap CI; paired comparison vs the frozen shipped head

**Method:** AUROC, recall of bad, false-reject on good, balanced accuracy, confusion by reason code, usable=true share, paired factory-clustered difference vs the shipped head.

**Owner:** dsh  **Started:** 2026-10-18 08:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 08:46): adopt.** Vision model (muse-spark-1.3-contributor), frozen quality prompt, ONE 4x3 12-tile sheet per window, no training. OOD-160 n=160 (138 good/22 bad; 19 of 22 bad are NO_ACTIVE_WORK; 47 factories; ONE labeller). AUROC of usable = 0.7701 [0.662,0.878] factory-grouped bootstrap (window bootstrap 0.664-0.877, optimistic). NO_ACTIVE_WORK subset (19 bad vs all good, n=157): 0.7641 [0.652,0.887]. PAIRED factory-clustered difference vs the frozen shipped 26-number head (0.5878 [0.450,0.728]): +0.1823 [+0.0428,+0.3141], EXCLUDES ZERO. Natural threshold (usable==true): 13/22 bad rejected (recall 0.5909), 7/138 good rejected (FRR 0.0507), balanced accuracy 0.7701 (positive class = good). HANDS_NOT_VISIBLE 2/2 caught; 8 of 19 NO_ACTIVE_WORK read as OK. Share of ALL answers usable=true = 0.875 (87.5%) - the documented success bias, reported as required. Parse rate 160/160. Cost 161 image calls (the 160 windows plus one smoke call for q000), 11.95 s mean / 11.69 s median per call, 666 s wall.

Results file: eval/results/exp_vlm/GRID.V02.json

Decision: ADOPT the model as a REVIEW-BAND RANKER / second reader on factory footage, NOT as an automatic verdict: AUROC 0.77 is well above the shipped head's 0.588 on the same windows, but 8 of 19 NO_ACTIVE_WORK windows still read as usable and 87.5% of all answers are usable=true, so a naive usable==true gate accepts 9 of 22 bad windows. It is probabilistic, one labeller, no inter-rater kappa; the load-bearing follow-up is V04 (stability) and V03 (does the sheet format carry the signal).

Cost: 161 image calls of a 480 cap; 11.95 s mean per call; 666 s wall; CPU sheet build only, no GPU
