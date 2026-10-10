# E6.04 — Motion cue conditioning robustness (quantiles vs fixed bins) + logistic controls

**Why:** The E6.01 conclusion must not depend on one arbitrary stratification; a vanishing effect must be logged as a negative result.

**Hypothesis:** Within-stratum AUROC and the partial correlation are stable across quantile and fixed-bin strata, and cue+ego logistic does not beat ego alone.

**Data / label source:** IndEgo human gold, 150 windows / 7 clips (one labeller), same cue/ego vectors as E6.01

**Data regime:** 480p/10fps IndEgo mp4_480

**Method:** same strata under 4 rules (terciles, quintiles, fixed 5/15 px, fixed 1/5/20 px) x 4 cue variants; LOCO logistic controls cue-only vs ego-only vs cue+ego

**Owner:** experimenter  **Started:** 2026-10-10 15:21

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
