# E6.04 — Motion cue conditioning robustness (quantiles vs fixed bins) + logistic controls

**Why:** The E6.01 conclusion must not depend on one arbitrary stratification; a vanishing effect must be logged as a negative result.

**Hypothesis:** Within-stratum AUROC and the partial correlation are stable across quantile and fixed-bin strata, and cue+ego logistic does not beat ego alone.

**Data / label source:** IndEgo human gold, 150 windows / 7 clips (one labeller), same cue/ego vectors as E6.01

**Data regime:** 480p/10fps IndEgo mp4_480

**Method:** same strata under 4 rules (terciles, quintiles, fixed 5/15 px, fixed 1/5/20 px) x 4 cue variants; LOCO logistic controls cue-only vs ego-only vs cue+ego

**Owner:** experimenter  **Started:** 2026-10-10 15:21

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 15:48): drop.** n=150 IndEgo human windows, 7 clips. Within-stratum oriented AUROC of the residual cue under 4 stratification rules x 4 cue variants: interpretable strata (>=3 bad) span 0.503-0.756 (terciles 0.517/0.682; quintiles 0.503-0.756; fixed 1/5/20 px 0.569-0.806; fixed 5/15 px only 1 interpretable stratum). LOCO logistic control: cue only 0.660, ego-motion magnitude only 0.798, cue+ego 0.793.

Results file: eval/results/exp_motion_cue/E6.04.json

Decision: Negative result logged: the within-motion-level effect is weak and stratum-dependent, and the cue adds nothing on top of ego-motion magnitude (-0.005). Keep the cue only as a global stability descriptor.

Cost: no extra extraction; 4 schemes x 4 variants x 2000-draw bootstraps; device A40 (CPU).
