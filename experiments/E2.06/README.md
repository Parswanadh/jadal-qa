# E2.06 — Motion mosaic (|I_t - I_{t-1}| tiles)

**Why:** Does a motion-energy mosaic carry the usable-window cue (part may be head stability)?

**Hypothesis:** Motion mosaic differs from intensity mosaic; may invert (good windows have less residual motion).

**Data / label source:** same windows/folds; label_source human

**Data regime:** same as E2.01

**Method:** N=9 tiles of abs consecutive-frame difference, same eval

**Owner:** experimenter  **Started:** 2026-10-10 15:20

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:07): maybe.** motion mosaic |I_t-I_{t-1}| N=9 IndEgo 0.863 [0.78,0.91] is the best in-domain variant (mean-of-frames N9 0.852) but OOD 0.401 (fails transfer, like all IndEgo-trained models). Motion/stability cue again looks inverted/regime-specific; part may be head stability, not hand work.

Results file: eval/results/exp_mosaic/E2.06.json

Decision: Maybe for in-domain regime profiles only; never claimed to transfer to Egocentric-10K.

Cost: 1 embed pass/window; A40
