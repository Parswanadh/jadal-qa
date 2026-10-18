# GRID.05 — S0 parity on the mosaic canvas: five shipped rules plus the calibrated blur rule recomputed from mosaic features

**Why:** The mosaic path scores a single composite canvas instead of the frame stream, so its S0 must be reconstructed from canvas statistics; whether it agrees with the shipped 2-second probe on known defects is untested.

**Hypothesis:** BLACK/TOO_DARK/OVEREXPOSED/LENS_BLOCKED parity is achievable from pooled tile luma percentiles and pooled spatial variance; FROZEN needs the inter-tile MAD feature; blur needs the per-tile calibrated sharpness. Agreement on synthetic defect clips is expected to be high for the first four and weaker for FROZEN and blur.

**Data / label source:** scripts/make_demo_clips.py synthetic defect clips generated under /tmp/opencode/ (never in the repo) from two IndEgo source windows, plus the 12 unlabelled clips in interview/demo_kit/; false-reject measured on IndEgo-150 good windows and OOD-160 good windows

**Data regime:** 480p source, 10 s windows, mosaic N=9 3x3 224 px tiles, timestamps drawn and masked

**Method:** jadal.mosaic.s0_parity reconstructs the five shipped rules on the canvas (pixel-pooled tile luma percentiles, pooled spatial variance, it_frozen_frac + inter-tile MAD, pooled lapvar_cal); verdicts compared to scripts/poc_run.py::s0_from_luma + jadal.poc.cascade.s0_decide on the same clips

**Owner:** dsh  **Started:** 2026-10-18 06:50

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 07:30): adopt.** Probe parity on 56 labelled synthetic clips (8 per defect, generated from 4 CALIB IndEgo source clips; black and lens_blocked added because make_demo_clips.py has neither). Per-defect catch (production-5 / production+BLUR / mosaic): black 8/8 / 8/8 / 8/8 all BLACK; lens_blocked 8/8 / 8/8 / 8/8 all LENS_BLOCKED; frozen 8/8 / 8/8 / 8/8 all FROZEN; blur 0/8 / 8/8 / 8/8 all BLUR; too_dark 1/8 / 7/8 / 8/8 (mosaic FROZEN 8/8 - right verdict, wrong rule); overexposed 0/8 all three (the injected gain 1.7+30 never reaches p5>0.70, so OVEREXPOSED parity is UNTESTED not verified); good control 0/8 all. Agreement: five-rules-vs-five 48/56=85.7% [74.3,92.6], six-vs-six 55/56=98.2% [90.6,99.7]; all 8 five-rule disagreements are the too_dark clips. Independent arm on 238 clean windows with defects injected on the canvas: frozen 238/238 caught, blur sigma=3 229/238=96.2% (EVAL 123/132=93.2%). Clean false-reject of the mosaic rules on human-good windows: pooled 18/238=7.56% [4.84,11.64], IndEgo-150 good 7/100=7.00% [3.43,13.75], OOD-160 good 11/138=7.97% [4.51,13.71]; the four INHERITED rules disagree with production on 0/238 clean windows. demo_kit (12 UNLABELLED real uploads, no labels exist - verdicts and agreement only): 11/12 agree; demo_09 BLUR both, demo_10 FROZEN both, demo_11 BLACK both. demo_03 is the one disagreement (production no-fire, mosaic FROZEN): production temp_e=0.003136 is 6x above its 5e-4 threshold while the mosaic inter-tile MAD falls just under its clean-floor 0.0547 on a very dark clip (mean 0.0602) - the same brightness-sensitivity that mis-attributes too_dark. BUG FOUND AND FIXED: for non-square sources, letterbox padding under the burned timestamp box survives features.letterbox_mask, so ~50% luma leaked into the pooled percentiles (bg_leak_frac 0.0823 on OOD-160 vs 0.0127 strict); it flipped the 640x480 all-black demo_11 from BLACK to TOO_DARK. CAVEATS: synthetic defects only; demo_kit unlabelled; per-defect n=8; FROZEN/BLUR thresholds not comparable with the GRID.09 full-pipeline values.

Results file: eval/results/exp_grid/GRID.05.json

Decision: Use jadal.mosaic.s0_parity.mosaic_probe as the mosaic S0: the four brightness/geometry rules inherit the shipped thresholds verbatim; FROZEN and BLUR use separately calibrated thresholds (0.05470779972318289 and 0.013163918969293153, both the clean-CALIB 2nd percentile) because 224px tiles change the units. Also adopt the strict letterbox detector: it fixes a real pooled-percentile contamination on 16:9 sources.

Cost: 56 synthetic defect clips + 12 demo-kit clips + 238 clean windows x 3 mosaic variants, CPU only, 2 workers, nice -n 10, ~4 min wall; clips under /tmp/opencode, no pixels committed
