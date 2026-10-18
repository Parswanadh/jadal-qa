# GRID.04 — Compact frame-count/layout/density control ladder after a failed kill-gate (incl. the 4x3 / 12-frame headline)

**Why:** The customer's headline configuration is 12 frames in a 4x3 grid. The GRID.02 kill-gate failed, so the brief permits only a SMALL control sweep rather than the full GRID.04 grid. This ladder answers the customer's specific configuration question and measures the Pareto of AUROC vs canvas pixels, without opening an adoption path.

**Hypothesis:** Frame count does not rescue the mosaic: more tiles give at most a small gain, tile resolution matters more than tile count (MR.01/MR.03 found 112-160 px tiles as good as 224 at 3x3), and no arm approaches the 26-number baseline 0.810 (mean-of-frames EG2 reference was 0.852/0.861).

**Data / label source:** IndEgo 150 windows, ONE human labeller, 7 clips; arms: m2x2_t224 (N=4), m4x3_t224 (N=12, HEADLINE), m4x4_t224 (N=16), m4x3_t160 (N=12 density control), m4x3_t112 (N=12 density control); m3x3_t224 (N=9) is the GRID.01/02 arm

**Data regime:** 480p ~10 fps IndEgo, CPU only

**Method:** identical LOCO protocol and head recipe as GRID.01/02; every arm compared with a paired clip-bootstrap interval against the N=9 arm and against the 26-number baseline; canvas px and ms/window reported per arm

**Owner:** dsh  **Started:** 2026-10-18 06:44

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 06:48): drop.** Compact ladder, LOCO n=150, identical features/head/folds per arm. N=4/2x2 0.6750 [0.504,0.762]; N=9/3x3 0.7002 [0.503,0.778]; N=12/4x3 HEADLINE 0.7044 [0.488,0.786]; N=16/4x4 0.6986 [0.516,0.764]; N=12 t160 0.7126 [0.521,0.785]; N=12 t112 0.7146 [0.536,0.774]. No arm differs from N=9 with a paired interval excluding zero (max +0.014 [-0.049,+0.076]); every arm is 0.095-0.135 BELOW the 26-number baseline 0.810. Frame count is flat; smaller tiles are marginally better, consistent with MR.01.

Results file: eval/results/exp_grid/GRID.04.json

Decision: DROP. The customer's headline 12-frame 4x3 mosaic is 0.7044, significantly below the full-window pipeline: automation on a 12-frame picture is not supported by our own extractors. The full sweep was not run (kill-gate).

Cost: ~172 ms/window/arm CPU build+features; 0 GPU; 5 arms x 150 windows = ~2.5 min
