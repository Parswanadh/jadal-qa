# GRID.03 — Difference-mosaic channel (|I_t - I_{t-1}|) as a post-kill-gate small control

**Why:** Pre-specified in the brief as conditional on the GRID.02 kill-gate passing. The kill-gate FAILED, so this runs only as the small control the brief permits; the feature group was fixed in eval/exp_grid/extract.py before any result was read, but the registry entry is written after the kill-gate outcome is known and is labelled as such.

**Hypothesis:** The motion channel adds nothing: the earlier EmbeddingGemma-2 test found the motion mosaic best in-domain (0.863) and worst out of domain (0.401).

**Data / label source:** same 150 IndEgo windows; 'diff' group (7 features) on m3x3_t224

**Data regime:** 480p ~10 fps IndEgo, CPU only

**Method:** LOCO, same head recipe; paired clip-bootstrap vs spatial+intertile

**Owner:** dsh  **Started:** 2026-10-18 06:43

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 06:43): drop.** Difference-mosaic (|I_t - I_{t-1}|) features on top of spatial+intertile: 0.7002 [0.503,0.778], paired delta vs spatial+intertile -0.006 [-0.016,+0.002], does NOT exclude zero: the motion channel adds nothing. Registered after the GRID.02 kill-gate failed (the small control the brief permits); the feature group and protocol were fixed before the run.

Results file: eval/results/exp_grid/GRID.03.json

Decision: DROP the diff channel. Consistent with the earlier EG2 result (motion mosaic 0.863 in-domain, 0.401 OOD) and with the withdrawn spatial-localisation claim.

Cost: 0 extra (same decode pass)
