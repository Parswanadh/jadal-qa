# GRID.02 — Inter-tile temporal-proxy features on the mosaic: added value and leave-one-group-out ablation

**Why:** A still mosaic has no time axis, so the temporal half of the shipped head cannot be computed the same way. The question is whether cheap inter-tile proxies (frame difference, pHash/ZNCC similarity, phase-correlation shift, cross-tile flicker) recover any of it.

**Hypothesis:** Inter-tile proxies add LESS than 0.02 nested-LOCO AUROC over spatial-only mosaic features (the pre-registered kill-gate), because an earlier motion mosaic was best in-domain (0.863) yet collapsed out of domain (0.401) and pooling beat spatial layout.

**Data / label source:** same 150 IndEgo windows; mosaic-native group 'intertile' (12 features) on the m3x3_t224 arm

**Data regime:** 480p ~10 fps IndEgo, CPU only

**Method:** leave-one-group-out ablation (spatial / spatial+intertile / spatial+intertile+diff) with the same LOCO protocol; paired clip-bootstrap on the AUROC difference; kill-gate: stop the mosaic-as-decision-input line if the intertile group adds < 0.02

**Owner:** dsh  **Started:** 2026-10-18 06:38

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 06:43): drop.** Inter-tile temporal-proxy features add +0.019 AUROC over spatial-only (0.7058 [0.514,0.785] vs 0.6864 [0.522,0.772]), paired clip-bootstrap delta +0.019 [-0.018,+0.053] does NOT exclude zero. PRE-REGISTERED KILL-GATE (delta >= 0.02) FAILS: 0.019 < 0.02. Per the brief the mosaic-as-decision-input line STOPS; GRID.03/04 run only as a small control. Best single intertile proxies are the frozen/duplicate/zncc family, not the shift proxies.

Results file: eval/results/exp_grid/GRID.02.json

Decision: KILL-GATE FAILED -> stop the mosaic-as-decision-input line: no full GRID.04 sweep, no adoption path. Keep jadal/mosaic/features.py + tests as the measured answer and the mosaic as an evidence picture. Spend the freed compute on AUTO and on the mosaic's OOD transfer (report-only), which is the part the customer actually asked about.

Cost: same extraction, 0 extra GPU
