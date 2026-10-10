# MR.03 — Low-res mosaic: equal-canvas duel + timestamp/order ablation

**Why:** User claim: 480p tiles, more of them, faster without loss at equal canvas pixels.

**Hypothesis:** At equal canvas px, 16x168px tiles match or beat 9x224px tiles at similar cost.

**Data / label source:** 150 IndEgo gold windows, human, LOCO over 7 clips

**Data regime:** in-domain IndEgo 480p

**Method:** EG2 mosaic embedding; equal-canvas 3x3/224 vs 4x4/168; ablation at 4x4/112

**Owner:** monitor  **Started:** 2026-10-10 16:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
