# MR.03 — Low-res mosaic: equal-canvas duel + timestamp/order ablation

**Why:** User claim: 480p tiles, more of them, faster without loss at equal canvas pixels.

**Hypothesis:** At equal canvas px, 16x168px tiles match or beat 9x224px tiles at similar cost.

**Data / label source:** 150 IndEgo gold windows, human, LOCO over 7 clips

**Data regime:** in-domain IndEgo 480p

**Method:** EG2 mosaic embedding; equal-canvas 3x3/224 vs 4x4/168; ablation at 4x4/112

**Owner:** monitor  **Started:** 2026-10-10 16:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:50): maybe.** Equal canvas (~462-465k px): 16x168px (0.845) beats 9x224px (0.785) by +0.060 AUROC at -2.4 ms/win -> supports 'more smaller tiles, no loss'. But mosaic still within 0.016 of mean-of-frames N16 (0.861) and only ties EG2 baseline 0.848. Timestamps help +0.021; order irrelevant.

Results file: eval/results/exp_mosaic_res/MR.03.json

Decision: adopt geometry only if a mosaic input is used; mean-of-frames still preferred

Cost: 66.1 ms/win, 465k canvas px
