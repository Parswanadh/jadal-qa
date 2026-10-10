# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| MR.01 | MR | done | adopt | Low-res mosaic: tile-resolution ladder (3x3, 224->64px) | 150 IndEgo gold windows (manifest_gold.json), human, LOCO ov / in-domain IndEgo 480p |
| MR.02 | MR | done | drop | Low-res mosaic: density ladder (tile112, 2x2->6x6) vs mean-of-frames | 150 IndEgo gold windows, human, LOCO over 7 clips / in-domain IndEgo 480p |
| MR.03 | MR | done | maybe | Low-res mosaic: equal-canvas duel + timestamp/order ablation | 150 IndEgo gold windows, human, LOCO over 7 clips / in-domain IndEgo 480p |

## MR.01 — Low-res mosaic: tile-resolution ladder (3x3, 224->64px)
- **Why:** E2 dropped mosaic as model input; test whether lower-res tiles change the verdict and cut cost.
- **Hypothesis:** Smaller tiles (<=112px) at fixed grid keep AUROC within 0.02 of the 224px mosaic at lower canvas cost.
- **Data / label source:** 150 IndEgo gold windows (manifest_gold.json), human, LOCO over 7 clips
- **Regime:** in-domain IndEgo 480p
- **Method:** EG2 mosaic embedding, logistic C=0.01, clip-bootstrap CI B=1000
- **Owner / started / finished:** monitor / 2026-10-10 16:29 / 2026-10-10 16:50
- **Verdict:** adopt — Tiles 112-160px at 3x3 match/better 224px (0.803-0.804 vs 0.785) at -36% ms/win and -74% canvas px; smallest within 0.02 of best = 112px. Still below mean-of-frames N9 0.851. Adopt small tiles for a mosaic pipeline.
- **Decision:** adopt 112-160px tiles at 3x3
- **Cost:** 43.7 ms/win, 118k canvas px @112px vs 68.5 ms/win, 462k px @224px
- **Results:** eval/results/exp_mosaic_res/MR.01.json

## MR.02 — Low-res mosaic: density ladder (tile112, 2x2->6x6) vs mean-of-frames
- **Why:** Does packing more lower-res frames recover the E2 gap vs mean-of-frame embeddings?
- **Hypothesis:** Higher density at small tiles narrows but does not close the mean-of-frames gap.
- **Data / label source:** 150 IndEgo gold windows, human, LOCO over 7 clips
- **Regime:** in-domain IndEgo 480p
- **Method:** EG2 mosaic embedding vs mean-of-frame embeddings, LOCO
- **Owner / started / finished:** monitor / 2026-10-10 16:29 / 2026-10-10 16:50
- **Verdict:** drop — At matched frame count mosaic loses to mean-of-frames at EVERY density (-0.048 N9, -0.063 N16, -0.113 N25, -0.081 N36). Best density 3x3 (0.803). E2 verdict stands; density does not recover the gap.
- **Decision:** keep as evidence artefact only
- **Cost:** 43.7 ms/win @3x3/112 to 107.4 ms/win @6x6/112
- **Results:** eval/results/exp_mosaic_res/MR.02.json

## MR.03 — Low-res mosaic: equal-canvas duel + timestamp/order ablation
- **Why:** User claim: 480p tiles, more of them, faster without loss at equal canvas pixels.
- **Hypothesis:** At equal canvas px, 16x168px tiles match or beat 9x224px tiles at similar cost.
- **Data / label source:** 150 IndEgo gold windows, human, LOCO over 7 clips
- **Regime:** in-domain IndEgo 480p
- **Method:** EG2 mosaic embedding; equal-canvas 3x3/224 vs 4x4/168; ablation at 4x4/112
- **Owner / started / finished:** monitor / 2026-10-10 16:29 / 2026-10-10 16:50
- **Verdict:** maybe — Equal canvas (~462-465k px): 16x168px (0.845) beats 9x224px (0.785) by +0.060 AUROC at -2.4 ms/win -> supports 'more smaller tiles, no loss'. But mosaic still within 0.016 of mean-of-frames N16 (0.861) and only ties EG2 baseline 0.848. Timestamps help +0.021; order irrelevant.
- **Decision:** adopt geometry only if a mosaic input is used; mean-of-frames still preferred
- **Cost:** 66.1 ms/win, 465k canvas px
- **Results:** eval/results/exp_mosaic_res/MR.03.json
