# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| E2.01 | E2 | done | drop | Mosaic embedding (adopt candidate) | IndEgo gold 150 human windows (LOCO, 7 clips) + OOD 60 human / 480p@10fps IndEgo, 854x480@30fps Egocentric-10K; 10-s windows |
| E2.02 | E2 | done | adopt | Mean-of-frames control (matched frames) | same windows/folds as E2.01; label_source human / same as E2.01 |
| E2.03 | E2 | done | drop | Shuffled tile order | same windows/folds; label_source human / same as E2.01 |
| E2.04 | E2 | done | drop | Mosaic without timestamps | same windows/folds; label_source human / same as E2.01 |
| E2.05 | E2 | done | maybe | Single middle frame | same windows/folds; label_source human / same as E2.01 |
| E2.06 | E2 | done | maybe | Motion mosaic (|I_t - I_{t-1}| tiles) | same windows/folds; label_source human / same as E2.01 |
| E2.07 | E2 | done | drop | Pooled per-tile statistics control (no layout) | same windows/folds; label_source human / same as E2.01 |

## E2.01 — Mosaic embedding (adopt candidate)
- **Why:** Test if ONE tiled timestamped image can replace N sequential frame evaluations, same accuracy at <=1/4 cost.
- **Hypothesis:** Mosaic embedding is within 0.02 AUROC of mean-of-frames at N frames with 1/N the embedding passes.
- **Data / label source:** IndEgo gold 150 human windows (LOCO, 7 clips) + OOD 60 human windows (51 good/9 bad); label_source human
- **Regime:** 480p@10fps IndEgo, 854x480@30fps Egocentric-10K; 10-s windows
- **Method:** EmbeddingGemma-2 image embedding of mosaic, logistic C=0.01 LOCO, clip-bootstrap CI; N in {4,6,9,12,16} tile 224
- **Owner / started / finished:** experimenter / 2026-10-10 15:20 / 2026-10-10 16:07
- **Verdict:** drop — IndEgo LOCO (n=150, 7 clips) mosaic AUROC 0.757-0.842 over N=4..100, always below matched mean-of-frames (0.822-0.860); deltas -0.001..-0.214 (only N30 within 0.02 but OOD 0.155). OOD (n=60, n_bad=9): mosaic 0.155-0.638 vs mean 0.420-0.627, inconsistent. Fails the within-0.02-on-BOTH-regimes bar -> NOT adopted as model input; adopted ONLY as the human-evidence artefact (dashboard why-view). 95% CIs clip-bootstrap, optimistic.
- **Decision:** Do not use mosaic as model input; ship mosaic PNG+sidecar as the reviewer 'why' artefact.
- **Cost:** 1 embed pass/window vs N for mean-of-frames (<=1/4 for N>=4); 224px tiles; build 1.6s/window, embed 1 pass; A40 VRAM peak 3967 MB
- **Results:** eval/results/exp_mosaic/E2.01.json

## E2.02 — Mean-of-frames control (matched frames)
- **Why:** Control: is the mosaic better/worse than the plain average of the SAME N frame embeddings; matched folds.
- **Hypothesis:** Mean-of-frames is the accuracy baseline the mosaic must match within 0.02 AUROC.
- **Data / label source:** same windows/folds as E2.01; label_source human
- **Regime:** same as E2.01
- **Method:** mean of per-frame EmbeddingGemma-2 embeddings (same decoded frames), logistic LOCO
- **Owner / started / finished:** experimenter / 2026-10-10 15:20 / 2026-10-10 16:07
- **Verdict:** adopt — mean-of-frames control on the SAME decoded frames: IndEgo LOCO 0.822-0.860 over N=4..100 (N9 0.852 [0.75,0.91]; N16 0.860 [0.76,0.92]) reproduces the cached EG2 window-embedding reference 0.848 [0.73,0.91]. OOD 0.420-0.627. Confirms the incumbent model-input baseline under the matched folds. n=150 (one labeller) + 60 OOD (n_bad=9, optimistic).
- **Decision:** Keep mean-of-frames EG2 as the model-input baseline; no change.
- **Cost:** N embed passes/window (N frames decoded/mosaic); build ms included; A40
- **Results:** eval/results/exp_mosaic/E2.02.json

## E2.03 — Shuffled tile order
- **Why:** Does temporal/spatial layout carry signal, or is it only the bag of tiles?
- **Hypothesis:** Shuffling tile placement costs little if content dominates; a drop means layout matters.
- **Data / label source:** same windows/folds; label_source human
- **Regime:** same as E2.01
- **Method:** mosaic N=9 tile 224 with deterministic placement permutation (seed 1), same eval
- **Owner / started / finished:** experimenter / 2026-10-10 15:20 / 2026-10-10 16:07
- **Verdict:** drop — shuffled tile placement IndEgo 0.801 [0.64,0.89] vs ordered mosaic N9 0.785 [0.64,0.86]: delta +0.016, inside the CI; OOD 0.423. Tile order does not carry load-bearing signal -> no reason to shuffle.
- **Decision:** Drop shuffled-order input.
- **Cost:** 1 embed pass/window; A40
- **Results:** eval/results/exp_mosaic/E2.03.json

## E2.04 — Mosaic without timestamps
- **Why:** Do the burned-in timestamps add signal beyond the pixels?
- **Hypothesis:** Removing timestamps does not change AUROC much if content dominates.
- **Data / label source:** same windows/folds; label_source human
- **Regime:** same as E2.01
- **Method:** mosaic N=9 tile 224 timestamps=False, same eval
- **Owner / started / finished:** experimenter / 2026-10-10 15:20 / 2026-10-10 16:07
- **Verdict:** drop — no-timestamp mosaic IndEgo 0.810 [0.68,0.88] vs with-timestamps 0.785: delta +0.025 inside CI; OOD 0.444. Burned-in timestamps do not add signal beyond pixels.
- **Decision:** Drop the timestamp-helps hypothesis; keep timestamps only for the human-evidence view.
- **Cost:** 1 embed pass/window; A40
- **Results:** eval/results/exp_mosaic/E2.04.json

## E2.05 — Single middle frame
- **Why:** Lower bound: one frame instead of N (cheapest input).
- **Hypothesis:** Single frame is clearly worse than N-tile mosaic and mean-of-frames.
- **Data / label source:** same windows/folds; label_source human
- **Regime:** same as E2.01
- **Method:** 1x1 mosaic of the middle frame (t=5s), same eval
- **Owner / started / finished:** experimenter / 2026-10-10 15:20 / 2026-10-10 16:07
- **Verdict:** maybe — single middle frame (1x1, t=5s) IndEgo 0.784 [0.69,0.85] ~= mosaic N9 0.785 [0.64,0.86] (1/9 the embedding cost) but 0.068 below mean-of-frames N9 0.852; OOD 0.514. Cheapest input but below the 0.02 bar to the baseline.
- **Decision:** Keep as a cost floor / dashboard thumbnail; not a model input.
- **Cost:** 1 embed pass/window; A40
- **Results:** eval/results/exp_mosaic/E2.05.json

## E2.06 — Motion mosaic (|I_t - I_{t-1}| tiles)
- **Why:** Does a motion-energy mosaic carry the usable-window cue (part may be head stability)?
- **Hypothesis:** Motion mosaic differs from intensity mosaic; may invert (good windows have less residual motion).
- **Data / label source:** same windows/folds; label_source human
- **Regime:** same as E2.01
- **Method:** N=9 tiles of abs consecutive-frame difference, same eval
- **Owner / started / finished:** experimenter / 2026-10-10 15:20 / 2026-10-10 16:07
- **Verdict:** maybe — motion mosaic |I_t-I_{t-1}| N=9 IndEgo 0.863 [0.78,0.91] is the best in-domain variant (mean-of-frames N9 0.852) but OOD 0.401 (fails transfer, like all IndEgo-trained models). Motion/stability cue again looks inverted/regime-specific; part may be head stability, not hand work.
- **Decision:** Maybe for in-domain regime profiles only; never claimed to transfer to Egocentric-10K.
- **Cost:** 1 embed pass/window; A40
- **Results:** eval/results/exp_mosaic/E2.06.json

## E2.07 — Pooled per-tile statistics control (no layout)
- **Why:** Control for E2.01/E2.03: does tiling layout/timestamps help beyond pooled per-tile stats?
- **Hypothesis:** concat(mean,std) of N per-tile embeddings is close to the mosaic (layout not load-bearing).
- **Data / label source:** same windows/folds; label_source human
- **Regime:** same as E2.01
- **Method:** concat(mean,std) of N=9 per-tile EmbeddingGemma-2 embeddings, same eval
- **Owner / started / finished:** experimenter / 2026-10-10 15:20 / 2026-10-10 16:07
- **Verdict:** drop — pooled per-tile stats concat(mean,std) of N=9 tile embeddings IndEgo 0.826 [0.69,0.88], OOD 0.586. Control confirms layout/timestamp are not load-bearing and it does not beat mean-of-frames 0.852.
- **Decision:** Drop as a method; record as the no-layout control.
- **Cost:** 9 embed passes/window; A40
- **Results:** eval/results/exp_mosaic/E2.07.json
