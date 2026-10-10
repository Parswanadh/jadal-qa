# E2.02 — Mean-of-frames control (matched frames)

**Why:** Control: is the mosaic better/worse than the plain average of the SAME N frame embeddings; matched folds.

**Hypothesis:** Mean-of-frames is the accuracy baseline the mosaic must match within 0.02 AUROC.

**Data / label source:** same windows/folds as E2.01; label_source human

**Data regime:** same as E2.01

**Method:** mean of per-frame EmbeddingGemma-2 embeddings (same decoded frames), logistic LOCO

**Owner:** experimenter  **Started:** 2026-10-10 15:20

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:07): adopt.** mean-of-frames control on the SAME decoded frames: IndEgo LOCO 0.822-0.860 over N=4..100 (N9 0.852 [0.75,0.91]; N16 0.860 [0.76,0.92]) reproduces the cached EG2 window-embedding reference 0.848 [0.73,0.91]. OOD 0.420-0.627. Confirms the incumbent model-input baseline under the matched folds. n=150 (one labeller) + 60 OOD (n_bad=9, optimistic).

Results file: eval/results/exp_mosaic/E2.02.json

Decision: Keep mean-of-frames EG2 as the model-input baseline; no change.

Cost: N embed passes/window (N frames decoded/mosaic); build ms included; A40
