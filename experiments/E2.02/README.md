# E2.02 — Mean-of-frames control (matched frames)

**Why:** Control: is the mosaic better/worse than the plain average of the SAME N frame embeddings; matched folds.

**Hypothesis:** Mean-of-frames is the accuracy baseline the mosaic must match within 0.02 AUROC.

**Data / label source:** same windows/folds as E2.01; label_source human

**Data regime:** same as E2.01

**Method:** mean of per-frame EmbeddingGemma-2 embeddings (same decoded frames), logistic LOCO

**Owner:** experimenter  **Started:** 2026-10-10 15:20

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
