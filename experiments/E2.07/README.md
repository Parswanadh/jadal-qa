# E2.07 — Pooled per-tile statistics control (no layout)

**Why:** Control for E2.01/E2.03: does tiling layout/timestamps help beyond pooled per-tile stats?

**Hypothesis:** concat(mean,std) of N per-tile embeddings is close to the mosaic (layout not load-bearing).

**Data / label source:** same windows/folds; label_source human

**Data regime:** same as E2.01

**Method:** concat(mean,std) of N=9 per-tile EmbeddingGemma-2 embeddings, same eval

**Owner:** experimenter  **Started:** 2026-10-10 15:20

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
