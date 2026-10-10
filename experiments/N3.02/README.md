# N3.02 — Deterministic regime router: routed system vs single scorers

**Why:** Choose the scorer per regime; report review/error

**Hypothesis:** A descriptor router (resolution/fps/regime) recovers the best scorer per regime and beats every single scorer on mean-domain AUROC

**Data / label source:** IndEgo gold 150 + OOD-160 with E4 regime descriptors

**Data regime:** n/a

**Method:** Deterministic routers (fps, E4 profile, aspect ratio, quality-gate, rules-review) evaluated by AUROC + calibrated-band review/error

**Owner:** N3  **Started:** 2026-10-10 19:47

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
