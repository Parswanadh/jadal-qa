# N3.02 — Deterministic regime router: routed system vs single scorers

**Why:** Choose the scorer per regime; report review/error

**Hypothesis:** A descriptor router (resolution/fps/regime) recovers the best scorer per regime and beats every single scorer on mean-domain AUROC

**Data / label source:** IndEgo gold 150 + OOD-160 with E4 regime descriptors

**Data regime:** n/a

**Method:** Deterministic routers (fps, E4 profile, aspect ratio, quality-gate, rules-review) evaluated by AUROC + calibrated-band review/error

**Owner:** N3  **Started:** 2026-10-10 19:47

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 19:59): maybe.** No single scorer wins both regimes (rules 0.822->0.427; adapted 0.684->0.759; frozen 0.834->0.566). E4 descriptors (fps/profile) catch only the 30fps OOD batch (OOD label windows are all 854x480). Aspect-ratio router = 0.822 in-domain / 0.759 OOD (mean 0.791), best of all; review 0.15/0.26. Content quality-gate is a negative (OOD 0.680).

Results file: eval/results/exp_acc/N3.02.json

Decision: Adopt aspect-ratio routing as the best configuration; flag it as a dataset shortcut needing a real in-domain detector

Cost: CPU, cached features, ~55s
