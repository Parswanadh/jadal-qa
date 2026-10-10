# E4.04 — Profiles product layer: schema + 5 profiles + deterministic router

**Why:** Builder requires a customisable product that adapts pipeline to whatever data type it receives

**Hypothesis:** Rule-based router on descriptors selects the right profile with a margin and falls back safely on unseen regimes

**Data / label source:** synthetic descriptors + E4.01 descriptor ranges (n=670)

**Data regime:** all

**Method:** profiles/schema.json, 5 profiles, jadal/regimes/router.py::choose_profile, >=9 unit tests (<30s, no video)

**Owner:** experimenter-a3  **Started:** 2026-10-10 15:18

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 15:30): adopt.** Deterministic router choose_profile + 5 profiles + schema; 10/10 unit tests pass in 0.09 s (synthetic descriptors, determinism, UNSEEN_REGIME -> default, schema conformance). Receipt records profile, margin, reasons, unseen_regime, descriptors_sha. Thresholds calibrated on E4.01 ranges (n=730).

Results file: eval/results/exp_regimes/E4.04.json

Decision: Ship router + profiles; product selects profile per descriptor regime per window

Cost: 10 unit tests <0.5 s, no video, no network
