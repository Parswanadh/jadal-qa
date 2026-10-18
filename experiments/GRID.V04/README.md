# GRID.V04 — Run-to-run variance of the model on the same sheets

**Why:** The model is probabilistic; without a variance number, no VLM result can be compared with a deterministic head.

**Hypothesis:** Unanimous share and kappa over the 3 runs are high enough to treat a single run as a measurement, or they are not and the track reports a ranker only.

**Data / label source:** The V04 40-window subset (frozen ids in PREREG), 3 runs each

**Data regime:** Same frozen prompt and sheets across runs; unanimous share and Cohen/Fleiss kappa over usable and over task_progress

**Method:** Unanimous share, kappa with CI, and the per-run AUROC spread.

**Owner:** dsh  **Started:** 2026-10-18 08:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
