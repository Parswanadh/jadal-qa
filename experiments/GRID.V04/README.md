# GRID.V04 — Run-to-run variance of the model on the same sheets

**Why:** The model is probabilistic; without a variance number, no VLM result can be compared with a deterministic head.

**Hypothesis:** Unanimous share and kappa over the 3 runs are high enough to treat a single run as a measurement, or they are not and the track reports a ranker only.

**Data / label source:** The V04 40-window subset (frozen ids in PREREG), 3 runs each

**Data regime:** Same frozen prompt and sheets across runs; unanimous share and Cohen/Fleiss kappa over usable and over task_progress

**Method:** Unanimous share, kappa with CI, and the per-run AUROC spread.

**Owner:** dsh  **Started:** 2026-10-18 08:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-18 08:52): maybe.** 3 runs of the same frozen quality prompt on the same pre-registered 40 windows with the EXACT same cached PNG bytes, so the only variation is the model. 80 new calls (cap 120); run 1 reused from V02. n=40, n_bad=7 (each single-run CI is therefore wide and uninformative alone). usable unanimous share 0.90 (36/40; votes-true 35x3, 2x2, 2x1, 1x0), Fleiss kappa 0.5195, mean pairwise Cohen kappa 0.5347 (pairwise 0.286/0.787/0.531). task_progress unanimous 0.875, Fleiss kappa 0.410. Per-run AUROC of usable: 0.6429, 0.6991, 0.7143 -> SPREAD 0.0714. Majority vote of 3: AUROC 0.7143 [0.500,0.900], share usable 0.925.

Results file: eval/results/exp_vlm/GRID.V04.json

Decision: A SINGLE CALL CANNOT BE TREATED AS A MEASUREMENT (agreement moderate at kappa ~0.52, and the AUROC itself moves 0.07 between identical runs). Minimum honest usage is repeated calls with a majority vote, and only as triage/ranking feeding a human, never as an automatic verdict. This is consistent with the PREREG fallback rule.

Cost: 80 new image calls of a 120 cap (241 total in the track ledger); 326.8 s wall; no GPU
