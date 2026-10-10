# C1.01 — Confirmatory one-shot: frozen O4 recipe on the 108 round-3 hold-out

**Why:** The O4 recipe was frozen (PR #145 / `configs/ood_recipe_o4.json`) *before* the round-3
labels existed, precisely so it could be scored **once** on a fresh hold-out. The pooled-160
claim (O4 (b) AUROC 0.781) is contaminated by q1 feature selection; this is the pre-registered
confirmatory test.

**Hypothesis:** The frozen O4 recipe (E5 stable-8 + Z3 transferable-6, IndEgo signs, per-domain
z-score, equal-weight mean, threshold 0) stays above chance with a factory-grouped CI excluding
0.5 on the 108-window round-3 hold-out, and beats the frozen two-view ensemble (0.566-class).

**Data / label source:** `labels/round3/ood_labels3_108_with_factory.json` (n=108, 92 good /
16 bad, 44 factories). Never used for tuning. Window videos transcoded from the source clips in
`data/ego10k_scale/{clips,clips_more}/**` with the queue-2 recipe (480p, 10 fps, libx264
veryfast crf 26, no audio); verified to reproduce the frozen q2 features to ~4 decimals.

**Data regime:** Egocentric-10K round-3 hold-out, human labels, evaluation only.

**Method:** FROZEN recipe applied as-is. Per-domain z-score on the 108's own unsupervised
mean/std; signs from the 150 IndEgo gold windows only. AUROC + balanced accuracy at the frozen
threshold; factory-grouped bootstrap CI (honest, n_boot=2000, seed=0). Compared once each with
the frozen two-view ensemble and the sharpness-3 zero-shot head. **Scored once; no tuning.**

**Owner:** C1  **Started:** 2026-10-10 19:40

## Results

See `docs/reports/exp_confirm.md` and `eval/results/exp_confirm/C1.01.json`.

Results file: eval/results/exp_confirm/C1.01.json
