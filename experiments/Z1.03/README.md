# Z1.03 — Few-shot calibration curve + honest self-training for OOD

**Why:** 12 labels already lifted a fresh head to ~0.71 (E5.02); need the error-reduction-per-label curve and test whether self-training helps

**Hypothesis:** A small factory-grouped logistic reaches OOD AUROC >0.7 by ~12 labels; self-training on unlabelled Ego10K gives a free boost

**Data / label source:** OOD-60 human labels + unlabelled Ego10K E8 corpus 1546 windows (no labels)

**Data regime:** EG2 vs pixel features

**Method:** factory-grouped CV (no factory crosses folds) at k in {5,12,24}; pseudo-label high-confidence Ego10K windows, retrain once

**Owner:** Z1  **Started:** 2026-10-10 17:25

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 17:25): maybe.** Factory-grouped few-shot pixel-26: k=12 ->0.717, k=24 ->0.763 (POST-HOC, optimistic; min n_bad train=1). Self-training on unlabelled Ego10K did not help (<=+0.02). Blend zero-shot+few-shot did not beat few-shot.

Results file: eval/results/exp_adapt/Z1.03.json

Decision: ~12 labels matches the label-free sharpness-3 head (0.706); needs 24 to clearly exceed.

Cost: CPU ~3 min
