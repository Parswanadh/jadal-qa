# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| Z1.01 | Z1 | done | maybe | Stable-feature selection for zero-shot OOD transfer | IndEgo gold 150 windows (human) + OOD-60 (human, 51 good/9 b / IndEgo ~10fps 480p vs Egocentric-10K transcodes |
| Z1.02 | Z1 | done | drop | Unsupervised alignment + per-clip normalisation for zero-shot OOD transfer | IndEgo 150 + OOD-60 26 feats + unlabelled Ego10K E8 corpus 1 / IndEgo ~10fps vs Egocentric-10K ~30fps |
| Z1.03 | Z1 | done | maybe | Few-shot calibration curve + honest self-training for OOD | OOD-60 human labels + unlabelled Ego10K E8 corpus 1546 windo / EG2 vs pixel features |

## Z1.01 — Stable-feature selection for zero-shot OOD transfer
- **Why:** IndEgo-trained pixel heads score near chance OOD (0.41-0.53 AUROC); the 26 features include sign-flipping motion/temporal features (E5.03)
- **Hypothesis:** Restricting the head to the 8 sign-stable features (and the 3 sharpness features) cuts OOD error vs all-26
- **Data / label source:** IndEgo gold 150 windows (human) + OOD-60 (human, 51 good/9 bad)
- **Regime:** IndEgo ~10fps 480p vs Egocentric-10K transcodes
- **Method:** LOCO logistic on IndEgo, zero-shot scored on OOD-60; C by inner LOCO
- **Owner / started / finished:** Z1 / 2026-10-10 17:25 / 2026-10-10 17:25
- **Verdict:** maybe — Sharpness-3 head (radial_slope,hf_ratio,dct_energy) cuts OOD AUROC 0.534->0.706 [0.518,0.867] (frozen 0.415); stable-8 only 0.551; in-domain LOCO drops 0.774->0.578. n_bad=9 exploratory.
- **Decision:** Adopt sharpness-3 as the no-label OOD view; keep full-26 for in-domain.
- **Cost:** CPU ~seconds
- **Results:** eval/results/exp_adapt/Z1.01.json

## Z1.02 — Unsupervised alignment + per-clip normalisation for zero-shot OOD transfer
- **Why:** Maybe source/target feature distributions are merely shifted; unlabelled alignment could fix it without labels
- **Hypothesis:** Per-domain standardisation/whitening, CORAL and per-clip normalisation improve OOD AUROC over raw features
- **Data / label source:** IndEgo 150 + OOD-60 26 feats + unlabelled Ego10K E8 corpus 1546 windows (no labels)
- **Regime:** IndEgo ~10fps vs Egocentric-10K ~30fps
- **Method:** per-domain z-score/ZCA, CORAL source<->target, per-clip z/rank on IndEgo; zero-shot OOD eval
- **Owner / started / finished:** Z1 / 2026-10-10 17:25 / 2026-10-10 17:25
- **Verdict:** drop — No unsupervised alignment beat raw: coral_t2s 0.547, stable8+coral_s2t 0.678, domain_whiten worse (0.33-0.37), perclip_z 0.283; Ego10K-corpus CORAL no gain (fps confound).
- **Decision:** Do not ship alignment; report negative.
- **Cost:** CPU ~2 min
- **Results:** eval/results/exp_adapt/Z1.02.json

## Z1.03 — Few-shot calibration curve + honest self-training for OOD
- **Why:** 12 labels already lifted a fresh head to ~0.71 (E5.02); need the error-reduction-per-label curve and test whether self-training helps
- **Hypothesis:** A small factory-grouped logistic reaches OOD AUROC >0.7 by ~12 labels; self-training on unlabelled Ego10K gives a free boost
- **Data / label source:** OOD-60 human labels + unlabelled Ego10K E8 corpus 1546 windows (no labels)
- **Regime:** EG2 vs pixel features
- **Method:** factory-grouped CV (no factory crosses folds) at k in {5,12,24}; pseudo-label high-confidence Ego10K windows, retrain once
- **Owner / started / finished:** Z1 / 2026-10-10 17:25 / 2026-10-10 17:25
- **Verdict:** maybe — Factory-grouped few-shot pixel-26: k=12 ->0.717, k=24 ->0.763 (POST-HOC, optimistic; min n_bad train=1). Self-training on unlabelled Ego10K did not help (<=+0.02). Blend zero-shot+few-shot did not beat few-shot.
- **Decision:** ~12 labels matches the label-free sharpness-3 head (0.706); needs 24 to clearly exceed.
- **Cost:** CPU ~3 min
- **Results:** eval/results/exp_adapt/Z1.03.json
