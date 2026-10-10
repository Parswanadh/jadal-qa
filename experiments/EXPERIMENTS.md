# Experiment log

Every experiment: why it was run, on what data, what came out, and the decision. Generated from `experiments/registry.jsonl` (append-only). Negative results are kept.

| id | track | status | verdict | title | data / regime |
|---|---|---|---|---|---|
| E5.04 | E5 | done | drop | Frozen ensemble re-score on 160 OOD windows (UPDATED by O2.01: labels2 delivered) | 160 Egocentric-10K human windows (138/22, labels/ood160_labe / 480p 10s transcodes |
| O2.01 | E5 | done | drop | E5.04 final: frozen ensemble on 160 human OOD labels (factory-grouped CI) | 160 Egocentric-10K human-labelled windows (138 good/22 bad,  / 480p 10s transcodes; ~1 window/factory |
| O2.02 | E5 | done | maybe | Few-shot adaptation curve on 160 OOD labels (factory-grouped) | 160 Egocentric-10K human windows (138/22, 47 factories; labe / 480p 10s, 10 fps-effective |
| O2.03 | E5 | done | maybe | E5.03 refresh: per-feature signed AUROC on the 160 OOD labels | 150 IndEgo human gold windows (manifest_gold) + 160 Egocentr / IndEgo native ~10fps vs OOD 10 fps-effective |

## E5.04 — Frozen ensemble re-score on 160 OOD windows (UPDATED by O2.01: labels2 delivered)
- **Why:** ood_labels2.json is now delivered; the confirmatory frozen-ensemble OOD eval that E5.04 left inconclusive can now run.
- **Hypothesis:** Frozen IndEgo ensemble does not transfer; OOD AUROC near chance with a CI that includes 0.5.
- **Data / label source:** 160 Egocentric-10K human windows (138/22, labels/ood160_labels.json) + frozen ensemble_v1 scores
- **Regime:** 480p 10s transcodes
- **Method:** frozen usable_prob vs labels; factory-grouped cluster bootstrap (see O2.01)
- **Owner / started / finished:** O2 / 2026-10-10 17:41 / 2026-10-10 17:41
- **Verdict:** drop — SUPERSEDED by O2.01 (same analysis, committed in eval/results/exp_fewshot/O2.01.json). 160 OOD windows, frozen ensemble_v1 usable_prob: AUROC 0.566, factory-grouped CI [0.445,0.674] includes 0.5, bal.acc 0.546. Was inconclusive (labels2 not delivered).
- **Decision:** E5.04 inconclusive -> drop. Frozen ensemble is not a standalone OOD gate; adapt with O2.02.
- **Cost:** CPU; cached scores
- **Results:** eval/results/exp_fewshot/O2.01.json

## O2.01 — E5.04 final: frozen ensemble on 160 human OOD labels (factory-grouped CI)
- **Why:** ood_labels2.json is now delivered, so E5.04 can leave 'inconclusive' and give the confirmatory frozen-ensemble OOD number with the honest factory-grouped CI.
- **Hypothesis:** The frozen IndEgo ensemble does not transfer: its 160-window OOD AUROC is near chance and the factory-grouped CI includes 0.5.
- **Data / label source:** 160 Egocentric-10K human-labelled windows (138 good/22 bad, 47 factories; labels/ood160_labels.json = q1 60 + q2 100) + frozen ensemble_v1 scores (caches/ensemble/ood_scores.json + ood_scores2.json)
- **Regime:** 480p 10s transcodes; ~1 window/factory
- **Method:** frozen usable_prob vs human labels; AUROC + balanced accuracy with factory-grouped cluster bootstrap (2000) and (optimistic) window bootstrap
- **Owner / started / finished:** O2 / 2026-10-10 17:40 / 2026-10-10 17:40
- **Verdict:** drop — 160 OOD windows (138/22, 47 factories), frozen ensemble_v1 usable_prob: AUROC 0.566, factory-grouped CI [0.445,0.674] (includes 0.5), window-bootstrap CI [0.437,0.694], bal.acc@0.5 0.546 [0.460,0.635], acc 0.481. Batch caveat: q1(old 60) 0.415 vs q2(new 100) 0.657. Frozen ensemble is NOT a reliable standalone OOD scorer.
- **Decision:** Update E5.04 inconclusive -> drop. Do not use the frozen ensemble as an OOD gate; adapt (O2.02).
- **Cost:** CPU; 160 cached frozen scores; 0 new GPU scoring
- **Results:** eval/results/exp_fewshot/O2.01.json

## O2.02 — Few-shot adaptation curve on 160 OOD labels (factory-grouped)
- **Why:** Given the frozen ensemble does not transfer, quantify how much human OOD labelling recovers: the error-vs-labels curve.
- **Hypothesis:** A small logistic trained on the 26 per-pixel features recovers usable OOD signal within tens of labels; the frozen ensemble views do not.
- **Data / label source:** 160 Egocentric-10K human windows (138/22, 47 factories; labels/ood160_labels.json); 26 per-pixel features (frozen 160x90 recipe, 10 fps-effective) + frozen view scores p_eg2/p_pixel
- **Regime:** 480p 10s, 10 fps-effective
- **Method:** factory-grouped CV: repeated grouped subsampling (train ~k / test factory-disjoint complement) for k in {10,20,40,80}; StratifiedGroupKFold(5) OOF for k=160; inner leave-one-factory-out picks C; class_weight=balanced
- **Owner / started / finished:** O2 / 2026-10-10 17:40 / 2026-10-10 17:40
- **Verdict:** maybe — Factory-grouped curve. pixel-26 logistic: k=10 0.544/err0.489, 20 0.609/0.452, 40 0.650/0.415, 80 0.701/0.389, 160 grouped-CV 0.689 [0.510,0.847]/err0.395. Frozen-view arms stay at chance (platt 0.49-0.56, two-view 0.47-0.52). Best label-efficient recipe = logistic on 26 pixel features; ~40-80 labels matches the 160-label level. Frozen 0-label baseline 0.566/0.454.
- **Decision:** Adapt with the 26-pixel logistic; spend labels on pixel features, not on ensembling frozen views. Operating point ~40-80 labels (AUROC 0.65-0.70, err 0.39-0.42).
- **Cost:** CPU; 160 windows; 20 repeats/k; no GPU
- **Results:** eval/results/exp_fewshot/O2.02.json

## O2.03 — E5.03 refresh: per-feature signed AUROC on the 160 OOD labels
- **Why:** The what-transfers table was built on 60 OOD windows (n_bad=9); refresh it against the 160 labels (n_bad=22) and check whether the stable/flip/chance calls hold.
- **Hypothesis:** The sharpness/spatial-spectrum features keep their sign; motion/temporal-band features still flip; counts barely move with more labels.
- **Data / label source:** 150 IndEgo human gold windows (manifest_gold) + 160 Egocentric-10K human windows (ood160_labels.json); 26 per-pixel features, frozen 160x90 recipe
- **Regime:** IndEgo native ~10fps vs OOD 10 fps-effective
- **Method:** signed AUROC per feature; IndEgo clip-bootstrap CI, OOD160 factory-grouped CI; classify_transfer stable/flip/chance; compare class vs the old 60-label table
- **Owner / started / finished:** O2 / 2026-10-10 17:40 / 2026-10-10 17:40
- **Verdict:** maybe — 160-label table: 8 stable / 14 flip / 4 chance (same counts as the 60). Only 2/26 change class: frozen_frac flip->stable, global_shift stable->flip. Sharpness trio stable (radial_slope 0.633/0.635, hf_ratio 0.618/0.657, dct_energy 0.631/0.700 with grouped CI [0.55,0.83]); motion_energy still flips (0.308->0.646). Transferable subset (stable & >=0.10): dct_energy, hf_ratio, radial_slope, luma_p05, illum_b, illum_r, frozen_frac.
- **Decision:** Use the sharpness/spatial-spectrum features for regime routing; do not reuse motion/temporal-band signs. Table robust to doubling labels.
- **Cost:** CPU; 150+160 windows x 26 features
- **Results:** eval/results/exp_fewshot/O2.03.json
