# O2.02 — Few-shot adaptation curve on 160 OOD labels (factory-grouped)

**Why:** Given the frozen ensemble does not transfer, quantify how much human OOD labelling recovers: the error-vs-labels curve.

**Hypothesis:** A small logistic trained on the 26 per-pixel features recovers usable OOD signal within tens of labels; the frozen ensemble views do not.

**Data / label source:** 160 Egocentric-10K human windows (138/22, 47 factories; labels/ood160_labels.json); 26 per-pixel features (frozen 160x90 recipe, 10 fps-effective) + frozen view scores p_eg2/p_pixel

**Data regime:** 480p 10s, 10 fps-effective

**Method:** factory-grouped CV: repeated grouped subsampling (train ~k / test factory-disjoint complement) for k in {10,20,40,80}; StratifiedGroupKFold(5) OOF for k=160; inner leave-one-factory-out picks C; class_weight=balanced

**Owner:** O2  **Started:** 2026-10-10 17:40

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 17:40): maybe.** Factory-grouped curve. pixel-26 logistic: k=10 0.544/err0.489, 20 0.609/0.452, 40 0.650/0.415, 80 0.701/0.389, 160 grouped-CV 0.689 [0.510,0.847]/err0.395. Frozen-view arms stay at chance (platt 0.49-0.56, two-view 0.47-0.52). Best label-efficient recipe = logistic on 26 pixel features; ~40-80 labels matches the 160-label level. Frozen 0-label baseline 0.566/0.454.

Results file: eval/results/exp_fewshot/O2.02.json

Decision: Adapt with the 26-pixel logistic; spend labels on pixel features, not on ensembling frozen views. Operating point ~40-80 labels (AUROC 0.65-0.70, err 0.39-0.42).

Cost: CPU; 160 windows; 20 repeats/k; no GPU
