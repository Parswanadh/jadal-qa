# E5.02 — Few-shot adaptation on human OOD labels (factory-grouped CV)

**Why:** If the frozen IndEgo models do not transfer (0.415 OOD AUROC), the next question is whether a handful of human OOD labels can recover usable signal, and how many labels are needed.

**Hypothesis:** With n_bad=9 and factory-grouped folds the OOD few-shot logistic is unstable and largely indistinguishable from chance; simple recipes will not beat chance reliably.

**Data / label source:** 60 Egocentric-10K human-labelled windows (51 good / 9 bad, single labeller, ood_labels.json) + frozen caches: EG2 OOD frame embeddings (60x20x768) and 26 per-pixel features per window; labels2 NOT delivered

**Data regime:** 480p 10s transcodes of 1080p/30fps factory clips; ~47 factories, ~1 window/factory

**Method:** grouped/leave-one-factory-out CV (StratifiedGroupKFold); recipes: logistic on 26 per-pixel features, logistic on EG2 window means, two-view probability average; report smallest n_bad in any train fold

**Owner:** experimenter  **Started:** 2026-10-10 15:17

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 15:40): maybe.** 60 OOD windows (51 good / 9 bad, 47 factories), factory-grouped CV: pixel-26 0.756 [0.575,0.902]; EG2-window-mean 0.546 [0.350,0.736]; two-view avg 0.767 [0.596,0.904]. Learning curve: ~12 train windows already ~0.71 (std 0.10). Smallest n_bad in any training fold = 7. Estimates unstable (n_bad=9); grouped folds ~1 window/factory so CV is optimistic.

Results file: eval/results/exp_domain/E5.02.json

Decision: A ~12-window OOD adaptation set can recover ~0.7 AUROC on THIS set (exploratory); needs the 100-window batch + labels2 before any rate is quoted.

Cost: CPU ; 60 windows; 26 features / 768 EG2 means

**Verdict (2026-10-10 16:20): maybe.** 60 OOD windows (51 good / 9 bad, 47 factories), factory-grouped CV: pixel-26 0.756 [0.575,0.902] balacc 0.634; EG2-window-mean 0.546 [0.350,0.736] balacc 0.520; two-view avg 0.767 [0.596,0.904] balacc 0.614. Learning curve (pixel-26, C=1 balanced): ~12 -> 0.709 (std 0.10, 17 repeats, min_n_bad_train 1), ~24 -> 0.715 (20), ~36 -> 0.703 (20), ~45 -> 0.691 (19). Smallest n_bad in any training fold = 7. Estimates unstable (n_bad=9); grouped folds ~1 window/factory so CV is optimistic.

Results file: eval/results/exp_domain/E5.02.json

Decision: A ~12-window OOD adaptation set can recover ~0.7 AUROC on THIS set (exploratory); needs the 100-window batch + labels2 before any rate is quoted.

Cost: CPU ; 60 windows; 26 features / 768 EG2 means
