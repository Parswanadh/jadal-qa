# N1.02 — Self-training on the 1,546 unlabelled Egocentric-10K windows

**Why:** Does pseudo-labelling the unlabelled pool improve OOD accuracy under factory-grouped CV?

**Hypothesis:** No: the pool is shifted (domain AUROC 0.902) and the base stable-8 model is weak; self-training is <= base

**Data / label source:** /workspace/artifacts/windows_rescored.jsonl (1546 unlabelled, label_source none) + 160 labels

**Data regime:** OOD Egocentric-10K 160 windows; 47 factories; n_bad=22

**Method:** per-fold pseudo-label unlabelled windows from non-test factories; retrain; grouped OOF

**Owner:** N1  **Started:** 2026-10-10 19:33

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 19:33): drop.** Self-training on 1546 unlabelled windows does not help (<= base stable-8 0.7319; pool shifted, domain AUROC 0.902).

Results file: eval/results/exp_acc/N1.02.json

Decision: Do not use self-training.

Cost: CPU only
