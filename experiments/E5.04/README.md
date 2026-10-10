# E5.04 — Frozen ensemble re-score on 160 OOD windows (UPDATED by O2.01: labels2 delivered)

**Why:** ood_labels2.json is now delivered; the confirmatory frozen-ensemble OOD eval that E5.04 left inconclusive can now run.

**Hypothesis:** Frozen IndEgo ensemble does not transfer; OOD AUROC near chance with a CI that includes 0.5.

**Data / label source:** 160 Egocentric-10K human windows (138/22, labels/ood160_labels.json) + frozen ensemble_v1 scores

**Data regime:** 480p 10s transcodes

**Method:** frozen usable_prob vs labels; factory-grouped cluster bootstrap (see O2.01)

**Owner:** O2  **Started:** 2026-10-10 17:41

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 17:41): drop.** SUPERSEDED by O2.01 (same analysis, committed in eval/results/exp_fewshot/O2.01.json). 160 OOD windows, frozen ensemble_v1 usable_prob: AUROC 0.566, factory-grouped CI [0.445,0.674] includes 0.5, bal.acc 0.546. Was inconclusive (labels2 not delivered).

Results file: eval/results/exp_fewshot/O2.01.json

Decision: E5.04 inconclusive -> drop. Frozen ensemble is not a standalone OOD gate; adapt with O2.02.

Cost: CPU; cached scores
