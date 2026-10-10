# O2.01 — E5.04 final: frozen ensemble on 160 human OOD labels (factory-grouped CI)

**Why:** ood_labels2.json is now delivered, so E5.04 can leave 'inconclusive' and give the confirmatory frozen-ensemble OOD number with the honest factory-grouped CI.

**Hypothesis:** The frozen IndEgo ensemble does not transfer: its 160-window OOD AUROC is near chance and the factory-grouped CI includes 0.5.

**Data / label source:** 160 Egocentric-10K human-labelled windows (138 good/22 bad, 47 factories; labels/ood160_labels.json = q1 60 + q2 100) + frozen ensemble_v1 scores (caches/ensemble/ood_scores.json + ood_scores2.json)

**Data regime:** 480p 10s transcodes; ~1 window/factory

**Method:** frozen usable_prob vs human labels; AUROC + balanced accuracy with factory-grouped cluster bootstrap (2000) and (optimistic) window bootstrap

**Owner:** O2  **Started:** 2026-10-10 17:40

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 17:40): drop.** 160 OOD windows (138/22, 47 factories), frozen ensemble_v1 usable_prob: AUROC 0.566, factory-grouped CI [0.445,0.674] (includes 0.5), window-bootstrap CI [0.437,0.694], bal.acc@0.5 0.546 [0.460,0.635], acc 0.481. Batch caveat: q1(old 60) 0.415 vs q2(new 100) 0.657. Frozen ensemble is NOT a reliable standalone OOD scorer.

Results file: eval/results/exp_fewshot/O2.01.json

Decision: Update E5.04 inconclusive -> drop. Do not use the frozen ensemble as an OOD gate; adapt (O2.02).

Cost: CPU; 160 cached frozen scores; 0 new GPU scoring
