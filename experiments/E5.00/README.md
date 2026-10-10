# E5.00 — Frozen ensemble OOD prior (usable_prob AUROC on 60 human OOD windows)

**Why:** The E5 report quotes the frozen-ensemble out-of-domain AUROC 0.415 [0.210,0.647], but that number lived only in docs/reports/ensemble_views.md on main / PR #79, not in this branch or in a committed results JSON. Reproduce it from the frozen bundle scores so every E5 number traces to a committed file.

**Hypothesis:** The frozen IndEgo ensemble scores ~0.415 AUROC on the 60 human-labelled Egocentric-10K windows (exploratory, n_bad=9); reproducing it from the frozen scores gives the same headline.

**Data / label source:** 60 Egocentric-10K human-labelled windows (51 good / 9 bad, single labeller, ood_labels.json) scored by frozen ensemble_v1 (bundle caches/ensemble/ood_scores.json); human labels

**Data regime:** 480p 10s transcodes of 1080p/30fps factory clips; ~1 window/factory

**Method:** load frozen usable_prob; AUROC vs human labels; window-bootstrap 95% CI (optimistic)

**Owner:** experimenter  **Started:** 2026-10-10 16:12

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 16:20): maybe.** Frozen ensemble_v1 usable_prob vs 60 human OOD windows (51 good/9 bad, exploratory): AUROC 0.4150, window-bootstrap CI [0.201,0.656] (optimistic); balacc@0.5 0.471. Reproduces the 0.415 headline previously cited from docs/reports/ensemble_views.md (main/PR #79, NOT on this branch) from $JADAL_BUNDLE/caches/ensemble/ood_scores.json.

Results file: eval/results/exp_domain/E5.00.json

Decision: Cite E5.00.json for the frozen-OOD prior in the E5 report; keep the PR-#79 note for the earlier CI [0.210,0.647].

Cost: CPU <1 s; 60 frozen scores; no GPU, no encode
