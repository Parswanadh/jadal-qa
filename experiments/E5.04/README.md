# E5.04 — Frozen ensemble re-score on 160 OOD windows (labels2 pending -> inconclusive)

**Why:** The 100-window second batch exists as mp4s with frozen scores cached, but the HUMAN labels ood_labels2.json are not delivered. Script the confirmatory eval to run automatically when the file appears.

**Hypothesis:** With labels2 absent the confirmatory OOD eval stays inconclusive; only descriptive score distributions over the 160 windows can be reported (no accuracy).

**Data / label source:** 160 Egocentric-10K windows (60 in caches/ensemble/ood_scores.json, 100 in ood_scores2.json) scored by frozen ensemble_v1; ood_labels2.json MISSING (n human labels = 0)

**Data regime:** 480p 10s transcodes of 1080p/30fps factory clips

**Method:** load frozen scores (or re-score via jadal.ensemble.scorer when missing); when ood_labels2.json appears, merge + eval_ood-style window-bootstrap AUROC; until then descriptive distributions only

**Owner:** experimenter  **Started:** 2026-10-10 15:17

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 15:40): inconclusive.** ood_labels2.json NOT delivered: result JSON records labels_missing=true, n=0, ci95=n/a. Descriptive only: 160 windows scored with frozen ensemble_v1, usable_prob mean 0.560 (median 0.460), accept 71 / review 35 / reject 54; batch1(60) mean 0.496, batch2(100) mean 0.598. No accuracy computed.

Results file: eval/results/exp_domain/E5.04.json

Decision: Watcher eval/exp_domain/watch_e5_04.sh re-scores and runs the confirmatory window-bootstrap eval automatically when ood_labels2.json appears.

Cost: 0 new GPU scoring (frozen scores cached); descriptive only

**Verdict (2026-10-10 16:20): inconclusive.** ood_labels2.json NOT delivered: the 160 scored windows have NO labels; result JSON records labels_missing=true, n=0, ci95=n/a (no accuracy computed). Descriptive only: 160 windows scored with frozen ensemble_v1, usable_prob mean 0.560 (median 0.460), accept 71 / review 35 / reject 54; batch1(60) mean 0.496, batch2(100) mean 0.598. Config: frozen sha 4da1068a recorded by the scorer; local raw sha d62a95a differs only by pod path remap; canonical sha f43654d7 matches and is asserted (observed shas stored in E5.04.json).

Results file: eval/results/exp_domain/E5.04.json

Decision: Watcher eval/exp_domain/watch_e5_04.sh re-scores and runs the confirmatory window-bootstrap eval automatically when ood_labels2.json appears.

Cost: 0 new GPU scoring (frozen scores cached); descriptive only
