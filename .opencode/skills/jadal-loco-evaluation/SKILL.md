---
name: jadal-loco-evaluation
description: How to evaluate correctly on this dataset: leave-one-clip-out over 7 IndEgo clips, clip-bootstrap confidence intervals, nested hyperparameter selection, out-of-domain scoring on human-labelled Egocentric-10K windows, selective-accuracy curves, baselines to reproduce. Load before computing any accuracy number.
license: Apache-2.0
---
- **In-domain**: leave-one-clip-out (train on 6 clips, test on the held-out clip, pool the 7 held-out predictions); AUROC with clip-level bootstrap (resample clips with replacement, 2,000 times, skip draws with one class); balanced accuracy at a threshold chosen on the TRAINING clips; per-clip AUROC table (pallet/pallet2 sit at chance for most models); per-reason recall (NO_ACTIVE_WORK, HANDS_NOT_VISIBLE, DUPLICATE, TOO_DARK, UNDECODABLE).
- **Nested**: any hyperparameter, feature selection, calibration or threshold is chosen by an inner leave-one-clip-out on the 6 training clips, never on the held-out clip. Fixed seeds.
- **Out-of-domain**: score the labelled Egocentric-10K windows (`labels/ood_labels*.json`, `latest` map) with the model FROZEN beforehand; CIs by window are optimistic (47+ factories, one window each); always print n_good, n_bad; refuse to report a metric with fewer than 10 bad windows (say "exploratory").
- **Selective accuracy**: sort windows by |p - 0.5|, report balanced accuracy and good-windows-wrongly-rejected at coverage 100/80/60/50%.
- **Baselines to reproduce first** (they must print the documented numbers): `scripts/baseline_logistic_loco.py` -> 0.810; `scripts/ensemble_views_exploratory.py` -> EG2 0.848, two-view average 0.853 (exploratory; nested 0.834-0.846).
- **Controls**: always include a pooled/no-structure control when claiming that structure (spatial layout, temporal order, timestamps) helps; a claim without its control is `inconclusive`.
- Word of caution: AUROC < 0.5 on a held-out clip means the learned direction does not transfer; it is not a usable "inverse feature".
