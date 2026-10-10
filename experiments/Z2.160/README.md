# Z2.160 — Z2 conformal/risk-control re-run on 160 OOD labels

**Why:** PR #126 only had an OOD-60 note; re-evaluate the IndEgo-calibrated rule on the merged 160 labels with grouped CIs.

**Hypothesis:** The distribution-free coverage guarantee does not transfer out-of-domain.

**Data / label source:** IndEgo oof_probs n=150 + labels/ood160_labels.json n=160 + frozen ood_scores{,2}.json

**Data regime:** fit on IndEgo only; OOD evaluation only

**Method:** split conformal (clip LOCO) + CRC/LTT (unchanged) + OOD-160 transfer eval + frozen review/error frontier

**Owner:** M5  **Started:** 2026-10-10 19:25

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 19:25): drop.** IndEgo rule at coverage 0.85 gives only 55.6% empirical coverage on OOD-160 (target 85%); frozen score near chance (AUROC 0.566 [0.413,0.704]); no OOD operating point.

Results file: eval/results/exp_conformal160/Z2.01_160.json

Decision: n/a

Cost: n/a
