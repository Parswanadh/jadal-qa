# Z4.01 — Clip-level aggregation + LOCO clip calibration

**Why:** Goal (a): decide at the clip level from window scores; which aggregator separates the 7 clip verdicts and how much window review is needed for a target clip error

**Hypothesis:** A high-quantile or rank aggregate beats the mean because a clip is bad when a tail of its windows is bad

**Data / label source:** IndEgo human gold 150 windows/7 clips (labels/manifest_gold.json) + cached oof_probs.npz

**Data regime:** 480p/10fps IndEgo assembly+logistics

**Method:** mean/median/q75/q90/trimmed/rank-mean of 1-p_ens; clip truth bad_frac>=0.30; LOCO threshold + isotonic; clip AUROC; uncertainty-targeted review curve

**Owner:** Z4  **Started:** 2026-10-10 17:29

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 17:30): maybe.** q75 aggregator best clip AUROC 0.917 (mean 0.833); LOCO clip BA <=0.75 (5/7) for every aggregator -> clip verdicts are discriminative but thin. Uncertainty-targeted review: 35% of windows -> error<=1/7, 65% -> 7/7 (vs every-2 51.5% -> 7/7).

Results file: eval/results/exp_clip/Z4.01.json

Decision: No single aggregator lifts clip BA above 5/7; use q75/mean only as a ranking signal, not a certified verdict.

Cost: CPU-only on cached labels+scores, no GPU
