# exp_phys — deterministic physics rule-list model (Q1.01)

Transparent, mathematics-first ACCEPT/REJECT/REVIEW scorer for 10-second
egocentric factory windows. **No learned weights, no probabilities**: every
threshold is a fixed quantile of the training clips, quantised to 4 decimals,
and the score is an integer count of crossed thresholds.

## Files
| file | purpose |
|---|---|
| `phys_rules.py` | candidate rule list, threshold derivation, integer scorer, deterministic decision |
| `run_exp_phys.py` | LOCO evaluation on the 150 IndEgo gold windows, writes results + figure |
| `../../eval/results/exp_phys/Q1.01.json` | full LOCO metrics, per-rule stats, ablations |
| `../../eval/results/exp_phys/rule_table.json` | frozen deployment rule table (all-clip thresholds) |
| `../../eval/results/exp_phys/roc_phys.png` | ROC (rules vs logistic vs EG2) + per-rule precision |
| `../../docs/reports/exp_phys_rules.md` | report |

## Reproduce
```bash
OMP_NUM_THREADS=2 python3 eval/exp_phys/run_exp_phys.py
OMP_NUM_THREADS=2 python3 -m pytest tests/test_exp_phys_rules.py -q
```
Data root defaults to `/workspace/jadal_pod_bundle`; override with `JADAL_DATA`.
Reads cached features only (no GPU, no decode). Never edits labels.

## Method in one paragraph
For each named rule and physics feature we take the p01/p05/p10/p25 quantiles
(mirrored to p99/p95/p90/p75 for high-is-bad features) of the **training** clips
in a leave-one-clip-out loop. A window's score is the number of crossed
thresholds (integer). A quality rule that crosses its *extreme* (p01/p99)
threshold forces REJECT; otherwise the integer score is compared to a reject
threshold chosen on training-good windows (5% false-reject budget) → REJECT /
REVIEW / ACCEPT. Every decision carries `{rule, feature, value, op, threshold}`.
