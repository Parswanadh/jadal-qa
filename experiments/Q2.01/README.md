# Q2.01 — Deterministic physics rule-list transfer on 160 OOD human labels

**Verdict: drop.** The frozen IndEgo deterministic physics rule list is
**below chance** out-of-domain (AUROC 0.475/0.493; bal.acc 0.441/0.456) and
worse than the frozen logistic baseline (0.566/0.546). It rejects more GOOD
windows (39–41 %) than BAD (27–32 %); review only makes accuracy worse.

- Result: `eval/results/exp_phys/Q2.01.json`
- Report: `docs/reports/exp_phys_ood.md`
- Figure: `docs/experiments/exp_phys/images/rule_transfer.png`
- Features: `eval/results/exp_phys/inputs/ood_phys.npz` (160/160, 0 errors)
- Rule table (frozen, reused read-only): `eval/exp_phys/frozen_perpixel_v1.json`,
  `eval/exp_phys/frozen_rulelist.json` (B/verify-rules, PR #124)

Reproduce:
```
OMP_NUM_THREADS=2 python3 eval/exp_phys/extract_ood.py --jobs 2
OMP_NUM_THREADS=2 python3 eval/exp_phys/run_phys_ood.py
python3 eval/exp_phys/make_figure.py
```
