# exp_fewshot — O2: E5.04 final + few-shot adaptation curve (160 OOD labels)

Owner: **O2** (pod `jadal-qa`, branch `m/fewshot160`). Registry: `experiments/registry.jsonl`.

- **O2.01** — E5.04 final: frozen `ensemble_v1` on the 160 human OOD windows,
  factory-grouped bootstrap. Verdict **drop** (AUROC 0.566, grouped CI
  [0.445, 0.674] includes 0.5).
- **O2.02** — few-shot adaptation curve, k ∈ {10, 20, 40, 80, 160} labels,
  factory-grouped CV. Best recipe: logistic on 26 per-pixel features
  (80 labels → AUROC 0.701 / error 0.389).
- **O2.03** — E5.03 refresh on the 160 labels: 8 stable / 14 flip / 4 chance
  (2/26 class changes vs the old 60).

Figure: `images/adaptation_curve.png`. Full write-up:
`docs/reports/exp_fewshot160.md`. Results: `eval/results/exp_fewshot/O2.0{1,2,3}.json`.

Reproduce:

```
OMP_NUM_THREADS=2 python3 eval/exp_fewshot/prep_inputs.py --jobs 2   # builds inputs/ood160_pixel.npz (git-ignored)
OMP_NUM_THREADS=2 python3 eval/exp_fewshot/run_o2.py all
python3 eval/exp_fewshot/make_figure.py
OMP_NUM_THREADS=2 python3 -m pytest tests/test_exp_fewshot.py -q
```
