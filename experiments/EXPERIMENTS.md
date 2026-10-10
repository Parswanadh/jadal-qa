# Experiments

Registry: `experiments/registry.jsonl`. One folder per experiment under
`experiments/<id>/`.

## R1 — Optimized + recalibrated pipeline over the expanded corpus

| id | status | verdict | summary |
|---|---|---|---|
| R1.01 | partial | maybe | 830/9,757 windows (8.5%) scored, 0 errors, 0.55 win/s, 5.50 processed footage-h/wall-h; extrapolated full-corpus wall 296 min (4.9 h). Descriptive verdicts: accept 25.3% / reject 12.0% / review 62.7%. Recalibrated heads restore the gold operating point (review 0.1867, confident BA 0.9073 vs frozen heads 0.7632); OOD AUROC 0.5231 inside the frozen grouped CI. Determinism 10/10. Run still in flight. |

See `experiments/R1.01/README.md` and `docs/reports/exp_opt_scale.md`.
