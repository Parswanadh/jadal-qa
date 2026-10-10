# rhythm STATUS — COMPLETE

Branch `feat/custom-algo-rhythm`, pushed. **PR #77 open** (do NOT merge):
https://github.com/Parswanadh/jadal-qa/pull/77

## Headline numbers (150 real human windows, LOCO, clip-bootstrap 2000)

| method | AUROC (95% CI) |
|---|---|
| **rhythm ensemble, no synthetic negatives** | **0.753 (0.651-0.813)** |
| rhythm ensemble, WITH synthetic negatives | 0.351 (0.232-0.490) — below chance |
| ablation (d): global channels only | 0.724 |
| flow_act-like scalar family | 0.764 |
| 26 GPU per-pixel features + logistic | 0.810 |

Per-clip: desk 0.707, ecu_full 0.740, ecu_late 0.625, logistics4 0.786,
organise 0.856, **pallet 0.515, pallet2 0.500** (chance).

**Outcome: documented negative result.** Did not reach AUROC >= 0.85 / CI lower
bound >= 0.72 / no clip below 0.60. Does not beat the 0.810 logistic baseline.

## The decisive finding
Tensor-level synthetic negatives make the model score **0.351 on real windows**
while looking fine on synthetic-inclusive data. It learns the approximation, not
the defect. Two full 7-fold runs differ only in that flag.

## Real bug found and fixed
`cv2.phaseCorrelate` mutates its inputs in place (OpenCV 5.0.0). The first cache
was corrupt and was discarded; fix guarded by 3 regression tests. `luma_mean`
0.094 -> 0.300, `under_frac` 0.538 -> 0.106.

## Frozen artifact
`configs/rhythm_v1.json` sha256 `0faf3609f4c849bc...`; `configs/rhythm_v1.npz`
1427 bytes. LOCO estimate 0.759, **not certified**.

## Tests
Full suite: **575 tests OK** (skipped=3).

## Independent tracks (`custom_algo_data/research/`, outside git)
- `papers/` — 24 verified-fetched sources; no fetched paper benchmarks a binary
  "hands visible AND actively working" label on egocentric video.
- `ablations/` — `sp_cellmax` (6 hand-built spatial features) 0.799 vs a
  size-matched global-only control averaging 0.725; beat 96.7% of matched draws.
- `rhythm_analysis/` — temporal-spectrum features, raw vs variance-normalised.
- `math_approaches/` — six framings; after removing magnitude NONE gave a
  defensible gain; concatenating 206 features hurt (0.629 vs 0.738).

## Not done (no number claimed)
Ablations for no-ego-motion, no-skin, G in {1,4,8}, no-augmentation, single-vs-
ensemble were NOT run in the time box. Streaming path not implemented. No GPU-vs-
CPU determinism run. Cost in windows/hour not measured.
