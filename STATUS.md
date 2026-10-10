# rhythm STATUS

Branch `feat/custom-algo-rhythm` — pushed to origin. Worktree
`/home/parshu/projects/jadal-qa-wt/custom-algo`.

## done
- Baseline reproduced exactly: 26 GPU per-pixel features + logistic, LOCO AUROC
  **0.810** (handover: 0.81). Per-clip: ecu_full 0.95, logistics4 0.90,
  organise 0.91, desk 0.80, ecu_late 0.65, **pallet 0.50, pallet2 0.50**.
  Log `custom_algo_data/logs/baseline_repro.log`.
- `jadal/rhythm/features.py` — representation R: 160x90, 10 fps, 6 grid channels
  (res_mean, res_p90, skin_frac, tvar, flow_mag, flow_coh) + 8 global channels,
  phase-correlation translation ego-motion compensation.
- `jadal/rhythm/model.py` — K-member ensemble vectorised in grouped convs.
  2,287 params/member. Gradients verified. G in {1,4,6} all work.
- `eval/rhythm/extract.py` — sharded deterministic extractor (150 gold windows cached).
- `eval/rhythm/train.py` — nested LOCO, per-member independent augmentation,
  inner-LOCO member selection, `--folds` sharding.
- `eval/rhythm/evaluate.py` — clip-bootstrap CI, per-clip, per-reason, coverage
  curve, same-fold baselines.
- `tests/test_rhythm_features.py` — 22 tests, all passing.
- `docs/reports/rhythm.md` — skeleton, all numbers PENDING (to be filled).
- Parallel tracks in `custom_algo_data/research/`: papers (24 verified sources),
  ablations, rhythm_analysis, math_approaches.

## running
- 4 parallel GPU fold shards (`eval/results/rhythm/loco_shard{0..3}.json`),
  K=48, top-M=24, 25 epochs, batch 8. GPU at 100%, ~4.6 GiB.

## blockers / corrections
- **BUG FOUND AND FIXED (by the test sub-agent, verified):**
  `cv2.phaseCorrelate` mutates its inputs in place on OpenCV 5.0.0. The first
  cache was computed on corrupted frames and has been DISCARDED
  (`cache/main_INVALID_phasecorrelate_bug/`). Any number from the first
  extraction is void. The fix (copying inputs) is now a regression test.
  Evidence the fix matters: g000 `luma_mean` 0.0944 -> 0.2995,
  `lap_var` 321 -> 1140, `under_frac` 0.538 -> 0.106.
- The 0.810 baseline is the 26 GPU per-pixel features, NOT this tensor; it is
  not directly comparable to numbers computed from `cache/main`.

## next
- Merge fold shards -> pooled LOCO AUROC + CI; run baselines + ablations; freeze.

## headline numbers
- baseline LOCO AUROC **0.810** (reproduced, other representation).
- rhythm ensemble LOCO AUROC: **pending** (training running).
