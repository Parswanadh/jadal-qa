# rhythm STATUS

Branch `feat/custom-algo-rhythm` (worktree `/home/parshu/projects/jadal-qa-wt/custom-algo`).

## done
- Worktree created off `origin/main` (b2f65de).
- Baseline reproduced exactly: 26 GPU features + logistic, LOCO AUROC **0.810**
  (handover says 0.81). Log: `custom_algo_data/logs/baseline_repro.log`.
  Per-clip: ecu_full 0.95, logistics4 0.90, organise 0.91, desk 0.80, ecu_late 0.65,
  pallet 0.50, pallet2 0.50 (chance). Within-clip z-scoring confirmed to hurt (0.601).
- Representation extractor `jadal/rhythm/features.py` (160x90, 10 fps, 6 grid
  channels + 8 global channels, phase-correlation translation ego-motion).
- Vectorised ensemble `jadal/rhythm/model.py`: K members in one set of tensor ops
  (grouped convs), 2,287 params/member, gradients verified, G in {1,4,6} all work.
- `eval/rhythm/extract.py` sharded extractor.
- `tests/test_rhythm_features.py` (22 tests, written by a sub-agent).
- `docs/reports/rhythm.md` skeleton (sub-agent), all numbers PENDING.
- Parallel research tracks launched (see `custom_algo_data/research/`).

## running
- Cache re-extraction (the first extraction is INVALID - see blockers).
- CPU ablation + rhythm-spectrum + alternative-maths sub-agents.

## blockers / corrections
- **BUG FOUND AND FIXED**: `cv2.phaseCorrelate` mutates its inputs in place on
  OpenCV 5.0.0. The first cache (227 files) was computed with corrupted frames and
  is being discarded. Fixed by copying inputs in `_phase_shift`. Any number from
  the first extraction is void.
- GPU training not yet run (needs the re-extracted cache).

## next
- train + LOCO eval of the ensemble vs the 4 required baselines, ablations, freeze.

## headline numbers
- baseline LOCO AUROC 0.810 (reproduced). rhythm model: NOT YET MEASURED.
