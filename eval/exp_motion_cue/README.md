# eval/exp_motion_cue — E6 motion cue + per-clip sign calibration

Registry ids **E6.01–E6.04** (see `experiments/EXPERIMENTS.md`); report
`docs/reports/exp_motion_cue.md`; results `eval/results/exp_motion_cue/E6.0*.json`.

## What is here

| file | role |
|---|---|
| `motion_cue.py` | deterministic cue extraction: decode 160x90 gray, per-pair phase-correlation translation + residual `mean|I_t - warp(I_{t-1})|` |
| `run_e6.py` | runs E6.01–E6.04 and writes the results JSONs |
| `../../eval/results/exp_motion_cue/cue_indego.npz` | cached per-window cue values, 150 IndEgo windows |
| `../../eval/results/exp_motion_cue/cue_ood.npz` | cached per-window cue values, 60 Egocentric-10K labelled windows (480p/30 fps transcode, stride 3) |

## Run

    source /workspace/env.sh
    export JADAL_EXP_DIR=/workspace/wt/exp-motion-cue/experiments OMP_NUM_THREADS=1
    JADAL_BUNDLE=$JADAL_BUNDLE python3 eval/exp_motion_cue/run_e6.py \
        --experiments E6.01,E6.02,E6.03,E6.04 --workers 3

Re-running reuses the cached cue arrays (delete them or pass `--force-extract` to recompute).
CPU only (no GPU, 0 MB VRAM); 657 ms/window for the 150 IndEgo windows on 3 workers.
Tests: `python3 -m pytest tests/test_exp_motion_cue.py -q` (< 30 s, synthetic frames only).

## Headline (all from the results JSONs)

- The strongest motion signal in the caches is **ego-motion magnitude** (`series_gold` `shift_px`
  mean): oriented AUROC **0.805 [0.675, 0.893]** on 150 IndEgo human windows (n_bad = 50).
- The residual-motion cue after translation-only compensation is weaker (0.692) and its Spearman
  partial correlation with the label, controlling for ego-motion magnitude, is
  **−0.102 [−0.347, +0.072]** (CI includes 0); cue + ego logistic 0.793 vs ego only 0.798.
- Label-free per-clip sign calibration (the clip's own unlabelled windows) does not reliably fix the
  `pallet` inversion: pooled AUROC raw 0.692 [0.477, 0.839] vs majority-prior sign 0.610
  [0.449, 0.767]; `pallet` 0.36→0.64 and `organise` 0.31→0.69 repaired but `desk` 0.73→0.27 broken.
- Out of domain (60 human-labelled Egocentric-10K windows, n_bad = 9, exploratory) the frozen
  IndEgo direction scores **0.281 [0.145, 0.434]** — the cue's orientation does not transfer.
