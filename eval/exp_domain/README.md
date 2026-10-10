# E5 — domain adaptation (experiment code)

Track E5 (see `../../experiments/EXPERIMENTS.md`, report
`../../docs/reports/exp_domain_adapt.md`).

- `e5lib.py` — loaders + bootstrap CIs + LOCO / grouped-CV helpers (CPU, fixed seeds).
- `prep_inputs.py` — builds `eval/results/exp_domain/inputs/` (git-ignored `.npz`):
  self-decoded OOD 26 features (frozen 160×90 ffmpeg recipe, native + 10 fps) and
  L2-normalised OOD EG2 window means. `--fetch-git` pulls the frozen EG2 frame
  embeddings from `origin/feat/ensemble-views:eval/results/ensemble/ood_frames.npz`.
- `run_e5.py {e5_01,e5_02,e5_03,e5_04,all}` — writes `eval/results/exp_domain/E5.0*.json`.
- `watch_e5_04.sh` — polls for `ood_labels2.json` and re-runs E5.04 when it appears.

Determinism: `OMP_NUM_THREADS=1`, fixed seeds, no sampling at inference. CPU only
(no GPU encode; cached EG2 embeddings are reused).
