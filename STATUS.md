# STATUS — pod wave (lead agent)

Updated: 2026-10-10 20:45 IST (15:15 UTC) · branch `pod/kit` (lead) · A40 pod

## Environment (verified this session)
- opencode v2.0.26; agents present: lead, experimenter, explorer, reviewer, tester, docs, infra.
- A40 46 GB, `nproc`=4, RAM 503 GB, `/workspace` 78 GB free. NVDEC present.
- Baselines reproduce: `tools/port/baseline_logistic_loco.py` → raw LOCO AUROC **0.810**;
  `tools/port/ensemble_views_exploratory.py` → EG2 **0.848**, prob-avg **0.853**. See `docs/pod/ENVIRONMENT.md`.
- Autosave running (`tmux ls` → `autosave`). GPU slots: `bash tools/pod/gpu_run.sh <cmd>` (2 slots).
- Data: 150 IndEgo gold windows, 60 human-labelled OOD windows (n_bad=9), 100 OOD queue-2 windows
  (mp4 only — **human labels `ood_labels2.json` NOT yet delivered**), 1,546-window Egocentric-10K manifest.

## Batch A (launched in parallel; base branch `pod/base` = ported laptop code, remapped to the pod)
| # | branch | agent | job | output (disjoint) | deadline |
|---|---|---|---|---|---|
| A1 | `feat/exp-mosaic` | experimenter | E2 mosaic (builder `jadal/mosaic/build.py` + tests + EG2 variants) | `eval/exp_mosaic/`, `jadal/mosaic/`, `eval/results/exp_mosaic/` | 17:00 UTC |
| A2 | `feat/exp-fps-pod` | experimenter | E1.03 OOD fps sweep + E1.04 frame selection | `eval/exp_fps_pod/`, `eval/results/exp_fps_pod/` | 17:00 UTC |
| A3 | `feat/exp-regimes-pod` | experimenter | E4.01-E4.04 descriptors, regime models, domain-gap, profiles+router | `eval/exp_regimes/`, `jadal/regimes/`, `profiles/` | 17:00 UTC |
| A4 | `feat/exp-motion-cue` | experimenter | E6 motion cue conditioning + per-clip sign calibration | `eval/exp_motion_cue/`, `docs/reports/exp_motion_cue.md` | 17:00 UTC |
| A5 | `feat/exp-domain-adapt` | experimenter | E5 in-domain heads on 10k AI frames, few-shot OOD, sign table | `eval/exp_domain/`, `docs/reports/exp_domain_adapt.md` | 17:00 UTC |
| A6 | `pod/prwatch` | infra | PR-triggered agent dispatcher + `eval/standard/run_all.py` | `tools/pod/pr_watcher.py`, `agents/`, `eval/standard/` | 17:00 UTC |
| A7 | `docs/pod-docs` | docs | keep docs/pod/*, experiments/EXPERIMENTS.md true | `docs/pod/*`, `experiments/EXPERIMENTS.md` | rolling |

## Done
- 15-min env verification (above). `pod/base` built + pushed (ported feature/experiment code, paths remapped).
- GPU slot helper `tools/pod/gpu_run.sh`; artifact dirs under `/workspace/artifacts/`.

## Running
- A1–A6 (headless workers, logs in `/workspace/logs/<name>.log`); A7 docs.

## Next (batch B, after A reports)
- E7 cost model; E8 re-score 1,546 Egocentric-10K windows into the PoC contract format (descriptive only);
  final reports + `docs/reports/pod_experiments_summary.md`.

## Blockers / notes
- `labels/ood_labels2.json` (100 human OOD labels) not delivered → OOD claims stay at n=60 / n_bad=9.
- `AGENT_API_KEY` env var absent (only HF_TOKEN, GH_TOKEN present).
- Registry: repo `experiments/registry.jsonl` empty; historical log seeded as `experiments/registry_snapshot.jsonl`.
