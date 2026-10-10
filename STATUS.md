# STATUS — pod wave (lead agent)

Updated: 2026-10-10 21:31 IST (16:01 UTC) · branch `pod/kit` (lead) · A40 pod

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
- A1 exp-mosaic, A2 exp-fps-pod, A4 exp-motion-cue, A5 exp-domain-adapt; B1 exp-cost-model, B2 exp-rescore;
  infra fix pass on `pod/prwatch` (reviewer findings I1/I2).
- Done: A3 exp-regimes-pod, A6 prwatch (PR #81), A7 pod-docs (PR #82).

## PRs open (mine)
- #81 `pod/prwatch` — dispatcher + standard eval harness (`eval/standard/run_all.py` reproduces pixel26 0.8100
  [0.6934,0.8872] / eg2 0.8480 [0.7344,0.9087], n=150 human IndEgo LOCO). Reviewer FAIL → **fixed and pushed**:
  I1 registered `.opencode/agents/{triage,review}.md`; I2 deny-edit + narrow-bash for those agents and the
  always-on tier now runs in a throwaway clone with the PR's own config/agents replaced by trusted ones;
  M1-M4 addressed (timeout/nobody stated, exec tier documented as harness, real HOME for the trusted tier,
  OOD features reordered to FEATURE_NAMES). `Ran 498 tests … errors=4` — the same 4 missing-dependency errors
  (`imagehash`, `fastapi`) as `main`; no new failures.
- #82 `docs/pod-docs` — env page (incl. the ~7.65-core cgroup quota correction vs `nproc`=4), union registry
  tool, summary skeleton. Reviewer FAIL → **fixed and pushed** (pod path cited for the 0.4-0.53 OOD figure;
  table-lags-registry note; RAM delta). Duplicate `union_registry.py` removed from the lead branch.
- #83 `feat/exp-regimes-pod` — E4.01-E4.04. Reviewer FAIL + tester finding → fix pass running (UNSEEN_REGIME
  ranges must cover `temp_energy`/`clip_frac`; the "receipt records it" claim must be wired or reworded; the
  router test was pytest-only and invisible to `unittest discover`).
- #87 `feat/exp-domain-adapt` — E5.01-E5.04. Headline: AI-frame heads 0.789-0.826 LOCO on the human windows
  (model-vs-model, AI labels); few-shot on the 60 OOD windows 0.756 [0.575,0.902] / two-view 0.767 [0.596,0.904]
  (trained ON the OOD labels, n_bad=9, smallest fold n_bad=7); E5.04 `inconclusive` (labels2 missing).
- #88 `feat/exp-fps-pod` — E1.03 (OOD fps sweep: 0.43-0.53 → confirmed failure to transfer) + E1.04
  (uniform20 0.787 [0.650,0.866] beats motion20 0.762, keyframe 0.780, random20 0.739 → negative result).
- #84/#85 `feat/exp-fps`, `feat/exp-temporal` — NOT ours: the human builder pushed branch-recovery commits at
  21:03 IST ("save local work") carrying the same E1/E3 code as `pod/base`. Note the overlap when merging.

## Facts worth carrying into the docs
- `nproc` prints 4 only because coreutils honours `OMP_NUM_THREADS=4` from `/workspace/env.sh`; the cgroup
  quota is ≈7.65 cores (`/sys/fs/cgroup/cpu.max` `765000 100000`), so 7-8 agents are within budget.
- `sentence-transformers` 6.1.0 IS importable (torch 2.8.0+cu128, transformers 5.19.0, cv2 5.0.0); `imagehash`
  and `fastapi` are NOT installed (4 environment-only test errors).

## Lead tools
- `tools/pod/gpu_run.sh` (2 GPU slots), `tools/pod/collect_results.py` (reads result files; flags done-without-file).
- `tools/pod/union_registry.py` — removed from the lead branch; canonical copy ships with PR #82.

## Next (batch B, after A reports)
- E7 cost model; E8 re-score 1,546 Egocentric-10K windows into the PoC contract format (descriptive only);
  final reports + `docs/reports/pod_experiments_summary.md`.

## Blockers / notes
- `labels/ood_labels2.json` (100 human OOD labels) not delivered → OOD claims stay at n=60 / n_bad=9.
- `AGENT_API_KEY` env var absent (only HF_TOKEN, GH_TOKEN present).
- Registry: repo `experiments/registry.jsonl` empty; historical log seeded as `experiments/registry_snapshot.jsonl`.
