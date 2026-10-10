# STATUS — pod wave (lead agent)

Updated: 2026-10-10 22:05 IST (16:35 UTC) · branch `pod/kit` (lead) · A40 pod

## Environment (verified this session)
- opencode v2.0.26; agents present: lead, experimenter, explorer, reviewer, tester, docs, infra.
- A40 46 GB, RAM 503 GB, `/workspace` 78 GB free. NVDEC present. `nproc` prints 4 (it honours `OMP_NUM_THREADS`
  from `env.sh`); the real cgroup quota is ≈7.65 cores (`/sys/fs/cgroup/cpu.max` `765000 100000`).
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
- A1/A2/A3/A4/A5/A6/A7 done (PRs #81, #82, #83, #87, #88, #89, #90, #95). Still running: the E1 and E5
  review-fix passes, B2 `exp-rescore` (E8.01), and the docs summary fill.

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

## Final state (2026-10-10 22:05 IST / 16:35 UTC)
All 8 experiment branches are committed, pushed and have open PRs to `main`; the docs refresh is the last writer.

| PR | branch | ids | verdict |
|---|---|---|---|
| #95 | `feat/exp-mosaic` | E2.01-E2.07 | mosaic **drop** as model input, **adopt** as evidence artefact |
| #88 | `feat/exp-fps-pod` | E1.03, E1.04, E1.05 | OOD no transfer at any fps; **drop** all selection arms (uniform wins) |
| #83 | `feat/exp-regimes-pod` | E4.01-E4.04 | descriptors **adopt**, splits **drop**, gap control **drop**, router **adopt** |
| #87 | `feat/exp-domain-adapt` | E5.00-E5.04 | heads **maybe**, few-shot **maybe**, sign table **maybe**, E5.04 **inconclusive** |
| #89 | `feat/exp-motion-cue` | E6.01-E6.04 | all **drop** — the cue is ego-motion magnitude |
| #90 | `feat/exp-cost-model` | E7.01 | **adopt** — 15.2 video-h per wall-h, feature-bound |
| #113 | `feat/exp-rescore` | E8.01 | **adopt** (descriptive) — 1,546 windows re-scored |
| #81 | `pod/prwatch` | E9.01 | **adopt** — dispatcher + standard eval harness |
| #82 | `docs/pod-docs` | — | docs: env page, union registry, wave-1 summary |

Not ours: #84/#85 (`feat/exp-fps`, `feat/exp-temporal`, builder branch-recovery), and the builder's parallel
`b/*` PRs (#91-#94, #96-#99). All reviews of my PRs are file-only (the `reviewer` agent has no `bash`); the
`tester` agent ran the suites.

E8.01 detail (descriptive, `label_source: "none"`): n=1,546 Egocentric-10K windows, verdicts accept 296 (19.2%)
/ reject 83 (5.4%) / review 1,167 (75.5%), accept-share 95% CI [0.136, 0.254] (58-clip bootstrap); profiles
factory_1080p30 1,540 + default 6 (UNSEEN_REGIME); 5,761 ms/window (p50 5,195, p95 9,371). The E3.04 sparse+exit
policy was NOT applied (the contract needs one verdict per window). Decode path is not bit-identical to the
frozen swscale (Δp_pixel ≤ 0.048, n=6 control). Full file `/workspace/artifacts/windows_rescored.jsonl`.

## Headline numbers for this wave (all from results JSONs; read with `tools/pod/collect_results.py`)
| id | verdict | headline (n, label source, protocol) | regime |
|---|---|---|---|
| E1.03 | — | frozen IndEgo model on the 60 human OOD windows: AUROC 0.3595 [0.180,0.554] @30 fps → 0.4314 [0.237,0.627] @10 → ~0.50 @5-1; Spearman rank stability 30-vs-1 fps 0.83. **Failure to transfer at every fps.** n_bad=9, exploratory | 1080p/30 HEVC OOD |
| E1.04 | drop | equal 20-frame budget on 150 IndEgo: uniform20 0.7866 [0.650,0.866] > keyframe 0.780 > motion20 0.762 > random20 0.739 → **no selection beats uniform** | 480p/10 in-domain |
| E2.01 | drop | mosaic as model input: best N=9 0.785 [0.64,0.86] vs matched mean-of-frames 0.852 [0.75,0.91]; OOD 0.155-0.638 vs 0.42-0.63 | IndEgo + OOD |
| E2.02 | adopt | mean-of-frames control 0.822-0.860 (reproduces cached 0.848 [0.73,0.91]) | IndEgo |
| E2.03/04/07 | drop | tile order, timestamps, pooled layout not load-bearing | IndEgo |
| E4.01 | adopt | deterministic descriptors; 6 content descriptors shrink after re-encode; resolution/fps/codec separate by construction | mixed |
| E4.02 | drop | no regime-specific split beats global (all deltas' CIs span 0) | IndEgo terciles |
| E4.03 | drop | re-encoding to the IndEgo spec narrows the descriptor gap but is descriptor-only — the frozen scorer was NOT re-scored, no transfer claim | 480p transcode |
| E4.04 | adopt | 5 profiles + deterministic router, UNSEEN_REGIME → default, `receipt_fragment()` helper (**not wired** into the receipt), 13 tests | all |
| E5.01 | maybe | AI-frame heads pooled → 0.789-0.826 LOCO on the 150 **human** IndEgo windows (frame-level 0.911/0.957 is AI-vs-AI, not accuracy) | 10k AI frames |
| E5.02 | maybe | few-shot ON the 60 OOD labels (grouped by factory): pixel26 0.756 [0.575,0.902], two-view 0.767 [0.596,0.904]; ~12 labels → 0.71 (std 0.10); smallest fold n_bad=7 | OOD n_bad=9 |
| E5.03 | maybe | sign table: 8 stable / 14 flip / 4 chance; sharpness holds, motion inverts | both |
| E5.04 | inconclusive | labels2 not delivered (n=0); 160 windows scored descriptively only | OOD |
| E6.01-04 | drop | the cue is essentially ego-motion magnitude (wearer not walking); residual motion after translation-only compensation adds nothing; frozen orientation does not transfer (0.281 [0.145,0.434]) | IndEgo + OOD |
| E7.01 | adopt | best profile ≈ **15.2 video-hours per wall-hour** on the A40 (vs default 29.84x speed-up); **feature/embedding-bound**, decode only 16.7% of the total | measured |
| E9.01 | adopt | standard eval harness: pixel26 0.8100 [0.6934,0.8872], eg2 0.8480 [0.7344,0.9087], n=150 human LOCO | IndEgo |
| E8.01 | pending | re-scoring all 1,546 Egocentric-10K windows (descriptive) — running | 1080p/30 |

## Handover notes for the builder (read before merging)
1. **Conflict cause is duplication, not disagreement.** `main` (as of `e320c58`, 21:49 IST) now contains the
   ported files `eval/exp_fps/*`, `eval/exp_temporal/*`, `jadal/gpu/*`, `eval/gpu/*`, `eval/ensemble/*`,
   `jadal/ensemble/*`, `configs/ensemble_v1.json` — merged from #79 (`feat/ensemble-views`), #84/#85
   (the "save local work" recovery branches) and #96. My branches (`feat/exp-mosaic`, `feat/exp-fps-pod`,
   `feat/exp-rescore`, `pod/prwatch`) also contain those files (via the shared base `pod/base`), so GitHub
   reports add/add conflicts on ~25 paths for #81, #88, #95, #113 (and #82 is being re-evaluated).
   #83, #87, #89, #90 have already been rebased by someone and are MERGEABLE.
2. **Resolution recipe** (normal merge commit, never force-push): `git merge origin/main` on the branch, then for
   the inherited duplicated paths take ONE side consistently — the `pod/base`/branch side is the version the
   reviewers and the tester actually exercised on this pod (paths resolved from `JADAL_BUNDLE`, `unittest`-clean),
   while the merged recovery copies are untested here. Keep the branch's own new files
   (`jadal/mosaic/*`, `jadal/regimes/*`, `profiles/*`, `eval/exp_mosaic/*`, `eval/exp_rescore/*`,
   `eval/standard/*`, `tools/pod/pr_watcher.py`, `eval/results/exp_*`) — they do not conflict.
   `.gitignore` and `configs/ensemble_v1.json` are the only two files worth a hand-merge rather than a
   side-pick.
3. **Do not merge my PRs for me** — I did not merge any, and none of my branches was force-pushed. The 4
   conflicting branches are clean and pushed; the conflict exists only at merge time.
4. `labels/ood_labels2.json` never arrived, so every OOD claim in this wave is n=60 / n_bad=9 (exploratory).
   E5.04 and E8.01 are the two ids that change the moment those 100 human labels land.
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
