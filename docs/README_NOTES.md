# README notes — claims table + hero candidates (2026-10-10, branch docs/readme-top, base e320c58)

## Hero candidates (8, all must be true; picked #1)

1. **QA for head-camera factory video that shows its evidence — including when it fails.** (PICKED)
2. A deterministic accept / reject / review funnel for 10-second egocentric windows.
3. Every reject carries its reason, its measured value, and its threshold.
4. It works on the footage it was trained on. Factory transfer is the open problem.
5. The team that kills its own results: negatives documented, not hidden.
6. Content-hashed receipts for every window. Re-run the clip, compare the hash.
7. In-domain signal, out-of-domain struggle — both in the same table.
8. Hand-checking factory video does not scale. This funnel tries, and says where it fails.

Picked #1 because it states what it is (QA), the differentiator (evidence),
and the honesty brand in one line, with no banned words. Rejected #3 (over-sells
S1 rejects: they carry only s1_score vs the 0.6 threshold) and #4 (softened to
'scores above chance in-domain on 7 clips, exploratory; weakest clip 0.606').

## Claim -> file -> status

| claim | file | status |
|---|---|---|
| 150 human-labelled IndEgo windows, 100 good / 50 bad, 7 clips | `eval/results/gold_labels_snapshot.json`, `docs/reports/funnel.md:421-425`, `docs/reports/perpixel_accuracy.md:3-9` | measured |
| 150-window label mix: NO_ACTIVE_WORK 23, HANDS_NOT_VISIBLE 14, DUPLICATE 6, TOO_DARK 6, UNDECODABLE 1 | `docs/reports/perpixel_accuracy.md:5-6`, `docs/reports/funnel.md:421-422` | measured |
| Nested leave-one-clip-out protocol, clip-bootstrap CIs | `docs/reports/perpixel_accuracy.md:7-9`, `docs/reports/ensemble_views.md:12-15`, `eval/results/ensemble/nested_loco.json:20` | measured protocol |
| Ensemble prob-average AUROC 0.834 [0.715, 0.914] in-domain; pre-registered gates NOT met (AUROC>=0.85, CI-low>=0.72) | `eval/results/ensemble/nested_loco.json:182-189`, `docs/reports/ensemble_views.md:21` | measured (exploratory, one labeller) |
| EG2 view 0.846 [0.728, 0.913]; pixel view 0.792 [0.666, 0.882] | `eval/results/ensemble/nested_loco.json:166-181`, `docs/reports/ensemble_views.md:19-20` | measured in-domain |
| Rank-average 0.844 [0.717, 0.919]; gates AUROC>=0.85 FAIL, CI-low>=0.72 FAIL | `eval/results/ensemble/nested_loco.json:190-238`, `docs/reports/ensemble_views.md:26-28` | measured |
| S1 head (retrained deterministically in-process from a recorded recipe on the 150 IndEgo windows; head sha256 in each receipt) LOCO AUC 0.802 [0.710, 0.870], thr 0.6, margin 0.25, n=150 | `eval/results/gpu/cascade.json:5-16`, `docs/reports/poc_run.md:5-7,28-33` | measured in-domain |
| Per-pixel conv1d 0.782 [0.627, 0.860] vs HGB 0.758 vs physics-LR 0.715 vs m1-LR 0.673; rule list 0.589 rejected as decider | `eval/results/perpixel/compare.json`, `docs/reports/perpixel_accuracy.md:47-56,196-197` | measured in-domain |
| OOD 160 windows, 138 good / 22 bad, 47 factories, one labeller | `docs/pod/SYNC_UPDATE_2026-10-10_2145.md:6` | measured (labels in pod bundle, txt files absent from this branch) |
| Frozen ensemble OOD AUROC 0.566 (0.433-0.693), balacc 0.546, 75/138 good below 0.5 | `docs/pod/SYNC_UPDATE_2026-10-10_2145.md:22` (source `docs/ood_eval_ensemble_v1_160.txt`, absent from branch) | measured on 160; CIs window-bootstrap, optimistic |
| PoC cascade OOD AUROC 0.600 (0.474-0.724), rejects 53/138 good (38%) and 13/22 bad | `docs/pod/SYNC_UPDATE_2026-10-10_2145.md:22` (source `docs/ood_eval_poc_cascade_160.txt`, absent) | measured on 160; optimistic CIs |
| OOD 60-window precursor: ensemble 0.415 [0.210, 0.647]; EG2 0.586, pixel 0.342; PoC 0.529 [0.316, 0.728], 21/51 good rejected | `eval/results/ensemble/ood_heads.json`, `docs/reports/ensemble_views.md:138-157`, `eval/results/poc/ood_check_60.json`, `docs/reports/poc_run.md:70-84` | exploratory (n_bad=9 < 10), superseded by the 160-window rows |
| PoC run 1587 unique windows, 4.404 h, 58 clips, 47 factories | `eval/results/poc/run_summary.json:6-12`, `docs/reports/poc_run.md:51` | measured, descriptive (unlabelled) |
| PoC verdicts accept 709 / reject 702 / review 176 (44.7/44.2/11.1%); stages S0 0 / S1 1411 / S2 176 | `eval/results/poc/run_summary.json:13-27`, `docs/reports/poc_run.md:53-57` | measured descriptive |
| PoC latency p50 548.1 ms, p90 699.2 ms per 10-s window, sequential single-process | `eval/results/poc/run_summary.json:37-44` | measured |
| fps: 10 fps 0.809 [0.693, 0.890] vs 3.33 fps 0.807 [0.697, 0.873] (all 26 feats); spatial-only 0.690 vs 0.678 | `docs/reports/exp_fps.md:11-17`, `eval/results/exp_fps/e1_01.json` | descriptive, IndEgo only, no equivalence test |
| Keyframe20 INVALID (median 1 I-frame per 10-s window, max 5) | `docs/reports/exp_fps.md:36-40` | INVALID, no AUROC |
| Every-2nd-window frac 0.51 errL 0.049 [0.019, 0.087] | `docs/reports/exp_temporal.md:23`, `eval/results/exp_temporal/E3.01_policies.json` | measured IndEgo n=150; tolerance post-hoc |
| Early exit 4 s 0.860 vs 10.5 s 0.848, saturates 4-6 s | `docs/reports/exp_temporal.md:36`, `eval/results/exp_temporal/E3.03_early_exit.json` | measured IndEgo-only |
| x4.9 combined saving (every-2 + 4 s) arithmetic estimate, decode-bound upper bound | `docs/reports/exp_temporal.md:38` | estimate, NOT measured |
| Receipt = SHA-256 of canonical JSON, NOT signed; same-machine determinism only | `docs/ARCHITECTURE.md:103-110`, `docs/reports/verify.md` | built |
| CPU feature path digest identical across repeats and 1-vs-4 cores | `docs/reports/edge.md:81-83`, `docs/ARCHITECTURE.md:128` | measured |
| 750 tests OK (skipped=9), 34 test files | `make test` (`Makefile:12-15`); ran 2026-10-10 | measured |
| S0 exact rules: BLACK p95<0.03, TOO_DARK mean<0.05, LENS_BLOCKED spatial_var<1e-5, OVEREXPOSED p5>0.70, FROZEN temp_e<5e-4 | `eval/results/gpu/cascade.json:17-48`, `docs/reports/poc_run.md:22-26` | built |
| Experiment snapshot lists 138 ids (E0/E1/E3/E4, L1.01-L1.40, L2.01-L2.34, L3.01-L3.35, B.01-B.15) | `docs/pod/SYNC_UPDATE_2026-10-10_2145.md:50` (snapshot in pod bundle, not in this repo; no `experiments/registry.jsonl` or `tools/exp.py` on this branch) | pod-bundle reference |
| IndEgo: Fraunhofer IPK, CC BY 4.0, clips not in git | `docs/ATTRIBUTIONS.md:9` (licences not checked, per header :5) | third-party |
| Egocentric-10K-Evaluation: Build AI, Apache-2.0 (brief); gemini-2.5-flash AI labels, not human-checked | `docs/ATTRIBUTIONS.md:10-11` (licences per the project brief, unverified) | third-party |
| EmbeddingGemma 2 optional helper, not core; RGB only, not hyperspectral; no generic VLM | `docs/ARCHITECTURE.md:25-26,158-159` | built direction |
| Spatial localisation WITHDRAWN (pooled 0.790 beats grid 0.753) | `docs/pod/SYNC_UPDATE_2026-10-10_2145.md:30` | negative |
| Motion cue inverted (good = less residual flow; wearer-not-walking) | `docs/pod/SYNC_UPDATE_2026-10-10_2145.md:31`, `docs/reports/ensemble_views.md:204-226` | negative |
| Synthetic negatives harmful (0.351) | `docs/reports/ensemble_views.md:119` | negative |
| Certified bounds not achievable at n=150 (support floor 86; effective n ~11-43) | `docs/reports/funnel.md:228,267,460-469`, `docs/pod/SYNC_UPDATE_2026-10-10_2145.md:33` | negative |
| VLM stopped as too probabilistic | `docs/ARCHITECTURE.md:25` | negative |
| Deterministic rule list ranks 0.589, GOOD kept 0.0 — rejected as decider | `eval/results/perpixel/compare.json:58-62`, `docs/reports/perpixel_accuracy.md:107-121` | negative |
| Per-pixel cost: processes 2.14 video-seconds per CPU-core-second (n=6 windows), 769 segments/h/core, ~2340 cores for 5000x real-time (a target, not reached) | `eval/results/perpixel/cost.json`, `docs/reports/perpixel_accuracy.md:199-208` | measured |
| Demo assets in repo: `web/` dashboard + PoC replay, `scripts/poc_*.py`, `docs/screens/poc_*.png` | `web/index.html`, `web/poc/index.html`, `scripts/`, `docs/DEMO.md`, `docs/reports/poc_demo.md` | built; no drop-videos portal with reference column in this repo |
