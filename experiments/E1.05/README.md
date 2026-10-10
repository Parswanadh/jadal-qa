# E1.05 — Cost / Pareto re-bench: clean minima for the E1 fps and selection arms

**Why:** The PR #88 review (F5/F6) found the committed Pareto artefact had no single-command regeneration (`run_all.sh` omitted `--benches`) and its `cost_note` quoted a shard-run multiplier that does not match the committed JSONs; the pod load also changed since the first bench, so re-measure the clean minima and let the script record the per-regime/stage ratio in the JSON instead of a hand-typed range.

**Hypothesis:** Re-benching keeps the same Pareto front (IndEgo 10/3.33/2/1 fps + uniform20, OOD 1-2 fps) and the shard-run means stay slower than the clean minima, by a regime/stage-dependent factor of roughly 2-9x.

**Data / label source:** cost only, no labels: 1 IndEgo window (labels/manifest_gold.json, human) + 1 Egocentric-10K labelled window (labels/ood_labels.json source_path, human); accuracy rows read from E1.01/E1.03/E1.04.json (n=150 in-domain LOCO, n=60 out-of-domain, 51 good/9 bad)

**Data regime:** in-domain IndEgo 480p 10fps H.264 + out-of-domain Egocentric-10K original 1080p 30fps HEVC, decode CPU cv2 160x90

**Method:** eval/exp_fps/pareto.py --reps 3 --benches 3: min over 3 reps x 3 bench runs of decode + cpu_window_features per fps setting, OMP_NUM_THREADS=1, cv2 threads=1; Pareto front over (AUROC, total ms/window) per regime; clean_vs_shard_ratio computed in-script from E1.01/E1.03/E1.04.json

**Owner:** experimenter  **Started:** 2026-10-10 16:12

## Results

**Verdict (2026-10-10 16:20): maybe.** cost re-bench, no labels (n=1 IndEgo window + 1 Egocentric-10K 1080p window), 3 reps x 3 bench runs, min-of-9, OMP_NUM_THREADS=1 cv2 threads=1, host load 17.6: clean minima in-domain 480p decode 112.0 ms/window (1.1198 ms/frame) + features 288.9/166.7/102.2/53.7/32.4 ms/window at 10/5/3.33/2/1 fps; OOD 1080p HEVC decode 2900.4 ms/window (9.6679 ms/frame) + features 843.0/391.8/246.5/128.7/74.9/55.5/32.4 ms/window at 30/15/10/5/3/2/1 fps. Pareto fronts unchanged (in-domain 10/3.33/2/1 fps + uniform20; OOD 1-2 fps). The shard-run means in E1.01/E1.03/E1.04.json are 2.24-8.01x these clean minima, regime/stage-dependent (OOD decode 2.24x, in-domain decode 5.08x, features 3.11-5.75x). Spread over the 3 bench runs: in-domain decode 1.12-1.84 ms/frame, OOD decode 9.67-11.54 ms/frame.

Results file: eval/results/exp_fps/pareto.json

Decision: Cite pareto.json for every clean cost number in the E1 docs. Regenerate with python3 eval/exp_fps/pareto.py --reps 3 --benches 3 (run_all.sh now does); read clean_vs_shard_ratio instead of quoting a hand-typed shard/clean multiplier.

Cost: about 80 s wall, 1 core, no GPU; CPU decode cv2 160x90

### Notes

* This re-bench supersedes the clean minima in the previous `pareto.json` (in-domain decode
  130.8 ms, OOD decode 3711.4 ms, OOD features at 30 fps 956.3 ms) which were measured at the
  same protocol but under a different pod load. The report, registry and READMEs were re-synced
  to the values above; the bench-to-bench spread is recorded in `bench_spread`.
* Decode is the same work in every fps setting (the whole native window is decoded), so the
  `decode_ms` column is identical across fps within a regime.
