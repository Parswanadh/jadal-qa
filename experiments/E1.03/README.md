# E1.03 — OOD 1080p30-source fps sweep {30,15,10,5,3,2,1}

**Why:** The OOD regime is original 1080p/30fps HEVC, not IndEgo 480p/10fps; the fps answer may differ per regime and the frozen IndEgo model is known not to transfer, so we also need label-free score stability across fps.

**Hypothesis:** Frozen IndEgo model AUROC stays 0.4-0.53 at every fps (no transfer); score rank-correlation across fps is high (>=0.8) so the fps choice does not change OOD ranking; per-feature direct AUROC is more stable than the frozen model.

**Data / label source:** 60 human-labelled Egocentric-10K OOD windows (labels/ood_labels.json: 51 good / 9 bad, human, 1 labeller) decoded from queues/label_queue_random.json source_path+source_start_s (original 1080p HEVC 30fps); plus 100 UNLABELLED queue-2 windows (labels/ood_labels2.json absent) for descriptive score stability only.

**Data regime:** out-of-domain Egocentric-10K 1080p 30fps HEVC, 47 factories, 1 window each

**Method:** decode each window at 160x90 with cv2 (CPU), index rule idx=arange(0,T,step), steps {30:1,15:2,10:3,5:6,3:10,2:15,1:30}, 26 cpu_window_features with TRUE effective fps=30/step; frozen logistic fit on ALL 150 IndEgo windows at 10fps (C=0.3, balanced, StandardScaler); window-bootstrap CI95 B=2000; spearman score rank-correlation across fps; per-feature signed AUROC.

**Owner:** experimenter  **Started:** 2026-10-10 15:17

## Results

**Verdict (2026-10-10 16:15): maybe.** OOD n=60 (51 good/9 bad human; n_bad=9 EXPLORATORY, CI by window optimistic): frozen IndEgo logistic AUROC 0.3595 [0.180,0.554] at 30fps, 0.4314 [0.237,0.627] at 10fps, 0.5468 [0.373,0.715] at 3fps, 0.5512 [0.375,0.717] at 2fps, 0.5468 [0.379,0.708] at 1fps - every CI95 includes 0.5, no transfer at any fps. Label-free score rank stability (spearman) 3vs2 0.9949, 5vs1 0.9813, 10vs5 0.9763, 30vs1 0.8285. Per-feature direct AUROC is fps-invariant for middle-frame features (dct_energy 0.6993, hf_ratio 0.6863, radial_slope 0.6449, range 0.0000) and collapses for band/rPPG features: rppg_pulse 0.7059 at 30fps -> 0.2854 at 2fps -> 0.5000 at 1fps (constant feature there), range 0.4205. 100 queue-2 windows UNLABELLED: descriptive stability only, no accuracy claimed.

Results file: eval/results/exp_fps/E1.03.json

Decision: Do NOT choose the OOD fps from an IndEgo-trained model. OOD fps stays open until an Egocentric-10K-trained head exists (n_bad=9 cannot fit one). For descriptive scoring a cheap fps is acceptable because the frozen ranking is stable below 10fps (rho>=0.98 for 5vs1), with no accuracy claimed. The OOD cost lever is DECODE (2900.4 ms/window clean on the 1080p HEVC source; 6495.7 ms/window in the loaded shard run), not fps.

Cost: decode 2900.4 ms/window clean (9.6679 ms/frame, pareto.json clean_bench_ood) and 6495.7 ms/window (21.652 ms/frame) in the loaded shard run (E1.03.json); features 843.0 -> 32.4 ms/window from 30 -> 1 fps clean (3312.4 -> 168.9 ms/window in the loaded shard run); 60 windows x 7 fps; CPU only, no GPU; loaded/clean ratio 2.24x decode and 3.93-5.21x features (pareto.json clean_vs_shard_ratio)

### Correction log (review of PR #88)

* **F1:** an earlier revision of this file (verdict line 2026-10-10 15:49, superseded above)
  quoted "decode 3004 ms/window (10.0 ms/frame)" and "features 798 -> 32 ms/window". Neither
  number exists in any results JSON; the cost figures above now come from
  `pareto.json` (`clean_bench_ood`, `clean_vs_shard_ratio`) and `E1.03.json`
  (`decode_ms_mean_300f`, `cost.feat_ms_per_window_by_fps`). The superseded line was removed
  from this file; the append-only registry keeps it.
* **F3:** the report's per-feature table cell for `rppg_pulse` at 1 fps read 0.2854 (the 2 fps
  value); it is 0.5000 at 1 fps in `E1.03.json` (`per_feature_signed_auroc["1"]["rppg_pulse"]`).
* **F8:** `E1.03.json` is regenerated from the committed shards by
  `python3 eval/exp_fps/merge_ood.py` at HEAD, so it carries the `ci95`/`cost` blocks and its
  `git_sha` is the HEAD at merge time (merge provenance, not decode-run provenance: the shards
  are git-ignored intermediates re-run by `run_all.sh`). Re-merging reproduces the previous
  numbers exactly apart from `git_sha`.
