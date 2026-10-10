# eval/exp_fps — E1: frame rate and frame selection for the pixel-feature view

Deterministic study of **how many frames per 10 s window** the 26 per-pixel
features (`jadal/gpu/features.py::cpu_window_features`) need, and **which frames**
to pick, per data regime.

## Files

| file | what it does |
|---|---|
| `fps_lib.py` | shared lib: decode-once at 160x90, `idx=arange(0,T,step)` subsample, LOCO + clip-bootstrap CI, paths resolved from `$JADAL_BUNDLE` |
| `run_fps.py` | `--mode sweep` (E1.01 IndEgo fps sweep) / `--mode select` (E1.04 equal 20-frame budget) |
| `run_ood.py` | E1.03: sweep the **original 1080p/30 fps HEVC sources** of the OOD queues at {30,15,10,5,3,2,1} |
| `run_eg2.py` | E1.02: EmbeddingGemma-2 view, cached 0.5 s embeddings subsampled (no re-embed) |
| `merge_e1_01.py` / `merge_e1_04.py` / `merge_ood.py` | merge shards -> `eval/results/exp_fps/*.json` |
| `pareto.py` | accuracy-vs-cost Pareto rows per regime (clean cost = min over repeats) |
| `run_all.sh` | full regeneration, 6 processes at a time, `OMP_NUM_THREADS=1` |

## Protocol

* in-domain: 150 IndEgo human-labelled windows, 7 clips, **leave-one-clip-out**,
  `StandardScaler` + logistic `C=0.3` balanced, clip-bootstrap CI95 (B=1000, seed 0).
* out-of-domain: 60 human-labelled Egocentric-10K windows (51 good / 9 bad,
  **n_bad = 9 -> exploratory, CI by window is optimistic**); the IndEgo-trained
  model is **frozen** and applied as-is. It does not transfer.
* every decode is CPU (`cv2`, 160x90 analysis resolution, `cv2.setNumThreads(1)`),
  every feature call gets the **true effective fps** (`30/step`, `10/step`).

## Regenerate

```bash
export JADAL_BUNDLE=/workspace/jadal_pod_bundle OMP_NUM_THREADS=1
bash eval/exp_fps/run_all.sh          # ~20 min on the A40 pod, 6 procs
```

Shards (`shard_*.json`, `ood*_shard_*.json`) are intermediates written into
`eval/results/exp_fps/` and are git-ignored; the merged JSONs are committed.

Tests: `python3 -m pytest tests/test_exp_fps.py -q` (no network, no GPU, <30 s).
