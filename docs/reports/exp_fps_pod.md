# exp_fps — E1: frame rate and frame selection for the pixel-feature view (pod)

Branch `feat/exp-fps-pod` · device **A40** (pod host; decode `cv2` 160×90 on CPU,
features `cpu_window_features` float64 on CPU, `OMP_NUM_THREADS=1`, `cv2` threads 1) ·
every number below comes from a results JSON in `eval/results/exp_fps/`.

Two regimes are kept separate everywhere:

| regime | data | labels | protocol |
|---|---|---|---|
| `indego_480p_10fps_in_domain` | IndEgo 480p ~10 fps H.264, 150 windows / 7 clips | **human** (100 good / 50 bad, one labeller) | leave-one-clip-out, clip-bootstrap CI95 (B=1000, seed 0) |
| `ego10k_1080p_30fps_ood` | Egocentric-10K original 1080p 30 fps HEVC, 60 windows | **human** (51 good / **9 bad**) | IndEgo model frozen, applied as-is; CI95 = window bootstrap (B=2000) — **optimistic**, ~1 window per factory |

> **OOD caveat, stated every time:** out-of-domain **n_bad = 9 (exploratory)**, CI by
> window is optimistic. No OOD number here is an accuracy claim for a product.

## Headline

1. **E1.04 (new): no frame-selection strategy beats uniform sampling** at an equal
   20-frame budget (n=150 LOCO). Uniform 0.7866 [0.650, 0.866]; motion-adaptive
   0.7620, keyframe-only 0.7798, seeded random 0.7386 — all CI95 overlap.
2. **E1.03 (new): the IndEgo-trained pixel model does not transfer to
   Egocentric-10K at any fps** (frozen AUROC 0.36–0.55, every CI95 includes 0.5).
   Individual *spatial* features still separate good/bad OOD windows
   (`illum_b` 0.754, `luma_p95` 0.712, `dct_energy` 0.699) and are **fps-invariant**;
   temporal-band and rPPG features are not.
3. **E1.01 / E1.02 (prior art, re-run on the pod, identical numbers):** the pixel view
   holds within 0.02 AUROC down to 3.33 fps; the EmbeddingGemma-2 view is *better* at
   fewer frames.

Prior art lives in `experiments/registry_snapshot.jsonl` (the imported laptop
registry). Its **E1.01/E1.02 lines were copied verbatim into the live
`experiments/registry.jsonl`** on this branch so that `tools/exp.py render` shows the
whole E1 log; E1.03/E1.04 were registered and finished here. Results files for all
four ids are in `eval/results/exp_fps/`.

## E1.01 — IndEgo fps sweep (prior art, regenerated)

`eval/results/exp_fps/e1_01.json` — n=150, LOCO logistic C=0.3 balanced.

| fps | frames/window | AUROC | CI95 | Δ vs 10 fps | feat ms/win |
|---|---|---|---|---|---|
| 10 | 100 | **0.8086** | [0.693, 0.889] | — | 1420 |
| 5 | 50 | 0.7932 | [0.665, 0.878] | −0.0154 | 745 |
| 3.33 | 34 | **0.8068** | [0.697, 0.873] | −0.0018 | 515 |
| 2 | 20 | 0.7866 | [0.650, 0.866] | −0.0220 | 315 |
| 1 | 10 | 0.7736 | [0.631, 0.849] | −0.0350 | 186 |

Decode is the full 100-frame window in every row (897 ms/window in that run).
Regenerating on the pod reproduces the laptop numbers **exactly** (same to 4 dp) —
a determinism check on the port.

## E1.02 — EG2 window-embedding subsample (prior art, regenerated)

`eval/results/exp_fps/e1_02.json` — n=150, LOCO, cached 0.5 s embeddings (no re-embed).

| fps | frames/window | AUROC | CI95 |
|---|---|---|---|
| 2 | 20 | 0.8480 | [0.734, 0.909] |
| 1 | 10 | 0.8596 | [0.760, 0.914] |
| 0.5 | 5 | 0.8702 | [0.763, 0.924] |

Two-view probability average with the pixel view: 0.8524 [0.724, 0.931] at 2 fps,
0.8538 [0.731, 0.927] at 1 fps. Fewer frames are **not** worse here.

## E1.03 — OOD fps sweep on the original 1080p/30 fps sources (new)

`eval/results/exp_fps/E1.03.json` — 60 human-labelled windows (51 good / 9 bad),
decoded from `queues/label_queue_random.json` `source_path` + `source_start_s`
(probed: 1920×1080, 30/1), 300 frames, `idx = arange(0, T, step)`, true effective
fps `30/step` passed to the feature code. Frozen model = logistic C=0.3 balanced fit
on **all 150** IndEgo windows at native 10 fps.

| fps | frames/window | frozen AUROC | CI95 (window boot) | feat ms/win |
|---|---|---|---|---|
| 30 | 300 | 0.3595 | [0.180, 0.554] | 3312 |
| 15 | 150 | 0.4096 | [0.222, 0.606] | 1729 |
| 10 | 100 | 0.4314 | [0.237, 0.627] | 1094 |
| 5 | 50 | 0.4989 | [0.320, 0.678] | 536 |
| 3 | 30 | 0.5468 | [0.373, 0.715] | 372 |
| 2 | 20 | 0.5512 | [0.375, 0.717] | 266 |
| 1 | 10 | 0.5468 | [0.379, 0.708] | 169 |

**Every CI95 includes 0.5.** At 30 fps the frozen direction is *below* chance (0.36).
The upward drift towards 1–2 fps is not evidence that low fps is better — it is the
frozen model being mis-calibrated on this domain; the experiment cannot separate the
two, which is why the verdict is not `adopt`.

Label-free score rank-correlation across fps (spearman, no labels needed):
`3vs2` 0.9949, `5vs1` 0.9813, `10vs5` 0.9763, `30vs1` 0.8285. Below 10 fps the frozen
ranking is essentially unchanged; 30 fps is the outlier.

Per-feature **signed AUROC directly against the OOD labels** (no transfer involved),
selected rows; full table in the JSON:

| feature | 30 fps | 10 fps | 1 fps | range |
|---|---|---|---|---|
| `illum_b` | 0.7538 | 0.7582 | 0.7538 | 0.0044 |
| `luma_p95` | 0.7124 | 0.7124 | 0.7092 | 0.0130 |
| `dct_energy` | 0.6993 | 0.6993 | 0.6993 | **0.0000** |
| `illum_g` | 0.6993 | 0.6972 | 0.6928 | 0.0065 |
| `pca_resid` | 0.6906 | 0.6928 | 0.6906 | 0.0044 |
| `hf_ratio` | 0.6863 | 0.6863 | 0.6863 | **0.0000** |
| `radial_slope` | 0.6449 | 0.6449 | 0.6449 | **0.0000** |
| `motion_energy` | 0.6928 | 0.7146 | 0.6187 | 0.0959 |
| `rppg_pulse` | 0.7059 | 0.6841 | 0.2854 | 0.4205 |
| `temp_b2` | 0.6950 | 0.6863 | 0.5000 | 0.1950 |
| `flicker_idx` | 0.6928 | 0.5000 | 0.5000 | 0.1928 |
| `temp_b0` | 0.2353 | 0.2397 | 0.2266 | 0.0131 |
| `illum_r` | 0.1699 | 0.1678 | 0.1612 | 0.0087 |

Reading: middle-frame-only features (`dct_energy`, `hf_ratio`, `radial_slope`,
`global_shift`) and pooled-pixel features are exactly fps-invariant; band-energy
features (`temp_b2/b3`, `flicker_idx`) collapse to chance once the band edge falls
below Nyquist; `rppg_pulse` is broken at low fps because the feature code still uses
its hard-coded `FPS_DEFAULT = 30` (already noted in `e1_01.json`
`fps_comparability`). Several features are strongly **inverted** on this domain
(`illum_r` 0.17, `temp_b0` 0.24, `luma_p50` 0.22) — another sign the OOD regime
differs from IndEgo (different exposure/lighting).

### Queue-2 sweep — UNLABELLED, descriptive only

`labels/ood_labels2.json` was absent at run time, so the 100 `label_queue_random2`
windows are **unlabelled**: **no accuracy is claimed** for them. What is reported is
score stability (`E1.03.json` → `queue2_unlabelled`):

* frozen-score spearman across fps: `3vs2` 0.9974, `5vs1` 0.9889, `10vs5` 0.9778,
  `30vs1` 0.8265 — same shape as the labelled set.
* frozen score mean by fps: 0.3605 (30) → 0.5077 (10) → 0.5958 (5) → 0.5201 (1).
  The level moves with fps; only the *ranking* is stable. A frozen IndEgo score is
  not a quality measure on this domain.
* decode 8779 ms/window (29.3 ms/frame), features 4952 → 205 ms from 30 → 1 fps.

## E1.04 — frame selection at an equal 20-frame budget (new)

`eval/results/exp_fps/E1.04.json` — n=150 IndEgo human windows, LOCO logistic C=0.3
balanced, clip-bootstrap CI95 B=1000 seed 0. Every arm decodes the **same** 100 native
frames, so the arms differ only in which 20 are fed to the features.

| strategy | AUROC | CI95 | bal. acc | Δ vs uniform | feat ms/win |
|---|---|---|---|---|---|
| **uniform-20** (`arange(0,T,5)`) | **0.7866** | [0.650, 0.866] | 0.725 | — | 220 |
| motion-adaptive (top-19 \|Δframe\| + frame 0) | 0.7620 | [0.675, 0.802] | 0.675 | −0.0246 | 174 |
| keyframe-only (I-frames + uniform padding) | 0.7798 | [0.662, 0.835] | 0.735 | −0.0068 | 167 |
| seeded random (seed 0) | 0.7386 | [0.586, 0.805] | 0.685 | −0.0480 | 176 |

Per-window overlap with uniform-20 and label-free score correlation:

| strategy | mean Jaccard vs uniform | frames shared / 20 | windows identical to uniform | score spearman vs uniform |
|---|---|---|---|---|
| motion-adaptive | 0.1347 | 4.72 | 0 / 150 | 0.7328 |
| keyframe-only | 0.9244 | 19.18 | 63 / 150 | 0.9052 |
| seeded random | 0.0811 | 3.00 | 0 / 150 | 0.8091 |

**Conclusion: no selection strategy beats uniform at equal cost.** Motion-adaptive
picks genuinely different frames (Jaccard 0.13, score ρ 0.73) and is *worse*
(−0.025). At 10 fps IndEgo a 10 s window contains a mean of **1.23 I-frames**, and
**150 / 150 windows have fewer than 20**, so the keyframe arm is always padded back to
uniform (that is why its Jaccard is 0.92); it cannot fill the budget on its own.
Uniform-20 is exactly the E1.01 2 fps row (0.7866), an internal consistency check.

**Correction logged:** the pre-existing partial `e1_04.json` in the bundle reported
keyframe AUROC **identical** to uniform (0.7866, same CI). Cause: the arm asked
ffprobe for `frame=pkt_pts_time,pict_type`, and in ffprobe 6.1 `pkt_pts_time` is
empty, so no I-frames were ever found and the arm silently degenerated to uniform.
Fixed (use `pts_time`, plus `-skip_frame nokey -read_intervals` — 43 s → 0.6 s per
window); the corrected numbers are above. No past registry line was edited.

## Cost / Pareto (`eval/results/exp_fps/pareto.json`)

`clean` cost = min over **3 bench runs × 3 reps** of one representative window
(`OMP_NUM_THREADS=1`, `cv2` threads 1); the shard-run means in
`E1.01/E1.03/E1.04.json` are 2–4× slower because the pod is shared (host load average
17.8 at bench; per-run spread is recorded in `bench_spread`). Feature cost fits
`feat_ms = intercept + slope·frames` with R² ≥ 0.998:
**IndEgo 2.48 ms/frame** (intercept 4.6 ms), **OOD 2.64 ms/frame** (intercept −5.9 ms).

`indego_480p_10fps_in_domain` (LOCO, n=150):

| setting | frames | AUROC | decode ms/win | feat ms/win | total ms/win |
|---|---|---|---|---|---|
| 10 fps | 100 | 0.8086 | 129 | 252 | 381 |
| 5 fps | 50 | 0.7932 | 129 | 130 | 259 |
| **3.33 fps** | 34 | 0.8068 | 129 | 86 | **215** |
| 2 fps / uniform-20 | 20 | 0.7866 | 129 | 53 | 182 |
| 1 fps | 10 | 0.7736 | 129 | 31 | 161 |
| motion-20 | 20 | 0.7620 | 129 | 53 | 182 |
| keyframe | 20 | 0.7798 | 129 | 53 | 182 |
| random-20 | 20 | 0.7386 | 129 | 53 | 182 |

Pareto front: 10 fps, 3.33 fps, 2 fps, 1 fps and uniform-20.

`ego10k_1080p_30fps_ood` (frozen model → **accuracy axis not usable**, n_bad = 9):

| fps | frames | frozen AUROC | decode ms/win | feat ms/win | total ms/win |
|---|---|---|---|---|---|
| 30 | 300 | 0.3595 | 3004 | 798 | 3802 |
| 15 | 150 | 0.4096 | 3004 | 374 | 3378 |
| 10 | 100 | 0.4314 | 3004 | 244 | 3247 |
| 5 | 50 | 0.4989 | 3004 | 123 | 3126 |
| 3 | 30 | 0.5468 | 3004 | 76 | 3079 |
| 2 | 20 | 0.5512 | 3004 | 55 | 3059 |
| 1 | 10 | 0.5468 | 3004 | 32 | 3035 |

**Decode dominates the OOD regime**: 3004 ms of the 3802 ms at 30 fps (10.0 ms/frame,
1080p HEVC, CPU). Dropping 30 fps → 1 fps removes 96 % of the feature cost but only
20 % of the total, because the whole 1080p window is decoded either way. In IndEgo
(480p H.264, 10 fps) decode is 129 ms and the feature stage dominates instead.

## Decisions (per regime)

1. **In-domain pixel view: keep 3.33 fps (34 frames/window).** It is within 0.02 AUROC
   of 10 fps (−0.0018) at 56 % of the total cost; 2 fps (−0.0220) misses the 0.02 rule
   and 1 fps (−0.0350) clearly does. This matches the E1.01 prior-art decision.
2. **Frame selection: dropped.** Uniform sampling stays (E1.04, verdict `drop`). No arm
   beats uniform, no arm reduces decode, and motion-adaptive is worse at equal cost.
3. **OOD fps: not decidable from this experiment** (E1.03, verdict `maybe`). The IndEgo
   model does not transfer at any fps, so no fps can be justified on accuracy. What is
   supported: the frozen *ranking* is stable below 10 fps (ρ ≥ 0.98 for 5 vs 1), so a
   cheap fps can be used for descriptive scoring **provided no accuracy is claimed**.
4. **The OOD cost lever is decode, not fps.** Any real OOD saving needs decode-time
   subsampling or a downscale/NVDEC decode path (not measured here).
5. **EG2 view (prior art): no denser sampling needed** — 0.5–1 fps mean embeddings
   match 2 fps (CIs overlap).

## What was not done, and why

* **No OOD-trained head.** Deciding the OOD fps question needs a model fit on
  Egocentric-10K labels; n_bad = 9 cannot support one (the second batch of ~24 bad
  windows was not merged at run time). The OOD fps question stays open.
* **No decode-time subsampling / NVDEC.** All decode is CPU `cv2`; the 1080p decode
  cost is the dominant OOD term and was not optimised.
* **Cost is measured on a shared pod**, so absolute ms are not portable; the
  run-to-run spread is recorded rather than hidden.
* **No per-reason breakdown** (NO_ACTIVE_WORK / HANDS_NOT_VISIBLE / DUPLICATE /
  TOO_DARK) for the selection arms; n_bad = 50 in-domain would make the per-reason
  cells small.

## Regenerate

```bash
export JADAL_BUNDLE=/workspace/jadal_pod_bundle OMP_NUM_THREADS=1
bash eval/exp_fps/run_all.sh                 # shards -> merged JSONs (~25 min, 6 procs)
python3 eval/exp_fps/pareto.py --reps 3 --benches 3
python3 -m pytest tests/test_exp_fps.py -q   # 17 tests, no network, <30 s
```

Results: `eval/results/exp_fps/{E1.03.json, E1.04.json, e1_01.json, e1_02.json,
pareto.json}`. Raw per-window shards are git-ignored intermediates
(`eval/results/exp_fps/.gitignore`).
