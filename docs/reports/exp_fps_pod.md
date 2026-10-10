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
3. **E1.01 / E1.02 (prior art, re-run on the pod):** the pixel view holds within
   0.02 AUROC down to 3.33 fps; the EmbeddingGemma-2 view is *better* at fewer frames.
   The re-run matches the laptop `registry_snapshot.jsonl` summary at the 3 dp that
   snapshot records (0.809 / 0.793 / 0.807 / 0.787 / 0.774); see the E1.01 section for
   what is and is not claimed.

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

Decode is the full 100-frame window in every row (897 ms/window in that run; 112 ms
clean, see the cost section). The pod re-run reproduces the laptop
`registry_snapshot.jsonl` summary **at the precision that snapshot records it**
(3 dp: 0.809 / 0.793 / 0.807 / 0.787 / 0.774 — the `e1_01.json` values round to those).
The laptop 4-dp result file is **not** in this repo, so no "identical to 4 dp" claim is
made. What can be checked in-repo: re-running `merge_e1_01.py` on the committed shards
reproduces this table exactly (the only diff is the `git_sha` field; the same holds for
`E1.03.json`, `E1.04.json` and `e1_02.json`).

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
| `rppg_pulse` | 0.7059 | 0.6841 | 0.5000 | 0.4205 |
| `temp_b2` | 0.6950 | 0.6863 | 0.5000 | 0.1950 |
| `flicker_idx` | 0.6928 | 0.5000 | 0.5000 | 0.1928 |
| `temp_b0` | 0.2353 | 0.2397 | 0.2266 | 0.0131 |
| `illum_r` | 0.1699 | 0.1678 | 0.1612 | 0.0087 |

Reading: middle-frame-only features (`dct_energy`, `hf_ratio`, `radial_slope`,
`global_shift`) and pooled-pixel features are exactly fps-invariant; band-energy
features (`temp_b2/b3`, `flicker_idx`) collapse to chance once the band edge falls
below Nyquist; `rppg_pulse` is broken at low fps because the feature code still uses
its hard-coded `FPS_DEFAULT = 30` (already noted in `e1_01.json`
`fps_comparability`) — it reads 0.7059 at 30 fps, 0.2854 at 2 fps and 0.5000 at 1 fps,
where the feature returns a constant (that 1 fps cell read 0.2854 in an earlier
revision of this table: it was the 2 fps value). Several features are strongly
**inverted** on this domain (`illum_r` 0.17, `temp_b0` 0.24, `luma_p50` 0.22) —
another sign the OOD regime differs from IndEgo (different exposure/lighting).

**`motion_energy` is shown across fps but is NOT comparable across fps**: it is the
mean |frame diff|, which grows with the inter-frame interval `dt`, so its value changes
with the sampling step rather than with the content (`e1_01.json`
`fps_comparability.NOT_comparable_across_fps`; the other rows in that list are
`temp_b0..b3`, `flicker_idx`, `rppg_pulse` and `frozen_frac`). Its column here is
descriptive only — no fps effect should be read from it.

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

**Sanity note on the CI widths:** motion-adaptive's clip-bootstrap CI is ~40 % narrower
than uniform-20's at the same n, clips and protocol (width 0.127 [0.675, 0.802] vs
0.215 [0.650, 0.866]; `E1.04.json` `rows[].ci95`). The bootstrap resamples only 7
clips, so CI width is itself noisy at this n; the narrower interval is recorded here so
it is not silently read as "motion-adaptive is a more stable model" — its point
estimate is *lower* than uniform's.

**Correction logged:** the pre-existing partial `e1_04.json` in the bundle reported
keyframe AUROC **identical** to uniform (0.7866, same CI). Cause: the arm asked
ffprobe for `frame=pkt_pts_time,pict_type`, and in ffprobe 6.1 `pkt_pts_time` is
empty, so no I-frames were ever found and the arm silently degenerated to uniform.
Fixed (use `pts_time`, plus `-skip_frame nokey -read_intervals` — 43 s → 0.6 s per
window); the corrected numbers are above. No past registry line was edited.

## Cost / Pareto (`eval/results/exp_fps/pareto.json`)

`clean` cost = min over **3 bench runs × 3 reps** of one representative window
(`OMP_NUM_THREADS=1`, `cv2` threads 1, host load average 17.6), regenerated by
**E1.05** (`python3 eval/exp_fps/pareto.py --reps 3 --benches 3`). The shard-run means in
`E1.01/E1.03/E1.04.json` were measured while the pod was shared, so they are slower than
these clean minima by a factor that depends on the regime and stage — **2.2–8.0×**
(`pareto.json` `clean_vs_shard_ratio`: OOD decode 2.24×, in-domain decode 5.08× (E1.04)
and 8.01× (E1.01), features 3.11–5.75×). An earlier revision of this report said
"2–4×", which no results JSON supported. Bench-to-bench spread over the three runs
(both directions recorded rather than hidden): decode 1.12–1.84 ms/frame in-domain and
9.67–11.54 ms/frame out-of-domain. Feature cost fits
`feat_ms = intercept + slope·frames` with R² ≥ 0.99:
**IndEgo 2.91 ms/frame** (intercept 4.3 ms), **OOD 2.79 ms/frame** (intercept −10.2 ms).

`indego_480p_10fps_in_domain` (LOCO, n=150):

| setting | frames | AUROC | decode ms/win | feat ms/win | total ms/win |
|---|---|---|---|---|---|
| 10 fps | 100 | 0.8086 | 112 | 289 | 401 |
| 5 fps | 50 | 0.7932 | 112 | 167 | 279 |
| **3.33 fps** | 34 | 0.8068 | 112 | 102 | **214** |
| 2 fps / uniform-20 | 20 | 0.7866 | 112 | 54 | 166 |
| 1 fps | 10 | 0.7736 | 112 | 32 | 144 |
| motion-20 | 20 | 0.7620 | 112 | 54 | 166 |
| keyframe | 20 | 0.7798 | 112 | 54 | 166 |
| random-20 | 20 | 0.7386 | 112 | 54 | 166 |

Pareto front: 10 fps, 3.33 fps, 2 fps, 1 fps and uniform-20.

`ego10k_1080p_30fps_ood` (frozen model → **accuracy axis not usable**, n_bad = 9):

| fps | frames | frozen AUROC | decode ms/win | feat ms/win | total ms/win |
|---|---|---|---|---|---|
| 30 | 300 | 0.3595 | 2900 | 843 | 3743 |
| 15 | 150 | 0.4096 | 2900 | 392 | 3292 |
| 10 | 100 | 0.4314 | 2900 | 246 | 3147 |
| 5 | 50 | 0.4989 | 2900 | 129 | 3029 |
| 3 | 30 | 0.5468 | 2900 | 75 | 2975 |
| 2 | 20 | 0.5512 | 2900 | 55 | 2956 |
| 1 | 10 | 0.5468 | 2900 | 32 | 2933 |

**Decode dominates the OOD regime**: 2900 ms of the 3743 ms at 30 fps
(9.67 ms/frame, 1080p HEVC, CPU). Dropping 30 fps → 1 fps removes 96 % of the feature
cost but only 22 % of the total, because the whole 1080p window is decoded either way.
In IndEgo (480p H.264, 10 fps) decode is 112 ms and the feature stage dominates instead.

## Decisions (per regime)

1. **In-domain pixel view: keep 3.33 fps (34 frames/window).** It is within 0.02 AUROC
   of 10 fps (−0.0018) at 53 % of the total cost (214/401 ms clean); 2 fps (−0.0220)
   misses the 0.02 rule and 1 fps (−0.0350) clearly does. This matches the E1.01
   prior-art decision.
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
bash eval/exp_fps/run_all.sh                 # shards -> merged JSONs -> pareto.json (~25 min, 6 procs)
python3 -m pytest tests/test_exp_fps.py -q   # 26 tests, no network, <30 s
```

`run_all.sh` finishes with `pareto.py --reps 3 --benches 3` (~80 s, E1.05), the same command
recorded in `pareto.json` (`command`) that produced the committed cost numbers — an earlier
revision called `--reps 3` only, which made the committed `n_bench_runs: 3` impossible to
reproduce.
`eval/exp_fps/merge_ood.py`, `merge_e1_01.py`, `merge_e1_04.py` and `run_eg2.py` reproduce
their committed JSONs from the committed shards exactly (only `git_sha` differs), so
`git_sha` in these files means "HEAD at merge/bench time" (provenance of the last
regeneration), not "the machine state the video was decoded on"; the decode shards are
git-ignored intermediates re-created by `run_all.sh`.

Results: `eval/results/exp_fps/{E1.03.json, E1.04.json, e1_01.json, e1_02.json,
pareto.json}`. Raw per-window shards are git-ignored intermediates
(`eval/results/exp_fps/.gitignore`).

## Corrections from the PR #88 review (F1–F8)

1. **F1 (E1.03 cost).** The E1.03 registry line / README / `EXPERIMENTS.md` quoted
   "decode 3004 ms/window (10.0 ms/frame)" and "features 798 → 32 ms/window". Neither
   number appears in any results JSON (and 3004 contradicts the committed Pareto). They
   are retracted: the E1.03 cost now reads 2900.4 ms/window clean (9.6679 ms/frame,
   `pareto.json` `clean_bench_ood`) / 6495.7 ms/window in the loaded shard run
   (`E1.03.json`), features 843.0 → 32.4 ms/window clean (3312.4 → 168.9 ms in the
   loaded run). The stale registry line stays in `registry.jsonl` (the registry is
   append-only; the latest line per id wins) and is flagged as superseded in
   `experiments/E1.03/README.md`.
2. **F2 (E1.04 cost).** "decode 129 ms / feature 53 ms clean" → decode 112.0 ms and
   feature 53.7 ms, read from `pareto.json` `clean_bench_indego["2"]` (130.8/55.6 were
   the previous bench, superseded by the E1.05 re-bench). Used consistently across the
   report, registry, `EXPERIMENTS.md` and README.
3. **F3 (rppg_pulse cell).** The 1 fps cell read 0.2854 (the 2 fps value); it is 0.5000
   (`E1.03.json` `per_feature_signed_auroc["1"]["rppg_pulse"]`, where the feature is
   constant). Fixed, plus the 2 fps value is now stated explicitly in the reading text.
4. **F4 (`motion_energy` caveat).** Added to the reading paragraph: mean |frame diff|
   grows with the inter-frame interval, so the column is not comparable across fps
   (`e1_01.json` `fps_comparability`).
5. **F5 ("2–4× slower").** Removed. The report and `pareto.json` now quote the ratio
   computed per regime/stage in the JSON itself (`clean_vs_shard_ratio`, range
   **2.24–8.01×**), instead of a hand-typed multiplier.
6. **F6 (regeneration).** `run_all.sh` now calls `pareto.py --reps 3 --benches 3`, and
   `pareto.json` records that command, so one command reproduces the committed
   `n_bench_runs: 3`.
7. **F7 (CI width).** One-line sanity note added to the E1.04 section and the E1.04
   verdict: motion20's clip-bootstrap CI is ~40 % narrower than uniform20's (0.127 vs
   0.215) at the same n/protocol — recorded, not read as a better model (7 clips).
8. **F8 (provenance).** `E1.03.json` (and `E1.04.json`, `e1_01.json`, `e1_02.json`) were
   regenerated at HEAD from the committed shards, so the `ci95`/`cost` blocks and the
   `git_sha` come from the same regeneration; `git_sha` is documented as merge/bench
   provenance (see Regenerate above). Also removed the unsupported "reproduces the laptop
   numbers exactly (same to 4 dp)" claim — the laptop 4-dp file is not in this repo; the
   in-repo check is against the 3-dp `registry_snapshot.jsonl` summary.
9. **Registry ids.** The cost/Pareto re-bench is logged as **E1.05** (its own registry line
   and `experiments/E1.05/README.md`); E1.03/E1.04 were closed again so that the live
   registry lines cite the numbers in the results files.
