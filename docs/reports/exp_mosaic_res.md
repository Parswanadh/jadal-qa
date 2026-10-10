# X3 — Low-res / dense mosaic tiles vs sequential frames

**Claim X3 (owner: monitor) · branch `m/mosaic-res` · experiments MR.01–MR.03**

**Question (Q3).** *"Many pictures in one picture"* — if the tiles inside a mosaic
are lower resolution (e.g. 480p or smaller) instead of 1080p, can we pack **more
frames per canvas** so processing is **FASTER with the SAME accuracy**? Does a
denser, lower-res mosaic change the E2 verdict (mosaic ≤ mean-of-frames → dropped
as a model input)?

**Data / label source.** 150 IndEgo gold windows (`manifest_gold.json`, human, one
labeller; 100 good / 50 bad), 7 clips (`ecu_full, desk, organise, pallet,
logistics4, ecu_late, pallet2`). All source clips are native **480p** (the bundle
ships `mp4_480` only; there is no 1080p source — see Limitations).

**Model / protocol.** EmbeddingGemma-2 (`embeddinggemma-2`, cached, offline,
`max_soft_tokens=140`, L2-normalised). In-domain **leave-one-clip-out** (LOCO)
over the 7 clips, logistic `C=0.01`, **clip-bootstrap CI B=1000 seed 0** — the
same folds/protocol as the group E2 run. Baselines: EG2 window embedding LOCO
**0.848**, 26-feature LOCO **0.810**.

**Timestamps (mandatory).** Every mosaic rendered here carries the per-tile
timestamp `t=X.Xs` in the **top-right** of each tile (`FONT_SIZE=12px`, small but
readable), and every mosaic has a **JSON sidecar with exact per-tile `t` values**
(`jadal/mosaic/build.py`, `save_mosaic`). Timestamps are ON by default; they are
also measured as an explicit ablation in MR.03.

**Cost.** Wall-ms/window measured per config on a 48-window sample, each config
encoded separately (`cfg_timing.json`), split into **build ms/win** (CPU, scales
with canvas px) and **EG2 embed ms/win** (GPU). `canvas_px` is the deterministic
cost proxy.

---

## Findings table (150 gold windows, LOCO)

| combo | frames | tile px | canvas px | AUROC [CI95] | build ms/win | embed ms/win | total ms/win | vs matched control | verdict |
|---|---|---|---|---|---|---|---|---|---|
| MR.01 3×3 tile224 | 9 | 224 | 462 400 | 0.785 [0.64, 0.86] | 34.2 | 34.3 | 68.5 | mean9 −0.066 | — (E2 reference) |
| MR.01 3×3 tile160 | 9 | 160 | 238 144 | **0.804** [0.65, 0.87] | 23.5 | 25.9 | 49.4 | mean9 −0.048 | **adopt** |
| MR.01 3×3 tile112 | 9 | 112 | 118 336 | 0.803 [0.63, 0.87] | 19.7 | 24.0 | **43.7** | mean9 −0.048 | **adopt (cheapest)** |
| MR.01 3×3 tile96 | 9 | 96 | 87 616 | 0.762 [0.57, 0.84] | 22.6 | 24.3 | 46.9 | mean9 −0.090 | drop |
| MR.01 3×3 tile64 | 9 | 64 | 40 000 | 0.736 [0.53, 0.86] | 17.8 | 24.2 | 42.0 | mean9 −0.116 | drop |
| MR.02 2×2 tile112 | 4 | 112 | 52 900 | 0.722 [0.54, 0.80] | 9.5 | 23.8 | 33.3 | mean4 −0.100 | drop |
| MR.02 3×3 tile112 | 9 | 112 | 118 336 | 0.803 [0.63, 0.87] | 19.7 | 24.0 | 43.7 | mean9 −0.048 | best density |
| MR.02 4×4 tile112 | 16 | 112 | 209 764 | 0.797 [0.62, 0.88] | 36.6 | 24.5 | 61.1 | mean16 −0.063 | drop |
| MR.02 5×5 tile112 | 25 | 112 | 327 184 | 0.731 [0.52, 0.85] | 71.5 | 25.7 | 97.2 | mean25 −0.113 | drop |
| MR.02 6×6 tile112 | 36 | 112 | 470 596 | 0.760 [0.60, 0.85] | 80.9 | 26.5 | 107.4 | mean36 −0.081 | drop |
| MR.03 duel 3×3 tile224 | 9 | 224 | 462 400 | 0.785 [0.64, 0.86] | 34.2 | 34.3 | 68.5 | — | loser of duel |
| MR.03 duel 4×4 tile168 | 16 | 168 | 465 124 | **0.845** [0.73, 0.89] | 40.7 | 25.4 | **66.1** | −0.016 vs mean16 | **adopt-geometry** |
| MR.03 abl 4×4 t112 (ts on) | 16 | 112 | 209 764 | 0.797 [0.62, 0.88] | 36.6 | 24.5 | 61.1 | — | — |
| MR.03 abl 4×4 t112 (no ts) | 16 | 112 | 209 764 | 0.777 [0.60, 0.87] | 29.6 | 25.4 | 55.0 | −0.021 (ts helps) | timestamps kept |
| MR.03 abl 4×4 t112 (shuffled) | 16 | 112 | 209 764 | 0.814 [0.65, 0.89] | 34.8 | 25.5 | 60.3 | +0.016 (order ~irrelevant) | — |
| control mean-of-frames N4 | 4 | — | — | 0.823 [0.70, 0.88] | — | — | 4 passes | — | E2 winner |
| control mean-of-frames N9 | 9 | — | — | 0.851 [0.75, 0.91] | — | — | 9 passes | — | E2 winner |
| control mean-of-frames N16 | 16 | — | — | **0.861** [0.76, 0.92] | — | — | 16 passes | — | E2 winner |
| control mean-of-frames N25 | 25 | — | — | 0.844 [0.73, 0.90] | — | — | 25 passes | — | E2 winner |
| control mean-of-frames N36 | 36 | — | — | 0.842 [0.73, 0.90] | — | — | 36 passes | — | E2 winner |
| baseline EG2 window emb (0.848) | ~20 | — | — | 0.848 [0.73, 0.91] | — | — | many passes | — | reference |
| baseline 26-feature (0.810) | — | — | — | 0.810 | — | — | — | — | reference |

Bold = best in column. Control = mean of the **same** frames' EG2 embeddings (E2's
winner), matched folds.

---

## Verdicts (per combo)

### MR.01 — tile-resolution ladder (3×3, timestamps on) → **ADOPT 112–160 px**
Lower-res tiles **do not hurt** down to 112 px and are **cheaper**. Accuracy is
flat/better from 224→112 px (0.785 → 0.803–0.804, within CI), then degrades below
96 px (0.762 @96, 0.736 @64). The smallest tile within 0.02 AUROC of the best is
**112 px** (best is 160 px at 0.804). At 112 px, build+embed drops **68.5 → 43.7
ms/win (−36%)** and canvas drops **462k → 118k px (−74%)**.
*Verdict: adopt 112–160 px tiles for 3×3 — same accuracy, lower cost.* Still
**below mean-of-frames N9 (0.851)** by 0.048.

### MR.02 — density ladder (tile 112, 2×2→6×6) → **DROP** (does not recover the E2 gap)
More frames at low res do **not** beat the mean-of-frames control at matched frame
count at **any** density: mosaic − control = −0.048 (N9), −0.063 (N16), −0.113
(N25), −0.081 (N36). Accuracy peaks at 9 frames (3×3, 0.803); denser grids get
*cheaper per tile but not more accurate*, and cost rises with canvas
(2×2 33 ms → 6×6 107 ms/win). The E2 conclusion **stands** at equal frame count.
*Verdict: drop as a model-input strategy; keep as evidence artefact.*

### MR.03 — equal-canvas duel (9×224 px vs 16×168 px) → **ADOPT the geometry**
At the **same ~462–465k canvas px**, the denser 4×4/168 px mosaic (16 frames)
scores **0.845** vs **0.785** for the coarser 3×3/224 px mosaic (9 frames):
**+0.060 AUROC at −2.4 ms/win** (build 40.7 vs 34.2, embed 25.4 vs 34.3 — the
smaller-tile encode is *cheaper* on GPU because EG2 resizes to a fixed input).
This **supports the user's core claim** at equal canvas pixels: *more, smaller
tiles beat fewer, larger tiles, at the same or lower cost.*
Caveats: the duel also raises the frame count (9→16), so it is "more frames at
equal canvas" more than "same frames at lower res"; and the mosaic (**0.845**)
is still **within 0.016** of mean-of-frames N16 (**0.861**) and **within 0.003**
of the EG2 baseline (**0.848**). Ablation: timestamps help **+0.021**; tile order
is ~irrelevant (shuffle +0.016, i.e. temporal order is not exploited).
*Verdict: adopt the equal-canvas small-tile geometry **if** a mosaic input is used
at all; it does not overturn mean-of-frames as the best input.*

### Overall
- **"Same accuracy, faster"** holds **relative to the 224 px mosaic** (112–160 px
  and the 4×4/168 equal-canvas shape) — genuine, ~35% cheaper, accuracy unchanged.
- The **E2 verdict is NOT overturned**: at matched frame count the mosaic still
  loses to mean-of-frames at every density (−0.05 … −0.11), and the best mosaic
  (0.845) is still below mean-of-frames (0.861) and only ties the EG2 baseline.
- **Adoptability:** the low-res dense mosaic is **adoptable as a cost
  optimisation for a mosaic-based pipeline** and as an **evidence artefact**;
  it is **not adoptable as a replacement for mean-of-frames**.

---

## Image proofs (committed, top-right timestamps on every tile)

Sample window `g001` (label 1, clip `ecu_full`, 480p source). One PNG per
grid × tile cell + the equal-canvas duel pair, each **< 200 KB**, with JSON
sidecars carrying exact per-tile `t` values.

- `docs/experiments/exp_mosaic_res/images/grid3x3_tile224px.png` ·
  `grid3x3_tile160px.png` · `grid3x3_tile112px.png` · `grid3x3_tile96px.png` ·
  `grid3x3_tile64px.png`
- `grid2x2_tile112px.png` · `grid4x4_tile112px.png` · `grid5x5_tile112px.png` ·
  `grid6x6_tile112px.png`
- **equal-canvas duel:** `duel_1080p_3x3_tile224px.png` (fewer, larger tiles) vs
  `duel_480p_4x4_tile168px.png` (more, smaller tiles)
- `index.json` — canvas px, bytes, encode mode, per-cell `t` values.
  (Duel labels denote the deployment framing; the bundle source is 480p.)

## Results JSON

- `eval/results/exp_mosaic_res/MR.01.json` — tile ladder (n, label_source,
  protocol, ci95, cost, per-config build/embed ms, canvas_px).
- `eval/results/exp_mosaic_res/MR.02.json` — density ladder + mean-of-frames control.
- `eval/results/exp_mosaic_res/MR.03.json` — equal-canvas duel + ablation.
- `eval/results/exp_mosaic_res/{raw_eval.json, features.npz, cfg_timing.json}`.

---

## Honest limitations

- **No true 1080p source exists in the bundle** (all gold clips are `mp4_480`).
  The "1080p vs 480p" duel is therefore implemented as the **canvas-geometry**
  comparison (fewer large tiles vs more small tiles at equal canvas pixels) on
  the same 480p source; source-resolution per se is not isolated.
- The equal-canvas duel **also changes frame count** (9→16); it is confounded
  with the density effect. A same-frame-count, same-tile-res, different-canvas
  control was not in scope of the combos requested.
- CIs are wide (n_bad = 50, 7 clips); all deltas < 0.02 are within noise. No OOD
  set was run (task scoped to the 150 gold windows).
- EG2 embed ms/win is **nearly flat** in tile size (~24–34 ms) because the model
  resizes to a fixed input; the real GPU lever is collapsing N frames into **one**
  encode pass, and the real CPU lever is canvas px. Smaller tiles are "faster"
  mainly through lower build cost.
- Single human labeller; no inter-annotator agreement.

## Reproduction

```bash
export OMP_NUM_THREADS=2 JADAL_BUNDLE=/workspace/jadal_pod_bundle
git -C /workspace/repo worktree add -b m/mosaic-res /workspace/wt/m-mosaic-res pod/base
cd /workspace/wt/m-mosaic-res
bash tools/pod/gpu_run.sh python3 eval/exp_mosaic_res/run_res.py --no-artifacts
bash tools/pod/gpu_run.sh python3 eval/exp_mosaic_res/bench_cfg.py --sample 48
python3 eval/exp_mosaic_res/image_proofs.py     # CPU only
python3 eval/exp_mosaic_res/finalize_res.py
```

**Registry:** MR.01 (tile ladder), MR.02 (density ladder), MR.03 (equal-canvas
duel + ablation). Git head is recorded in each `MR.*.json` (`git_sha`).
