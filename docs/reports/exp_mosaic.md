# EXP-MOSAIC (track E2) — can ONE tiled image replace N frame evaluations? (2026-10-10)

Deterministic QA funnel, builder experiment E2. Tests whether a single mosaic image (grid of N
frames with `t=X.Xs` burned into the top-right of every tile) can replace N sequential frame
evaluations as a **model input**, versus the mean of the same N frame embeddings, on the two
labelled regimes.

## Data & protocol
- **In-domain**: IndEgo human gold, 150 windows / 7 clips (one labeller); leave-one-clip-out
  (LOCO), logistic C=0.01 on L2-normalised features, **clip-bootstrap 95% CI** (B=1000, seed 0).
- **Out-of-domain**: 60 human-labelled Egocentric-10K windows, 51 good / **9 bad** (say it every
  time); logistic **frozen on all IndEgo**, scored OOD; CI by **window bootstrap (optimistic**,
  47 factories over 60 windows (~1.3 windows/factory). OOD n_bad=9 ⇒ exploratory.
- Embedding: `google/embeddinggemma-2` from the local HF cache (offline), thumbnail 512 BILINEAR,
  normalize, `max_soft_tokens=140`, batch 64, bf16. Device **A40** (VRAM peak 3967 MB).
- Frame rule (all datasets): `t_i=(i+0.5)*10/N`, `src_frame_index=round(t_i*native_fps)`
  (IndEgo 480×480@10fps; Egocentric-10K label windows 854×480@30fps).
- Baselines in-repo: cached EG2 window-embedding logistic 0.848 [0.73,0.91] (reproduced here as the
  mean-of-frames control); 26 per-pixel features 0.81 — **single observed run** of
  `tools/port/baseline_logistic_loco.py`, not a logged experiment / results JSON.
- Regen: `bash tools/pod/gpu_run.sh python3 eval/exp_mosaic/run_variants.py` (cwd=worktree).
  Results: `eval/results/exp_mosaic/E2.0{1..7}.json`. Tests: `tests/test_mosaic_build.py` (15 pass;
  collected by `python3 -m unittest discover -s tests -p 'test_mosaic_build.py'`).

## E2.01/E2.02 — mosaic vs mean-of-frames, matched frames (N = spec grids + fps{1,2,3,5,10}×10s)
| N (fps_used) | mosaic IndEgo | mean-of-frames IndEgo | Δ | mosaic OOD | mean OOD |
|---|---|---|---|---|---|
| 4 (0.4) | 0.757 [0.61,0.83] | 0.822 [0.70,0.88] | −0.065 | 0.451 [0.20,0.72] | 0.420 |
| 6 (0.6) | 0.791 [0.63,0.86] | 0.824 [0.69,0.90] | −0.033 | 0.638 [0.39,0.84] | 0.571 |
| 9 (0.9) | 0.785 [0.64,0.86] | 0.852 [0.75,0.91] | −0.067 | 0.329 [0.17,0.49] | 0.627 |
| 12 (1.2) | 0.787 [0.66,0.88] | 0.853 [0.76,0.91] | −0.066 | 0.423 [0.27,0.58] | 0.623 |
| 16 (1.6) | 0.813 [0.68,0.88] | 0.860 [0.76,0.92] | −0.048 | 0.390 [0.20,0.59] | 0.580 |
| 10 (1.0) | 0.784 [0.62,0.87] | 0.830 [0.71,0.90] | −0.046 | 0.525 [0.30,0.72] | 0.562 |
| 20 (2.0) | 0.756 [0.61,0.82] | 0.851 [0.75,0.91] | −0.095 | 0.336 [0.16,0.52] | 0.525 |
| 30 (3.0) | 0.842 [0.74,0.89] | 0.843 [0.74,0.90] | −0.001 | 0.155 [0.05,0.28] | 0.553 |
| 50 (5.0) | 0.727 [0.60,0.80] | 0.847 [0.74,0.91] | −0.120 | 0.331 [0.15,0.53] | 0.610 |
| 100 (10.0) | 0.633 [0.51,0.68] | 0.847 [0.74,0.90] | −0.214 | 0.322 [0.19,0.48] | 0.573 |

The mean-of-frames control reproduces the cached EG2 reference (0.848 → 0.852–0.860). The mosaic
is **below** it on almost every N (deltas −0.001 … −0.214); the only N within 0.02 in-domain
(N=30) has OOD 0.155 (worse than chance). Accuracy falls with N: 10×10 tiles at 224 px are crushed
to ~50 px each by the 512-px preprocessor.

**The IndEgo CIs overlap at every N** — e.g. N=9 mosaic [0.64,0.86] vs mean-of-frames [0.75,0.91]
— so there is **no significant separation**. The "drop" verdict is the *E2 point rule* (within 0.02
AUROC of mean-of-frames on BOTH regimes) applied to a *consistent* direction (mosaic below the
control at every N) plus the OOD collapse at N=30, not a statistically significant difference.

## E2.03–E2.07 — structure controls
| variant | IndEgo AUROC [CI95] | OOD | vs mosaic N9 | vs mean N9 |
|---|---|---|---|---|
| E2.03 shuffled tile order | 0.801 [0.64,0.89] | 0.423 | +0.016 | −0.051 |
| E2.04 no timestamps | 0.810 [0.68,0.88] | 0.444 | +0.025 | −0.042 |
| E2.05 single middle frame | 0.784 [0.69,0.85] | 0.514 | −0.001 | −0.068 |
| E2.06 motion mosaic \|I_t−I_{t−1}\| | 0.863 [0.78,0.91] | 0.401 | +0.078 | +0.011 |
| E2.07 pooled per-tile stats (mean,std) | 0.826 [0.69,0.88] | 0.586 | +0.041 | −0.026 |

Order and timestamps are **not load-bearing** (shuffled and no-timestamp are within CI of the
ordered mosaic). A **single middle frame** (0.784, 1/N the embedding passes) equals the N=9 mosaic
— tiling N frames adds nothing over one frame in-domain. The motion mosaic is the best in-domain
variant but **fails to transfer** (OOD 0.401); consistent with the earlier "motion cue inverted /
part may be head stability, not hand work" note. The pooled no-layout control confirms layout is
not load-bearing.

## Decision (per the E2 rule: within 0.02 AUROC of mean-of-frames at ≤1/4 the cost on BOTH regimes)
**DROP as a model input.** The mosaic is not within 0.02 of mean-of-frames on IndEgo for most N and
is inconsistent / often worse on OOD (0.155–0.638 vs 0.420–0.627); the one in-domain-near point
(N=30) collapses on OOD. **ADOPT only as the human-evidence artefact** (dashboard "why" view): the
deterministic PNG + sidecar is a compact reviewer-facing record of what the verdict saw. The
`mean-of-frames` EG2 baseline stays the model input (E2.02 `adopt`, control).

**Scope note (receipt integration NOT in this PR).** The mosaic spec says the receipt should hash
the mosaic bytes + sidecar. **Nothing here wires that in**: this PR only builds the PNG + sidecar
and runs the E2 evaluation. The dashboard/receipt integration is prospective and not implemented.

## Cost (measured, A40)
Full 210-window sweep: decode 873 ms/window, build 1628 ms/window (all 10 mosaics + variants),
embed 5409 ms/window; 1 embedding pass for the mosaic vs N for the control (≤1/4 for N≥4); VRAM
peak 3967 MB. Per-mosaic PNG (N=9, 224 px) measured on the 210 reference artefacts: mean 590 KB,
range 349–743 KB (`python3 -c "import glob,os; s=[os.path.getsize(p) for p in glob.glob('/workspace/artifacts/mosaics/*.png')]; print(len(s), min(s), max(s))"`);
this is a filesystem measurement of the artefacts, not a logged experiment.

## Artefacts
`/workspace/artifacts/mosaics/` — 210 windows × (`<id>.png` + `<id>.json` sidecar), reference N=9,
tile 224, outside git. Sidecar keys: `window_id, source, grid, tile_px, fps_used, tiles
[{index,t_s,src_frame_index,x,y,w,h}], mosaic_sha256, builder_version, font{file,size}`. PNG is
byte-identical across repeat builds (test). The `font` field records the font file actually used
(falls back to `PIL:load_default`), so a PNG-hash mismatch across environments is explainable:
identical pixels need the identical `(font file, size)`.

## Honesty / not done
One human labeller, 7 clips, ~3 wearers (effective n ≈ 11–43); OOD n_bad=9 (exploratory, optimistic
CI); IndEgo-trained models did not transfer. Tile-px sweep {112,160,224} was built/tested but the
accuracy sweep used 224 only. The fps axis is interpreted as N = fps×10 s tiles (grids beyond the
spec set 4/6/9/12/16 use the exact near-square factorisation); documented deviation. No "first/
SOTA/novel/signed" claims; receipts are content-hashed, not signed.
