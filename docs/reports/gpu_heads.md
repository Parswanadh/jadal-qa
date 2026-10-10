# GPU multi-head training (DRAFT — numbers filling in)

Branch `feat/gpu-heads`, issue #62. Engine: `jadal/gpuheads/` — K heads in one
module, trained simultaneously via `torch.bmm` (K = 12 per hidden-size group,
4 groups → 48 configs: hidden {0, 64, 256, 1024} × wd × lr × seeds × class-weight,
dropout-free). Deterministic: `torch.manual_seed`, deterministic algorithms,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, fixed batch order, float32.
Peak VRAM ≤ 3 GiB for our process (measured per run, see JSONs).

## Label sources (never mixed in one headline number)
- Frame-level: AI labels from the EG2/spectral workers (hand_count, active_labor),
  n = 10,000 frames (spectral) / 7,000 EG2 shards at time of writing.
- Window-level: HUMAN gold, `spectral_data/manifest_gold.json`
  (snapshot 2026-10-10T10:53:24Z, sha 43ab44e5…, 150 windows / 7 clips,
  usable=100 / bad=50: NO_ACTIVE_WORK 23, HANDS_NOT_VISIBLE 14, DUPLICATE 6,
  TOO_DARK 6, UNDECODABLE 1). READ-ONLY, never written.

## M1 frame-level (5-fold stratified CV, nested selection inside train folds)
AI labels, protocol: outer 5-fold stratified; per-fold winner picked by inner
2-fold CV on the outer-train only, refit, scored held-out. Bootstrap CI95 (n=1000).
`eval/results/gpu_heads/m1_*.json`. (CPU-fit interim; GPU full-grid queued under flock.)

| features | target (AI) | n (pos rate) | many-heads AUROC [CI95] | single balanced-logistic AUROC | delta |
|---|---|---|---|---|---|
| spec-53 | hands≥1 | 10000 (0.964) | 0.824 [0.802, 0.845], bal-acc 0.755, F1 0.877 | 0.821 | **+0.002** |
| spec-53 | active_labor | 10000 (0.917) | 0.770 [0.753, 0.786], bal-acc 0.704, F1 0.825 | 0.763 | **+0.007** |
| EG2-768 | hands≥1 | 8000 (0.964) | 0.948 [0.934, 0.961], bal-acc 0.889, F1 0.962 | 0.959 | **−0.011** |
| EG2+spec | hands≥1 | pending (run in progress) | | | |

Honest read: many-heads + nested selection ≈ a well-specified single
(class-weighted) logistic head (±0.01) on this AI-labelled frame task — the
"many heads" win is nil here, not a breakthrough. The big effect is FEATURES:
EG2-768 (0.95) >> spectral-53 (0.82) for hands-visible. (An unweighted
single logistic collapses to ~0.50 on these imbalanced labels — reported only
to document the baseline pitfall, not as a comparison.)

## M2 human windows (THE KEY TEST)
Frame head (selected on AI frame labels only) → all 101 frames/window →
pool (mean, max, q10/25/50/75/90, frac>0.5, std) → logistic head, LOCO over 7 clips.
- AUROC 0.668, seg-bootstrap CI95 [0.563, 0.774], **clip-bootstrap CI95 [0.525, 0.814]**
  (n=150, human labels). Bal-acc 0.575, F1 0.815.
- Frame-level inner AUROC was 0.811 → window numbers FAR below frame numbers:
  domain gap (JPEG stills vs H.264 video, factory vs assembly) + label gap
  (AI hand/work proxy vs human usable judgement). Reported honestly.
- Per-reason recall@0.5: TOO_DARK 1.00 (6/6), NO_ACTIVE_WORK 0.09,
  HANDS_NOT_VISIBLE 0.07, DUPLICATE 0.00, UNDECODABLE 0.00.
  Per-reason-vs-usable LOCO AUROC: NO_ACTIVE_WORK 0.735, TOO_DARK 0.453,
  HANDS_NOT_VISIBLE 0.354, DUPLICATE 0.142 (all n≤23 — noise dominates).

## M3 determinism — PENDING (GPU queue)
## M4 cost — PENDING
- Window scoring: 0.07 ms/window (CPU forward of 1 head over 101×53).

## Feeder-bottleneck note (orchestrator RESOURCE FIX, recorded as a ladder rung)
CPU JPEG decode+resize starves the GPU (machine load >100, GPU ~5-20% util):
reference `spectral_data/nvjpeg_decode_example.py` — GPU nvjpeg decode +
on-GPU resize = 192 frames/s vs 24.7 frames/s CPU PIL (1 thread), 0.66 GiB peak
on real 1920×1080 frames. Our training maths already lives in batched torch.bmm
(tiny VRAM); rules followed: OMP/MKL/OPENBLAS=2, ≤3 workers, no n_jobs=-1,
float32 tensors, no new heavy CPU jobs while queued.
