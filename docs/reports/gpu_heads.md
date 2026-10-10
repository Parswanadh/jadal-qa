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
| EG2+spec-821 | hands≥1 | 8000 (0.964) | 0.908 [0.886, 0.930] | 0.959 | **−0.051** |

Honest read: many-heads + nested selection ≈ or below a well-specified single
(class-weighted) logistic head (−0.05…+0.007) on this AI-labelled frame task —
no "many heads" win found; at 821-d the batched engine underfits at 20–40
epochs (needs more epochs, noted as a limitation). The big effect is FEATURES
(single-head numbers, same folds): EG2-768 (0.959) ≈ EG2+spec (0.959) >>
spectral-53 (0.821) for hands-visible — adding spectral to the embedding adds
+0.000. (An unweighted single logistic collapses to ~0.50 on these imbalanced
labels — reported only to document the baseline pitfall, not as a comparison.)
Alignment sanity: EG2/spectral AI hand labels agree 1.00 on overlapped ids.

## M2 human windows (THE KEY TEST)
Frame head (selected on AI frame labels only) → all 101 frames/window →
pool (mean, max, q10/25/50/75/90, frac>0.5, std) → logistic head, LOCO over 7 clips.
- hands-head: AUROC 0.668, seg-bootstrap CI95 [0.563, 0.774],
  **clip-bootstrap CI95 [0.525, 0.814]** (n=150, human labels).
  Bal-acc 0.575, F1 0.815.
- work-head (active_labor proxy): AUROC **0.408**, clip-CI95 [0.361, 0.577]
  spans 0.5 — no evidence of transfer; per-clip AUROC swings 0.05–0.88.
  The AI activity proxy does not predict human usable.
- Frame-level inner AUROC was 0.81 (hands) → window numbers FAR below frame
  numbers: domain gap (JPEG stills vs H.264 video, factory vs assembly) +
  label gap (AI hand/work proxy vs human usable incl. capture quality).
  Reported honestly; EG2-window transfer not run (upstream
  `human_segments/` embeddings still missing at report time).
- Per-reason recall@0.5: TOO_DARK 1.00 (6/6), NO_ACTIVE_WORK 0.09,
  HANDS_NOT_VISIBLE 0.07, DUPLICATE 0.00, UNDECODABLE 0.00.
  Per-reason-vs-usable LOCO AUROC: NO_ACTIVE_WORK 0.735, TOO_DARK 0.453,
  HANDS_NOT_VISIBLE 0.354, DUPLICATE 0.142 (all n≤23 — noise dominates).

## M3 determinism
5 repeated trainings, same seed: sha256 `e8af4bb8…` 5/5 identical,
per-window pooled outputs identical 5/5 — **on CPU**. GPU-lock congestion
(~10 waiters) meant no GPU training slot completed before report time, so the
on-GPU bit-identity claim is NOT established; GPU M3 job remains queued
(`eval/gpu_heads/run_m3.py --device cuda`). Explicitly NOT claimed: anything
across devices, drivers, or library versions.

## M4 cost (measured; CPU-fit numbers, GPU pending lock)
- Training throughput: 8.1 heads/s (spec-53, 10k frames) … 1.4 heads/s
  (EG2-768, 8k frames) on 2 CPU threads; batched bmm keeps per-process tensors
  < 100 MB (≤3 GiB VRAM budget unthreatened by design; GPU peak unmeasured).
- Window scoring: 0.07 ms/window (1 head × 101 frames × 53 feats).
- Time/window end-to-end (M2 incl. LOCO fit): ~10 s total / 150 windows.

## Feeder-bottleneck note (orchestrator RESOURCE FIX, recorded as a ladder rung)
CPU JPEG decode+resize starves the GPU (machine load >100, GPU ~5-20% util):
reference `spectral_data/nvjpeg_decode_example.py` — GPU nvjpeg decode +
on-GPU resize = 192 frames/s vs 24.7 frames/s CPU PIL (1 thread), 0.66 GiB peak
on real 1920×1080 frames. Our training maths already lives in batched torch.bmm
(tiny VRAM); rules followed: OMP/MKL/OPENBLAS=2, ≤3 workers, no n_jobs=-1,
float32 tensors, no new heavy CPU jobs while queued.
