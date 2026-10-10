# FRAMES-10K report (draft — numbers filled as runs complete)

## Provenance (read first)
- Frame labels (`hand_count`, `active_labor`) on the 10k-frame parquet files are
  **AI-produced (gemini-2.5-flash with a prompt; NOT human)** — see
  `/home/parshu/projects/awing/data/ego10k_eval/README.md`. Every agreement
  number below measures agreement with the AI labeller, not with humans.
- Segment labels (150 ten-second IndEgo segments: 100 usable / 50 bad) are the
  **human's** (single labeller, read-only snapshot of
  `spectral_data/manifest_gold.json`). Bad reasons: NO_ACTIVE_WORK 23,
  HANDS_NOT_VISIBLE 14, DUPLICATE 6, TOO_DARK 6, UNDECODABLE 1. 7 clips.
- Backbone: EmbeddingGemma-2 (`/home/parshu/projects/awing/data/eg2`),
  bfloat16, thumbnail longest side 512 BILINEAR, L2-normalised, batched encode
  (serving uses batch-size-1 receipts; noted, not comparable bit-for-bit).

## Methods
- (1) Embed `egocentric_10k.parquet` (factory domain, 10k frames) with EG2,
  float16 checkpoint shards under `eval/results/frames10k/egocentric_10k/`.
- (2) Frame heads: logistic regression on frozen embeddings for `active_labor`
  (yes/no) and `hands_ok` (hand_count>=1); stratified 5-fold CV, C in
  {0.01,0.1,1,10} via inner 3-fold on train folds only.
- (3) KEY TEST: same heads applied to every 0.5 s frame of each human-labelled
  segment (~20 frames/segment, whole-segment coverage); per-segment pooling
  (mean/max/p25/p75/frac>0.5/var x {p_work, p_hands}); leave-one-clip-out CV
  vs HUMAN label; arms: (i) pretrained-pooled, (ii) human-only embedding head,
  (iii) combo. AUROC + clip-bootstrap 95% CI, balanced accuracy.
- (4) Cost: encode frames/s, wall frames/s, VRAM peak; Matryoshka truncation
  768/256/128 on frame heads.

## Results (INTERIM, 2026-10-10 ~16:30 IST — partial; full 10k + transfer next)

### (2) Frame heads vs AI labels — first 2,000 factory frames
`eval/results/frames10k/partial_6k/frame_cv.subset_2k.json`. Labels AI
(gemini-2.5-flash; NOT human). Protocol: stratified 5-fold CV, C in
{0.01,0.1,1,10} via inner 3-fold on train folds only. n=2000,
hands>=1 frac=0.960, work=yes frac=0.917 (skewed — balacc at 0.5 suffers).

| target | dim | CV AUROC | OOF AUROC [95% boot CI] | balacc@0.5 |
|---|---|---|---|---|
| active_labor | 768 | 0.911 | 0.911 [0.882, 0.934] | 0.700 |
| active_labor | 256 | 0.906 | 0.905 [0.877, 0.929] | 0.681 |
| active_labor | 128 | 0.902 | 0.901 [0.877, 0.923] | 0.630 |
| hands_ok | 768 | 0.968 | 0.970 [0.956, 0.981] | 0.711 |
| hands_ok | 256 | 0.966 | 0.967 [0.952, 0.980] | 0.663 |
| hands_ok | 128 | 0.952 | 0.954 [0.933, 0.971] | 0.577 |

### (2b) Lean heads on 8,000 AI frames (C=1 fixed, dim 768)
`eval/results/frames10k/partial_8k/frame_cv.lean_8k.json` (shards 0-7, all CPU
PIL decode path).
- active_labor: AUROC 0.9026 [0.8906, 0.9141], balacc 0.7394
- hands_ok: AUROC 0.9587 [0.9481, 0.9684], balacc 0.7484
(The C-grid 6k run died to system OOM; lean fixed-C rerun used float32 +
thread caps and survived. Label skew persists: work=yes 0.917, hands>=1 0.964.)

Matryoshka truncation 768->128 loses ~0.01-0.015 AUROC; balacc drops more
(threshold not retuned — operating point, not ranking, degrades).
### (3) KEY TEST: pooled frame heads -> HUMAN label (LOCO, 7 clips)
`human_segments/transfer.json`. n=150 (100 usable/50 bad), ~20 frames/seg
@0.5 s (3005 frames, GPU-nvjpeg). Heads: lean_8k (8000 AI factory frames,
CPU-PIL decode; gap noted, shared across arms). LOCO CV, inner-LOCO C pick,
class_weight=balanced. AUROC + clip-bootstrap 95% CI:

(i) pretrained_pooled (12 feats): 0.808 [0.699,0.862], balacc 0.715.
(ii) human_only_embed (768-d): 0.816 [0.710,0.892], balacc 0.730.
(iii) combo (780-d): 0.817 [0.716,0.863], balacc 0.745.
Per reason: HANDS_NOT_VISIBLE ~0.87; NO_ACTIVE_WORK ~0.74 (hardest, n=23);
DUPLICATE ~0.83; TOO_DARK 0.85-0.97 (n=6); UNDECODABLE ~0.98 (n=1).
CONCLUSION: no evidence 8k-AI pretraining helps beyond the 150 human labels
(CIs overlap fully; combo ties human-only). All CIs span ~0.70-0.89: with 7
clips nothing is resolved. Domain gap + AI-label skew (92% yes) limit transfer.

### (4) Cost (measured, RTX 4070 Laptop 8 GB, shared-loaded box)
Segments (`human_segments/meta.json`): 3005 frames — decode 229 s (cv2
seeks, CPU) + encode 205 s => 14.6 fps encode, 6.9 fps wall, VRAM peak
2.82 GB @batch 32 bfloat16, GPU-nvjpeg path. 10k shards (CPU-PIL):
~1000 frames / 2-9 min wall per shard incl. model load (~4-8 fps under
contention); GPU-decode shards 8-9 + concat pending. Feeder rung:
orchestrator nvjpeg 192 vs 24.7 fps stands; our seg wall rate is
encode-bound. Throughput verdict: EG2 needs CUDA (no Pi-CPU path); at
~7 fps wall, EG2-on-everything cannot touch the 5000x funnel target —
a cheap whole-video tier must gate it (tiering analysis for the PoC).

## Resource-ladder note (feeder bottleneck, orchestrator 16:43 IST)
Machine CPU/RAM saturated while GPU starved by CPU-side JPEG decode/resize +
oversubscribed BLAS threads. Reference measurement (orchestrator,
`spectral_data/nvjpeg_decode_example.py`, real 1920x1080 frames):
GPU nvjpeg + cuda resize = 192 frames/s vs CPU PIL decode+resize = 24.7
frames/s (1 thread), 0.66 GiB peak. Compliance: new GPU-feeding stages use
`eval/frames10k/gpu_pre.py` (nvjpeg + cuda bilinear to longest-side 512);
every new python starts with OMP/MKL/OPENBLAS_NUM_THREADS=2; no n_jobs=-1;
float16 + small shards (no memmap needed at these sizes). Shards 0-7 used the CPU PIL path; shards 8-9 and segments use the GPU path
— preprocessing differs slightly (gap unmeasured); the transfer comparison
stays valid because all arms share the same segment embeddings.
(End of report.)
