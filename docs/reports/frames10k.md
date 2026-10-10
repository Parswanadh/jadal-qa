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

## Results
(TBD — runs in progress; interim note pushed at ~min 45.)
