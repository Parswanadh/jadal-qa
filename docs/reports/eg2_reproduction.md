# EG2 embedding cache: reproduction from the source video (LB1-eg2-repro)

Plain-language summary: the EG2 view of the ensemble is the source of the headline
0.848 (EG2-only) / 0.8524 (two-view) AUROCs. Until now those numbers had only ever
been verified against the *cached* embeddings, never against the video. This report
re-embeds IndEgo windows from the source MP4s on this pod, at the recipe the cache
was built with, and compares.

**Verdict: reproduced in substance, not bit-exactly.** All 150 windows regenerate
with identical frame counts and window cosine 0.99905–0.99979. The residual is
consistent with a ~2–3 LSB per-pixel decode difference — not with any recipe
difference, and not with a wrong frame. End-to-end the published numbers reproduce:
EG2-only **0.848 → 0.8468**, two-view ensemble **0.8524 → 0.8524** (identical to 4 dp).

## Recipe used (copied from the cache producer, not re-tuned)

`eval/frames10k/embed_segments.py`, run as recorded in
`configs/ensemble_v1.json["embed_command"]` — i.e. **without** `--gpu-decode`:

- frames: `np.arange(start_s, min(end_s, dur), 0.5)` (2 fps, 20–21 per 10 s window),
  `cv2.set(CAP_PROP_POS_MSEC, t*1000)` + `read()`; unreadable frames skipped;
- preprocess: BGR→RGB, `PIL.Image.convert("RGB")`, `thumbnail((512,512), BILINEAR)`
  — the videos are 480×480, so the thumbnail is a **no-op** (no resize);
- model: `google/embeddinggemma-2`, local snapshot
  `914f7f89142e33e77833254d9c9b90c3cef7303b`, bf16, **no prompt/prefix** (pure image
  encode), `batch_size=32`, `normalize_embeddings=True`,
  `processing_kwargs={"image": {"max_soft_tokens": 140}}`;
- pooling: per-window mean over L2-normalised frames, then L2-normalise
  (`eval/exp_fps/run_eg2.py::window_mean`).

## Results (all 150 windows, re-embedded from video)

| metric | value |
|---|---|
| frame counts vs `counts` | **150/150 agree** (3005 = 3005 frames) |
| window cosine | min **0.999048**, median **0.999616**, max **0.999785** |
| windows below 0.999 / 0.99 | 0 / 0 |
| max abs elementwise diff (normalised) | 0.00599 |
| EG2-only AUROC (LOCO C=0.01) | cache 0.848 → regenerated **0.8468** (Δ −0.0012) |
| two-view ensemble AUROC | cache 0.8524 → regenerated **0.8524** (Δ 0.0000) |

The cache-side values recomputed here match `eval/results/exp_fps/e1_02.json`
exactly (0.848 / 0.8524), so the comparison uses the published protocol.

## What the residual is — and what it is not

Frame-level statistics are quoted **norm-invariant** (each frame renormalised before
the dot product); raw fp16 dot products are dominated by ±5e-4 norm jitter.

| control | result | reading |
|---|---|---|
| same pixels, different batch order | frame cosine ≥ 0.99997 | bf16 is directionally deterministic here |
| ±1 video frame (0.1 s) re-decode | frame cosine median 0.78 | frame indices are certainly correct |
| gpu-decode preprocessing (512×512 upscale) | window cosine 0.943–0.996 | the cache was NOT built with `--gpu-decode` |
| + uniform ±4 LSB pixel noise | frame cosine 0.99006 vs 0.99533 unperturbed | the residual ≡ a few-LSB pixel difference |
| bf16 vs fp32 | not run (fp32 weights = 2.98 GB > governors) | bf16 rounding is ‖δ‖≈2e-3, 50× smaller than the observed ‖δ‖≈0.097 |

Observed residual: frame cosine median 0.99533 (‖δ‖ ≈ 0.097). A wrong frame index
costs 0.78; the GPU preprocessing variant costs 0.94; **a 2–3 LSB per-pixel
difference costs exactly what we see**. That is the signature of a different
cv2/FFmpeg decoder build (this pod: cv2 5.0.0) or of a re-encoded source copy —
i.e. a difference in the pixels, not in the method.

## Reproduce

```bash
OMP_NUM_THREADS=2 python3 eval/frames10k/verify_cache_reproduction.py --phase decode   # 10-window sample
OMP_NUM_THREADS=2 python3 eval/frames10k/verify_full_reproduction.py --phase decode    # all 150
flock -w 3600 /workspace/state/gpu.slot2.lock env HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
  python3 eval/frames10k/verify_full_reproduction.py --phase encode
OMP_NUM_THREADS=2 python3 eval/frames10k/verify_end_to_end_auroc.py
```

## Caveats (honest)

1. `manifest_gold.json` was rewritten at 2026-10-10T17:19:49Z to the cache producer's
   host paths (`/home/parshu/projects/awing/...`), which do not resolve on this pod
   (0/150). Videos were resolved **by basename** to the bundle copies; the labels
   (`snapshot_utc`, `labels_sha256`, ids, clips, labels, reasons) are unchanged, and
   `segments.npz` + `manifest_gold.json` both match the pristine `SHA256SUMS.txt`.
2. The producer's own video files are not on this pod, so "the bundle MP4 is
   byte-identical to the file the cache was built from" is **not proven**. Given the
   controls above, the only remaining unknown in the chain is a few-LSB pixel
   difference; its effect on the headline metric is ≤0.0012 AUROC.
3. The cache-producing environment's cv2/FFmpeg version is unknown.

Results: `eval/results/pod_verify/eg2_embedding_reproduction.json` (+ `_full`,
`eg2_end_to_end_auroc.json`, `eg2_diagnostics_*.json`). Embeddings stay out of git.
