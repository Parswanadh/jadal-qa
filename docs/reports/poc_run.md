# PoC run report — Egocentric-10K through the per-pixel funnel

**Honesty banner (binding): there are NO human labels for this Egocentric-10K
video; every number below is a descriptive rate on unlabelled video, NOT
accuracy.** Accuracy comes only from the 150 human-labelled IndEgo windows
(LOCO AUROC 0.80, 95% CI 0.71–0.87 — wide, n=150 too small for certification).
Thresholds are not certified. Not measured on a Raspberry Pi. Receipts are
content-hashed, not signed. RGB only (3 bands) — hyperspectral is NOT claimed.

## Pipeline

1. **Normalise (domain shift, documented):** 1080p/30fps HEVC input is scaled
   to the head's 160×90 working resolution. Temporal/spectral features run on
   an effective 10 fps stream produced by 3-tap boxcar + decimate (anti-alias
   approximation of a native 10 fps camera, the IndEgo training condition);
   ALL decoded frames feed the S0 probe/noise stats. Two sequential,
   O(1)-memory passes per clip: pass A (160×90, features+S0+verdict), pass B
   (480 px wide, 3 thumb frames per window). No per-window seeks.
2. **S0 exact rules** (first 2 s probe; 2×2 box of the working-res luma as the
   documented equivalent of the 80×45 gray probe; same ops/thresholds as
   `eval/gpu/cascade.py`): BLACK p95<0.03, TOO_DARK mean<0.05, LENS_BLOCKED
   spatial_var<1e-5, OVEREXPOSED p5>0.70, FROZEN temp_e<5e-4 — each firing
   carries `{feature, value, op, threshold}` evidence.
3. **S1 logistic head:** StandardScaler + LogisticRegression(max_iter=2000),
   full fit on all 150 IndEgo gold windows from `eval/results/gpu/features.json`
   (LOCO was reporting-only). No frozen-coefficient file exists in the repo, so
   the head is retrained deterministically in-process; sha256 of the canonical
   coefficient/scaler/threshold JSON is recorded per window
   (`versions.head_sha256`). Operating point: threshold 0.6 (Youden on LOCO
   out-of-fold p), margin 0.25 → |p−0.5|<0.25 is `review`/stage **S2** (handed
   on; heavy tier NOT run in the PoC).
4. **Receipt:** `receipt_sha256 = sha256(canonical({window_id, verdict, reasons,
   features_digest, versions}))`; `features_digest` over 4-dp features.
   Re-verify: `scripts/poc_run.py --verify WINDOW_ID` re-decodes the clip
   sequentially to the same chunk and recomputes (must use the same decode
   path — NVDEC pixels differ from CPU by meanabsdiff ≈ 0.05, so the run
   records `decode_path` per window).

## Decode path

Production: `cpu-sequential` (deterministic, no GPU contention). NVDEC chain B
(`scale_cuda=160:90:format=nv12,hwdownload`) verified working under
`flock tools/gpu.lock` (300 frames/10 s window in 2.33 s) but its pixels differ
from CPU decode, so it is NOT mixed into one run. VRAM peak via nvidia-smi.

## Results (`poc_data/summary.json`, generated 2026-10-10 ~12:15 UTC)

- **1587 windows, 4.404 h video from 58 clips** (47 factories/workers), `partial: false`
- by_verdict: **accept 709 / reject 702 / review 176** (44.7% / 44.2% / 11.1%)
- by_reason: S1_USABLE 709, S1_UNUSABLE 702, UNCERTAIN 176
- by_stage: **S0 0 (0.0) / S1 1411 (0.889) / S2-review 176 (0.111)** — the S0
  exact rules never fired on this well-lit moving factory footage, so the
  early-reject tier exited nothing; all decisions fell to S1/S2.
- throughput: **10.81 h video per wall hour (rtf 10.8)** across 3 parallel
  CPU-sequential processes, ~24 min wall (first-to-last thumb mtime + startup);
  features latency p50 548 ms/window (decode amortised per clip, O(1) memory)
- decode_path: cpu-sequential ×1587; VRAM: none consumed (CPU-only run)
- head: full-fit on 150 IndEgo gold, `head_sha256=d59955b512db1a1d…`,
  threshold 0.6, margin 0.25
- verify: `--verify` recompute MATCH; `--selfcheck 20` identical digests
  (see below); `tests.test_poc_run` 9/9 pass

## Domain-shift finding

The IndEgo head rejects most factory footage (bright, high-motion:
luma_mean 0.53 vs 0.30 gold; luma_p05 z≈+26; temp_b2/motion_energy high) with
p≈0 — the head's negatives were dark/frozen/blurred IndEgo, so a bright busy
factory floor lands far from its training manifold. This is a descriptive
finding about operating-point transfer, NOT an accuracy claim: the 0.6
threshold does not transfer to Egocentric-10K without labelled calibration.

## Reproduce

```
OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python scripts/poc_run.py --all --shard 0/3
OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python scripts/poc_run.py --verify <WINDOW_ID>
OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python scripts/poc_run.py --selfcheck 20
OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python scripts/poc_run.py --summary
PYTHONNOUSERSITE=1 JADAL_MODELS_DIR=/home/parshu/projects/jadal-qa/models \
  /home/parshu/envs/jadal/bin/python -m unittest tests.test_poc_run
```
