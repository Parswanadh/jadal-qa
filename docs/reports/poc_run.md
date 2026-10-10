# PoC run report — Egocentric-10K through the per-pixel funnel

**Honesty banner (binding): there are NO human labels for this Egocentric-10K
video; every number below is a descriptive rate on unlabelled video, NOT
accuracy.** Accuracy comes only from the 150 human-labelled IndEgo windows
(shipped S1 logistic head, LOCO AUROC 0.80, 95% CI 0.71–0.87 — `eval/results/gpu/cascade.json`;
wide, n=150 too small for certification). The one labelled check on THIS footage
(60 windows, 51 good / 9 bad, n_bad below the pre-registered 10, so exploratory)
is at chance: see "Out-of-domain result" below.
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

## Results (`eval/results/poc/run_summary.json`, from `eval/poc/summarize.py --complete`)

- **1587 unique windows, 4.404 h video from 58 clips** (47 factories), `partial: false`
  (the windows file has 1587 lines and 0 duplicates or malformed lines)
- by_verdict: **accept 709 / reject 702 / review 176** (44.7% / 44.2% / 11.1%)
- by_reason: S1_USABLE 709, S1_UNUSABLE 702, UNCERTAIN 176
- by_stage: **S0 0 / S1 1411 (88.9%) / S2 176 (11.1%)**. The S0 exact rules never
  fired on this footage, so the early-reject tier exited nothing; all decisions
  fell to S1 or S2 (handed on, not run).
- latency per 10-s window (sequential, single process): p50 548.1 ms,
  p90 699.2 ms, p99 1202.2 ms
- throughput: **17.91 h video per h of summed sequential compute** (concurrency
  and wall-clock not included; the dashboard tile shows the same value)
- decode_path: cpu-sequential for all windows; no GPU memory used by the batch
- head: full fit on the 150 IndEgo gold windows, `head_sha256=d59955b5…`,
  threshold 0.6, margin 0.25 (`versions` in each window record)

Re-verify: `scripts/poc_run.py --verify <WINDOW_ID>` recomputes one window and
compares its receipt. Five sampled windows reproduced their stored receipt in the
run log (not committed as a file).

## Out-of-domain result (60 human-labelled windows)

`eval/results/poc/ood_check_60.json` (from `eval/poc/ood_check.py`): the windows
were labelled by a human (51 good, 9 bad) and joined to their verdicts by clip and
start time.

- PoC score AUROC **0.529** (window bootstrap 95% CI 0.316–0.728): chance.
- **21 of 51 good windows falsely rejected** (41%); 24 accepted, 6 review.
- bad windows: 4 accepted, 5 rejected.
- balanced accuracy 0.51.
- Exploratory: n_bad = 9 is below the pre-registered minimum of 10, and the
  window-level CI is optimistic because windows from one clip are not independent.
- Two other scorers, reported and not recomputed here: frozen two-view ensemble
  0.415 (`handover/ood_eval_ensemble_v1_exploratory.txt`), per-pixel scorer 0.515
  (`jadal-ops/explainer/FACTS.md`).

So the ~44% reject share on this footage is mostly false rejects, not a quality
measurement.

## Domain-shift finding

The IndEgo head rejects most of this footage. The cause is operating-point
transfer: the head was calibrated on 480p/10 fps IndEgo, and this input is
1080p/30 fps HEVC. The out-of-domain check above shows the 0.6 threshold does not
transfer without labelled calibration. Brightness statistics (luma_mean and
similar) that suggest why are observations from the run notes and are not in a
committed results file.

## Reproduce

```
# env: JADAL_PY (interpreter, default python3), JADAL_POC_DATA, JADAL_CLIP_ROOT
OMP_NUM_THREADS=2 $JADAL_PY scripts/poc_run.py --all --shard 0/3
OMP_NUM_THREADS=2 $JADAL_PY scripts/poc_run.py --verify <WINDOW_ID>
OMP_NUM_THREADS=2 $JADAL_PY scripts/poc_run.py --selfcheck 20
python3 eval/poc/summarize.py --complete
python3 eval/poc/ood_check.py
PYTHONNOUSERSITE=1 JADAL_MODELS_DIR=<models dir> $JADAL_PY -m unittest tests.test_poc_run
```
