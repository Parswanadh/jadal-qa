# E8 re-score of 1,546 Egocentric-10K windows — report (track E8, B2)

**Registry id:** E8.01 · **Owner:** experimenter-e8-rescore · **Device:** A40 ·
**Branch:** `feat/exp-rescore` · **git_sha:** `d23f79e852f8309b9bbb0e45b05876cb8eb5e6b7`

## What this is (and is not)

The PoC contract file `/workspace/artifacts/windows_rescored.jsonl` scores
**1546 of 1546 windows** of `queues/manifest_scale.json`
(58 clips, 4.404 h, 1920x1080 HEVC @30 fps; 0 decode failures)
with the frozen in-domain recipe **ensemble_v1** and the deterministic profile router.

> **These 1,546 Egocentric-10K windows have NO human labels: label_source is 'none' and every accept/reject/review verdict here is DESCRIPTIVE, not accuracy. IndEgo-trained models did not transfer out-of-domain (0.4-0.53 AUROC on the 60 human-labelled Egocentric-10K windows, n_bad=9), so the accept/reject share below is NOT a quality measure.**

`label_source` is `"none"` on every line. The verdict split below is **descriptive**.
The in-domain reference number for the recipe is AUROC **0.8534** (full-OOF, 150 IndEgo
windows / 7 clips, human gold, **one labeller**; `configs/ensemble_v1.json`); it does **not**
carry over here — IndEgo-trained models scored **0.4-0.53 AUROC** on the 60 human-labelled
Egocentric-10K windows (**n_bad = 9, exploratory**). "Receipt" means SHA-256 of the canonical
JSON: **content-hashed, not signed**.

## Recipe and router

- Score = equal-weight average of two frozen logistic heads on frozen features:
  **EG2** (EmbeddingGemma-2 mean embedding, 0.5 s grid, 20 frames/window, C=0.003) and
  **pixel26** (160x90 rgb24, all frames, 26 per-pixel features, C=0.3).
- **Router:** `profiles/*.json` + `jadal/regimes/router.py::choose_profile` exist and were
  used; the router branch **has landed**, so no fallback recipe was needed and none was invented.
- **Profile distribution:** {'default': 6, 'factory_1080p30': 1540} — every window is 1920x1080/30, so the
  hires rule fires before the quality/motion rules. The 6
  windows that fell out of the calibrated descriptor ranges went to `default` (widest abstain band)
  as designed: 3 with `shift_mean` 245-642 (range [0,230], fast camera motion) and 3 with
  `luma_mean` 0.66-0.67 (range [0.05,0.65], overexposed). Margins: min 0.000, p50 1.000.
- `stage` is `"L1"` on every line (funnel vocabulary: L0 rules, L1 model heads, L2 verdict).
- **Verdict rule** = the selected profile's operating point
  (`factory_1080p30`: score ≥0.75 accept, ≤0.25 reject, else review). The frozen band
  (score >0.5 accept / <0.4 reject) is reported as a secondary distribution
  {'accept': 452, 'reject': 706, 'review': 388} — it is **not** the one used.

## Verdict distribution (n=1546, descriptive)

| verdict | n | share | 95% CI (window, Wilson) | 95% CI (clip bootstrap, 58 clips) |
|---|---|---|---|---|
| accept | 296 | 0.1915 | [0.1726, 0.2118] | [0.1357, 0.2535] |
| reject | 83 | 0.0537 | [0.0435, 0.0661] | [0.0255, 0.0971] |
| review | 1167 | 0.7549 | [0.7328, 0.7756] | — |

Window-level CIs are anti-conservative (windows inside one clip are correlated); the clip
bootstrap is the honest one. Score stats: mean 0.4858,
p05/p50/p95 0.2472/0.4085/0.9072,
share inside the frozen abstain band 0.2962.
**This split is not a quality measure: it measures an untransferred model plus a router.**

## Cost (measured, 4 workers, shared A40)

- **0.6831 windows/s** overall; **5761.4 ms/window mean**,
  p50 5195.0 ms, p95 9370.8 ms; **wall 2263.1 s**
  for 1546 windows on one shared pod (load average ~15 from parallel tracks).
- Per stage (mean): decode 2354.3 ms · pixel features
  1154.3 ms · regime descriptors 1055.1 ms ·
  EG2 embed 1195.9 ms · score 1.7 ms.
- Per window: **one** ffmpeg/NVDEC pass with three branches (300 x 160x90 pixel frames,
  20 x 512x288 EG2 frames, 8 native frames for descriptors), one EG2 forward, no sampling at
  inference, fixed seed.
- **E3.04 policy (every-2nd window + 4 s early exit) was NOT applied**: the contract needs one
  verdict per window, and the frozen heads were fit on full-window features, so a 4 s window
  would change the feature distribution. `profiles/factory_1080p30.json` asks for
  `fps=5` / `probe_every_k_windows=2`; the pixel view was run at the frozen recipe's full frame
  rate instead (documented deviation, not silently dropped). E3.04's x4.9 saving remains an
  unapplied, decode-bound estimate from the E3 track.

## Verification

- All 1546/1546 receipts and `features_digest` values
  **re-verify** from the line content (`eval/exp_rescore/contract.py::verify_line`).
- `receipt_sha256` = sha256(canonical JSON of the line); `features_digest` = sha256(canonical
  `features{}`); the 768-d EG2 mean vector is stored by content hash, with the full matrix at
  `/workspace/artifacts/exp_rescore/eg2_mean_l2.npy`
  (sha256 `88381d7c493040c214de8c70eaa90dfc9cde419e3b3bb2e775788ec5187fdfe3`); per-window router inputs at
  `/workspace/artifacts/exp_rescore/descriptors.jsonl`
  (sha256 `d2c35dfd0a398796198d6cff409cb723446328c3ae018bd6c76ec82b856885e5`).
- **Decode-path control** (frozen `ffmpeg_cpu`+swscale vs the fast NVDEC+scale_cuda path, pixel
  view only): n=6 windows, |Δ p_pixel| max 0.0483, mean 0.0141 (exploratory). The fast path is **not bit-identical** to the frozen path; features and
  `p_pixel` differ slightly, so re-running the frozen path would give slightly different
  receipts. Not claimed identical.
- **Config note:** `configs/ensemble_v1.json`'s raw `.sha256` sidecar is stale (written before the
  pod path remap); the head numbers were verified equal to the frozen
  `$JADAL_BUNDLE/caches/ensemble/ensemble_v1.json`, and both digests are recorded per line in
  `versions{}`.

## Files

- runner `eval/exp_rescore/run_e8.py`, helper `eval/exp_rescore/contract.py`,
  result writer `eval/exp_rescore/write_results.py`, tests `tests/test_exp_rescore.py` (11 pass, 23 s)
- results `eval/results/exp_rescore/E8.01.json`, sample
  `eval/results/exp_rescore/windows_rescored_sample.jsonl` (20 lines = header + 19 windows)
- contract (not committed, `artifacts/` is ignored): `/workspace/artifacts/windows_rescored.jsonl`,
  body sha256 `74495d0059e41c93ece1922327abffd9dc6a15a689ccb18a404f67aecbed7c93`
- regenerate: `source /workspace/env.sh && export JADAL_EXP_DIR=/workspace/wt/exp-rescore/experiments OMP_NUM_THREADS=1 && cd /workspace/wt/exp-rescore && python3 eval/exp_rescore/run_e8.py --workers 4 --resume --out /workspace/artifacts/windows_rescored.jsonl && python3 eval/exp_rescore/write_results.py`

## Not done / open

- no human labels for these windows -> no accuracy, no AUROC out-of-domain claim
- no transfer claim: IndEgo-trained heads scored 0.4-0.53 AUROC on the 60 labelled Egocentric-10K windows (n_bad=9)
- the fast NVDEC decode path is not bit-identical to the frozen ffmpeg_cpu path (see decode_path_control)
- no per-window wall-clock budget on the edge device (pod A40 numbers only)
