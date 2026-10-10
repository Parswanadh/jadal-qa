# Per-pixel accuracy track (issue #58)

FINAL human gold: `spectral_data/manifest_gold.json` — 150 ten-second windows
from 7 IndEgo 480p ~10 fps clips (ecu_full, ecu_late, desk; pallet, pallet2,
organise, logistics4). Label 1 = usable (100), 0 = bad (50): NO_ACTIVE_WORK 23,
HANDS_NOT_VISIBLE 14, DUPLICATE 6, TOO_DARK 6, UNDECODABLE 1. READ-ONLY.
All headline numbers are nested leave-one-clip-out (segments of a clip never
in train and test) with clip-bootstrap CIs. n=150 from 7 clips and ~3 wearers
means wide intervals; nothing here claims generalisation beyond IndEgo.

> Numbers below are filled from `eval/results/perpixel/*.json` (see
> Regenerate lines). Placeholders marked TODO until the runs land.

## M1 — unified whole-video feature table

`eval/perpixel/make_table.py` → `eval/results/perpixel/{gold,synth}.npz`,
`series_gold.npz`, `dup_sigs.json`, `meta.json`.
121 features x 150 gold + 60 synth: 40 arm-B T1 (`B_*`, units in
`docs/reports/spectral_features.md`), 26 GPU CPU-reference (`G_*`), 55
per-frame-series descriptors (`P_*`, units in `jadal/perpixel/series.py`).
Every decoded frame at 160x90 feeds the series (no N-frame sampling).

## M2 — per-reason detectors (`eval/perpixel/detectors.py` → `m2.json`)

Regenerate: `PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/perpixel/detectors.py`

- Quality (TOO_DARK 6 + UNDECODABLE 1 vs good 100), nested LOCO rule choice:
  AUROC **0.849, CI [0.53, 1.00]** — wide (n=7 positives), CI spans 0.5.
  Sensitivity at 5% FAR on good: **0.0** (pooled operating point fails; the
  honest number is the AUROC). Winning rules per fold: `flat_lap`
  (P_lap_var_mean < ~0.0075) on 6/7 folds, `darkfrac` once.
- Synth transfer (rule refit on all gold): black/blur/darken/overexpose
  detected 1.0 at 5% FAR; lowbitrate 0.67 (AUROC 0.917); frozen 0.33 (0.528);
  shake 0.17 (0.611); **noise 0.0 (AUROC 0.222, inverse)**;
  **static 0.0 (AUROC 0.000, perfectly inverse)** — a flatness rule cannot see
  noise/static by construction. Negative result, kept.
- Duplicates: constructed pairs (threshold 0.841 on held-out half):
  P 0.957 / R 1.0 / AUROC 1.0 (n_test=64). Human 6 DUPLICATE labels:
  recall@1 = recall@3 = **0.5**, precision of top matches **0.048**
  (63/150 windows above the constructed threshold — it over-flags real data).
  No exact-hash pairs among the 150. NNs of the 6 labels: g072→g070 (0.897),
  g077→g076 (0.855, same-clip desk loops); g088→g146 (0.859), g089→g144
  (0.801), g090→g121 (0.807), g149→g144 (0.796, cross-clip similar scenes).
  Human "duplicates" are repeated scenes/loops, not near-copies; the
  signature saturates (~0.8 for everything), so ranking works weakly but any
  fixed threshold is useless on real data.
- Activity (good 100 vs NO_ACTIVE_WORK 23 + HANDS_NOT_VISIBLE 14), nested LOCO:
  L2-logistic **0.673 [0.55, 0.82]**, HistGradientBoosting **0.758
  [0.57, 0.87]**. Per-reason recall at 5% FRR (LR): GOOD kept 0.95,
  HANDS_NOT_VISIBLE 0.286, NO_ACTIVE_WORK 0.217 (descriptive: pooled-OOF
  threshold evaluated on the same scores — fix round 1/I5). Per-clip LR AUROC: desk 0.929,
  ecu_full 0.870, ecu_late **0.425**, logistics4 0.857, organise 0.907,
  pallet **0.407**, pallet2 0.500 — learned direction flips on 2–3 clips
  (AUROC < 0.5 = no transfer, not an inverse feature).
- Tiny 1D-conv on per-frame series (CPU, inner-selected conv/40 epochs):
  **0.782 [0.63, 0.86]**, seeds 0.776/0.792/0.765, mean run-sd 0.050,
  determinism maxdiff 0.0.
- Abstain curve (LR): balanced accuracy 0.53 at full coverage, flat 0.50 as
  coverage drops — abstention does not rescue this track.

## Amendment A — signal noise: RAW quality, CLEANED activity, ablation

Quality estimators (Immerkaer σ, wavelet-HH1 MAD σ, 8x8 blockiness) run on RAW
pixels. Activity features run on the cleaning ladder
raw → flicker-removal → global-motion-compensation → temporal-median-3 →
spatial-Gaussian (+ Hann-Welch spectra, median/MAD floors).
`eval/perpixel/theory_clean.py` → `theory.json`.
Same nested-LOCO logistic protocol per rung (activity task):

| rung | AUROC | 95% CI | verdict |
|---|---|---|---|
| raw | 0.532 | [0.42, 0.68] | baseline |
| +flicker-removal | 0.604 | [0.49, 0.72] | STAYS (+0.072) |
| +global-motion-comp | 0.491 | [0.41, 0.67] | dropped (−0.114) |
| +temporal-median-3 | 0.527 | [0.45, 0.67] | +0.037 vs GMC but base rung rejected |
| +spatial-Gaussian (full) | 0.475 | [0.39, 0.62] | dropped |

All CIs overlap heavily: weak evidence, said plainly. Kept: per-frame luma
normalisation only. GMC hurts (egocentric parallax/rotation breaks integer
phase alignment; it injects artefacts). Operating thresholds (motion frac at
5% FRR) move per rung — thresholds are rung-specific, refit inside train.
Noise σ per clip (Immerkaer median, code values): ecu_full 0.0094, ecu_late
0.0102, desk 0.0083, pallet2 0.0108 (wavelet-HH1 reads ~2x lower, same order;
blockiness ≈ 1.0 everywhere = no visible 8x8 blocking at 160x90). Clips are
similarly noisy; noise does not explain the per-clip AUROC flips.

## Amendment D — deterministic signal-theory track (PRIMARY)

Nulls first: frozen χ² (CFAR), Fisher g-test for 0.3–8 Hz rhythm, duplicate
empirical null from cross-clip pairs; empirical FAR vs nominal α reported.
Decision list (≤8 integer thresholds, monotone: defects only vote BAD),
exhaustive seed-free grid search inside train clips, lexicographic ties.
Per-rule Clopper–Pearson uppers are descriptive: at n=150 no rule is
Bonferroni-certifiable (CERTIFY-by-clip would leave n≈84 < ~86 clean fires
needed — same finding as funnel3). Determinism: 10 repeats + 1v8 threads.
`eval/perpixel/theory_clean.py` → `theory.json`.

- Frozen χ² null: empirical FAR on 100 good windows **0.0 at α = 0.01 and
  0.001** — the null (dof ~10⁶, razor sharp) is conservative on real footage;
  sensitivity on true frozen content not measured (limitation; M2 synth
  transfer covers the flatness rule instead).
- Rhythm g-test: empirical FAR **0.11 at α = 0.01, 0.02 at α = 0.001** —
  11–20x the nominal rate. The white-noise null is miscalibrated for motion
  signals, exactly as documented in the code; g is used as a ranking
  statistic, not a CFAR gate.
- Duplicate empirical null (cross-clip sims): p99 = 0.817, max 0.859;
  recall on the 6 human labels 0.5/0.5 at that threshold; 0 exact-hash pairs.
- Rule list, nested LOCO: balanced accuracy **0.470**, AUROC (bad-score)
  **0.589 [0.42, 0.66]**. Per-reason recall: TOO_DARK/UNDECODABLE/DUPLICATE
  1.0, NO_ACTIVE_WORK 0.913, HANDS_NOT_VISIBLE 0.929 — but GOOD kept **0.0**:
  every good window is rejected by some rule. Thresholds do not transfer
  across clips (dark 0.009–0.019, blur_norm 94–127 per fold). Per-rule CP
  uppers on false rejects: frozen 0.023, overexp 0.066, blur 0.038,
  nomotion 0.066 — but dark 0.499 and norhythm 0.569 (reject up to half of
  good windows). Coverage–accuracy curve flat at ~0.47: margins don't separate.
- Determinism receipts: 10 repeats identical, digest
  `6438cfc1…`, 1-vs-8 threads identical.
- Verdict: the deterministic track ranks weakly (0.589) but its operating
  thresholds fail (0.470) — worse than L2-logistic (0.673) and HGB (0.758) by
  0.08–0.29 AUROC points. The gap sits in GOOD false rejects: single-feature
  thresholds fit on train clips do not survive clip shift. Kept as primary
  by requirement, with the gap stated.

## Physics features (research3/physics_features.md specs 1,2,3,5,6 + dHash + clip/entropy)

`jadal/perpixel/physics.py`; `eval/perpixel/physics_run.py` → `physics.json`.
SKIPPED: heat-equation scale-space, CCT, MSCN, max-entropy. Deviation: pixels
are the fixed 160x90 stream decode (thresholds transfer across sources at
that scale, not at 480p); Farneback capped at ≤25 pairs/window.
- Spec 1 (Poisson-Gaussian blocks): per-frame fits failed 100% (only ~50
  candidate 16x16 blocks at 160x90 → <20 after the 30% flat cut → NaN by
  spec rule). Fixed by pooling blocks across the 12 sampled frames (same
  block size, cut, 20-minimum): now finite on 98% of windows (a ≈ −0.05…0,
  read-noise-dominated at this scale; b ≈ 5–33 DN²; SNR 12–18 dB).
  Remaining 2% NaN recorded, never guessed. ten_norm/mad_frozen use the
  spec-1 floor, Immerkaer fallback.
- Spec 2 needed a syntax fix (`%+N`, not `%#+N`, on this ffprobe build).
- Nested-LOCO LR, activity task: physics-only **0.715 [0.57, 0.81]**;
  theory+physics 0.691; m1+theory+physics 0.702. Quality task: physics-only
  **0.963 [0.74, 1.0]** (n=7, CI still wide but lower bound 0.74).
- Single-feature ranking (sign = which side is "bad"): flow_act 0.806 and
  flow_med 0.801 (more residual hand-region flow → USABLE, as it should be),
  mad_mean 0.749, p_size_med 0.745, frm_bytes_med 0.744 (more bytes/change →
  usable); clip_frac 0.666 +bad; snr_db 0.635; ten_raw 0.604 +bad;
  dhash_loop_min 0.648; pg_a/pg_b 0.56–0.57 (weak); hist_ent 0.522 (nothing);
  mad_frozen_frac 0.583; frm_frozen_frac 0.500 (frozen content barely exists
  in gold).
- Cost: full physics extract ~106 s / 150 windows ≈ 0.7 s per 10-s window.

## Amendment B — label-noise audit

OOF confident-learning audit of AI 10k-frame labels (frames10k shards) and the
150 human labels. Human file untouched; review queue in
`eval/results/perpixel/review_queue.json` (segments.json schema).
Retrain with/without flagged evaluated on HUMAN windows only.
`eval/perpixel/audit.py` → `audit.json`.
- AI labels (7000 frames, 7 shards at audit time; embedding worker still
  writing): active_labor OOF AUROC 0.932, flagged 258 (3.7%); hands_ok OOF
  0.976, flagged 108 (1.5%). Flag = OOF prob disagrees with AI label by >0.7.
- Human labels: 12/150 flagged for re-check (in review_queue.json with
  reasons/probabilities).
- Retrain activity-LR (compact C=1 protocol; same 137 human windows unless
  noted): all 0.704 (m2 tuned-C LR: 0.673, same ballpark); downweight flagged
  ×0.25 → 0.729 (+0.025, within CI width); drop flagged → 0.846 **on the
  remaining 125 windows — not comparable** (removes the hardest windows from
  test too). Conclusion: label noise is present but not the main story; the
  clip-shift gap dominates.

## Amendment C — what the training algorithm is (plain language)

- Model class: decision list of single-feature threshold rules (primary);
  baselines: L2-logistic (linear), HistGradientBoosting (≤100 trees),
  tiny 1D-conv on per-frame series (~1k params).
- Features in: §M1/§D/§Physics tables above. Labels: human gold 0/1 (+reason).
- Loss: logistic log-loss (LR/HGB/conv); rule list maximises train balanced
  accuracy over exhaustive grids (no gradient, no randomness).
- Regularisation: LR C ∈ {0.01,0.1,1,10}; HGB min_samples_leaf=10, ≤100 iters;
  conv AdamW wd=1e-2, 20–40 epochs; rule list ≤8 thresholds + monotonicity.
- CV: nested leave-one-clip-out everywhere (grids/thresholds/calibration
  inside train clips); Platt calibration on inner-LOCO OOF.
- NOT trained: no deep nets (conv is 2 layers, CPU), no VLM, no embeddings in
  the primary track; EG2 embeddings used only inside the label-audit probe.

## M3 — frozen scorer + compare (`eval/perpixel/freeze_v2.py`)

`configs/perpixel_v1.json` (sha256 `27b350ac…`, full hash in
`configs/perpixel_v1.json.sha256`) + `configs/perpixel_conv.pt`.
PRE-REGISTERED for one later scoring on a new human-labelled Egocentric-10K
sample; do NOT tune after freezing.
Scorer (`jadal/perpixel/scorer.py`): quality tripwires {lowent(hist_ent<6.486),
blur(blur_norm<95.09), ten(ten_norm<339.28)} with integer evidence, then
frozen conv1d activity score (seed 0, threshold 0.362 at refit-all 5% FRR).
Digest excludes `B_mv_per_frame` (ffmpeg MV debug counts vary run to run —
measured 1597.8 vs 1593.4; feature kept, never hashed). Receipts
(`freeze_receipts.json`): 3× repeats identical digests + decisions on 2
windows (g001 GOOD 0.6715; g000 BAD:loop 0.0 — loop-tripwire fires on the
still, self-similar window); theory 10× + threads-1v8 identical.
Compare table: `eval/results/perpixel/compare.json` (conv 0.782 vs HGB 0.758
vs physics-LR 0.715 vs m1-LR 0.673; 7-rule list rejected as decider).

## M4 — cost (`eval/perpixel/cost.py` → `cost.json`)

1 CPU core, 6 windows (60 video-s): decode 3.6 s, series 5.1 s, theory
6.9 s, physics 6.2 s, M1 row (armB+gpu-ref+series) 15.0 s → full 28.1 s.
RTF **2.14** (video-s per core-s, decode included); **769 ten-second
segments/hour/core**; 5000x needs **2340 cores**, 3333x needs 1560.
Where time goes: M1/armB decode+features 53%, theory 25%, physics 22%.
GPU probe failed (rc=1, no output — recorded, not retried under deadline);
GPU path therefore unclaimed. No deployment reaches 5000x on CPU with this
pipeline: it is a verification-grade scorer, not the streaming tier.

## Caveats

- n=150 / 7 clips / ~3 wearers: all CIs wide; several span 0.5.
- NO_ACTIVE_WORK / HANDS_NOT_VISIBLE are semantic; per-pixel statistics carry
  most of the gap (expected).
- Duplicate threshold from constructed pairs over-flags real windows.
- Nothing generalises beyond IndEgo until the frozen Egocentric-10K scoring.

## Rebuild manifest (fix round 1/M5)

Absolute input dir: `/home/parshu/projects/jadal-qa-wt/spectral_data`
(manifests + videos, read-only, outside git). Committed binaries rebuild by:
`eval/perpixel/make_table.py` → gold/synth/series/dup_sigs/meta;
`eval/perpixel/detectors.py` → m2.json; `theory_clean.py` → theory.json +
theory_feats.npz; `physics_run.py` → physics.json + physics_feats.npz;
`rules_fix.py` → rulelist.json; `audit.py` → audit.json + review_queue.json;
`train_conv_frozen.py` → configs/perpixel_conv.{pt,json};
`freeze_v2.py` → configs/perpixel_v1.json (+ canonical sha) + compare.json;
`receipts.py` → freeze_receipts.json; `cost.py` → cost.json;
`ego10k_score.py` → ego10k.json (placeholder until the human sample lands).

## Fix round 1 (review `reviews/pr72-review.md`)

Real command outputs (abridged; full logs alongside the JSONs):
- C1 unsafe deserialisation: FIXED. Conv weights now `configs/perpixel_conv.json`
  (plain floats, no pickle); scorer hash-verifies `weights_sha256` BEFORE use
  and refuses on mismatch; model cached (no per-call `torch.load`, no global
  seed/thread mutation). `np.load` switched to `allow_pickle=False` everywhere
  except ragged `series_gold.npz`, which is validated element-wise on load
  (`detectors.load_series`); `theory_feats.npz:F` converted from object-dicts
  to an int64 fixed-point matrix (verified reload with `allow_pickle=False`).
- C2 series selection leak: FIXED. `(kind,epochs)` max-selection deleted;
  `(conv,40)` pre-specified as a fixed capacity prior, docstring says so, and
  `m2.json:series` re-ran under it (series_rerun.log). 0.782 is now the LOCO
  estimate of a fixed procedure (unbiased); freezing conv over HGB is stated
  as a judgment call (point estimate + determinism receipts, overlapping CIs).
- I1 silent fallbacks: FIXED. `frame_bytes` raises `FileNotFoundError` /
  `RuntimeError` on missing file, ffprobe rc, unparseable output, or <3
  in-window packets. Test `test_frame_bytes_raises` asserts the raise; the
  old succeed-on-`/nonexistent.mp4` test is gone. (`jadal/spectral`
  `compressed_stats` swallowing is read-only / another worker's — noted, not
  touched.)
- I2 duplicate in frozen scorer: WIRED as single-window loop-tripwire
  (`dhash_loop_min <= 4 AND still_run > 0.5`, frozen ints, evidence attached);
  cross-window corpus-NN stays offline (M2, same threshold). Config + report
  downgrade the frozen claim to quality + activity + loop-tripwire.
- I3 determinism receipts: PARTLY REBUTTED, partly fixed. The `6438cfc1…`
  claim IS backed — `theory.json:rule_list.determinism`
  (`repeats_identical: true`, `threads_1_vs_8_identical: true`; reviewer
  checked top-level keys only). Real gaps fixed: `eval/perpixel/receipts.py`
  is now the in-repo writer of `freeze_receipts.json` with full 64-hex
  digests (re-ran; receipts2.log).
- I4 `np.resize` tiling: FIXED. `jadal/perpixel/series.fit_length`
  (truncate + zero-pad) used by scorer, detectors, and frozen training. All
  150 gold windows are T=100 (verified counter), so frozen weights are
  unaffected; padding path covered by `test_fit_length_no_tiling`.
- I5 pooled thresholds: LABELLED. Activity per-reason recalls now carry the
  same descriptive caveat as the quality sens@5%FAR (this section, M2).
- I6 Ego10K holdout: protocol script `eval/perpixel/ego10k_score.py` +
  placeholder `eval/results/perpixel/ego10k.json`. Record: the audit probe
  read Ego10K *embeddings + AI labels* for noise diagnosis only; frozen
  weights/thresholds/decisions never saw Ego10K; no Ego10K human label exists
  yet. No re-freeze (nothing trained on Ego10K).
- I7 tests: ADDED `TestScorerContract` (synthetic ffmpeg clip: digest-repeat
  identical + 64-hex, config thresholds + int evidence, MV-exclusion,
  duplicate wiring) plus raise/pad unit tests. `tests/test_perpixel.py`: 16
  tests OK. Full suite re-run below.
- M1: `meta.json` "27 GPU" → 26 (code has 26).
- M2: `freeze_v2.py` reads stage numbers from `rulelist.json`
  (`rules_fix.py` now stores `stage_lowent_blur_ten`).
- M3: `.sha256` now covers the canonical payload excluding `created_utc` /
  `git_sha`; re-freeze hash `edb61f7c…` (previous `27b350ac…` covered the
  volatile envelope — superseded, reason recorded here).
- M4: scorer loads the net once (module cache), no per-call global torch
  mutation.
- Full suite: `PYTHONNOUSERSITE=1 timeout 250 .../python -m unittest discover
  -s tests 2>&1 | grep -E '^(Ran|OK|FAILED|ERROR|FAIL:)'` → (see below).
