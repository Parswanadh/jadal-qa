# E4 regimes + profiles — experiment report (SUPERSEDED)

> **SUPERSEDED — do not cite.** This laptop-era draft predates the pod re-run
> and is kept only for history. It still marks E4.01–E4.03 "pending" and
> incorrectly calls the queue-1 set a "480p/10fps transcode" (it is
> **480p/30fps**). The authoritative, finished report is
> **[`docs/reports/exp_regimes_pod.md`](exp_regimes_pod.md)** (results in
> `eval/results/exp_regimes/E4.01.json` … `E4.04.json`).

Track E4 (exp-regimes). Builder intent: "for blurry data one algorithm/model,
for clear data another; we should be able to differentiate or select between
multiple types; the end product should be customizable and adapt to whatever
data we use." Evidence: every IndEgo-trained scorer failed on real
Egocentric-10K windows (AUROC 0.4–0.53) — a data-type problem.

## E4.01 regime descriptors (domain-shift diagnosis)

Method: deterministic per-window descriptors (`eval/exp_regimes/descriptors.py`,
no GPU, 8 sampled frames): container (resolution, fps, codec, bitrate,
bytes/frame) + sharpness (Laplacian variance, normalised), noise sigma
(Immerkaer), luma mean/p05/p50/p95, exposure clipping, edge density,
blockiness (8px-boundary ratio), chroma spread, ego-motion (phase-correlation
shift stats, static fraction, temporal energy, dup fraction).

Sets: 150 IndEgo gold windows (human labels) · 60 OOD queue-1 windows on BOTH
the 480p/10fps transcode and the 1080p/30fps source sub-window · 100 queue-2
transcodes (labels pending) · stratified Ego10K sample (n=300, seed 7).

**Result (ranked separators, Cohen's d):** _pending OOD decode — table in
`eval/results/exp_regimes/diagnosis.json`, figure `figs/descriptors_hist.png`._

## E4.02 regime-specific vs global (nested LOCO on IndEgo gold)

Terciles of sharpness / lighting / motion; global vs regime-only logistic for
the 26-feature view and the EG2 view. Adopt a split only if regime beats
global by > 0.03. _Results in `regime_vs_global.json`, `figs/regime_delta.png`._

## E4.03 domain gap control

Paired transcode-vs-source descriptor ratios + frozen-ensemble
(`ood_scores.json`) score audit incl. p_pixel-vs-descriptor correlations.
Honest verdict on whether matching the input spec fixes transfer:
_results in `gap_control.json`._

## E4.04 profiles product layer

`profiles/schema.json` + 5 profiles (`clear_stationary`, `blurry_lowlight`,
`walking_logistics`, `factory_1080p30`, `default`); deterministic router
`jadal/regimes/router.py::choose_profile` (rule-based, margin reported,
`default` + `UNSEEN_REGIME` fallback); 9 unit tests, all pass, < 1 s.
Router thresholds initial (v0.1.0); calibrated against E4.01 ranges on close.

## Verdicts

| id | verdict | summary |
|----|---------|---------|
| E4.01 | _pending_ | descriptors computed for IndEgo 150/150; OOD+Ego10K decoding |
| E4.02 | _pending_ | runs on E4.01 output |
| E4.03 | _pending_ | runs on E4.01 output |
| E4.04 | adopt | router + profiles + tests shipped; thresholds to confirm vs E4.01 |
