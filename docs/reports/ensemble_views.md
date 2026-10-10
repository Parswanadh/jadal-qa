# Ensemble of views (EG2 window embedding + 26 per-pixel features)

Plain-language summary: two fixed classifiers vote. View 1 looks at *what the
frames show* (EmbeddingGemma-2 image embeddings averaged over the 10 s window).
View 2 looks at *how pixels move and flicker* (26 hand/GPU features: motion,
temporal bands, sharpness, light, skin, duplicates). Each view outputs a
probability; the ensemble uses the plain average (no learned weighting).
Windows scoring near 0.5 are labelled **review** (abstain band [0.4, 0.6]).

## 1. Nested leave-one-clip-out (the honest numbers)

150 human windows, 7 clips. Per outer fold, C for each view is picked by inner
leave-one-clip-out on the 6 training clips (grid {0.003, 0.01, 0.03, 0.1, 0.3,
1}); calibration (Platt/isotonic) and the 5%/10% false-reject thresholds are
fit on inner out-of-fold train probs only. Clip-bootstrap 95% CI (2,000).

| view | AUROC | 95% CI | bal.acc@0.5 |
|---|---|---|---|
| EG2 mean-embedding (nested C) | 0.846 | [0.728, 0.913] | 0.770 |
| 26 per-pixel feats (nested C) | 0.792 | [0.666, 0.882] | 0.760 |
| **ensemble, prob-average** | **0.834** | **[0.715, 0.914]** | **0.795** |
| ensemble, rank-average | 0.834 | [0.712, 0.910] | 0.785 |
| ensemble + Platt (per-fold) | 0.799 | [0.701, 0.895] | 0.655 |
| ensemble + isotonic (per-fold) | 0.805 | [0.693, 0.897] | 0.735 |

Gates: AUROC >= 0.85 **FAIL** (0.834); CI lower bound >= 0.72 **FAIL** (0.715,
marginal); no clip < 0.60 **PASS** (min 0.606, pallet). Nested validation
lowered the exploratory 0.853 to 0.834 — reported as-is.

Per-fold chosen C: EG2 mostly 0.003–0.01, pixel 0.03–1.0 (full table in
`eval/results/ensemble/nested_loco.json`, key `chosen_C`).

### Per-clip AUROC (prob-average; same-fold singles in brackets EG2/pixel)

| clip | n (good/bad) | ensemble | bal.acc |
|---|---|---|---|
| desk | 29 (16/13) | 0.822 | 0.767 |
| ecu_full | 48 (43/5) | 0.981 | 0.600 |
| ecu_late | 14 (10/4) | 0.675 | 0.700 |
| logistics4 | 17 (14/3) | 0.905 | 0.821 |
| organise | 19 (9/10) | 0.956 | 0.833 |
| pallet | 17 (6/11) | 0.606 | 0.538 |
| pallet2 | 6 (2/4) | 0.750 | 0.750 |

Weakest clips stay pallet (0.606) and ecu_late (0.675), as in exploration.

### Per-reason recall (reject at 0.5) and usable-vs-reason AUROC

| reason | n_bad | recall | ovr AUROC |
|---|---|---|---|
| NO_ACTIVE_WORK | 23 | 0.70 | 0.80 |
| HANDS_NOT_VISIBLE | 14 | 0.86 | 0.85 |
| DUPLICATE | 6 | 0.67 | 0.77 |
| TOO_DARK | 6 | 1.00 | 0.96 |
| UNDECODABLE | 1 | 1.00 | 0.93 |

### Coverage vs balanced accuracy (most-confident first, 0.5 threshold)

| coverage | k | bal.acc | 95% CI | good wrongly rejected |
|---|---|---|---|---|
| 100% | 150 | 0.795 | — | 19/100 |
| 80% | 120 | 0.842 | (in JSON) | 13/82 |
| 60% | 90 | 0.852 | (in JSON) | 8/59 |
| 50% | 75 | 0.840 | (in JSON) | 8/50 |

Frozen-train thresholds applied OOF: 5%-FR rule -> bal.acc 0.645 (rejects too
little); 10%-FR rule -> 0.710. The shipped operating point stays 0.5 with the
[0.4, 0.6] review band (heuristic; nested 60%-coverage bal.acc 0.852 vs
exploratory 0.89 — lower, reported honestly).

## 2. Reconciling EG2 0.848 (exploratory) vs 0.816 (frames10k worker)

Same preprocessing on both sides (L2-normalised segment-mean, StandardScaler,
class-balanced) — verified in `frames10k/eval/frames10k/analyze_transfer.py`.
The difference is the C grid/selection: frames10k tuned C over {0.01, 0.1, 1,
10} by inner LOCO and its folds picked larger C values that generalise worse;
the exploratory run fixed C=0.01 a priori. Our nested grid {0.003…1} selects
0.003–0.01 and gives **0.846**, reproducing the exploratory number. Verdict:
both numbers were right under their own protocol; the small-C procedure
(C=0.003 frozen) is the better, stable one. Frozen full-data inner LOCO also
picks C=0.003 (OOF 0.851).

## 3. Frozen config

`configs/ensemble_v1.json` (+ `.sha256`):
sha256 `4da1068a78d5d15021f4e167d6aa24fd2e0bb08aa88f83c83c2f167cc086fe31`.
C: EG2 0.003, pixel 0.3. Heads: StandardScaler + logistic weights (float64
JSON). Fusion: equal prob-average. Threshold 0.5, abstain [0.4, 0.6].
`jadal/ensemble/scorer.py: score(video_path, start_s, end_s, config_path=None)`.
Numpy-only head math (`head_prob`, `ensemble_prob`); feature code reused
read-only (`jadal/gpu` from `origin/feat/gpu-pipeline`; EG2 sampling mirrors
`frames10k/eval/frames10k/embed_segments.py`: 0.5 s steps ≈ 20 frames, thumb
512, `max_soft_tokens` 140). Pixel decode: ffmpeg CPU 160×90 rgb24,
luma=0.299R+0.587G+0.114B float64, fps=n/10 capped 30.

Not done (time box): optional third view (sp_cellmax, physics tripwire gate)
— unevaluated, so excluded. Synthetic negatives not used (known harm 0.351).

## 4. Determinism

`eval/results/ensemble/determinism.json`: 10 repeated `score()` of 5 windows
(g000, g029, g100, g120, g140) — identical digests required. EG2 GPU numerics
are NOT claimed identical across devices.

## 5. Pre-registered OOD scoring

`eval/results/ensemble/PREREGISTRATION.md` (sha written before labels read).
60 Egocentric-10K windows; frozen scores `eval/results/ensemble/ood_scores.json`;
`eval/ensemble/eval_ood.py --labels <labels.json>` prints AUROC (window
bootstrap, optimistic), bal.acc at frozen 0.5, accept/reject split; works on
partial labels, refuses with < 10 per class. Nothing tuned after freezing.

## 6. Honest limits

n=150, 7 clips, ~3 wearers, one labeller; clip-correlated windows; wide CIs.
Pallet-type logistics motion does not transfer (0.606). Per-fold calibration
hurt ranking (Platt 0.799) so usable_prob stays uncalibrated; Platt (a,b) kept
as reference only. 12 labels were audit-flagged as possibly mislabelled
(perpixel review_queue.json) — not adjusted here. Full-OOF 0.853 is
resubstitution-optimistic (C picked on same OOF), not a claim.

## 7. Regenerating commands (all with OMP_NUM_THREADS=2)

- nested LOCO: `OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/ensemble_nested_loco.py` -> `eval/results/ensemble/nested_loco.json`, `oof_probs.npz`
- freeze: `OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1 PYTHONPATH=. /home/parshu/envs/jadal/bin/python eval/freeze_ensemble.py` -> `configs/ensemble_v1.json` (+`.sha256`)
- batch scoring: `OMP_NUM_THREADS=2 ... /home/parshu/envs/jadal/bin/python eval/ensemble_score_windows.py --windows <json> --out <json>` (EG2 under `/home/parshu/.local/bin/flock -w 3600 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock`)
- determinism: same flock wrapper around `eval/ensemble_determinism.py` -> `eval/results/ensemble/determinism.json`
- OOD eval: `eval/ensemble/eval_ood.py --labels /home/parshu/projects/jadal-qa-wt/ood_labels/labels.json`
- head unit tests: `.../python -m unittest tests.test_ensemble`
- full suite: `PYTHONNOUSERSITE=1 JADAL_MODELS_DIR=/home/parshu/projects/jadal-qa/models /home/parshu/envs/jadal/bin/python -m unittest discover -s tests`
