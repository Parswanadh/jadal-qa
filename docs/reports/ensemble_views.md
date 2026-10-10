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
| ensemble, rank-average | 0.844 | [0.717, 0.919] | 0.805 |
| ensemble + Platt (per-fold) | 0.799 | [0.701, 0.895] | 0.655 |
| ensemble + isotonic (per-fold) | 0.805 | [0.693, 0.897] | 0.735 |

Gates: AUROC >= 0.85 **FAIL** (0.834; rank-average 0.844 also fails); CI lower
bound >= 0.72 **FAIL** (0.715, marginal; rank-average 0.717); no clip < 0.60 **PASS** (min 0.606, pallet). Nested validation
lowered the exploratory 0.853 to 0.834 — reported as-is.

Per-fold chosen C: EG2 mostly 0.003–0.01, pixel 0.03–1.0 (full table in
`eval/results/ensemble/nested_loco.json`, key `chosen_C`).

### Per-clip AUROC (prob-average ensemble; single views for reference)

| clip | n (good/bad) | ensemble | EG2 | pixel | ensemble bal.acc@0.5 |
|---|---|---|---|---|---|
| desk | 29 (16/13) | 0.822 | 0.813 | 0.755 | 0.767 |
| ecu_full | 48 (43/5) | 0.981 | 0.926 | 0.949 | 0.600 |
| ecu_late | 14 (10/4) | 0.675 | 0.675 | 0.650 | 0.700 |
| logistics4 | 17 (14/3) | 0.905 | 0.786 | 0.905 | 0.821 |
| organise | 19 (9/10) | 0.956 | 0.967 | 0.911 | 0.833 |
| pallet | 17 (6/11) | 0.606 | 0.576 | 0.530 | 0.538 |
| pallet2 | 6 (2/4) | 0.750 | 0.875 | 0.500 | 0.750 |

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
| 100% | 150 | 0.795 | [0.723, 0.861] | 19/100 |
| 80% | 120 | 0.842 | [0.770, 0.907] | 13/82 |
| 60% | 90 | 0.852 | [0.770, 0.926] | 8/59 |
| 50% | 75 | 0.840 | [0.744, 0.925] | 8/50 |

Frozen-train thresholds applied OOF: 5%-FR rule -> bal.acc 0.645 (rejects too
little); 10%-FR rule -> 0.710. The shipped operating point stays 0.5 with the
[0.4, 0.6] review band (heuristic; nested 60%-coverage bal.acc 0.852 vs
exploratory 0.89, from the earlier exploratory run, not in these JSONs — lower,
reported honestly).

## 2. Reconciling EG2 0.848 (exploratory) vs 0.816 (frames10k worker)

(0.848 and 0.816 come from the exploratory and frames10k runs, not from a JSON in
this PR. The 0.851 full-data inner-LOCO OOF is from the freeze log, not a committed
JSON. Nested EG2 0.846 is in `nested_loco.json`.)

Same preprocessing on both sides (L2-normalised segment-mean, StandardScaler,
class-balanced) — verified in `frames10k/eval/frames10k/analyze_transfer.py`.
The difference is the C grid/selection: frames10k tuned C over {0.01, 0.1, 1,
10} by inner LOCO and its folds picked larger C values that generalise worse;
the exploratory run fixed C=0.01 a priori. Our nested grid {0.003…1} selects
0.003–0.01 and gives **0.846**, reproducing the exploratory number. Verdict:
both numbers were right under their own protocol; the small-C procedure
(C=0.003 frozen) is the better, stable one. Frozen full-data inner LOCO also
picks C=0.003 (OOF 0.851, freeze log).

## 3. Frozen config

`configs/ensemble_v1.json` (+ `.sha256`):
sha256 `4da1068a78d5d15021f4e167d6aa24fd2e0bb08aa88f83c83c2f167cc086fe31`.
C: EG2 0.003, pixel 0.3. Heads: StandardScaler + logistic weights (float64
JSON). Fusion: equal prob-average. Threshold 0.5, abstain [0.4, 0.6].
`jadal/ensemble/scorer.py: score(video_path, start_s, end_s, config_path=None)`.
Numpy-only head math (`head_prob`, `ensemble_prob`). Feature code is a VENDORED
COPY, not read-only reuse: `jadal/gpu/features.py` and `jadal/gpu/__init__.py` are
byte-identical (cmp-verified 2026-10-10) to the versions on `feat/gpu-pipeline`,
`feat/poc-run` and `integration/poc`; identical blobs do not conflict on merge.
Config sha convention: sha256 of the raw file bytes (= `sha256sum`), used by
`load_config`, `score()`, the batch scorer and `eval_ood.py`. EG2 sampling mirrors
`frames10k/eval/frames10k/embed_segments.py`: 0.5 s steps ≈ 20 frames, thumb
512, `max_soft_tokens` 140). Pixel decode: ffmpeg CPU 160×90 rgb24,
luma=0.299R+0.587G+0.114B float64, fps=n/10 capped 30.

### Post-freeze scorer logic fix (2026-10-10; config and weights unchanged)

Before this fix, `score()` and `eval/ensemble_score_windows.py` tested
`prob > 0.5` before the abstain band, so (0.5, 0.6] returned accept and the upper
half of the [0.4, 0.6] review band was dead code. The fix is one shared function,
`jadal/ensemble/scorer.py: verdict_for(prob, lo, hi, thr)`, which tests the band first.
The frozen config (sha `4da1068a…`), weights, threshold and band are unchanged, and
so are all cached probabilities. Only the `verdict` field was recomputed. In
`ood_scores.json` 0 of 60 scores lie in (0.5, 0.6], so its verdict split is unchanged:
good 14 accept / 26 reject / 11 review; bad 3 / 3 / 3. The queue-2 caches
(`ood_scores2.json`, not yet labelled) had 7 verdicts change (3 in `2a`, 4 in `2b`).
Unit tests: `tests/test_ensemble.py::TestVerdictBand`.

Not done (time box): optional third view (sp_cellmax, physics tripwire gate)
— unevaluated, so excluded. Synthetic negatives not used (known harm 0.351).

## 4. Determinism

`eval/results/ensemble/determinism.json`: 10 repeated `score()` of 5 windows
(g000, g029, g100, g120, g140) — identical digests required. EG2 GPU numerics
are NOT claimed identical across devices.

## 5. Pre-registered OOD scoring — RESULT: does not transfer

`eval/results/ensemble/PREREGISTRATION.md` (sha written before labels read).
60 Egocentric-10K windows; frozen scores `eval/results/ensemble/ood_scores.json`
(cache config sha `4da1068a…`, matches frozen; 60/60 scored, 0 errors);
`eval/ensemble/eval_ood.py --labels /home/parshu/projects/jadal-qa-wt/ood_labels/labels.json --exploratory`.

Human labels: 51 good / 9 bad (8 NO_ACTIVE_WORK, 1 SHAKY). The pre-registered
minimum of 10 bad windows is NOT met, so this is exploratory, not confirmatory
(the script refuses without `--exploratory`).

| metric | value |
|---|---|
| OOD AUROC (window-bootstrap, optimistic) | **0.415 [0.210, 0.647]** |
| mean usable_prob good / bad | 0.488 / 0.542 (inverted) |
| good windows below 0.5 | 37/51 |
| bal.acc@0.5 / acc | 0.471 / 0.333 |
| accept / reject-or-review | 17 / 43 |

The IndEgo-trained ensemble does NOT transfer to Egocentric-10K factory video:
AUROC below chance with the sign flipped.

Per-head AUROCs of the same frozen scores (`eval/results/ensemble/ood_heads.json`,
produced by `eval/ensemble/ood_heads.py`; exploratory, n=60, n_bad=9, window-bootstrap
CIs):

| head | AUROC | 95% CI | note |
|---|---|---|---|
| EG2 head (`p_eg2`) | 0.586 | [0.333, 0.818] | near chance |
| per-pixel head (`p_pixel`) | 0.342 | [0.180, 0.516] | inverted (0.658 if sign flipped) |
| usable_prob (fusion) | 0.415 | [0.210, 0.647] | as above |

Reported from other branches (see PR #76 `eval/poc/ood_check.py` and the frozen
per-pixel scorer on PR #72), not recomputed here: per-pixel scorer 0.515 and PoC
cascade 0.529 on the same windows, both near chance.

The frozen model was not tuned afterwards. Hypotheses under investigation
(see §8): IndEgo is ~3 wearers doing assembly/logistics with visible
hand-motion signal; OOD factories differ in viewpoint, activity mix and
illumination, and the 26 pixel features' operating points (plus the EG2 head's
small-C direction) do not carry over. n_bad=9 keeps every OOD CI very wide.

## 6. Honest limits

n=150, 7 clips, ~3 wearers, one labeller; clip-correlated windows; wide CIs.
Pallet-type logistics motion does not transfer (0.606). Per-fold calibration
hurt ranking (Platt 0.799) so usable_prob stays uncalibrated; Platt (a,b) kept
as reference only. 12 labels were audit-flagged as possibly mislabelled
(perpixel review_queue.json) — not adjusted here. Full-OOF 0.853 is
resubstitution-optimistic (C picked on same OOF), not a claim. The config key
`full_oof_auroc_resubstitution_free` is a misnomer: that number is the optimistic
resubstitution value, as stated here. Out of domain (§5) the ensemble does not
transfer (usable_prob AUROC 0.415, exploratory); this headline is in-domain only.

## 8. EXPLORATORY: why the frozen ensemble fails OOD and what transfers

Setup (all exploratory, n=60 OOD windows, 51 good / 9 bad; nothing frozen
tuned; window-bootstrap CIs, optimistic): lean_10k frame heads (logistic on
frozen EG2 embeddings, trained on 10,000 AI-labelled factory frames from the
SAME Egocentric-10K dataset; whether their source videos are disjoint from the 60
OOD windows was NOT checked, so (a) and (c) may be optimistic) applied to every 0.5 s
frame of the 60 OOD windows (`eval/results/ensemble/ood_frames.npz`, 1200 frames), pooled per
window (mean/max/p25/p75/frac>0.5/var). Pixel side: the same 26 features
recomputed on CPU (`ood_pixel.npz`). Full numbers:
`eval/results/ensemble/transfer_probe.json`.

(a) In-domain frame heads DO transfer:

| head | pool | OOD AUROC | 95% CI | IndEgo pooled ref |
|---|---|---|---|---|
| active_labor | mean | 0.863 | [0.72, 0.97] | 0.813 |
| active_labor | frac>0.5 | 0.850 | [0.67, 0.97] | — |
| hands_ok | mean | 0.752 | [0.53, 0.93] | 0.817 |

The semantic direction learned from same-dataset factory frames carries over;
the IndEgo-fitted operating points do not.

(b) Sign table, selected rows (all 29 features in `transfer_probe.json`, key
`sign_table`). Oriented = OOD AUC in the IndEgo direction; "free" = OOD AUC with
its own sign; CI is the oriented CI.

| feature | IndEgo | OOD oriented | OOD free | oriented CI | agrees |
|---|---|---|---|---|---|
| head_work_mean | 0.813 | 0.863 | 0.863 | [0.71,0.97] | yes |
| head_hands_mean | 0.817 | 0.752 | 0.752 | [0.54,0.93] | yes |
| motion_energy | 0.308 | 0.309 | 0.691 | [0.16,0.49] | no |
| skin_frac | 0.667 | 0.255 | 0.745 | [0.10,0.43] | no |
| temp_b0 | 0.534 | 0.240 | 0.760 | [0.07,0.44] | no |
| luma_p50 | 0.517 | 0.222 | 0.778 | [0.06,0.39] | no |
| luma_p95 | 0.499 | 0.277 | 0.723 | [0.11,0.48] | no |
| luma_mean | 0.509 | 0.294 | 0.706 | [0.11,0.48] | no |
| dct_energy | 0.631 | 0.702 | 0.702 | [0.53,0.86] | yes |
| frozen_ens_prob | 0.834 | 0.415 | 0.585 | [0.22,0.65] | no |

Pattern: the two semantic EG2 rows keep their sign; motion/skin/luma pixel
features invert (e.g. in IndEgo more residual hand-region motion points one
way, in OOD factories the other — same sign-inversion the harness saw across
clips). Caveats: features with IndEgo AUC ≈ 0.5 (illum_r 0.49, flicker 0.50)
carry no meaningful sign, so their "agreement" is vacuous; with n_bad=9 only
head_work_mean's CI clearly excludes chance.

The frozen heads' own OOD AUROCs (EG2 0.586, per-pixel 0.342 inverted) are in
`ood_heads.json`, cited in §5.

(c) Leave-one-window-out feasibility (each window ≈ its own factory; UPPER
BOUNDS — they use OOD labels): single best feature (head_work_mean, selected
on the same 60) 0.80 [0.55,0.97]; frozen-ens-prob LOO 0.44; two-view LOO on
frozen heads 0.468. Read: refitting just the fusion on ~100 more OOD labels
cannot rescue the inverted views (0.44–0.47); the feasible path is the
in-domain head features (~0.8), to be confirmed when the 100 extra labels land.

## 9. Next round: 160-window re-evaluation (scores cached, labels pending)

Queue 2 (`label_queue_random2.json`, 100 windows) is scored with the frozen
model, cached at `eval/results/ensemble/ood_scores2.json` (merged from
`ood_scores2a/b.json`, 100/100, 0 errors, cache config sha matches frozen
`4da1068a…`). `labels2.json` had not appeared at push time. Verdict fields in these caches were
recomputed with the post-freeze `verdict_for` (§3); probabilities are unchanged.
`eval_ood.py` refuses `--labels2` without `--scores2`. When it lands,
run the frozen model unchanged with:

```
OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1 PYTHONPATH=. /home/parshu/envs/jadal/bin/python \
  eval/ensemble/eval_ood.py --labels /home/parshu/projects/jadal-qa-wt/ood_labels/labels.json \
  --labels2 /home/parshu/projects/jadal-qa-wt/ood_labels/labels2.json \
  --scores2 eval/results/ensemble/ood_scores2.json
```

Queue-2 scoring commands used (frozen model; no labels touched — repeatable):

```
OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1 PYTHONPATH=. /home/parshu/envs/jadal/bin/python \
  eval/ensemble_score_windows.py --windows /tmp/ood2a_windows.json --out eval/results/ensemble/ood_scores2a.json
# same under /home/parshu/.local/bin/flock -w 3600 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock for the EG2 part
```

(`/tmp/ood2*_windows.json` are `{id,path,start_s,end_s}` projections of the
queue file; rebuild with the one-liner in `eval/ensemble_score_windows.py`
docstring if needed.)

## 10. Regenerating commands (all with OMP_NUM_THREADS=2)

- nested LOCO: `OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/ensemble_nested_loco.py` -> `eval/results/ensemble/nested_loco.json`, `oof_probs.npz`
- freeze: `OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1 PYTHONPATH=. /home/parshu/envs/jadal/bin/python eval/freeze_ensemble.py` -> `configs/ensemble_v1.json` (+`.sha256`)
- batch scoring: `OMP_NUM_THREADS=2 ... /home/parshu/envs/jadal/bin/python eval/ensemble_score_windows.py --windows <json> --out <json>` (EG2 under `/home/parshu/.local/bin/flock -w 3600 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock`)
- determinism: same flock wrapper around `eval/ensemble_determinism.py` -> `eval/results/ensemble/determinism.json`
- OOD eval: `eval/ensemble/eval_ood.py --labels $JADAL_OOD_LABELS` (default `../ood_labels/labels.json`)
- per-head OOD AUROCs: `eval/ensemble/ood_heads.py` -> `eval/results/ensemble/ood_heads.json`
- head unit tests: `.../python -m unittest tests.test_ensemble`
- full suite: `PYTHONNOUSERSITE=1 JADAL_MODELS_DIR=/home/parshu/projects/jadal-qa/models /home/parshu/envs/jadal/bin/python -m unittest discover -s tests`

## 11. Data licensing (private repo only)

The Egocentric-10K-derived files are licensed data for the PRIVATE repo only and
must never reach a public snapshot. They are listed in `docs/PUBLIC_SNAPSHOT_EXCLUDE.txt`:
`eval/results/ensemble/ood_frames.npz` (EmbeddingGemma-2 per-frame embeddings of 1200
Egocentric-10K frames, despite the name: no images), `ood_pixel.npz`, `ood_scores*.json`,
`transfer_probe.json`, `ood_heads.json`. Details in `eval/results/ensemble/README.md`.
The IndEgo-derived files (`nested_loco.json`, `oof_probs.npz`, `configs/ensemble_v1.json`)
are not on that list.

