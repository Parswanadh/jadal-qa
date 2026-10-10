# C1.01 — Confirmatory one-shot: frozen O4 recipe on the 108 round-3 hold-out

**Status: CONFIRMATORY, ONE-SHOT.** The recipe was frozen *before* the round-3 labels
existed. It was scored **once** here. **No tuning, no feature re-selection, no threshold
changes, no re-fitting of any kind.** Result is reported as-is, whether good or bad.

## 1. Frozen artifacts verified (recorded before scoring)

| artifact | branch | sha256 (measured) | expected | match |
|---|---|---|---|---|
| `configs/ood_recipe_o4.json` | `m/recipe-freeze` (PR #145) | `be321ec13654d57021353aa4128b3357a99286a82fe88f12257f5b58824a0f4d` | `be321ec1…0f4d` | ✅ |
| `eval/exp_adapt160/o4_recipe.json` | `m/adapt160` (PR #131) | `5725e76f4b9d1c08cf0b849abafff69c3bb81ee760ebdcb457c32210a85b8ae9` | sidecar `449c27e0…416c` | ⚠️ **MISMATCH** |
| `eval/exp_adapt160/o4_recipe.sha256` (sidecar contents) | `m/adapt160` | `449c27e0fdd8c2bde7461e63a62dc67e080bfb4413c19cc876b3569dec3b416c` | — | — |

**Score-definition check:** the two recipe documents **agree** on the score definition
(features, IndEgo-only signs, unsupervised per-domain z-score, equal-weight mean,
threshold 0.0, `score > 0 ⇒ good`). The brief's STOP rule (disagreement on the score
definition) therefore does **not** trigger, and the run proceeded.

**Discrepancy flagged (not a definition conflict):** the `m/adapt160` sidecar
`o4_recipe.sha256` records `449c27e0…`, but the `o4_recipe.json` blob on that branch
hashes to `5725e76f…`. The sidecar was committed in the same commit (`06571c9`) that
added the JSON, so the recorded hash was never recomputed after a last edit. The
authoritative freeze is `configs/ood_recipe_o4.json` on `m/recipe-freeze`, whose sha256
matches exactly. Both documents describe the same recipe; the mismatch is an integrity
defect in the adapt160 sidecar only. Frozen ensemble config used:
`caches/ensemble/ensemble_v1.json`, raw sha256
`4da1068a78d5d15021f4e167d6aa24fd2e0bb08aa88f83c83c2f167cc086fe31` (the sha recorded in
the frozen `ood_scores.json`).

## 2. Data & regime

* Hold-out: `labels/round3/ood_labels3_108_with_factory.json`, **n=108, 92 good / 16 bad,
  44 factories** (single labeller). Never used for tuning.
* Window videos were **not** transcoded on disk. Each window's source clip + `start_s`
  was resolved from the round-3 queue/provenance and transcoded from
  `data/ego10k_scale/{clips,clips_more}/**` with the queue-2 recipe (854×480, **10 fps**,
  libx264 veryfast **crf 26**, no audio, 10 s).
* **Recipe reproduction check:** a fresh transcode of the frozen q2 window
  `r000_53a83eb4b7f1` reproduces the frozen Z3/pixel features to ~4 decimals
  (`ct_fsize_std` 6952.1 vs 6951.4; all motion/invariant/stable-8 features ≤0.3 %), i.e.
  the test-time feature pipeline matches the frozen training pipeline.
* Features: Z3 transferable-6 + E5 stable-8 extracted with the **frozen** extractor code
  (`m/adapt160:z3extract.py`; `jadal.gpu.features.cpu_window_features`). Per-domain
  z-score uses the **108's own** unsupervised mean/std; signs come from the 150 IndEgo
  gold windows only.
* 108/108 windows resolved and decoded; **0 failures**.

## 3. Headline result — frozen O4 recipe (b), scored once

**O4 (b) = E5 stable-8 + Z3 transferable-6, IndEgo-signed, per-domain z-score,
equal-weight mean, threshold 0.0:**

| metric | value | factory-grouped 95% CI (honest) | window-boot CI (optimistic) |
|---|---|---|---|
| AUROC | **0.676** | **[0.525, 0.827]** | [0.537, 0.805] |
| balanced accuracy @ 0.0 | **0.648** | [0.533, 0.768] | — |

Confusion @ 0.0: TP 56 / FN 36 / FP 5 / TN 11 (92 good, 16 bad).

The recipe is **above chance but weak**: the grouped AUROC CI only just excludes 0.5
(lower bound 0.525), and it sits **far below** the pooled-160 claim of 0.781. The
pre-registered hope — that the frozen O4 recipe stays clearly above chance and beats the
frozen ensemble on the hold-out — **does not hold** (see §4).

## 4. Comparison scorers (each scored once, no tuning)

| scorer | AUROC | grouped 95% CI | BA | threshold | confusion (TP/FN/FP/TN) |
|---|---|---|---|---|---|
| (a) O4 Z3 transferable-6 | 0.675 | [0.523, 0.822] | 0.534 | 0.0 | 58/34/9/7 |
| (b) **O4 E5-8 + Z3-6** | **0.676** | [0.525, 0.827] | 0.648 | 0.0 | 56/36/5/11 |
| (c) sharpness-3 zero-shot head | 0.715 | [0.579, 0.841] | 0.552 | 0.5 | 90/2/14/2 |
| (d) frozen ensemble_v1 (two-view) | 0.690 | [0.534, 0.823] | 0.640 | 0.5 | 43/49/3/13 |

**Paired factory-grouped bootstrap, AUROC(recipe) − AUROC(frozen ensemble), same resamples:**

| recipe | ΔAUROC | 95% CI | P(Δ>0) |
|---|---|---|---|
| (a) O4 Z3-6 | −0.011 | [−0.189, 0.173] | 0.462 |
| (b) O4 E5-8 + Z3-6 | −0.009 | [−0.197, 0.177] | 0.466 |
| (c) sharpness-3 head | +0.030 | [−0.163, 0.218] | 0.616 |

### Honest reading

* The **frozen O4 recipe does not beat the frozen two-view ensemble** on the round-3
  hold-out: ΔAUROC ≈ −0.01, CI straddling 0, P(Δ>0) ≈ 0.46.
* The **pooled-160 ranking did not replicate.** On the 160, O4 (b) = 0.781 vs frozen =
  0.566 (a +0.21 gap). On the 108, O4 (b) = 0.676 vs frozen = 0.690 — a **tie**.
  Two effects combine: the O4 recipe generalises much worse than advertised
  (0.781 → 0.676), and the frozen ensemble is much stronger on this hold-out
  (0.566 → 0.690).
* The sharpness-3 zero-shot head has the best raw AUROC (0.715) but a poor balanced
  accuracy at its frozen 0.5 threshold (0.552), because the logistic probabilities are
  shifted high (TP 90 / FN 2 / FP 14 / TN 2). It is **not** separable from the ensemble
  either (ΔCI includes 0).
* All three learned/adapted scorers are only marginally above chance on this hold-out;
  none clears a useful operating point.

## 5. Verdict

**Verdict: do NOT adopt the O4 recipe on the strength of the confirmatory test.** The
recipe is directionally above chance but its factory-grouped CI barely excludes 0.5, and
it does **not** reproduce the pooled-160 advantage over the frozen ensemble. The pooled-160
0.781 should not be used as a generalisation claim; the honest confirmatory number for
this recipe on fresh data is **AUROC 0.676 [0.525, 0.827], BA 0.648**.

This is reported as the pre-registered one-shot result. **Scored once on the hold-out;
no tuning.**

## 6. Reproduce

```
# features (CPU, ~13 min)
OMP_NUM_THREADS=2 python3 eval/exp_confirm/prep_confirm.py
# frozen two-view ensemble (GPU via the slot helper, ~15 min)
bash tools/pod/gpu_run.sh env OMP_NUM_THREADS=2 python3 eval/exp_confirm/score_ensemble_fast.py
# confirmatory scoring (CPU, ~1 min)
OMP_NUM_THREADS=2 python3 eval/exp_confirm/run_confirm.py
python3 eval/exp_confirm/make_figure.py
```

Results: `eval/results/exp_confirm/C1.01.json`. Figure:
`docs/reports/images/exp_confirm.png`.
