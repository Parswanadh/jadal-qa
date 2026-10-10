# Cascade + selective-accuracy verification (claim LB1-cascade)

Independent verification (worker LB1-cascade) of the early-reject cascade and
selective-accuracy sentences in the pod bundle's `docs/FACTS.md`.
Everything below was recomputed on the pod from **read-only** artifacts; no GPU
was used and no input file was modified.

Reproduce with:

```
OMP_NUM_THREADS=2 python eval/pod_verify/cascade_selective_verify.py
# -> eval/results/pod_verify/cascade_and_selective.json
```

Artifacts used

| artifact | sha256 | note |
|---|---|---|
| `labels/manifest_gold.json` (bundle, read-only) | labels sha `43ab44e5…` | 150 human gold windows, 100 good / 50 bad, 7 clips |
| `eval/results/gpu/features.json` (= `caches/gpu/features.json`) | `6d27d872e9ac285c4abddcb61eef70a2eed565aa3adc16ab0900e3cb3d5b1483` | 210 windows: 150 gold + 60 synthetic, carries the `degradation` tag |
| `caches/frames10k/segments.npz` | — | EG2 mean window embeddings (150) |
| `caches/ensemble/oof_probs.npz` | — | nested-LOCO OOF probs (150) |

Protocols (stated exactly, both implemented in the script)

* **cascade** (`eval/gpu/cascade.py`): S0 exact rules on the 2-s probe
  (`black p95<0.03`, `dark mean<0.05`, `lens_blocked spatial_var<1e-5`,
  `overexposed p5>0.70`, `frozen temp_e<5e-4`); S1 = StandardScaler +
  LogisticRegression on the 26 features, **no** class weights, `max_iter=2000`,
  leave-one-clip-out; operating point `argmax(tpr+tnr-1)` over
  `linspace(0.05,0.95,37)` with `pred=(p>=t)`; `MARGIN=0.25` around `p=0.5`
  routes to S2 (interface stub).
  Note: the `argmax` operating point is stored as a legacy `threshold` field in
  receipts and never decides a verdict. `decide_window` in `jadal/poc/cascade.py`
  decides with bands: reject at p <= 0.25, accept at p >= 0.75, review (S2) between.
* **selective**: confidence `|p-0.5|`, order `np.argsort(-conf)` (stable),
  `k = int(cov*n)`, decision `p>0.5`, balanced accuracy computed on the selected
  `k` only. Two views: **nested** (`oof_probs.npz`, inner-LOCO C selection,
  `class_weight=balanced`) and **exploratory** (fixed C: EG2 0.01 / pixel 0.3,
  `class_weight=balanced`, plain LOCO — `scripts/ensemble_views_exploratory.py`).

## Verdicts

| # | claim (FACTS.md) | view | published | recomputed | verdict |
|---|---|---|---|---|---|
| 1 | ~2% S0 / 50% S1-accept / 18% S1-reject / 30% uncertain | 26-feature cascade | 2 / 50 / 18 / 30 % | 3/150, 75/150, 27/150, 45/150 = 2 / 50 / 18 / 30 % | REPRODUCED |
| 1 | dark/black/frozen caught 6/6 each at S0 on synthetic | synthetic | 6/6 each | darken 6/6, black 6/6, frozen 6/6 (all other degradations 0/6 at S0) | REPRODUCED |
| 2 | decided: balanced acc 0.81; 8/75 good falsely rejected; 8/30 bad accepted | 26-feature cascade | 0.81, 8/75, 8/30 | 0.8133, 8/75, 8/30 (coverage 105/150 = 0.70) | REPRODUCED |
| 3 | all 150: 0.74 (26/100 good wrongly rejected) | pixel26, **exploratory** | 0.74, 26/100 | 0.7400, 26/100 | REPRODUCED |
| 3 | 80% most confident: 0.84 (15/80 good wrongly rejected, 88% of bad rejected) | pixel26, **exploratory** | 0.84, 15/80, 88% | 0.8438, 15/80, 35/40 = 87.5% → 88% | REPRODUCED |
| 3 | 50% most confident: 0.81; curve not monotone | pixel26, **exploratory** | 0.81 | 0.8051; curve 0.74 → 0.84 → 0.83 → 0.81 (rises then falls) | REPRODUCED |
| 4a | 0.74 deciding all 150 → 0.84 on the 80% most confident | pixel26, **exploratory** | 0.74 → 0.84 | 0.7400 → 0.8438 | REPRODUCED |
| 4b | balanced acc 0.79 at threshold 0.5; 0.89 at 60% most confident (7/63 good wrongly rejected) | prob-average (EG2+pixel26), **exploratory** | 0.79; 0.89, 7/63 | 0.7900; 0.8889, 7/63 | REPRODUCED |

No numeric discrepancy was found in any of the four claims. The caveats below
are about **view attribution**, not about the numbers.

## Caveats that matter

1. **Claims 3 and 4 are exploratory-protocol numbers, not nested ones.** FACTS
   line 42 says "26 features + logistic, LOCO, n=150" but does not say
   "exploratory". Recomputing the identical selective curve from the **nested**
   OOF probabilities (`oof_probs.npz`, `p_pixel`) gives **0.76 at 150 windows
   (same 26/100 good wrongly rejected), 0.81 at 80%, 0.80 at 50%** — i.e. the
   published 0.74/0.84/0.81 only reproduces under the fixed-C non-nested
   protocol. The nested view's own numbers are also on record in
   `eval/results/ensemble/nested_loco.json` (pixel AUROC 0.7916, bal@0.5 0.76),
   which my recomputation matched exactly for every view (EG2 0.77, pixel 0.76,
   prob-avg 0.795, rank-avg 0.805). **The selective sentence should carry the
   same "exploratory, not yet nested" caveat as the AUROC sentence in the 18:25
   update.**
2. **Claim 4b's view is ambiguous in FACTS.** The sentence lists EG2 (0.848),
   pixel26 (0.810), prob-average (0.853) and then says "balanced accuracy 0.79 at
   threshold 0.5, 0.89 … (7 of 63 …)" without naming the model. Only the
   **probability average** matches both numbers (EG2: 0.77 @0.5, 0.87 @60% with
   7/65; pixel26: 0.74 @0.5, 0.83 @60% with 11/59). A reader could mis-attach
   the 0.79/0.89 to EG2 because it is the last model named with a value.
3. **Claim 1 mixes two populations in one sentence**: "~2% exit at stage 0"
   is the *gold* rate (3/150), while "6/6 each" is the *synthetic* rate. Both are
   correct; the parenthetical makes them look like one measurement.
4. **Tie handling.** Confidence values `|p-0.5|` have no exact ties for any
   claimed view (checked; `boundary_ties = 1` for every coverage point). The
   rank-average view does have 54–58 exact rank ties and is therefore the only
   order-dependent number, but no published claim rests on it.
5. **Synthetic rule counts were verified from a different artifact than the
   producer used.** `eval/gpu/cascade.py` reads
   `/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_synth.json`, which is
   **not present on this pod**. The same 60 synthetic windows are however inside
   `features.json` tagged with `degradation` (10 degradations × 6), so the 6/6
   counts are recomputable and were reproduced from there.
6. **The raw-video → feature step was not re-run** (CPU-only mandate; GPU
   forbidden). The S1 model was re-fit from the committed `features.json`, whose
   repo copy and pod cache copy are byte-identical (sha256 above).
