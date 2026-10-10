# N4 — Label strategy + certification budget: exactly what to label next

**Track:** N4 (labels / accuracy) · **Branch:** `m/acc-labels` · **Owner:** N4
**Timebox:** 35 min · **Device:** CPU only (numpy/sklearn, cached inputs; no GPU, no network)
**Code:** `eval/exp_acc/acc_lib.py`, `eval/exp_acc/run_acc.py` · **Tests:** `tests/test_acc_labels.py`
**Results:** `eval/results/exp_acc/N4.01.json`, `N4.02.json`, `labels_next.json`
**Figure:** `docs/reports/images/exp_acc_labels.png`
**Reproduce:** `OMP_NUM_THREADS=2 python3 eval/exp_acc/run_acc.py`

**Data (read-only, never edited):**
- 160 human OOD windows — `labels/ood160_labels.json` (138 good / 22 bad, 47 factories, one labeller).
- Frozen 26 per-pixel features for those 160 — `eval/results/exp_acc/inputs/ood160_pixel.npz`.
- 1,546 unlabelled Egocentric-10K windows — `artifacts/windows_rescored.jsonl`
  (26 per-pixel features + two view scores; **the same 47 factories**, 58 video clips, 25 workers).
- Prior context: O2 few-shot curve (`m-fewshot160`), Z2 certification math (`m-conformal`),
  Z4 uncertainty finding (`m-clip-sequential`).

---

## 0. TL;DR for the human

1. **Label next (model error, from the current pool): the top-200 list in
   `eval/results/exp_acc/labels_next.json`.** Selection rule = **diversity / coverage**
   (farthest-first over the 26 regime features vs the 160 already labelled), balanced to
   **4–5 windows per factory across all 47 factories**. **Do not** pick by model uncertainty.
2. **Certification is a data-count problem, not a model problem.** To *certify* a marginal
   false-accept bound at 95 % confidence you need independent **factories**: ~**86** for
   r = 0.20, ~**342** for r = 0.10, ~**1,365** for r = 0.05 (grid-union). We have **47**, which
   certifies only **r ≈ 0.27**. The current "review 12 %, FA 4 %" point is **descriptive, not
   certified**.
3. **The current 1,546-window pool adds ZERO clusters** (all 47 factories already have labels).
   Extra labels from it cut *model* error; they do **not** by themselves make a claim certifiable.
   Certification requires **new factories** (new data).
4. **Strategy ranking (paired, vs random):** diversity/maximin **wins** at mid budget
   (k = 80: −0.081 paired error, better in 4/5 folds); breadth (new-factory round-robin) gives a
   small consistent edge at low budget (k = 20–80); **uncertainty sampling is worst**
   (worse than random in 4/5 folds at k = 80 — confirms Z4 AUROC 0.349); depth (more windows per
   existing factory) is harmful early. All gaps are modest and noisy — see §1.4.

---

## 1. N4.01 — label-efficiency analysis

### 1.1 Protocol

26-feature per-pixel logistic (`class_weight=balanced`, C = 1), **factory-grouped** 5-fold
`StratifiedGroupKFold`. For each fold, labels are acquired from the training pool in the
strategy's order; at each budget *k* the model is fit on the first *k* windows and scored on the
held-out factories. `error = 1 − balanced accuracy @ 0.5`. `random` is averaged over 5 seeds ×
5 folds; deterministic strategies run once per fold. The full-160 point is the grouped-CV OOF
anchor. **A factory is never in both train and test** (a held-out factory = a new deployment
cluster).

### 1.2 Strategies

| id | rule | intuition |
|---|---|---|
| `random` | seeded uniform permutation | baseline |
| `breadth_new_factories` | round-robin: one window per new factory before a second from any | maximise independent clusters per label |
| `depth_existing_factories` | exhaust factories in descending size | maximise windows-per-factory |
| `diversity_maximin` | greedy farthest-first over z-scored 26-D regime features | coverage of the input space |
| `uncertainty_boundary` | closest to p = 0.5 (max 4p(1−p)) | Z4 active-learning control (expected bad) |

### 1.3 Error vs labels (grouped CV, lower = better)

| k labels | random | breadth (new factories) | depth (existing factories) | **diversity/maximin** | uncertainty |
|---:|---:|---:|---:|---:|---:|
| 10 | **0.446** | 0.463 | 0.578 | 0.505 | 0.557 |
| 20 | 0.421 | **0.384** | 0.411 | 0.435 | 0.413 |
| 40 | 0.383 | **0.373** | 0.564 | 0.378 | 0.423 |
| 80 | 0.338 | 0.309 | 0.382 | **0.257** | 0.401 |
| 120† | 0.338 | 0.347 | 0.335 | 0.359 | **0.293** |
| 160 (OOF anchor) | 0.342 | — | — | — | — |

† at k = 120 the training pool is nearly exhausted (~128 windows/fold), so the order only
decides which ~8 windows are dropped — this row is noise and should be ignored.

### 1.4 Paired comparison vs random (the honest read)

Per-fold paired deltas (negative = better than random; `(·)` = fraction of folds better):

| strategy | k = 10 | k = 20 | k = 40 | k = 80 | k = 120 |
|---|---:|---:|---:|---:|---:|
| breadth_new_factories | +0.027 (40 %) | **−0.039 (60 %)** | −0.010 (60 %) | −0.029 (60 %) | +0.009 (40 %) |
| depth_existing_factories | +0.141 (20 %) | −0.012 (60 %) | +0.182 (20 %) | +0.044 (20 %) | −0.003 (60 %) |
| diversity_maximin | +0.053 (33 %) | +0.012 (40 %) | −0.005 (60 %) | **−0.081 (80 %)** | +0.021 (40 %) |
| uncertainty_boundary | +0.120 (40 %) | −0.009 (40 %) | +0.040 (40 %) | **+0.063 (20 %)** | −0.045 (60 %) |

**Which wins and by how much.**
- **Diversity/maximin is the single best strategy at mid budget**: at k = 80 it is **8.1 points of
  error below random** (0.257 vs 0.338) and better in 4/5 folds. It also has the best AUROC
  (0.79 vs 0.72). This is the recommended acquisition rule.
- **Breadth (new factories) is best at low budget** (k = 20–40): −3.9 pts at k = 20, better in
  3/5 folds. If you can source new factories, take one window from each before deepening.
- **Depth (more windows per existing factory) is harmful early**: +18 pts at k = 40 (only 20 % of
  folds better) — a model trained on 3–4 factories does not generalise to held-out factories.
- **Uncertainty sampling is the worst mid-budget strategy** (+6.3 pts at k = 80, worse in 4/5
  folds). This is the label-level confirmation of Z4's clip-level finding (uncertainty AUROC 0.349,
  anti-correlated with true OOD error). **Do not label by model uncertainty.**

**Honesty note.** n = 160, n_bad = 22, 5 folds → the independent-seed error bars are ±0.11–0.19
and most single-point differences are within noise. The paired deltas above are the defensible
read, and even they are only "suggestive" except the diversity@80 signal and the
uncertainty-is-bad signal, which are consistent across folds. Treat the ranking as a prior for the
next collection, not a certified effect.

---

## 2. N4.02 — certification budget

### 2.1 The bound

The exchangeable unit is the **cluster** (a factory), not the window: windows inside a factory are
correlated (Z2 measured ICC and design-effect n_eff). A cluster-Hoeffding upper confidence bound
on the marginal false-accept rate R(t) = P(accept AND bad) is

```
UCB = Rhat + sqrt( ln(1/delta') / (2K) )        =>   K_min = ceil( ln(1/delta') / (2 (r - Rhat)^2) )
```

with K = number of independent clusters. `delta' = delta` for a **pre-registered single threshold**
or `delta' = delta / grid` for a **union bound over the LTT threshold grid** (Z2 uses a grid of 46
thresholds in [0.05, 0.50] step 0.01). Here **1 − δ = 0.95** (δ = 0.05) and Rhat = 0 (best case).

### 2.2 Factories needed (95 % confidence)

| target false-accept r | K, single pre-registered threshold | K, union over 46-threshold grid |
|---|---:|---:|
| **0.05** | 600 | **1,365** |
| **0.10** | 150 | **342** |
| **0.20** | 38 | **86** |

(If the observed risk floor is Rhat = 0.02 instead of 0, the r = 0.05 counts rise further — see
`N4.02.json` → `budget_rhat_0p02`.)

### 2.3 What we can certify today, and the review/error claim

- **We have K = 47 factories** → the tightest certifiable marginal false-accept is **r ≈ 0.27**
  (grid-union) or **r ≈ 0.18** (pre-registered threshold). **A 5 % or 10 % bound is not certifiable
  today.**
- The Z2 recommended operating point (**coverage 0.85 → review 12 %, marginal false-accept 4 %**)
  is therefore a **descriptive** statement on the 150-window IndEgo gold set, **not a certified
  guarantee**. Every published "review %/error %" number must carry `label_source` + `n` + protocol
  and the phrase *descriptive / not certified*.
- This is a **data-count limitation, not a model or tail-bound limitation** (Z2 §3.4). To promote
  an r-bound to production, **collect more independently-labelled factories**.
- **Clip-level labels.** There are 58 video clips (vs 47 factories) in the pool, and a clip-level
  label (one verdict per video) is ~17× cheaper per cluster than window labels. But clips **nest
  inside factories** (2 clips/factory), so the honest exchangeable unit stays the **factory**;
  clip-level labels buy certification power **only if the deployed claim is itself clip-level**
  (accept/reject a whole video). Recommendation: window labels for model error, **new factories**
  for certification.

---

## 3. Labeling spec — `ood_labels3.json`

### 3.1 File + schema

**File name:** `ood_labels3.json` (third batch; do not overwrite `ood_labels.json` / `ood_labels2.json` /
`ood160_labels.json`).

```json
{
  "description": "Batch q3: N4 next-labels (diversity/coverage selection, 47 factories). Human verdicts.",
  "n": 200, "n_good": null, "n_bad": null,
  "labeller": "<name>", "labelled_utc_start": "<ISO8601>",
  "selection_rule": "N4.01 diversity_maximin + per-factory balance (eval/results/exp_acc/labels_next.json)",
  "windows": [
    {
      "window": "q3_000",
      "segment_id": "s_a310f3da3d17",
      "verdict": "good",
      "reason": "",
      "factory": "factory_026",
      "video": "data/ego10k_scale/label_windows/s_a310f3da3d17.mp4"
    }
  ]
}
```

Per-window fields (exactly as requested):

| field | type | meaning |
|---|---|---|
| `window` | str | batch-local id, `q3_000`… in the order of `labels_next.json` |
| `segment_id` | str | the pool `window_id` from `labels_next.json` (e.g. `s_a310f3da3d17`) |
| `verdict` | str | `"good"` (usable / active work) or `"bad"` |
| `reason` | str | for `bad`: `NO_ACTIVE_WORK` \| `SHAKY` \| `HANDS_NOT_VISIBLE` \| other (free text); `""` for good |
| `factory` | str | `factory_0xx` (copied from `labels_next.json`, for grouped analysis) |
| `video` | str | path to the 10-s window clip (from `labels_next.json`) |

Keep the same convention as `ood160_labels.json` so the three batches merge cleanly
(`y = 1` good/usable, `y = 0` bad; `error = 1 − balanced accuracy`).

### 3.2 Which windows to label — the top-N list

**`eval/results/exp_acc/labels_next.json`** contains the ordered **top-200** windows:
- source pool = the 1,486 unlabelled candidates (1,546 rescored **minus** the 60 already-labelled q1 windows);
- ranking = **greedy farthest-first (maximin)** over z-scored 26 regime features against the 160
  already-labelled windows, with a per-factory redundancy penalty ×0.5;
- **not** ranked by model uncertainty (Z4: anti-correlated, AUROC 0.349);
- result = **200 windows across all 47 factories, 4–5 windows each** (deliberately balanced).

The first 10 rows (ids + factory + why):

| # | window_id | factory | novelty dist | labels in factory before |
|---:|---|---|---:|---:|
| 1 | `s_a310f3da3d17` | factory_026 | 20.68 | 1 |
| 2 | `s_c4fb549cd4a0` | factory_084 | 18.53 | 2 |
| 3 | `s_5eb801bdc5f9` | factory_037 | 17.42 | 3 |
| 4 | `s_a76a721b4a31` | factory_077 | 15.42 | 4 |
| 5 | `s_4c25b8f129fb` | factory_040 | 14.84 | 10 |
| 6 | `s_c737a8fce4fc` | factory_009 | 12.76 | 13 |
| 7 | `s_34212ab19135` | factory_085 | 11.28 | 2 |
| 8 | `s_e1da6ff314c7` | factory_033 | 11.14 | 2 |
| 9 | `s_0bb78a712271` | factory_079 | 10.72 | 4 |
| 10 | `s_50cb14457f49` | factory_016 | 10.52 | 4 |

`novelty dist` = distance from the nearest of the 160 already-labelled windows in z-scored
26-feature space; the selection deliberately favours **under-labelled factories** (few labels so
far) as well as feature novelty. The full committed `labels_next.json` is authoritative: each of the
200 rows carries `window_id`, `factory`, `clip`, `worker`, `start_s`, `end_s`, `p_good`,
`novelty_dist`, `rank_score`, `labels_in_factory_before` and a `why` string.

**How many.** Two separate budgets, because the two goals are different:

| goal | labels | where | what it buys |
|---|---:|---|---|
| **cut model error fastest** | **200** | existing 47 factories (top-200 list) | extends the diversity/maximin curve; the k = 80 point is the best-supported gain (−8 pts paired) |
| **make the claim certifiable** | **+39 new factories** (86 total) | **new data** (not in this pool) | certifies **r = 0.20** at 95 % (grid-union) |
| | **+295 new factories** (342 total) | new data | certifies r = 0.10 |
| | **+1,318 new factories** (1,365 total) | new data | certifies r = 0.05 |

Practical recommendation for the next collection: **label the 200** (model error, cheap, in-pool) and
**open a data request for ≥39 new factories** (the smallest certified r, 0.20). If the product needs
r = 0.10, budget ~300 new factories; r = 0.05 is a large multi-week collection and should be
scheduled, not assumed.

### 3.3 Acceptance criteria

A batch is **accepted** only if all hold:

1. **Schema:** every window has all six fields; `verdict ∈ {good, bad}`; `bad ⇒ reason` non-empty
   from the controlled vocabulary; `segment_id` exactly matches a `labels_next.json` id; `factory`
   matches the pool metadata.
2. **Coverage:** ≥ 90 % of the requested ids labelled (≥ 180/200); no more than 5 % skipped without
   a note; the batch spans all 47 factories.
3. **No leakage / no overwrite:** `ood_labels3.json` is a new file; existing batches are never
   edited; the 60 already-labelled ids are absent (the generator enforces this).
4. **Second-labeller check:** ≥ 20 windows independently re-labelled; agreement reported. If
   disagreement on `good/bad` > 10 %, the batch is quarantined and the label guideline is revised
   before merging (the OOD set is the only human ground truth we have — keep it clean).
5. **Merge:** after acceptance, merge into a new `ood_labels_merged.json` with a `latest verdict per
   window` rule (as `ood160_labels.json` did); re-run `N4.01`/`O2.02`; report the new curve with
   `n`, `label_source`, `n_bad` and factory-grouped CIs.
6. **Certification:** certification counts only **factories**, and only after each new factory has
   ≥ 1 labelled window; report the achieved K and the resulting certifiable r (N4.02 inversion).

---

## 4. Caveats (binding honesty rules)

- n = 160, n_bad = 22, 47 factories — **small**. All N4.01 differences are noisy; the report gives
  paired deltas and fold win-rates rather than claiming significance.
- The OOD set is **out-of-domain**: IndEgo-trained models did not transfer (AUROC ~0.4–0.7).
  Every error here is a *descriptive* OOD estimate, never a transfer claim.
- The 1,546 pool is **unlabelled** (`label_source = none`); its scores are used only as metadata.
- The certification budget is a **closed-form data requirement** (cluster-Hoeffding / LTT), not a
  measured guarantee. It assumes factories are exchangeable and that the risk is stationary; both
  are approximations for crowd-captured video and should be stated whenever a bound is quoted.
- No labels were edited; the pool, the 160 labels and all prior artifacts are read-only inputs.

## 5. Reproduce

```bash
git checkout m/acc-labels
OMP_NUM_THREADS=2 python3 eval/exp_acc/run_acc.py        # writes N4.01.json, N4.02.json, labels_next.json, figure
python3 -m unittest tests.test_acc_labels                # formula + schema checks (<30 s)
```

**Input note.** `eval/results/exp_acc/inputs/ood160_pixel.npz` (the frozen 26-feature matrix for the
160 windows) is a generated input cache (`.npz` is gitignored) produced by
`eval/exp_fewshot/prep_inputs.py` on branch `m-fewshot160`. `acc_lib._find_input` searches that
worktree as a fallback; to regenerate, run the O2 prep on the same frozen decode recipe.
