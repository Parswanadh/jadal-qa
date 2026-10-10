# O1 — Frozen PoC out-of-domain evaluation on the 160 human labels

**Claim:** O1 · **Branch:** `m/ood160-eval` · **Type:** evaluation only (nothing fitted, tuned or re-thresholded)
**Data:** `jadal_pod_bundle/labels/ood160_labels.json` — 160 Egocentric-10K 10-s windows, 47 factories, one human labeller, 138 good / 22 bad (bad: 19 `NO_ACTIVE_WORK`, 2 `HANDS_NOT_VISIBLE`, 1 `SHAKY`). Merge of batch q1 (60) + q2 (100).
**Scores:** pod-bundle caches `caches/ensemble/ood_scores.json` (q1) + `ood_scores2.json` (q2) of the **frozen** two-view `ensemble_v1` (config sha256 `4da1068a…`, threshold 0.5, abstain band `[0.4, 0.6]`). No video decode, no GPU.
**Artifacts:** `eval/results/exp_ood160/O1.01.json`, `eval/results/exp_ood160/labels160.json` / `.csv`, `eval/exp_ood160/run_ood160_eval.py`, figure `docs/experiments/exp_ood160/images/ood160_auroc_confusion.png`.

This is the pre-registered confirmatory test: n_bad = 22 now exceeds the 10/class minimum. Nothing about the frozen scorer was changed to produce it.

## Headline

| scorer | OOD AUROC | 95% CI window (optimistic) | 95% CI factory-grouped (honest) |
|---|---|---|---|
| **frozen ensemble `usable_prob`** | **0.566** | 0.433 – 0.693 | **0.414 – 0.697** |
| EG2 view only | 0.589 | 0.407 – 0.743 | 0.407 – 0.743 |
| pixel-26 view only | 0.520 | 0.393 – 0.646 | 0.393 – 0.646 |
| *shipped PoC cascade\** | *0.600* | *0.474 – 0.724* | *not available* |

The frozen two-view ensemble carries a **weak positive** out-of-domain signal. Factory-grouping barely widens the CI here (47 factories, 160 windows) but the honest lower bound **drops below 0.5** — the point estimate is not statistically separable from chance at this sample size.

## Shipped operating point (frozen rule: `p>0.5` accept, `0.4≤p≤0.5` review, `p<0.4` reject)

Verdict distribution over all 160: **accept 71 · review 35 · reject 54**.

| | accept | review | reject | n |
|---|---|---|---|---|
| **human GOOD** | 63 (46%) | 28 (20%) | **47 (34%)** | 138 |
| **human BAD** | 8 (36%) | 7 (32%) | 7 (32%) | 22 |

- Balanced accuracy **0.387** with review counted as a miss (good-accept 0.457, bad-reject 0.318); CI window 0.285–0.502, factory-grouped 0.277–0.485.
- Balanced accuracy **0.546** as the pre-registered scalar decision at 0.5 (review ⇒ reject); this reproduces the handover's `balacc@0.5 0.546`.
- **The shipped threshold is a bad binary classifier here:** it hard-rejects **47 of 138 good windows (34% false rejects)** while rejecting only 7 of 22 bad windows. A third of good footage is thrown away and two-thirds of bad footage survives.

## What improved vs the old n = 60 result

| | n=60 (old, 51/9) | n=160 (now, 138/22) |
|---|---|---|
| frozen ensemble AUROC | 0.415 | **0.566** |
| shipped PoC cascade AUROC | 0.529 | **0.600**\* |
| cascade balanced accuracy | 0.51 | 0.549\* |

- The old **0.415** was noise at n_bad = 9, not an inverted score. At n = 160 the ensemble point estimate is **0.566** — a weak but consistent direction (both views > 0.5, EG2 0.589 > pixel 0.520).
- The warning "all three IndEgo scorers are at chance out of domain" is still *broadly* true: no scorer clears chance at the 95% level. But "at chance / inverted" was too strong; the correct statement is **weak positive signal, not yet usable as a gate**.

## What remains unreliable

1. **Not separable from chance.** Both CIs contain 0.5 (grouped lower bound 0.414). Any claim that the frozen PoC "works" out of domain is unsupported.
2. **The decision rule is worse than the ranking.** AUROC 0.566 but 34% of good windows rejected at 0.5. The ranking carries a little signal; the operating point discards it.
3. **Batch heterogeneity.** q1 (n=60) AUROC 0.415 vs q2 (n=100) 0.657. Two batches from one labeller disagree; label-severity drift or sample composition is confounded with any "improvement".
4. **Factory structure.** 47 factories, but a few contribute many windows and both classes (`factory_001` 10, `factory_003` 11, `factory_009` 13). Window bootstrap treats them as independent; it is not.
5. **The shipped PoC cascade** (AUROC 0.600) is a *different* IndEgo-calibrated artifact whose heads are not in the bundle caches — carried over from `docs/ood_eval_poc_cascade_160.txt` / `FACTS.md:97`, **not recomputed**, and has no factory-grouped CI. Do not quote a grouped CI for it.
6. **Cache hygiene.** `ood_scores2.json` stores 7 windows with `usable_prob>0.5` as `review` (it used the wider `[0.4,0.6]` band). Verdicts here are recomputed from the frozen config rule, not copied; 7/160 differ.

## Reproduction

```
OMP_NUM_THREADS=2 python3 eval/exp_ood160/run_ood160_eval.py   # -> eval/results/exp_ood160/O1.01.json
OMP_NUM_THREADS=2 python3 eval/exp_ood160/make_fig.py          # -> docs/experiments/exp_ood160/images/
```

\* cascade numbers are provenance-labelled carry-overs; everything else in this report is recomputed from the frozen config and the shipped label file.
