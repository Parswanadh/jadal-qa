# Z4 — Clip-level sequential decisions + next-100 active labels

**Branch:** `m/clip-sequential` · **Owner:** Z4 (monitor) · lineage `pod/base`
**Data:** IndEgo human gold 150 windows / 7 clips (`labels/manifest_gold.json`, sha `43ab44e52eaa…`); cached LOCO OOF `caches/ensemble/oof_probs.npz`; 1,546 Egocentric-10K windows (`artifacts/windows_rescored.jsonl`); OOD-60 (`labels/ood_labels.json`, EVAL ONLY).
**Clip truth:** bad fraction ≥ 0.30 → 4 reject / 3 accept (desk=rej, ecu_full=acc, ecu_late=acc, logistics4=acc, organise=rej, pallet=rej, pallet2=rej).

> **Caveats up front.** In-domain n = **7 clips** (150 windows); the OOD set has **n_bad = 9** (51 good). Neither supports certified bounds — every number below is descriptive. The clip-verdict task is *hard*: the model reaches only ~4/7–6/7 clip verdicts at full coverage (the known IndEgo→Ego10K / pallet calibration problems). The sequential frontier is therefore unstable and its best cell is illustrative, not deployable.

**Regen:** `OMP_NUM_THREADS=2 python3 eval/exp_clip/run_clip.py --out eval/results/exp_clip` then `python3 eval/exp_clip/make_report.py`. Figures: `docs/experiments/exp_clip/images/{{clip_aggregation,sequential_frontier,active_labels}}.png`.

---

## Headline

**1 — Clip aggregation (Z4.01).** A high-quantile aggregate (q75) of the window bad-score separates the 7 clip verdicts best (clip AUROC **0.917**), but LOCO threshold balanced accuracy tops out at **0.75** (5/7 clips). The simple mean is close (AUROC 0.833, BA 0.71).

**2 — Sequential decision (Z4.02).** A Wald SPRT on the calibrated log-odds, processing windows in time order, produces a usable clip error / windows-processed / abstain frontier. The cheapest cell meeting **agreement ≥ 0.90 and abstain ≤ 0.40** is `A=4.9, B=3.7` (hard mode): **5/7 decided, 5/5 correct, 2/7 abstained, 66.5% of windows processed**. **But it is not cheaper than the fixed every-2 grid (51.5% at 7/7)** — sequential early-stopping does not drop review at equal error here, because the bad rate climbs with time (E3: 0.09 sec1 → 0.58 sec5), so a stopping rule must see the tail.

**3 — Active labels (Z4.03).** The top-100 to label next are written to `eval/results/exp_clip/top100_active_labels.json` (and the pod state file); they spread over all 47 factories. **Honest negative:** on the held-out OOD-60 windows the uncertainty score is *anti-correlated* with the true OOD errors (AUROC = 0.35 < 0.5; only 1/9 bad in the top-25%). The model is *confidently wrong* out of domain, so proximity-to-boundary does not find the domain-shift errors; the factory-diversity penalty is what buys coverage. Use the list for coverage/audit, not as a guaranteed error reducer.

---

## Z4.01 — clip-level aggregation + LOCO calibration

Aggregate the window bad-score (`1 − p_ens`) per clip; clip truth = bad fraction ≥ 0.30; calibrate on the **clip** level with leave-one-clip-out (threshold and isotonic); clip AUROC over the 7 clips.

| aggregator | clip AUROC | LOCO-threshold BA | correct | LOCO-isotonic BA |
|---|---|---|---|---|
| mean | 0.833 | 0.71 | 5/7 | 0.54 |
| median | 0.750 | 0.75 | 5/7 | 0.38 |
| q75 **←** | 0.917 | 0.71 | 5/7 | 0.71 |
| q90 | 0.833 | 0.71 | 5/7 | 0.67 |
| trimmed | 0.750 | 0.42 | 3/7 | 0.38 |
| rankmean | 0.750 | 0.42 | 3/7 | 0.38 |

**Window-level review needed to reach a clip error target** (start from the `mean` aggregate; review the *most uncertain* windows first, within each clip, and give them their true labels):

- clip error ≤ 1/7 (6/7 correct): review **0.35** of windows per clip.
- clip error = 0 (7/7 correct): review **0.65** of windows per clip.

For comparison the every-2 grid reviews 51.5% and reaches 7/7 in the *label-oracle* sense (E3.01). Targeted uncertainty review reaches 6/7 at 35% — a modest saving that buys one clip of error.

---

## Z4.02 — sequential SPRT frontier

Wald SPRT, windows in time order, `θ0=0.15, θ1=0.45` on the binary decision (`p̂_bad ≥ 0.5`). Stop **reject** if `S ≥ A`, **accept** if `S ≤ −B`, else abstain when the clip's windows run out. Cost = mean fraction of the clip's windows processed; error = 1 − agreement on decided clips.

| A | B | mode | windows processed | abstain | decided | error (decided) | wrong |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.5 | hard | 0.12 | 0.00 | 7/7 | 0.43 | desk, ecu_full, pallet2 |
| 1.5 | 0.5 | hard | 0.17 | 0.00 | 7/7 | 0.29 | desk, pallet2 |
| 1.5 | 1.0 | hard | 0.29 | 0.14 | 6/7 | 0.17 | desk |
| 5.0 | 4.0 | hard | 0.72 | 0.43 | 4/7 | 0.00 | — |

**Best tolerated cell** (`agreement ≥ 0.90`, `abstain ≤ 0.40`; 170 of 9860 fine cells qualify):

- `A=4.9, B=3.7`, mode **hard**: processed **0.665**, decided **5/5 correct**, abstain **0.286**.
- decided: `desk, ecu_full, ecu_late, organise, pallet`; abstained: `logistics4, pallet2`.
- robustness: the same partition holds for A ∈ [4.9, 11.5], B ∈ [3.7, 3.9] — but the decide/abstain boundary sits between **desk** (running min −3.69) and **ecu_late** (−3.92): a 1-clip-wide margin.

**θ sensitivity (hard mode):**

| θ0 | θ1 | tolerated cells | best frac | agree | abstain |
|---|---|---|---|---|---|
| 0.1 | 0.4 | 253 | 0.66 | 1.00 | 0.29 |
| 0.15 | 0.45 | 170 | 0.67 | 1.00 | 0.29 |
| 0.2 | 0.5 | 0 | — | — | — |
| 0.25 | 0.55 | 0 | — | — | — |

**Soft-mode finding.** the soft expected-LLR SPRT never reaches the accept bound on any clip: every clip's mean (1-p_platt) is positive, so good clips only abstain or (wrongly) drift to reject. The clip-level model probability is miscalibrated (a sign/offset shift, cf. the pallet inversion) -- binarising at 0.5 is what recovers the accept side. This is a calibration limitation, not a sequential-rule one.

**Label-oracle comparison.** Using the *true* window labels as the SPRT evidence (each processed window = one human review), the cheapest 7/7 cell is `A=1.25, B=4.5` at **0.655** processed — still above every-2's 0.515. 

**Review-savings finding.** Neither the deployable model SPRT (best tolerated cell 66.5% of windows) nor the label-oracle SPRT (65.5% at 7/7) processes fewer windows than the fixed every-2 grid (51.5% at 7/7). Sequential early-stopping does NOT drop review at equal clip error on these clips: because the bad rate climbs with time (E3: 0.09 sec1 -> 0.58 sec5), an early stop is biased good and must wait for the tail. This confirms E3's preference for a fixed time-spread grid.

**Recommendation.** Keep the fixed every-2 grid as the sampler. Use the SPRT only as a clip-level ABSTAIN gate: decide by threshold, and if the cumulative evidence is still inside the bounds at the end, send the clip to a human. The best tolerated cell shows the mechanism (5/5 correct, 2/7 abstained) but is not cheaper.

---

## Z4.03 — best cell + next-100 active labels

**Method.** greedy diversified uncertainty sampling: raw = 4p(1-p) (closeness to the decision boundary) + 0.5*|p_eg2 - p_pixel| (two-view epistemic disagreement); redundancy penalty = 0.5^(#picked in same factory) * 0.5^(#picked in same clip)

**Candidates.** 1,486 unlabelled Egocentric-10K windows (1,546 minus the 60 already-labelled OOD-60). Top-100 covers **47/47 factories** and 58 clips (max 3 per factory).

**OOD-60 sanity check (evaluation only).** Uncertainty U = 4p(1−p) vs the 9 human OOD-bad windows: AUROC = **0.349** (below 0.5), top-25% capture = 1/9. The 9 bad windows sit at *high* p (confidently accepted), i.e. *low* uncertainty — uncertainty sampling does not find domain-shift errors. Treat the ranking as diversity/coverage sampling.

**Top 25 of the next-100 (full list: `eval/results/exp_clip/top100_active_labels.json`):**

| # | window_id | factory | p(good) | U | view-gap | rank score |
|---|---|---|---|---|---|---|
| 1 | `s_843f651a660f` | factory_068 | 0.466 | 0.995 | 0.932 | 1.462 |
| 2 | `s_c442f553680a` | factory_036 | 0.462 | 0.994 | 0.919 | 1.454 |
| 3 | `s_51faaf27f1ac` | factory_066 | 0.460 | 0.994 | 0.920 | 1.454 |
| 4 | `s_e840f9634e29` | factory_069 | 0.448 | 0.989 | 0.896 | 1.437 |
| 5 | `s_5cf8b8157e7e` | factory_024 | 0.447 | 0.989 | 0.889 | 1.433 |
| 6 | `s_8118f35cc1b7` | factory_061 | 0.447 | 0.989 | 0.885 | 1.431 |
| 7 | `s_4eabccf5cd6b` | factory_057 | 0.443 | 0.987 | 0.885 | 1.429 |
| 8 | `s_c2728eb32a76` | factory_003 | 0.438 | 0.985 | 0.868 | 1.418 |
| 9 | `s_8a16dd9adf76` | factory_040 | 0.469 | 0.996 | 0.836 | 1.414 |
| 10 | `s_0b39377a261d` | factory_019 | 0.472 | 0.997 | 0.833 | 1.414 |
| 11 | `s_d51c707f1e79` | factory_080 | 0.439 | 0.985 | 0.857 | 1.414 |
| 12 | `s_0bb78a712271` | factory_079 | 0.426 | 0.978 | 0.852 | 1.404 |
| 13 | `s_7cddb1dd78ab` | factory_076 | 0.424 | 0.977 | 0.849 | 1.401 |
| 14 | `s_6754317afc29` | factory_014 | 0.425 | 0.977 | 0.847 | 1.401 |
| 15 | `s_476a3e094e56` | factory_001 | 0.444 | 0.988 | 0.825 | 1.400 |
| 16 | `s_7ecda5c5bd9d` | factory_084 | 0.423 | 0.976 | 0.845 | 1.399 |
| 17 | `s_e2cf61a40709` | factory_013 | 0.428 | 0.980 | 0.812 | 1.385 |
| 18 | `s_d45827fbe802` | factory_073 | 0.413 | 0.970 | 0.825 | 1.383 |
| 19 | `s_cf7c926ecf62` | factory_042 | 0.411 | 0.969 | 0.823 | 1.380 |
| 20 | `s_d6b5243cfa66` | factory_056 | 0.467 | 0.996 | 0.763 | 1.377 |
| 21 | `s_ec2b7e6e1909` | factory_058 | 0.409 | 0.967 | 0.818 | 1.376 |
| 22 | `s_f256094b19c5` | factory_039 | 0.408 | 0.966 | 0.811 | 1.372 |
| 23 | `s_1cb7de9e60ea` | factory_009 | 0.407 | 0.965 | 0.812 | 1.371 |
| 24 | `s_08850ee70dd5` | factory_016 | 0.395 | 0.956 | 0.787 | 1.349 |
| 25 | `s_7ee0350f7dde` | factory_062 | 0.394 | 0.955 | 0.787 | 1.348 |

---

## Caveats (small-n, stated plainly)

- **7 in-domain clips** → 4 reject / 3 accept; clip AUROC/BA, LOCO thresholds and the SPRT best cell are all computed on 7 points. The best cell's accept bound is separated by ~0.2 log-odds between two clips' running minima — an artefact, not a deployable rule.
- **OOD-60 = 9 bad** → the uncertainty-vs-error check is descriptive; an AUROC of 0.35 on n_bad=9 has a very wide CI and we do **not** fit on it.
- **Windows are ~15 s apart** and sparse; 'fraction of windows' is a proxy for footage, and a sequential stop saves feature compute, not decode/seeks.
- **Label source** is one human on IndEgo only; Ego10K windows are unlabelled. No transfer claim is made.
- Negative results kept: soft-LLR SPRT never accepts; sequential review does not beat the every-2 grid; uncertainty sampling does not enrich OOD errors.

