# DSH-PHYS: raising the deterministic physics rule model (v2)

Track: `dsh/phys-v2` (base `origin/main`). Owner: DSH research lead, jadal-qa pod.
Device: **CPU only** — `nvidia-smi` fails on this pod (`Failed to initialize NVML:
Unknown error`), so every number here is CPU/ffmpeg; no GPU was used.
Repo policy followed: nothing tuned on the 108-window confirmatory hold-out before
the freeze; OOD labels never used to fit, weight, threshold or select anything.

## 0. What was asked and what actually happened

The mission was to raise the accuracy of the **deterministic physics rule model** —
fixed constants, integer Q=4 comparisons, no learned weights in the decision path,
byte-identical receipts — in-domain first, then survive the domain shift without labels.

Three things were achieved, one was not:

| # | Result | Number | Verdict |
|---|---|---|---|
| 1 | In-domain v2 table beats the Q1 table (LOCO AUROC) | **0.8472** vs 0.822 | achieved, +0.025 |
| 2 | Deterministic re-weighting (LR / exhaustive grid) | 0.7965 / 0.7145 | **negative** — worse than uniform |
| 3 | Label-free per-domain re-quantiling rescues OOD-160 | 0.427 → **0.546** (all rules), 0.504 → **0.638** (a-priori scale subset) | partially achieved |
| 4 | One-shot confirmatory hold-out (108 windows, frozen) | **0.5166** [0.367, 0.695] absolute, 0.5099 [0.348, 0.693] re-quantiled | **honest negative — no transfer** |

The hold-out result is the one that matters for deployment and it is bad. It is
reported first-class, not buried.

## 1. Validation gate: is the rebuilt pipeline faithful?

Before trusting any new number, the frozen Q1 table (PR #141) was applied to a matrix
rebuilt from scratch with the frozen vendored kernels
(`eval/exp_phys_dsh/build_indego.py` → `inputs/indego_full.npz`, 150×42,
`full.*`+`phys.*`), under Q1's own LOCO protocol.

* Rebuilt features vs the Q1 caches: `max|Δ| ≤ 4e-16` for 9 of 10 probe features,
  `full.temp_b0` differs by up to 0.156 (corr 0.999995) — a float-path difference that
  A4 showed cancels in the per-fold quantile thresholds (0/150 decision differences).
* **LOCO AUROC of the frozen Q1 table on the rebuilt matrix = 0.8222**, exactly Q1's
  reported 0.822 (`LEAD_v2.json`, `part_A_validation`).

Gate passed. Every OOD and hold-out number below uses this same 42-feature space.

## 2. New deterministic rules from physical references (A1, commit `2ad71fd`)

Instead of raw quantiles of the feature pool, A1 derived thresholds as
`round(k × a physical reference measured on TRAINING clips × 1e4)` — a sharpness
floor, a skin-fraction floor, a shake ceiling, a motion-energy ceiling. Five of A1's
eight rules live in the packaged 42-feature space and are reproduced verbatim in
`configs/phys_rules_v2.json`:

| rule | feature | op | thr (Q=4 int) | weight | derivation |
|---|---|---|---|---|---|
| HANDS_NOT_VISIBLE | `full.skin_frac` | < | 261 | 2 | 0.75 × median training skin fraction |
| BLUR | `full.dct_energy` | < | 65 | 3 | 0.75 × median training DCT AC energy |
| SHAKY | `full.global_shift` | > | 211595 | 3 | 2.0 × median training global shift |
| BLUR | `full.hf_ratio` | < | 127 | 3 | 0.75 × median training HF share |
| NO_ACTIVE_WORK | `full.motion_energy` | > | 577 | 3 | 2.0 × median training motion energy |

A1's own best configuration (per-fold measured thresholds, not frozen constants)
reached **LOCO AUROC 0.8411 [0.7319, 0.8972]**, BA 0.7650, FRR 5.0% on IndEgo
(`A1_indomain_v2.json`). Its negative results: per-fold re-selection of rules/weights
overfits (0.7764, FRR 14%); the QUALITY veto is inert in-domain; lexicographic
priority-OR is far weaker (0.6650); `P_diff_e_min` and `G_dup_hash_mean` are constant
on all 150 windows (zero information); BLUR rules dominate (4 of 8 selected) — the
in-domain AUROC is driven by **acuity, not exposure**.

## 3. The frozen v2 table

`configs/phys_rules_v2.json` — selected candidate **C3 = A1's five fixed rules + the
Q1 16-spec graded ladder**, uniform integer weights on the Q1 part, A1's integer
weights on the new part, `t_reject = 25`, `t_accept = 0`. All frozen; no threshold is
fitted inside a test fold, so the LOCO numbers are a zero-degree-of-freedom evaluation.

| candidate | LOCO AUROC | LOCO BA | LOCO FRR | tau* |
|---|---|---|---|---|
| C1 = Q1 uniform (control) | 0.8230 | 0.7050 | 5.0% | 13 |
| C2 = A1 five rules only | 0.8231 | 0.6250 | 5.0% | 7 |
| **C3 = merged (FROZEN)** | **0.8472** | 0.6700 | 6.0% | 25 |
| C4 = C3 minus FROZEN_STATIC | 0.8469 | 0.6200 | 6.0% | 25 |
| C5 = C4 without the QUALITY veto | 0.8469 | 0.6200 | 6.0% | 25 |

Protocol: IndEgo 150 (100 good / 50 bad, human labels `manifest_gold.json`), 7 clips,
clip-bootstrap B=2000 seed 0. Command: `python3 eval/exp_phys_dsh/freeze_v2.py`.
Source: `LEAD_freeze.json`.

**The in-domain gain is +0.024 AUROC over Q1 (0.8472 vs 0.8230) and it is not
statistically established** — with 7 clips one clip moves AUROC by >0.05 and the
clip-bootstrap CIs overlap heavily. The BA is *lower* than Q1's 0.808 because Q1's
0.808 is a decided-subset-conditional BA at a different operating point; the numbers
are not directly comparable and are not claimed to be.

## 4. Deterministic aggregation: uniform beats everything (negative result)

The obvious idea — replace the uniform integer count with derived integer weights —
was tested three ways, all on training clips only, all under the same LOCO:

| weighting | derivation | LOCO AUROC |
|---|---|---|
| uniform (Q1) | every threshold crossing = +1 | **0.8222** |
| integer LR | `w = clip(round(ln(p_bad/p_good)), 0, 4)`, Laplace-smoothed, training rows only | 0.7965 |
| exhaustive integer grid | `argmax` over {0..3}^8 of training BA s.t. training FRR ≤ 5% | 0.7145 |

Source: `LEAD_v2.json`, `part_B_aggregation`. Both principled re-weightings are
**worse**. The grid result is the classic failure mode: maximising a threshold metric
(BA) on 6 training clips picks weights that do not generalise.

There is a reason, and it is the mathematical core of this report: counting how many
*fixed quantile levels* a value crosses is an **order-statistic (rank) functional of
each feature**. It is invariant to any strictly monotone per-feature reparametrisation
of the data. Log-likelihood-ratio weights, by contrast, estimate the source domain's
*marginal* likelihoods — precisely the quantity that the domain shift breaks. The
uniform count is therefore the more robust statistic, and it is also the one with no
free parameter to overfit. This is why (a) reweighting loses in-domain under LOCO and
(b) only the *location* of the thresholds needs adapting across domains.

## 5. The domain shift: label-free per-domain re-quantiling

`LEAD_v2.json` part C and `A2_ood_transfer.json` (A2, independent implementation).
OOD = 160 human-labelled Egocentric-10K windows (138 good / 22 bad), window-bootstrap
CI (optimistic). The target's own **unlabelled** values supply the order statistics;
rule list, directions, quantile levels and integer weights stay frozen from IndEgo.

| variant | OOD AUROC | CI95 | reject rate | in-domain cost |
|---|---|---|---|---|
| raw frozen IndEgo thresholds | 0.4272 | — | 99.6% (A2) / 38% (lead, 42-feat rebuild) | none (0.823) |
| **per-domain re-quantile** | **0.5456–0.556** | — | 8.1% | 0.7128 (requantile on a held-out clip) |
| per-domain z-score | 0.6794 in-domain / weaker OOD | — | 12% | — |
| 32-window unlabelled calibration slice | 0.5422 (mean of 20 seeds, sd 0.027) | — | — | — |

The reject-everything pathology disappears and AUROC moves from below chance to
slightly above it, but **it does not approach the in-domain 0.82**. A2's per-rule
diagnostics explain why: under re-quantiling the *scale-like* rules recover
(BLUR 0.460 → **0.630**, CODING_NOISE 0.509 → **0.613**, DUPLICATE 0.427 → 0.512)
while the *semantic/motion* rules stay at chance (NO_ACTIVE_WORK 0.503,
TOO_DARK 0.498, HANDS_NOT_VISIBLE 0.408 — the last one gets worse). The OOD bad
windows are overwhelmingly NO_ACTIVE_WORK (19 of 22), and that is exactly the mode the
IndEgo-derived motion rules do not carry across.

Acting on that diagnostic with an **a-priori** subset (sharpness / bitrate / hash
features — chosen from feature physics, *not* from OOD labels) gives a real gain:

| subset | in-domain AUROC | OOD raw | OOD re-quantile |
|---|---|---|---|
| all 16 specs | 0.823 | 0.4272 | 0.5456 |
| **a-priori scale subset (BLUR, CODING_NOISE, DUPLICATE)** | 0.789 | 0.5040 | **0.6380** |
| motion only | 0.711 | 0.4042 | 0.4834 |
| quality only | 0.659 | 0.3124 | 0.5275 |

Source: `LEAD_subset_ood.json`. This is the best OOD number produced here and it is
still far below the 0.781 of the adapted transferable score, and far below the
in-domain 0.82. It is a partial rescue, not a solution.

## 6. One-shot confirmatory hold-out (108 windows) — honest negative

The frozen `configs/phys_rules_v2.json` (sha256 recorded in the results JSON) was
scored **once** on the 108-window confirmatory hold-out (92 good / 16 bad, human
labels, `labels/round3/ood_labels3_108_with_factory.json`); the script writes a lock
and refuses a second run. Command: `python3 eval/exp_phys_dsh/score_holdout.py`.

| mode | AUROC | CI95 (factory bootstrap) | BA | reject rate | FR(good) |
|---|---|---|---|---|---|
| absolute (frozen thresholds) | **0.5166** | [0.3673, 0.6952] | 0.504 | 5.6% | — |
| re-quantile (label-free) | **0.5099** | [0.3481, 0.6932] | 0.524 | 8.3% | — |

**The frozen v2 table does not transfer to the hold-out: both modes are at chance and
the CIs include 0.5.** The hold-out's bad reasons are NO_ACTIVE_WORK 10, OVEREXPOSED 4,
HANDS_NOT_VISIBLE 2 — a different mix from OOD-160, and again dominated by a mode the
physics rules do not detect. With n_bad = 16 this is a wide, single point estimate;
but it is the honest one-shot answer, and the answer is no.

## 7. Review ↔ error frontier and per-rule value (A4, commit `250eb63`)

A4 reproduced the Q1 operating point exactly (decided 0.400, REVIEW 0.600,
BA_decided 0.8080, AUROC 0.8222 = Q4.01 `pure_rules`) and then swept the integer band
(`A4_review_band.json`):

* At the pre-registered 5% training-FR budget, moving the band's lower edge to
  `t_accept = 3` gives BA 0.8320 (+0.0240), REVIEW 0.520 — but this point was chosen on
  test metrics, so it is an upper bound.
* Best *deployable* point (per-fold inner selection of `(t_accept, tau)` on training
  windows only): BA **0.8512** (+0.0431), REVIEW 0.667, FR 0.04 — paired clip-bootstrap
  ΔBA +0.0349 [−0.109, +0.429], P = 0.61. **Not statistically distinguishable.**
* Frontier mechanics: `t_accept` moves ACCEPT↔REVIEW only; `t_reject` moves
  REJECT↔REVIEW and is where the false-reject rate is actually controlled.
* Drop-one-rule LOCO: FROZEN_STATIC is the one harmful rule (+0.0503 BA when dropped);
  OVEREXPOSED mildly harmful (+0.0202); BLUR (−0.0203 AUROC), TOO_DARK (−0.0245),
  NO_ACTIVE_WORK (−0.0182), CODING_NOISE (−0.0148) are load-bearing.
* Axis-2 (margin-to-nearest-threshold) is an **honest negative**: it does not sharpen
  the frontier.

## 8. Determinism re-proof

* A4's independent self-contained integer scorer (`det_scorer_v2.py`): receipt sha256
  `c0d09f83fb1565ef820bb1eed2582487ab5ba2bcd300feaa70f163dbfe997b63`, **10/10 in-process
  and 10/10 separate processes → 1 unique hash, diff count 0**; also identical under row
  permutations, feature-key order, `PYTHONHASHSEED` 0/1/random, `LC_ALL`, `TZ`,
  `PYTHONUTF8=0`, CWD. 985 evidence entries, 0 non-integer fields after JSON round-trip.
* The packaged v2 scorer (`tools/phys/model_v2.py` + `score_window_v2.py`) re-proved
  independently here: 10 runs with alternating `PYTHONHASHSEED` → **1 unique receipt
  hash** `3dfefc8156a2e9629ed2d6b4993fa7f0209f0c71fb265328f56cd384c0854d14`.
* Every decision emits `{rule, feature, value_q, op, threshold_q, weight, grade}`.
* **Not proven**: cross-machine / cross-Python identity, and the upstream feature
  extraction, which is still float (the decision path begins at the Q=4 quantisation).

## 9. What is shipped

* `configs/phys_rules_v2.json` — frozen rule table, integer weights, `t_reject`/`t_accept`,
  per-threshold derivations, the two domain modes, provenance and limitations.
* `tools/phys/model_v2.py` — integer-only decision engine + canonical receipt.
* `tools/phys/score_window_v2.py` — CLI scorer (absolute and label-free re-quantile modes).
* `eval/exp_phys_dsh/` — build scripts, `run_v2_lead.py`, `freeze_v2.py`,
  `run_subset_ood.py`, `score_holdout.py`, `BRIEF.md`.
* Results: `LEAD_v2.json`, `LEAD_freeze.json`, `LEAD_subset_ood.json`,
  `HOLDOUT108_final.json`, `A1_*.json`, `A2_ood_transfer.json`, `A4_*.json`.
* Feature matrices (`inputs/*.npz`, not committed): IndEgo 150×42, OOD-160 160×42,
  hold-out 108×42 — all built with the frozen vendored kernels.

## 10. Limitations (read before quoting any number)

1. **IndEgo is 150 windows over 7 clips.** One clip moves AUROC by >0.05; all in-domain
   CIs are wide and the v2-vs-Q1 gain (+0.024) is inside the noise band.
2. **OOD-160 has 22 bad windows, the hold-out 16.** Both CIs are wide; the hold-out CI
   includes 0.5. The OOD bootstrap is window-level and therefore optimistic.
3. **The hold-out says no.** Whatever the in-domain gain, the model does not transfer.
4. **Re-quantiling is transductive**: it uses the target's own unlabelled distribution.
   That is label-free but not assumption-free — a target with a different bad-rate moves
   the order statistics. The 32-window calibration slice result (0.5422 ± 0.027) shows
   the effect survives a small slice but is weaker.
5. **Candidate rule families were shortlisted on the same IndEgo pool** (mild selection
   bias); thresholds and weights themselves are training-only.
6. **CPU only.** The A40 was unavailable (`nvidia-smi` NVML failure); no GPU number is
   claimed.
