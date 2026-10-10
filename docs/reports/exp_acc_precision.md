# A4 — High-precision deterministic rule: coverage at ≥95 % decided-accuracy

**Track:** A4 (jadal-qa pod) · **Branch:** `m/acc-precision` → `main` · **Owner:** A4
**Device:** CPU only (cached scores; numpy). `OMP_NUM_THREADS=2`. Seed 7, B=2000 bootstrap.
**Results:** `eval/results/exp_acc/A4.01.json` (frontiers), `A4.02.json` (deterministic rule, per-factory, receipt)
**Code:** `eval/exp_acc/run_a4.py` (imports `eval/exp_acc/o4_lib.py`), `eval/exp_acc/make_fig_a4.py`
**Figure:** `docs/experiments/exp_acc/images/a4_coverage_accuracy.png`
**Reproduce:** `OMP_NUM_THREADS=2 python3 eval/exp_acc/run_a4.py`
**Registry:** A4.01, A4.02

---

## 0. Question and decision rule

"95 % on what we decide, abstain the rest": maximise the **coverage** (fraction of windows
given ACCEPT/REJECT instead of REVIEW) subject to **≥95 % decided balanced-accuracy** (BA on
the decided windows), in-domain and OOD. We also report decided **accuracy** (which, at the
low OOD bad-rate, is a much weaker bar).

**Decided BA** = ½·(share of decided *good* windows accepted + share of decided *bad* windows
rejected). A rule that decides only good windows and no bad ones has accuracy 1.0 but BA 0.5 —
so BA is the honest bar and the one used for the headline.

All rules are deterministic and prior-free. The sweep covers five families:

| family | form | deterministic? |
|---|---|---|
| symmetric band (prob) | decide iff \|p−0.5\| ≥ w | fixed permille thresholds |
| symmetric band (z) | decide iff \|s\| ≥ w (adapted z-score) | fixed permille thresholds |
| global quantile | reject bottom ⌊q·n⌋, accept top ⌊q·n⌋ | integer counts on the batch |
| asymmetric top/bottom-k | accept top ⌊a·n⌋, reject bottom ⌊r·n⌋ | integer counts on the batch |
| split conformal (LAC) | leave-one-group-out threshold q | fixed threshold after calibration |
| within-group quantile | per clip (in-domain) / per factory (OOD) rank fraction | integer counts per group |

---

## 1. Data and cached scores (nothing re-trained, nothing tuned on the hold-out)

| regime | n | good / bad | groups | protocol | scorer |
|---|---|---|---|---|---|
| in-domain | 150 | 100 / 50 | 7 clips | LOCO (OOF) | frozen two-view `p_ens`; adapted O4 LOCO |
| OOD-160 | 160 | 138 / 22 | 47 factories | factory-grouped | frozen `ood_scores*`; adapted O4 (`stable-8`+`Z3-6`) |
| hold-out 108 | 108 | 92 / 16 | 44 factories | **evaluate only, one-shot** | frozen `confirm108_ensemble`; adapted O4 |

* Label sources: `manifest_gold.json` (IndEgo, 1 labeller), `ood160_labels.json` (rounds 1+2,
  1 labeller), `round3/ood_labels3_108_with_factory.json` (**confirmatory; never used for
  tuning**). n_bad = 50 / 22 / 16 respectively — the OOD CIs are wide and honest.
* The **adapted O4** score = `stable-8` (E5) + `Z3 transferable-6`, IndEgo-only signs,
  unsupervised per-domain z-score, equal-weight mean (frozen recipe `configs/ood_recipe_o4.json`).
* The adapted score is an **OOD recipe**: in-domain it is weak (LOCO AUROC **0.563**), while the
  frozen two-view score is strong in-domain (**0.834**). On OOD-160 the adapted score is strong
  (**0.781**) and the frozen score near chance (**0.566**); on the hold-out both are ~0.68
  (frozen 0.690, adapted 0.676, a tie — consistent with C1.01).

---

## 2. Coverage–accuracy curve (headline)

**Maximum coverage at ≥95 % decided-BA** (point estimate; rule chosen on the same set, so
optimistic) and the **honest** version (grouped-bootstrap 95 % CI lower bound ≥95 %):

| regime | scorer | AUROC | full-decision BA | **max cov @ BA≥0.95** | rule | **cov @ BA-CI-lower≥0.95** |
|---|---|---|---|---|---|---|
| in-domain | frozen | 0.834 | 0.795 | **0.120** (18/150) | within-clip quantile q=0.075 | **0.027** (4/150) |
| in-domain | adapted | 0.563 | 0.510 | 0.013 (2/150) | band_z w=1.0 | 0.013 |
| OOD-160 | adapted | 0.781 | 0.733 | **0.300** (48/160) | asym a=0.27, r=0.03 | **unreachable** |
| OOD-160 | frozen | 0.566 | 0.546 | **unreachable** | — | unreachable |
| hold-out 108 | adapted | 0.676 | 0.648 | **0.269** (29/108) | asym a=0.21, r=0.06 | **0.148** (16/108) |
| hold-out 108 | frozen | 0.690 | 0.640 | 0.204 (22/108) | asym a=0.18, r=0.03 | unreachable |

Decided BA vs coverage (upper envelope over all rules; from `A4.01.json`):

| cov | in-dom frozen BA | in-dom adapted BA | OOD-160 adapted BA | hold-out adapted BA |
|---|---|---|---|---|
| 0.05 | 1.000 | 0.733 | 0.786 | 1.000 |
| 0.10 | 0.929 | 0.741 | 0.867 | 1.000 |
| 0.15 | 0.875 | 0.726 | 0.913 | 1.000 |
| 0.20 | 0.833 | 0.708 | 0.935 | 1.000 |
| **0.25** | 0.833 | 0.692 | **0.949** | 0.960 |
| **0.30** | 0.828 | 0.665 | **0.957** | 0.911 |
| 0.40 | 0.840 | 0.667 | 0.884 | 0.795 |
| 0.50 | 0.848 | 0.665 | 0.826 | 0.745 |

If the bar is plain **accuracy** (not BA) the answer is much more generous, because ~86 % of
OOD windows are good: OOD-160 frozen reaches accuracy ≥0.95 at coverage **0.169**, OOD-160
adapted at **0.281**, hold-out at **0.287**. But those points decide almost **no bad windows**
(e.g. OOD-160 frozen accuracy 0.963 with BA 0.5), so we do not headline them.

**Key reading.** The OOD-160 "0.300 @ BA 0.957" point is fragile: it accepts 43 windows
(all good) and rejects 5 (only **1 bad**), so its BA rests on a single bad decision out of 22.
That is exactly why the grouped CI lower bound never reaches 0.95 on the 160: with n_bad = 22
and 47 factories, no coverage gives a confident ≥95 % BA. The 108 hold-out behaves better
(CI lower bound ≥0.95 at 14.8 %) but n_bad = 16 there too.

---

## 3. Deterministic rule (A4.02)

The chosen rules expressed as integer operations (no floats in the verdict path; the score is
quantised to permille):

* **OOD-160, adapted O4:** `accept top 43`, `reject bottom 5`, `review middle 112` (n = 160)
  — i.e. rank the batch, ACCEPT the 43 highest scores, REJECT the 5 lowest, REVIEW the rest.
  Coverage 0.300, decided BA 0.957, accuracy 0.917.
* **Hold-out 108, adapted O4:** `accept top 23`, `reject bottom 6`, `review middle 79` (n = 108).
  Coverage 0.269, decided BA 0.960, accuracy 0.931. Honest variant: `accept top 14, reject
  bottom 2` → coverage 0.148, decided BA 1.0, grouped CI [0.956, 1.0].
* **In-domain, frozen two-view:** per-clip quantile q = 0.075 (reject the bottom 7.5 %, accept
  the top 7.5 % **within each clip**), coverage 0.120, BA 0.950. Fixed-threshold alternative:
  symmetric band around 0.5; global asymmetric `accept top 2 %, reject bottom 4 %` → coverage
  0.060 at BA 1.0.

**Receipt path verified.** The rule is serialised as
`{reject_at_or_below, accept_at_or_above, scale=1000}` into a `jadal.receipts.make_receipt`
receipt; two builds with the same inputs are **byte-identical**
(`receipt_sha256 = a65b0701…4ead`) and `verify_receipt` re-derives the same hash
(`match = true`). See `A4.02.json → receipt`.

---

## 4. Per-factory / per-regime variants

Within-group (per-clip in-domain, per-factory OOD) rank calibration vs the global rule at
matched coverage (`A4.02.json → per_factory`):

| regime / scorer | within-group max cov @ BA≥0.95 | global max cov @ BA≥0.95 |
|---|---|---|
| in-domain / frozen | **0.120** (per-clip) | 0.040 |
| in-domain / adapted | — | 0.013 |
| OOD-160 / frozen | — | — |
| OOD-160 / adapted | — | — |
| hold-out 108 / frozen | — | — |
| hold-out 108 / adapted | — | 0.037 |

**Per-factory calibration does not raise OOD coverage.** OOD factories hold only 1–4 windows
each (47 factories / 160 windows; 44 / 108 on the hold-out), so a per-factory quantile decides
almost nothing per factory and its pooled BA collapses. Per-**clip** calibration does help
in-domain (0.120 vs 0.040) because IndEgo clips hold ~21 windows each — but the honest CI lower
bound is still only 0.027.

---

## 5. Honest verdict

1. **≥95 % decided-BA is achievable today only at low coverage, and only as a point estimate.**
   In-domain: **≈12 % coverage** (18/150) with the frozen score; OOD: **≈30 % coverage**
   (48/160) with the adapted O4 score; hold-out: **≈27 %** (29/108, one-shot).
2. **The ≥95 %-with-confidence claim is not supported OOD.** With n_bad = 22 (160) and 16 (108)
   and 44–47 groups, the grouped-bootstrap CI lower bound on decided BA never reaches 0.95 on
   the OOD-160 at any coverage; it reaches **14.8 %** only on the 108. In-domain it reaches
   **2.7 %**. So the deployable, defensible statement is "**~3 % in-domain / ~15 % on the
   108 hold-out, and no confident claim on the 160**".
3. **Plain accuracy is a trap here.** Accuracy ≥0.95 is easy (17–31 % coverage) but those points
   decide almost no bad windows (BA ≈ 0.5–0.7). Only BA-based abstention is meaningful.
4. **The O4 adaptation helps OOD but not in-domain** (0.781 vs 0.566 OOD; 0.563 vs 0.834
   in-domain) — the two scorers are complementary, not one replacing the other.
5. **Per-factory calibration is not a lever** at 1–4 windows/factory.

**What would raise the coverage@95 % (→ A1–A3):**

* **Labels (A1/N4):** more OOD **bad** windows and more windows per factory. The BA bar is
  limited by n_bad = 22 / 16 and 1–4 windows/factory; ~2–3× more bad windows would make a
  95 % CI lower bound attainable at 20–30 % coverage.
* **Signals (A3):** the O4 score is the best available OOD signal (0.781); additional
  cheap, transferable signals (motion/hand/invariant features) that lift OOD AUROC toward
  0.85+ would push the 95 %-BA crossing from 0.30 toward 0.45+.
* **Models (A1/A2):** an in-domain scorer for OOD that transfers (the frozen ensemble does
  not: 0.566) — e.g. grouped-CV model selection / domain adaptation. The confirmatory 108
  already shows the pooled-160 0.781 does **not** replicate (0.676), so any model claim must
  be confirmed on fresh factories, not on the pooled 160.

**Caveats.** (i) The quantile / top-bottom-k rules are transductive (thresholds from the batch)
— deterministic and prior-free, but they need the batch; fixed permille bands are the
deployment-safe form. (ii) The coverage@95 % rule was *selected on the evaluation set*
(argmax over a large rule grid), so the point estimates are optimistic; the grouped CI lower
bound is the honest column. (iii) The hold-out 108 was scored once here and never used to
choose the rule family. (iv) Single labeller per set.

---

## Artifacts

`eval/exp_acc/run_a4.py`, `eval/exp_acc/make_fig_a4.py`, `eval/results/exp_acc/A4.01.json`,
`A4.02.json`, `eval/results/exp_acc/inputs/` (cached scores/features, read-only copies),
`docs/experiments/exp_acc/images/a4_coverage_accuracy.png`, registry A4.01–A4.02.
