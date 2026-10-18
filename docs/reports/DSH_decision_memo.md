# DSH decision memo — GRID / AUTO / TASK (2026-10-18)

Author: DSH (lead agent). Base `origin/main` at `53983d6` + tooling restore (`dsh/infra`).
Environment and base decision: `docs/pod/ENV_grid_auto.md`. Track reports:
`docs/reports/grid_pipeline.md`, `docs/reports/auto_policy.md`, `docs/reports/task_completion.md`.
PRs: **#174** GRID, **#175** AUTO, **#173** TASK. Every number below is from a results JSON under
`eval/results/` with n, n_bad, label source, protocol, CI type, device and command. All of it is
CPU on the builder's laptop; no GPU result is claimed.

## 1. The recommendation

**Do not ship full automation. Ship tiered automation with an audit sample, and keep the review band.**

The brief's decision rule is explicit: if the lower bound of the 100 %-coverage balanced accuracy on
OOD-160 is below 0.70, the recommendation must be tiered automation with audit. Measured:

| 100 % coverage, factory footage (OOD-160: n=160, n_bad=22, **19 of 22 bad are `NO_ACTIVE_WORK`**, 47 factories, ONE labeller) | value | 95 % CI (factory-grouped) |
|---|---|---|
| Frozen shipped 26-number head, AUROC | 0.588 | **[0.450, 0.728]** — includes 0.5 |
| Forced accept iff p ≥ 0.5, balanced accuracy | **0.556** | **[0.448, 0.673]** |
| ... false-accept rate (bad windows accepted) | 0.409 | |
| ... false-reject rate (good windows rejected) | 0.478 | |

At full automation the current pipeline accepts **41 %** of bad factory windows and rejects **48 %** of
good ones, and the balanced-accuracy interval **includes 0.5**. The same policy in-domain (IndEgo-150,
n=150, n_bad=50, LOCO, clip bootstrap) is 0.740 [0.631, 0.789] at p ≥ 0.5 and 0.755 [0.646, 0.797]
with a per-fold Youden threshold chosen on training clips only. So: the funnel is usable in-domain,
it is not established to work on factory footage, and **100 % coverage is not supportable today.**

## 2. What to ship, what to drop

**SHIP**
* The existing funnel unchanged (S0 exact rules → 26-number head → 0.25/0.75 band). Nothing in this
  pass justified moving it.
* **An audit policy** (AUTO.07): auto-decide, send a random 2–5 % sample **plus every low-confidence
  window** to humans, track error and verdict mix per site, alarm on drift. All hours columns are
  arithmetic on assumed review time and are labelled as such.
* **The brightness-calibrated sharpness rule** (GRID.09) — the one clear win in this pass. It fixes a
  hole before proposing the rule: the shipped five S0 rules reject **0 of 16** blurred windows and the
  head accepts **6 of 16 at σ=3 with max p = 0.9999**. With the rule (threshold 0.0211, chosen on a
  clip/factory-grouped CALIB split that never saw a blurred window), on held-out EVAL (n=132 each):
  catch **18.9 % [5.8, 40.4]** at σ=1.5, **72.0 % [38.8, 89.4]** at σ=3, **100 %** at σ=8; AUROC at
  σ=3 **0.955 [0.876, 0.991]**. Clean false-reject on human-good windows **3.00 % [0.62, 8.52]**
  (IndEgo, 3/100) and **6.52 % [3.04, 12.0]** (OOD-160, 9/138) — and **0 of 238 are marginal**: all 12
  exits were already rejected by the shipped five rules, so enabling it costs nothing on clean data.
  The brightness control confirms the calibration does the work (surface-only gain ×1.4 rejects
  1.6 % / 7.2 %), but the statistic is **not gain-invariant** — gain ×0.5 rejects 22.2 % / 15.9 %.
  **It ships OFF** (`BLUR_LAPVAR_CAL_MIN = 0.0`); enabling is one constant flip, and that is the
  builder's call. Requirements: all positives are **synthetic** Gaussian blur at 480p (no real defocus
  footage exists in the inputs), and the catch at σ=3 is 100 % on IndEgo but 46 % on OOD-160, so
  per-dataset operating points are still needed.

**DROP**
* **The mosaic as a decision input.** Pre-registered kill-gate **FAILED**: inter-tile features add
  **+0.019** AUROC (paired clip bootstrap [−0.018, +0.053], does not exclude zero), under the 0.02
  threshold. The margin against the 0.02 bar is thin (0.0006), so the **stronger** reason to stop is
  that the interval includes zero — the effect is not established at all. Spatial-only mosaic features
  score **0.6864 [0.522, 0.772]** against the 26-number baseline **0.810 [0.683, 0.892]** — a paired
  difference of **−0.124 [−0.202, −0.071]** that excludes zero, and the mosaic head is **below chance
  on the `pallet` clip** (0.197) which holds 11 of the 50 bad windows. The customer's headline
  **12-frame 4×3 picture scores 0.7044 [0.488, 0.786]**; frame count is flat from N=4 to N=16, and
  smaller tiles are marginally *better* (t112 0.7146). Keep the mosaic as an **evidence picture**,
  which is what it already is. All of this was **independently re-derived by a different stream with
  its own code and reproduced exactly** (TASK.07, PR #173).
* **Mosaic out of domain.** The IndEgo-trained mosaic head **inverts** on factory windows:
  **0.3557 [0.224, 0.515]** vs the shipped head's 0.588, paired **−0.232 [−0.380, −0.077]**.
* **The EmbeddingGemma-2 second view as a review-band resolver.** In-band AUROC 0.811 vs the head's
  0.820, paired Δ **−0.010 [−0.115, +0.052]**; the hybrid at 100 % coverage is Δ **0.000
  [−0.044, +0.059]**. No improvement. (A band rank-average looked like +0.060 [+0.015, +0.136] but
  equal-coverage controls show it is an operating-point change: false-accept 0.26 → 0.12 while
  false-reject 0.26 → 0.28.)
* **Conformal-style error guarantees at useful coverage.** Under exchangeability with the assumptions
  listed in AUTO.03, a 5 % target supports at most **7.3 % coverage [1.1, 16.9]** — and the realized
  held-out risk was **18.2 %**, so the bound failed at every target on held-out clips, reproducing the
  repo's earlier nominal-2.7 %-became-46 %. This is not a shippable guarantee.
* **The mosaic S0 probe as the production gate** (GRID.05). It reaches six-rule agreement **98.2 %
  [90.6, 99.7]** on 56 synthetic clips, but its clean false-reject rate is **7.00 % [3.43, 13.75]**
  (IndEgo) and **7.97 % [4.51, 13.71]** (OOD-160) — all of it *marginal* windows that the shipped
  rules already reject, against **0 of 238** for the full-pipeline blur rule. Keep the shipped probe.
* **The mosaic as a cost story** (GRID.07). The mosaic path is genuinely faster — **190.5 ms/window
  → 52.5 footage-h/wall-h** vs the shipped **348.4 ms → 28.7** — but that is **0.55×**, not the
  ≤ 0.33× the pre-registered decision rule (b) requires, so cost alone does not buy it either.

## 3. Human workload (arithmetic from measured band shares, not a measurement)

Per 10,000 video-hours = 3.6 M ten-second windows. Measured band shares: **48.0 %** in-domain
(clip CI [0.347, 0.570], n=150) and **13.75 %** on factory footage (factory CI [0.075, 0.206],
n=160). Human time per review is **assumed 30 s** — no measured review time exists in the repo or the
bundle — and every hours figure scales linearly in that assumption.

| Dataset | band reviews / 10k video-h | band-only human-hours | +2 % audit | +5 % audit | +10 % audit |
|---|---|---|---|---|---|
| IndEgo (in-domain) | 1.728 M | 14,400 | 15,000 | 15,900 | 17,400 |
| Factory (OOD-160) | 0.495 M | 4,125 | 4,725 | 5,625 | 7,125 |

Expected share of **all** errors that the audit actually detects is much lower than the audit rate,
because most errors are outside the audited set: 0.623 / 0.635 / 0.654 in-domain and
**0.177 / 0.202 / 0.244** on factory footage at 2/5/10 %.

**The band, not the audit, drives the workload.** Removing the band without fixing the error is how
you get 41 % of bad windows accepted; shrinking the band is a business decision about which error you
can tolerate, and that is the builder's/Awign's call, not a modelling result.

## 4. Risks

1. **The decision-relevant weakness is `NO_ACTIVE_WORK`, and pixels barely see it.** 19 of 22 bad
   factory windows are "the worker is not doing task-relevant work". The frozen score separates them
   from good windows at AUROC **0.5755 [0.4275, 0.7386]** (n=157, n_bad=19) — no separation
   established. More labels alone will not fix this; the lever is new signals (hand visibility, motion,
   task context), which this pass did not build.
2. **One labeller, everywhere.** Every CI here is conditional on one person's labels. The TASK stream
   found that on IndEgo the human `NO_ACTIVE_WORK` label and the independent VIA step annotation
   **disagree on 12 of the 16 such windows** — direct evidence that this label needs adjudication.
3. **Small n out of domain.** n_bad = 22 (19 of one reason) across 47 factories. The shipped head's own
   interval includes 0.5. Nothing about factory performance is *established*, in either direction.
4. **Band-share drift is real** (the repo measured accept share moving 19 %→57 % when frame sampling
   changed), so the workload table is a snapshot, and per-site monitoring is not optional.
5. **Round-3 hold-out is consumed** (two prior scorings, C1 #162 and physics v2 #163). It was **not**
   scored again here — nothing earned the last permitted look. It therefore cannot support adoption.
6. **Pre-registration ordering is recorded but not independently attested.** GRID.01/02 PREREG
   configs re-hash to the sha256 in their results JSON, but PREREG and results landed in the same
   commit, so git alone cannot show the config was frozen before the run. The registry timestamps and
   the direction-predicting hypothesis are consistent with good faith, not proof. A future pass should
   commit PREREG in its own commit before the run and record the commit sha.

## 5. Questions only the builder / Awign can answer

1. **What error is tolerable at 100 % coverage?** Our measured factory balanced accuracy is
   0.556 [0.448, 0.673]. If Awign's tolerance is below 0.70 we need new signals, not new thresholds.
2. **What is the cost ratio (false accept : false reject)?** Cost-sensitive thresholds were measured at
   1:1, 3:1 and 1:3 (AUTO.04); only at 1:3 does a tuned threshold beat the fixed 0.5 on IndEgo
   (373 vs 607 cost units per 1,000 windows). We need the real ratio to pick an operating point.
3. **A fresh label batch: round 4, at least 150 windows, new factories if possible.** The 108-window
   round-3 hold-out is consumed and is report-only from now on; a clean confirmation requires new
   labels. Please also request **a second labeller on a subset** so inter-rater agreement can be
   reported — one labeller caps every claim in this memo.
4. **Permission for the paid vision-model arm (AUTO.06 / TASK.02).** Not run: no quota authorisation
   was given, and the brief requires it. The earlier 11-clip agreement (10 of 11) is anecdote.
5. **For TASK: the task list, the end-state that counts as done, whether a 3-minute clip is one task
   or several repeats, whether collectors repeat tasks across clips (episode grouping), and which
   partial states are billable.** Without these, clip-level "task completed" labels are guesswork and
   must stay `UNVERIFIED`.

## 6. What this pass did not do

* Full GRID.04 layout/density sweep — stopped by the pre-registered kill-gate.
* GRID.08 round-3 hold-out scoring — deliberately not spent.
* **The blur rule is left disabled** and all its positives are synthetic; **OVEREXPOSED parity on the
  mosaic is untested** (the injected defect never reaches the p5 > 0.70 rule), and the mosaic's FROZEN
  rule mis-attributes dark windows (verified: the 8 `too_dark` clips fire FROZEN — right verdict,
  wrong rule).
* AUTO.02 (Q4 hybrid-fallback replication), AUTO.05 (per-site calibration), AUTO.06 (paid vision
  resolver), AUTO.08 (final spec) — not run; AUTO.02/05 need artifacts that live only on unmerged
  branches, and AUTO.06 needs the builder's go-ahead.
* TASK.01/02/03/04/05/06 — TASK.01 needs the builder's labelling hours, TASK.02/03 need the paid
  quota, TASK.04 needs Awign's task definition. TASK.00b showed the free IndEgo annotations are **not**
  a substitute: the clip-level truth has zero variance (4/4 "completed") and the slice label agrees
  with a always-advancing baseline at κ = 0.208.
* No A40 numbers, no edge-board measurement, no multi-pod run, no 1080p-source cost row.

## The one honest sentence the product may say

**On 150 IndEgo windows the shipped funnel decides 100 % of windows with balanced accuracy 0.740
[0.631, 0.789], but on 160 labelled factory windows it is 0.556 [0.448, 0.673] with 41 % of bad
windows accepted and 48 % of good windows rejected, so automation today must be tiered with a human
audit sample rather than fully automatic; a 12-frame 4×3 picture scored by our own mosaic-native
features does not change this (0.704 in-domain, 0.356 when transferred to factory footage), the
missing ingredient for factory footage is new signals and fresh multi-labeller labels rather than a
new picture format, and the one change measured here that is worth shipping is a brightness-calibrated
sharpness rule that catches 100 % of σ=8 and 72 % of σ=3 blur at zero marginal clean false-rejects
(0 of 238) and ships disabled until the builder turns it on.**
