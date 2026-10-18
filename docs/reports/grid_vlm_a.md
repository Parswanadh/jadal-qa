# GRID.V (V-A stream) — the mosaic shown to a vision-language model

Stream V-A (`dsh/grid-vlm-a`), forked from `dsh/grid-vlm`. Experiment ids **GRID.V02**
and **GRID.V04**; results `eval/results/exp_vlm/GRID.V02.json` and `GRID.V04.json`.
Frozen pre-registration: `experiments/GRID.VLM/PREREG.json` (prompt sha256
`a59f387e…` for `quality`, window lists, call budgets, decision rules).

**What was run.** One 4×3 contact sheet per window — 12 tiles at 426×240, each labelled
`#N  MM:SS.s` in bold yellow — built by `tools/dsh/vlm_sheet.py` (`vlm-sheet-1.0.0`), and
one frozen `quality` prompt call per sheet to `opencode-go/muse-spark-1.3-contributor`.
No training, no fitting, no threshold selection. **The model is probabilistic**: nothing
here is deterministic and no number below is an accuracy claim.

**Context every number carries:** OOD-160 = 160 human-labelled Egocentric-10K windows,
**138 good / 22 bad, 19 of the 22 bad are `NO_ACTIVE_WORK`, 47 factories, ONE labeller.**

## GRID.V02 — quality verdict on all 160 windows

**161 image calls** (the 160 windows plus one smoke call for `q000` made before the run
began; the smoke reply was then reused, not repeated). Mean **11.95 s** per call, median
11.69 s, **666 s wall** with 3 workers. Parse rate **160/160**. Cap was 480.

| Metric | Value |
|---|---|
| AUROC of `usable`, all 160 | **0.7701** [0.662, 0.878] factory-grouped |
| ... window bootstrap (**optimistic**) | [0.664, 0.877] |
| AUROC of `task_progress == "advancing"` | 0.7701 (identical on these windows) |
| **`NO_ACTIVE_WORK` (19 bad) vs all good**, n=157 | **0.7641** [0.652, 0.887] |
| `HANDS_NOT_VISIBLE` (2 bad) vs good | 0.9746 — **descriptive only, n_bad=2** |
| **Paired vs the frozen shipped head** (0.5878 [0.450, 0.728]) | **+0.1823 [+0.0428, +0.3141], excludes zero** |

At the natural threshold (`usable == true`): **13 of 22 bad rejected** (recall 0.5909),
**7 of 138 good rejected** (false-reject 0.0507), **balanced accuracy 0.7701**
(convention: 0.5 × (good accepted + bad rejected), positive class = good).

**The success bias is real and large: 87.5 % of all answers were `usable = true`.**
A naive `usable == true` gate therefore still accepts **9 of 22** bad windows. The AUROC
is respectable; the raw verdict is not usable as-is.

Confusion, human reason → model `reason_code`:

| human | OK | NO_ACTIVE_WORK | HANDS_NOT_VISIBLE | BLOCKED_VIEW | DUPLICATE_OR_FROZEN |
|---|---|---|---|---|---|
| GOOD (138) | 131 | 3 | 2 | 1 | 1 |
| NO_ACTIVE_WORK (19) | **8** | 8 | 0 | 3 | 0 |
| HANDS_NOT_VISIBLE (2) | 0 | 0 | 2 | 0 | 0 |
| SHAKY (1) | 1 | 0 | 0 | 0 | 0 |

The 8 `NO_ACTIVE_WORK` windows read as `OK` are the whole residual error: the model does
see that no relevant work is happening in 11 of 19, and its `HANDS_NOT_VISIBLE` reading
matches the human label 2 of 2.

## GRID.V04 — run-to-run variance (3 runs, same 40 windows, same PNG bytes)

**80 new calls** (runs 2 and 3; run 1 reused from V02), 326.8 s wall, cap 120. The subset
is the pre-registered random 40 windows (**n_bad = 7**, so each single-run CI is wide and
uninformative on its own). The same cached PNG bytes were fed every run, so the only
variation is the model.

| | value |
|---|---|
| `usable` unanimous across 3 runs | **0.90** (36/40; votes-true histogram 35×3, 2×2, 2×1, 1×0) |
| `usable` Fleiss κ / mean pairwise Cohen κ | **0.5195** / 0.5347 (pairwise 0.286, 0.787, 0.531) |
| `task_progress` unanimous / Fleiss κ | 0.875 / **0.410** |
| AUROC `usable` per run | 0.6429, 0.6991, 0.7143 → **spread 0.0714** |
| Majority vote of 3 | AUROC 0.7143 [0.500, 0.900], share usable 0.925 |

**Verdict: NO — a single call cannot be treated as a measurement.** Agreement is moderate
(κ ≈ 0.52), not high, and the AUROC itself moves by 0.07 between identical runs. Repeated
calls with a majority vote are the minimum honest usage, and even then only as triage.

## Two implementation findings (both fed back to the lead)

1. **The frozen `vlm_mosaic.parse_tile_lines` recovers zero per-tile lines from real
   replies.** It matches only the *requested* `tile N @ t :: text` form; the model instead
   echoes the sheet label, `#N  MM:SS.s: text`. It returned **0** lines on all 160 replies;
   the tolerant parser added in `eval/exp_vlm/vlm_run.py` recovers **1920** (exactly 12 per
   sheet). V01/V03 share the frozen parser and will hit the same thing.
2. **A balanced-accuracy convention bug in my own first run** (`0.5 × (rejected-good +
   rejected-bad)` = 0.3208 instead of `0.5 × (accepted-good + rejected-bad)` = 0.7701) was
   caught and fixed before anything was reported; the convention is now pinned by a unit
   test.

## Honest limits

* The model is **probabilistic**; one labeller, no second-rater kappa. This is a
  **ranker / reviewer**, not an automatic verdict.
* 87.5 % `usable = true`, 9 of 22 bad windows accepted at the natural threshold.
* 19 of the 22 bad windows are one reason (`NO_ACTIVE_WORK`); `HANDS_NOT_VISIBLE` (2) and
  `SHAKY` (1) cannot support any claim.
* The V04 subset is **n_bad = 7** — its per-run CIs are wide, and it is not representative
  of the base rate.
* 108-window round-3 hold-out: **not touched**. No 1080p source, no video input, no L2/L1
  task labels — those belong to V05/V06, which this stream did not run.
* Costs are API calls plus CPU sheet building; **no GPU was used**.

## The one honest sentence

**On 160 labelled factory windows a frozen vision model reading one 12-tile contact sheet
per window ranks usable-vs-unusable at AUROC 0.770 [0.662, 0.878] — significantly better
than the frozen shipped head's 0.588 on the same windows — but it calls 87.5 % of windows
usable, still accepts 9 of 22 bad windows, and moves by κ ≈ 0.52 and ±0.07 AUROC between
identical repeat runs, so it is a triage ranker requiring repeated calls and a human on
disagreement, not an automatic verdict.**

---

# GRID.V06 — the model's role in the funnel (both pre-registered roles)

Prompt, model id and window sets unchanged from the frozen PREREG. The model is
**probabilistic** and GRID.V04 showed a single call is not a measurement (Fleiss κ 0.52,
AUROC spread 0.071 across identical runs), so **every number here is repeat-needed**. One
labeller. Parser used: **the tolerant parser in `eval/exp_vlm/vlm_run.py`** — after the
lead's `beb742e` fix the shared `vlm_mosaic.parse_tile_lines` still requires a timestamp
and recovers only **372 of 1920** lines, because 113 of 160 replies emit `#N: text` with no
timestamp at all.

## Role (a) — veto on high-confidence accepts, OOD-160, **zero new calls**

Reuses the V02 run-1 replies. `P_head` = accept iff p26 ≥ 0.75; `P_head_veto` = accept iff
p26 ≥ 0.75 **and** the model says usable.

| | value |
|---|---|
| Windows the shipped head accepts | **71** (63 good, **8 bad**) |
| Head false accepts vetoed by the model | **4 of 8 (50 %)** |
| Good windows the head accepted that the veto loses | **3** (4.76 % of them) |
| Balanced accuracy: head → head+veto | **0.5464 → 0.6265** |
| False-accept rate: head → head+veto | **0.3636 → 0.1818** |
| False-reject rate: head → head+veto | 0.5435 → 0.5652 |
| **Paired factory-clustered difference** | **+0.08 [0.0000, 0.1591] — does NOT exclude zero** |

The veto halves the head's false accepts on factory footage and roughly doubles the
false-accept improvement into a +0.08 balanced-accuracy gain — but at n_bad = 22 the paired
interval's lower bound sits exactly on zero, so **the gain is not established**. It is a
promising signal, not a result.

## Role (b) — resolver for the review band, IndEgo-150, **115 new calls** (cap 120)

Head scores from `common.loco_proba` on the frozen 26 features (the AUTO.00 recipe); the
forced-0.5 baseline reproduces **exactly** (0.740 here vs 0.740 documented). Band =
0.25 < p < 0.75 → **72 windows**; high-confidence accepts (p ≥ 0.75) → **43**. Both were
called (115 ≤ 120) so the model's two roles are measured in-domain as well.

| Policy (100 % coverage) | balanced acc | false accept | false reject | accepted bad | rejected good |
|---|---|---|---|---|---|
| `P_forced05` — AUTO.00 baseline | **0.740** | 0.26 | 0.26 | 13 | 26 |
| `P_band` — throw the band away | 0.640 | 0.10 | **0.62** | 5 | 62 |
| `P_band_resolve` — model decides the band | **0.755** | 0.38 | 0.11 | 19 | 11 |
| `P_full_veto` — resolve + veto accepts | **0.755** | 0.36 | 0.13 | 18 | 13 |

| Paired clip-clustered difference | Δ balanced accuracy | 95 % CI | established |
|---|---|---|---|
| band_resolve − band_thrown_away | **+0.115** | [0.0361, 0.1794] | **yes** |
| band_resolve − forced05 | +0.015 | [−0.0609, 0.1089] | no |
| full_veto − forced05 | +0.015 | [−0.0448, 0.0876] | no |

**The honest reading:** resolving the review band with the model is **significantly better
than throwing the band away** (+0.115, interval excludes zero) — it recovers most of what
the band costs. But it is **not significantly better than the free forced-0.5 rule**
(+0.015, interval includes zero), so as an accuracy device in-domain it does not earn its
API cost. The success bias reappears: the model called **65 of 72** band windows usable
(90 %) and **105 of 115** targets (91 %), which is why false-accept rises from 0.26 to 0.38
even as false-reject falls from 0.26 to 0.11 — the resolver mostly converts rejects into
accepts rather than separating the classes. The veto on in-domain high-confidence accepts
changed almost nothing (1 of 5 false accepts caught, 2 good lost), leaving BA unchanged at
0.755.

## Call accounting (checked before and after)

| | image calls |
|---|---|
| Ledger before V06 | 241 |
| **V06 calls (all tag `v06`, cap 120)** | **115** |
| Ledger after V06 | 356 |
| Failures / retries in V06 | **0 / 0** |

Role (a) cost nothing: it reuses the V02 replies. The 115 calls took 11.3 s mean / 11.23 s
median each. `JADAL_VLM_LEDGER` accounting is by ledger tag, not by invocation, so a re-run
after a crash reports the **true** expenditure (a first attempt crashed in the final
assembly after the calls were already paid for; the fix re-used them and made zero new
calls).

## The one honest sentence (addendum)

**The model halves the shipped head's false accepts on 160 factory windows (8 → 4) for the
loss of 3 good ones, but the paired factory-clustered balanced-accuracy gain of +0.08
[0.000, 0.159] is not established at n_bad = 22; in-domain it resolves the review band
significantly better than discarding it (+0.115 [0.036, 0.179]) yet no better than the free
forced-0.5 rule (+0.015, interval includes zero), so it is a triage ranker for a human — never
an automatic verdict — and every number here needs repeat runs.**
