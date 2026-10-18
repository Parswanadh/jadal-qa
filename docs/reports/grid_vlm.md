# GRID.V — the multi-frame picture shown to a vision-language model

Track owner: DSH stream `dsh/grid-vlm` (sub-streams V-A `dsh/grid-vlm-a`, V-B `dsh/grid-vlm-b`).
This is **Addendum 2** work and is a different question from PR #174: there, the mosaic was scored by
our own extractors (kill-gate failed) or by an embedding model; here it is shown to a
vision-language model, which is the configuration the customer actually described.

Model: `muse-spark-1.3-contributor` (OpenCode Go Responses endpoint, `input_image` content part).
Prompts and their sha256 are frozen in `experiments/GRID.VLM/PREREG.json`; every call is logged with
the model id, prompt text, raw reply, seconds and PNG sha256. The API key is read in-process from
OpenCode's auth store and is never printed or committed. **The model is probabilistic — nothing here
is deterministic and no call is a guarantee.**

## Sheet design

New builder `tools/dsh/vlm_sheet.py` (`vlm-sheet-1.0.0`). `jadal/mosaic/build.py` was **not** edited:
its 12 px `t=3.5s` label is too small for a model to rely on. The new sheet is 12 tiles in 4x3 at
426x240 px, row-major time order, with a large bold yellow-on-black label `#N  MM:SS.s` in every tile.
For the 30 s task slices the labels carry **clip-absolute** time.

## Results

### GRID.V01 — reading check (adopt) · 20 calls

240 of 240 tiles returned a line (coverage **1.000**) and **240/240 matched our own sidecar on both
the tile number and the timestamp** (exact **1.000**); 0 failed, 0 unparsed, 0 out-of-range. An
independent re-derivation sharing no code with the analysis path gave the identical 240/240.
*Reading only* — a model that reads perfectly can still judge badly, so this supports no quality claim.
The builder's human spot check of the 20 sheets is still outstanding; the PNG sha256s and raw replies
are in the results JSON for it.

### GRID.V02 — quality verdict on OOD-160 (adopt) · 161 calls

OOD-160 = 160 human-labelled Egocentric-10K windows, **138 good / 22 bad, 47 factories, ONE labeller,
19 of the 22 bad are `NO_ACTIVE_WORK`**. Every number carries that context.

| Metric | Value | Interval |
|---|---|---|
| AUROC `usable` | **0.7701** | [0.662, 0.878] factory-grouped |
| `NO_ACTIVE_WORK` (19) vs good, n=157 | **0.7641** | [0.652, 0.887] factory-grouped |
| Paired vs the frozen shipped head (0.5878) | **+0.1823** | **[+0.0428, +0.3141] excludes zero** |
| Balanced accuracy at `usable == true` | 0.7701 | 13/22 bad rejected, 7/138 good rejected |
| **Share of all answers `usable = true`** | **87.5 %** | accepts 9 of 22 bad windows |

This is the first measurement in the whole DSH pass that **beats the frozen shipped head out of
domain with a paired, factory-clustered interval that excludes zero**, and it answers the question the
pixel pipeline failed: the model does separate `NO_ACTIVE_WORK` from good windows (0.764).

But the success bias is real and stated up front: 87.5 % of windows read `usable`, so the naive gate
still accepts 9 of 22 bad windows. Confusion on good windows is small (131 OK, 3 NO_ACTIVE_WORK,
2 HANDS_NOT_VISIBLE, 1 BLOCKED_VIEW, 1 DUPLICATE_OR_FROZEN); `HANDS_NOT_VISIBLE` was caught 2/2.

### GRID.V03 — ablations (inconclusive) · 122 calls

Arm AUROCs on the frozen stratified 20-window set (10 bad / 10 good, 14 factories; the subset is
balanced and therefore **not** representative of the base rate):

| Arm | AUROC | 95 % CI |
|---|---|---|
| one middle frame only | 0.600 | [0.379, 0.800] |
| 24 tiles | 0.650 | [0.470, 0.846] |
| no timestamps | 0.700 | [0.500, 0.896] |
| JSON-only prompt (reference) | 0.700 | [0.500, 0.896] |
| 6 tiles | 0.750 | [0.542, 0.944] |
| shuffled tile order | 0.800 | [0.586, 1.000] |

**Every paired difference against the reference includes zero** (largest +0.100 [0.000, +0.278]), and
read against V04's 0.0714 run-to-run AUROC spread **no difference in this table is interpretable**.
So the headline question — *does a 12-frame sheet beat a single frame?* — is **not answered** here
(0.700 vs 0.600, paired −0.100 [−0.250, +0.111], and the single-frame arm also used a different
prompt). Two incidental findings: arms "no timestamps" and "JSON-only" returned **identical verdicts
on 20/20 windows** on genuinely different sheets (0/20 shared PNG sha256), so on this subset neither
the timestamps nor the reasoning step changed any verdict.

**Negative control — PASS.** A flat grey sheet read `usable=false`, `HANDS_NOT_VISIBLE`,
`cannot_tell`, and under the reading prompt it refused to invent times (every timestamp written
`--:--.-`). The model is reading the image, so the track is not invalidated.

### GRID.V04 — run-to-run variance (maybe) · 80 calls

Three runs on the same 40 frozen windows with **byte-identical sheets**:

| Quantity | Value |
|---|---|
| `usable` unanimous share | **0.90** (36/40) |
| Fleiss kappa, `usable` | **0.5195** (mean pairwise Cohen 0.535) |
| `task_progress` unanimous / kappa | 0.875 / 0.410 |
| per-run AUROC | 0.6429 / 0.6991 / 0.7143 → **spread 0.0714** |
| majority vote of 3 | 0.7143 [0.500, 0.900] |

**A single call cannot be treated as a measurement.** This triggers the pre-registered fallback: every
VLM number in this track is a **triage ranker / second reader**, not an accuracy claim, until it is
repeated and voted.

### GRID.V05 — task level on IndEgo (drop) · 24 image + 4 text + 24 video calls

4 annotated IndEgo clips, 6 evenly spaced 30 s slices each (24 slices, **2 of them
`no_relevant_work`** — the free truth is 22:2, so the always-advancing baseline is 0.9167).

| Arm | L1 agreement | vs baseline | kappa | gap slices caught |
|---|---|---|---|---|
| 12-tile sheet | 0.7083 | 0.9167 | **−0.1351** | **0 / 2** |
| video (same slices, CRF 28) | 0.7917 | 0.9167 | — | **0 / 2** |

Both forms are **worse than always saying "advancing"**, and neither caught either true gap. Reading
is not the problem (V01: 240/240 exact), and the sheet is not a cheaper substitute: sheet
**10.04 s/call**, video **9.63 s/call** at a median 1.06 MB.

**L2 (was the task completed?)** used the **builder's working definition** (Addendum 1: pick an object
from one place, put it in another, accurately — right object, right destination, no drop and no redo;
judge the visible start and end state). The free IndEgo truth is **4/4 "completed" — zero variance**,
so **no L2 accuracy can be computed** against it. The model answered `not_completed` once and
`cannot_tell` three times, disagreeing with the annotations on all four clips, with coherent reasons
("no final placed object shown", "no clear pick and place start end states"). Reported as
**the builder's definition, not Awign's**.

### GRID.V06 — role in the funnel (maybe) · 115 calls

**(a) Veto on high-confidence accepts — OOD-160, zero new calls.** The shipped head accepts 71
windows: 63 good, **8 bad**. The model vetoes **4 of those 8 (50 %)** at a cost of 3 good windows
(4.76 %). Balanced accuracy **0.5464 → 0.6265**; false-accept **0.3636 → 0.1818**. The paired
factory-clustered difference is **+0.08 [0.0000, 0.1591] — it does not exclude zero.** Promising,
not established at n_bad=22.

**(b) Resolver for the review band — IndEgo-150.** Forced-0.5 reproduces AUTO.00 exactly (0.740).

| Policy | Balanced acc | false-accept | false-reject |
|---|---|---|---|
| forced 0.5 (AUTO.00) | 0.740 | 0.26 | 0.26 |
| band discarded (all review → reject) | 0.640 | 0.10 | 0.62 |
| band resolved by the model | **0.755** | 0.38 | 0.11 |
| resolve + veto accepts | 0.755 | 0.36 | 0.13 |

Paired clip-clustered: band-resolve − band-discarded **+0.115 [+0.0361, +0.1794], established**;
band-resolve − forced-0.5 **+0.015 [−0.0609, +0.1089], NOT established**. The model significantly
beats *throwing the band away* but **not the free forced-0.5 rule**. The success bias reappears
in-domain (65/72 band windows and 105/115 targets read `usable`), so it mostly converts rejects into
accepts — false-accept 0.26 → 0.38 while false-reject 0.26 → 0.11.

## Call accounting (caps respected)

| Stream | image | text | video |
|---|---|---|---|
| V-A (V02 161 + V04 80 + V06 115) | 356 | 0 | 0 |
| V-B (V01 20 + V03 122) | 142 | 0 | 0 |
| V05 (24 sheets + 4 L2) | 24 | 4 | 24 |
| **total** | **522** | **4** | **24** |

Caps: 880 image, 40 text. No cap was breached; both sub-streams self-stopped once on an accounting
bug and resumed without re-spending calls. One pre-run smoke call (q000) is included in the count.

## What was NOT done

* **The builder's human spot check** of the V01 sheets — evidence is in the JSON, the check is his.
* **Repeated runs per ablation arm** — so the V03 table is uninterpretable, as stated.
* **The single-frame vs 12-frame question** — not answered; it needs a clean, prompt-matched,
  repeated design on more bad windows.
* **Any use of the 108-window hold-out** — untouched, as required.
* **No `NO_ACTIVE_WORK`-specific ablation** — the reason is the one that matters and it deserves its
  own stratum with more windows than 19.
* TASK.02 as originally planned (video-only pilot) is **replaced** by this track for the sheet-vs-video
  question, per Addendum 2.

## The one honest sentence the product may say

**A 12-frame 4x3 sheet of factory windows shown to Muse Spark 1.3 is read perfectly (240/240 tiles and
timestamps exact) and separates good from bad factory windows at AUROC 0.770 [0.662, 0.878] with
n_bad=22, beating the frozen shipped head by +0.182 [+0.043, +0.314] and catching the
`NO_ACTIVE_WORK` windows the pixel pipeline misses (0.764 [0.652, 0.887]) — but 87.5 % of its answers
are "usable", a single run is not a measurement (Fleiss kappa 0.52, AUROC spread 0.071), and as an
accuracy device it does not beat the free forced-0.5 rule in-domain (+0.015, not established), so it
belongs in the funnel as a second reader that flags accepts for human audit, not as a resolver that
replaces the review band.**
