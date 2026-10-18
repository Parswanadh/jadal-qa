# GRID.V (sub-stream V-B) — reading check and sheet ablations

Owner: sub-stream V-B (`dsh/grid-vlm-b`), for the GRID.V track (mosaic shown to a vision-language model).
Interpreter `/home/parshu/envs/jadal/bin/python`. Model `opencode-go/muse-spark-1.3-contributor`
(`muse-spark-1.3-contributor`), image input, key read in-process from the OpenCode auth store and
never logged. My ledger: `/tmp/opencode/dsh/vlm_ledger_b.jsonl` (mine alone).
Frozen contract: `experiments/GRID.VLM/PREREG.json`. Foundation used unchanged:
`tools/dsh/vlm_sheet.py`, `tools/dsh/vlm_mosaic.py`, `eval/exp_vlm/vlm_common.py`.

**My scope was exactly two experiments: GRID.V01 (reading check) and GRID.V03 (ablations).**
V02, V04, V05, V06 belong to the other streams; V-A's numbers appear here only as context and are
marked as such.

## 0. The negative control comes first (it decides whether anything else means anything)

Two contentless flat-grey sheets (no content, no labels) were sent, one through `quality` and one
through `read_tiles`. **Both passed**, which is the evidence that the model is actually reading the image:

| Control | Result |
|---|---|
| grey sheet through `quality` | `usable=false`, `reason_code=HANDS_NOT_VISIBLE`, `task_progress=cannot_tell`, explanation *"All twelve tiles are blank gray, no hands or work visible."* |
| grey sheet through `read_tiles` | 12 lines, every one *"uniform gray blank tile"*, and every timestamp written as `--:--.-` — **the model refused to invent a time** rather than hallucinating one |

So the object it is judging is the picture, not the prompt's expectation. Had either read as
`usable=true`, the whole track would be invalid; it did not.

## 1. GRID.V01 — reading check (20 windows, 20 calls)

One `read_tiles` call on the default 12-tile 4×3 426×240 labelled sheet, for each of the 20
pre-registered V01 windows. Ground truth is **our own builder's sidecar**, not the model.

| Metric | Value |
|---|---|
| windows | 20/20 called OK, 0 failed, 0 unparsed |
| sheet tiles | 240 |
| returned lines | 240 |
| **tile coverage rate** | **1.000** (240/240) |
| **exact match, tile number AND timestamp** | **1.000** per sheet tile and per returned line (240/240) |
| right tile number, wrong timestamp | 0 |
| out-of-range tile numbers | 0 |
| mean seconds per call | ~12 s |

Independently re-derived by `eval/exp_vlm/verify_v01.py`, which shares no code with the analysis
path: its own regex over the raw ledger replies, its own re-read of the sidecars, its own comparison
— **same numbers: 240/240, coverage 1.000, exact 1.000** (`eval/results/exp_vlm/GRID.V01_verify.json`).

**Reply-shape instability (worth knowing before anyone trusts a parser).** V-A's 160 V02 replies came
back echoing the sheet's own label shape, and the original strict `parse_tile_lines` recovered **0
lines** from them. The shared parser has since been fixed twice; the current version recovers
**1920/1920** tile lines from V-A's 160 real replies (160/160 sheets with >= 12 lines) and explicitly
rejects the trailing JSON verdict line so it can never be counted as a tile.

My run drew a **different shape from the same model and the same prompt**: of my 240 V01 reply lines,
**240 matched the asked-for `tile N @ MM:SS.s :: text`** and **0** matched the label-echo or the
hash-only shape — the inverse of V-A's distribution. That is why my strict parse was perfect, and it
is itself evidence that *the reply format is not stable across runs* for a probabilistic model, so a
parser must never be trusted on the strength of one run's shape. I re-parsed my 20 existing replies
with the original strict parser, the first tolerant parser and the **current** parser: all three give
**240/240, coverage 1.000, exact 1.000**. The count is at ceiling, so no parser can improve it, and the
number is parser-independent. No calls were re-spent for this.

**What V01 does NOT say.** It is a reading measurement. A model that reads every tile perfectly can
still judge badly, so nothing here is a quality claim. One call per window measures no run-to-run
variance (that is V04). The builder's human spot check of the sheets is **OUTSTANDING** — the results
JSON carries every sheet PNG sha256 and every raw reply for exactly that purpose; PNGs stay under
`/tmp/opencode/dsh/sheets` and are never committed.

## 2. GRID.V03 — ablations (20 windows, 6 arms, 122 calls)

The pre-registered stratified subset: **10 bad / 10 good, all 10 bad are `NO_ACTIVE_WORK`, 14
factories. The subset is balanced and therefore NOT representative of the base rate.** Same 20 windows
in every arm; factory-grouped bootstrap CI (2000 draws, seed 0).

**Reference arm: `e_json_only_prompt`** — the default 12-tile labelled sheet with only the reasoning
step removed, the closest in-budget analogue of V02's primary arm. The true primary (`quality` on 12
tiles) is V02's job and was deliberately **not** re-bought here.

`auroc_usable_binary` is a **binary** score, so it is exactly balanced accuracy at 0.5; the 3-level
`auroc_progress_graded` (advancing / cannot_tell / no_relevant_work) is the more informative number.

| arm | what it changes | AUROC (usable) | 95 % CI (factory) | AUROC (progress) | 95 % CI | usable=true share | recall bad | false-reject good |
|---|---|---|---|---|---|---|---|---|
| a_no_timestamps | timestamps not drawn | 0.700 | [0.500, 0.896] | 0.700 | [0.500, 0.896] | 0.70 | 0.50 | 0.10 |
| b_shuffled_order | tiles permuted | 0.800 | [0.586, 1.000] | 0.795 | [0.578, 1.000] | 0.60 | 0.70 | 0.10 |
| **c_single_middle_frame** | **ONE middle frame** | **0.600** | **[0.379, 0.800]** | 0.610 | [0.400, 0.800] | 0.80 | 0.30 | 0.10 |
| d6_tiles | 6 tiles | 0.750 | [0.542, 0.944] | 0.745 | [0.529, 0.944] | 0.65 | 0.60 | 0.10 |
| d24_tiles | 24 tiles | 0.650 | [0.470, 0.846] | 0.650 | [0.470, 0.846] | 0.75 | 0.40 | 0.10 |
| **e_json_only_prompt (REF)** | reasoning step removed | **0.700** | **[0.500, 0.896]** | 0.700 | [0.500, 0.896] | 0.70 | 0.50 | 0.10 |

Paired differences against the reference (same windows, factory-clustered, 2000 draws):

| arm vs (e) | Δ AUROC (usable) | 95 % CI | excludes zero? |
|---|---|---|---|
| a_no_timestamps | **0.000** | [0.000, 0.000] | no |
| b_shuffled_order | +0.100 | [0.000, +0.278] | no |
| c_single_middle_frame | −0.100 | [−0.250, +0.111] | no |
| d6_tiles | +0.050 | [0.000, +0.182] | no |
| d24_tiles | −0.050 | [−0.167, 0.000] | no |

**No arm differs from the reference with an interval that excludes zero.** With 10 bad windows that
is the expected outcome, not a failure.

### 2.1 Does the 12-frame sheet beat a single frame?

**Not established, and the comparison is confounded.** The 12-tile reference arm scores 0.700
[0.500, 0.896]; the single middle frame scores 0.600 [0.379, 0.800]; the paired difference is
**−0.100 [−0.250, +0.111]**, which includes zero. Worse, arm (c) uses the `quality` prompt while the
reference arm (e) uses `quality_json_only`, so a c-vs-e reading mixes "one frame vs twelve" with
"reasoning step vs none". That confound is stated with the number rather than hidden. Note also that
the single frame is judged `usable` more often (0.80) than any 12-tile arm — its failure mode is
agreeing with everything, not seeing less. **The honest statement is: on 20 windows and one call per
cell, this experiment cannot tell whether twelve frames beat one.**

### 2.2 The unexpected result: timestamps and the reasoning step change nothing at all

Arms (a) and (e) returned the **same verdict on 20/20 windows** — identical `usable` *and* identical
`task_progress` — despite genuinely different sheets (0/20 shared PNG sha256, verified) and different
prompts. Their paired difference is exactly 0.000. Given the V04 noise floor below, 20/20 identical is
well beyond chance. Within this subset the verdict does not depend on the burned timestamps, nor on
whether the model is asked to show its reasoning first. It is one call per cell on 20 windows, so this
is a strong hint, not a measured law.

### 2.3 Read against the noise floor (this is the frame for the whole table)

V-A's V04 gave three runs on the same 40 sheets: unanimous share 0.90, Fleiss κ 0.52, **per-run AUROC
spread 0.0714**. A single call is therefore not a measurement. Every arm-to-arm difference above is
≤ 0.10 and every paired interval includes zero, so **no arm-to-arm difference in this table is
interpretable**; the table shows what one call per cell produces, not what the sheet variants do. Any
future sheet comparison needs repeated runs per cell (V04-style) before its differences can be read.

For context (V-A's results, not mine): V02 on all 160 OOD windows scored `usable` **AUROC 0.7701
[0.662, 0.878]** factory-grouped, **0.7641 [0.652, 0.887]** on the 19 `NO_ACTIVE_WORK` windows, and
**+0.1823 [+0.0428, +0.3141]** paired against the frozen shipped head. So the model does rank these
windows; V03's job was to ask *which part of the sheet* does the work, and on this budget the answer
is: not measurably any of them.

## 3. Call accounting, and one honest incident

| Experiment | calls | cap |
|---|---|---|
| GRID.V01 | 20 | 20 in the PREREG (the task brief said "cap 160") |
| GRID.V03 | 122 (120 arms + 2 negative controls) | 122 |
| sub-stream total | **142** | — |

No cap was breached. **One incident, reported because it is the kind of thing that gets hidden:** the
first V03 run stopped itself at 122 *total ledger* calls with `BUDGET STOP` — my guard had compared
the shared ledger (including V01's 20 calls) against V03's cap of 122, so it refused 20 calls early.
That is a bug in my accounting, not an overspend: V03 had spent 102 of its 122 at the time. I fixed
the guard to count **per experiment by arm** (plus an explicit 142 sub-stream ceiling), added a
regression test for exactly this case, and resumed; the ledger-based resume meant the two arm-(e)
calls already made were not re-spent. V03 finished at exactly 122.

The parser fix was merged from `dsh/grid-vlm` and changed no V01 number (§1). V03's scores come from
`parse_quality` (JSON extraction), **not** from `parse_tile_lines`, so the parser bug could not have
touched the ablation table; the per-arm `n_unparsed` column is the analogue of a recovery rate and is
0 for all six arms.

## 4. What I did not do

* V02, V04, V05, V06 — other streams. No text-only (`l2_text`, `slice_l1`) calls were made by me.
* The builder's human spot check of the 20 V01 sheets — evidence provided, check outstanding.
* Repeated runs per arm (the only way to read the V03 table) — V04's three-run design covers 40
  windows for the primary arm only; repeating all six arms was outside my call budget.
* No 30-second slices, no clip-level completion, no real defocus footage, no second labeller. One
  labeller throughout; these are ranker/reviewer results, not accuracy claims.

## The one honest sentence this sub-stream may say

**The model reads our contact sheet perfectly — 240 of 240 tiles with the right number and timestamp
across 20 windows, and it correctly refuses a contentless sheet rather than hallucinating one — but on
20 stratified windows with a single call per cell it cannot be shown that twelve frames beat one
(0.700 vs 0.600, paired −0.100 [−0.250, +0.111], confounded with the prompt), nor that timestamps or
the reasoning step change any verdict (identical on 20/20 windows), and because the model's own
run-to-run AUROC spread is 0.0714 no difference in the ablation table is interpretable without repeated
runs.**
