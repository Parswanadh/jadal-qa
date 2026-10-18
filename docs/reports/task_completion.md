# TASK — task progress (L1) and task completion (L2): what TASK.00 measured

Stream S5 (`dsh/task`), DSH GRID/AUTO/TASK pass, 2026-10-18. CPU only, no paid API calls.
Everything here is **descriptive agreement between human/dataset labels and frozen scores**, not
accuracy on unlabelled video. Both measurements are zero-label and zero-training: no model was
fitted for a headline number.

**Registry:** `TASK.00` (verdict `inconclusive`), `TASK.00b` (verdict `drop`)
**Results JSON:** `eval/results/task/TASK.00.json`, `eval/results/task/TASK.00b.json`
**Code:** `tools/task/task00.py`, `tools/task/task00b.py`; tests `tests/test_task00_zero_label.py` (10 tests, <1 s, the band regression test was shown to fail when the band is moved)
**Interpreter:** `/home/parshu/envs/jadal/bin/python` (3.12.15); `JADAL_BUNDLE=/home/parshu/projects/jadal-qa-wt/handover/jadal_pod_bundle/pod_bundle`

---

## 1. The shipped band, quoted from the source (checked, not assumed)

The brief defines the shipped verdict as `accept iff p >= 0.75`, `reject iff p <= 0.25`, else
`review`. `jadal/poc/cascade.py` says:

```python
THRESHOLD = 0.6
MARGIN = 0.25

def decide_window(s0, feats, clf, scaler):
    """Returns (verdict, stage, reasons, score_or_None)."""
    rule, ev = s0_decide(s0)
    if rule is not None:
        return ("reject", "S0", [{"code": rule, "evidence": ev}], None)
    p = s1_score(feats, clf, scaler)
    if abs(p - 0.5) < MARGIN:
        return ("review", "S2",
                [{"code": "UNCERTAIN", ...}], round(p, 4))
    if p >= THRESHOLD:
        return ("accept", "S1", [{"code": "S1_USABLE", ...}], round(p, 4))
    return ("reject", "S1", [{"code": "S1_UNUSABLE", ...}], round(p, 4))
```

`THRESHOLD = 0.6` is **inert**: the `p >= THRESHOLD` branch is only reached when
`|p - 0.5| >= MARGIN = 0.25`, i.e. `p <= 0.25` or `p >= 0.75`. So the brief's reading is right and
the code's `0.6` never decides anything. One thing the brief's sentence omits: `decide_window`
applies the S0 exact rules **before** the score, and an S0 hit rejects with `score = None`.
On the 160 windows used here no S0 rule fired (all 160 carry a numeric score and stage `S1`/`S2`),
so the band alone reproduces the file.

**Band check [MEASURED]:** re-deriving the verdict from the frozen `old26` score reproduces the
verdict recorded in `ood160_heads_old26_vs_new22.json` on **160 / 160** windows, 0 mismatches
(`TASK.00.json → band_check`). A regression test asserts this and was shown to go red when the band
was deliberately widened by 0.01.

Source of the verdicts: `interview/dsh_pack/ood160_heads_old26_vs_new22.json`, key `old26`
(the original portal). `new22` (the hardened head) is **not** used anywhere in this report.

---

## 2. TASK.00(a) — do `NO_ACTIVE_WORK` labels predict the pipeline's verdicts?

Data: 160 human-labelled Egocentric-10K 10-s windows, 138 good / 22 bad, **one** human labeller,
47 factories. Bad reasons: `NO_ACTIVE_WORK` 19, `HANDS_NOT_VISIBLE` 2, `SHAKY` 1. Frozen `old26`
scores for all 160 (0 null scores). Factory is the grouping unit for every interval.

### 2.1 Verdict shares [MEASURED] (`verdict_shares`)

| Group | n | accept | review | reject | reject + review |
|---|---|---|---|---|---|
| `NO_ACTIVE_WORK` | 19 | 7 (36.84 %) | 2 (10.53 %) | 10 (52.63 %) | 12 (**63.16 %**) |
| all other windows | 141 | 64 (45.39 %) | 20 (14.18 %) | 57 (40.43 %) | 77 (**54.61 %**) |
| gold good only | 138 | 64 (46.38 %) | 20 (14.49 %) | 54 (39.13 %) | 74 (53.62 %) |
| all 160 | 160 | 71 (44.38 %) | 22 (13.75 %) | 67 (41.88 %) | 89 (55.63 %) |

The honest reading: `NO_ACTIVE_WORK` windows are reject+review **63.2 %** of the time against
**54.6 %** for everything else — the direction the brief expects, but **7 of 19 (36.8 %)
`NO_ACTIVE_WORK` windows are ACCEPTED** by the shipped head. The label is therefore not a usable
stand-in for the verdict: for every three windows a human marked "not doing anything relevant",
one is accepted.

Breakdown by bad reason (all counts, no rate claim on n=1 or 2): `NO_ACTIVE_WORK` 7/2/10,
`HANDS_NOT_VISIBLE` 0/0/2, `SHAKY` 1/0/0 (accept/review/reject).

Score medians (`score_separation`): good 0.6259, `NO_ACTIVE_WORK` 0.1467, other bad 0.0001.
The reason labels do carry information the score partly reflects — the medians are far apart — but
the overlap is large enough that the per-window verdict is not implied.

### 2.2 Does the per-factory NO_ACTIVE_WORK share predict the per-factory reject+review share? [MEASURED]

47 factories, shares computed per factory (`per_factory.rows` in the results JSON):

| Statistic | Value |
|---|---|
| Spearman rho | **0.1217** |
| Pearson r | **0.2040** |
| Permutation p, two-sided, 20 000 draws, seed 0 | **0.1708** |
| n (factories) | 47 |
| Factories with any `NO_ACTIVE_WORK` | **11** |
| Factories with >= 5 windows | 6 |

**This is a weak test and should be read as one.** 41 of 47 factories have <= 5 windows; the
"share" is a ratio of small counts (0, 1/2, 1/3, 2/3, ...); only 11 factories have a non-zero
`NO_ACTIVE_WORK` share at all, so the correlation is effectively estimated from 11 informative
points and 36 stacked zeros. The point estimates lean positive, and nothing here establishes the
association; a factory-level regression on these counts would be fitting noise.

### 2.3 AUROC of the frozen score, `NO_ACTIVE_WORK` vs good [MEASURED]

Headline: **AUROC 0.5755, 95 % CI [0.4275, 0.7386]**, factory-grouped bootstrap (2000 draws,
seed 0, all 2000 valid), n = 157, n_good = 138, n_bad = 19, 47 factories. Window-bootstrap
(optimistic) CI [0.4413, 0.7077] is reported in the JSON but is **never** used here to call
anything significant. Flipped direction (`NO_ACTIVE_WORK` as the positive class): 0.4245
[0.2614, 0.5725].

For context on the same head: good vs all 22 bad = 0.5878 [0.4503, 0.7284] (factory-grouped).
Both intervals include 0.5. Note this is the same head family whose in-domain LOCO AUROC on
IndEgo is 0.810 (`GRID.00`); the drop out of domain is the known domain-shift result, not a new
finding, and the interval here is wide enough that it is not evidence of a specific value.

**Answer to the question as asked:** on this set the reason label and the shipped verdict are
weakly and not decisively associated (63.2 % vs 54.6 % reject+review; AUROC 0.576 with a CI that
includes 0.5; per-factory rho 0.12, p 0.17). The label cannot be substituted for the head, and the
head is not merely reproducing the label — 7 `NO_ACTIVE_WORK` windows are accepted. Registry
verdict: **inconclusive**.

---

## 3. TASK.00(b) — are the IndEgo VIA step annotations usable as free L1 / L2 truth?

### 3.1 What the annotations actually contain [MEASURED] (`segments_engine`)

Format inspected, not assumed. They are **VGG Image Annotator (VIA)** documents:

* `attribute["1"].aname` — the attribute name; observed values `"steps"` and `"TEMPORAL-SEGMENTS"`.
* `attribute["1"].options` — map `option-id -> step label`, e.g.
  `{"0": "00_preparation", "1": "01_inspect_and_clean", ..., "14": "14_turn_on"}`. This is the
  **step vocabulary available for that clip**, not the list of steps actually performed.
* `file["1"].fname` — the annotated video, e.g. `User_15_410_0411_2_480.mp4`.
* `metadata[*].z = [start_s, end_s]` — one temporal segment. `metadata[*].av["1"]` — the chosen
  option-id. `xy` is empty in every file (no spatial regions).
* `project.pid/rev` are the literal placeholders `__VIA_PROJECT_ID__` etc. (unfilled template),
  so there is no project-level revision to cite.

`z` is in **seconds**, verified rather than assumed: `cv2` reports 10.000 fps and durations of
698.30 / 208.40 / 416.10 / 280.40 s for `ecu_full` / `ecu_late` / `desk` / `organise`, matching the
last gold window end for each clip to 0.01 s (`duration_check_cv2` in each `per_clip` block).

Inventory: **180** `AS_*.json` files on disk (179 under `1_Assembly/annotated_keysteps_assembly_disassembly`,
1 under `3_Logistics_Organization/annotated_keysteps_logistics_organisation`), **180 parsed, 0
unparseable**. The `annotated_keysteps.txt` files carry the per-task step lists
(`Assembly_ECU` 15 steps, `Assembly_Table ...` 10, `Organize` 3, ...) used as the task list for L2.

Coverage of the 7 gold clips: **4 have an annotation** — `ecu_full`, `ecu_late`, `desk` (assembly)
and `organise` (logistics). **3 have none** — `pallet`, `pallet2`, `logistics4`. Note the brief's
phrasing "the 3 logistics clips have no L2 truth" is off by one clip in the other direction:
`organise` *is* a logistics clip and *does* have an annotation; the three without one are all
logistics clips. 110 gold windows live on the 4 annotated clips (78 good / 32 bad).

### 3.2 L2 (per clip): `completed` iff the annotation reaches the last step of its task list [MEASURED]

| Clip | matched task list (`keysteps_*.txt`) | steps in list | steps annotated | last step of list | present? | L2 |
|---|---|---|---|---|---|---|
| `ecu_full` | `Assembly_ECU` (15) | 15 | 14 | `14_turn_on` | yes | **completed** |
| `ecu_late` | `Assembly_ECU` (15) | 15 | 5 | `14_turn_on` | yes | **completed** |
| `desk` | `Assembly_Table (without top: leave out 08)` (10) | 10 | 3 | `09_clean_up` | yes | **completed** |
| `organise` | `Organize` (3) | 3 | 2 | `02_clean_up` | yes | **completed** |

**L2 is degenerate on this subset: 4 of 4 clips are `completed`, so the flag has zero variance and
carries no discriminative information here.** Two caveats make "completed" weaker than it sounds:
(i) the rule is "reaches the *last* step", not "performed every step" — the annotations omit
task-list steps (`desk` 7 of 10 missing: `00_preparation` through `06_attach_crank`; `ecu_late` 10
of 15 missing, including `10_attach_wooden_plate` between two annotated steps; `ecu_full` 1 of 15;
`organise` 1 of 3 — every clip omits `00_preparation`); (ii) the annotations look like they cover a
**sub-segment** of the task, not the whole task: `desk` is a 416 s video whose first annotated step
is `07_adjust_height` at 4.065 s. So L2 here means "the recording ends after the last step", which
is a property of how the clips were cut as much as of the work.

### 3.3 L1 (per 10-s slice): `advancing` inside a step, `no_relevant_work` in a gap [MEASURED]

Primary rule: a gold window is `advancing` if annotated step intervals cover **more than half** of
it. Result: **105 `advancing` / 5 `no_relevant_work`** of 110. The annotated steps are nearly
contiguous — measured gaps total about 5 % of annotated time (`gaps_s` per clip: e.g. `ecu_full`
0-8.79, 36.36-42.72, 274.82-280.22, 680.52-698.30 s) — and the gold windows are sampled sparsely
(110 windows over 1603 s), so only **5** windows land in a gap. 96 windows are fully inside a step,
12 are partial, 2 have zero overlap.

**Cross-tabulation against the human gold window labels** (`agreement_with_human_gold`):

| | gold good | gold bad | total |
|---|---|---|---|
| derived `advancing` | 78 | 27 | 105 |
| derived `no_relevant_work` | 0 | 5 | 5 |
| total | 78 | 32 | 110 |

* Agreement rate **0.7545** — against an **always-`advancing` baseline of 0.7091** (78/110). The
  gain over the constant predictor is 5 windows.
* Cohen's kappa **0.208**; balanced accuracy 0.5781; sensitivity on gold-good 1.000,
  specificity on gold-bad 0.1562. All comparison intervals for these rates are not reported
  because a 5-window cell cannot support one honestly.
* Restricted to gold reasons that are about work activity (`""` or `NO_ACTIVE_WORK`, n = 94):
  agreement 0.8723, kappa 0.3562, balanced accuracy 0.6250.
* Rule sensitivity (same windows, different gap definition): `any_overlap` kappa 0.0864,
  `midpoint_in_step` kappa 0.1886, `fully_inside_step` kappa 0.0991. The label is fragile to the
  definition, which is expected when only 2-17 windows change side.

**The 5 gap windows are all gold-bad (precision 5/5) but recover only 5 of 32 gold-bad windows
(15.6 %).** The remaining 27 gold-bad windows are inside annotated steps and are bad for reasons the
step annotation cannot see: `DUPLICATE` 5, `HANDS_NOT_VISIBLE` 6, `TOO_DARK` 4, `UNDECODABLE` 1,
plus `NO_ACTIVE_WORK` 12 that the annotation places inside a step.

**Direct source-vs-source disagreement:** of the **16** windows on these 4 clips that the human
labelled `NO_ACTIVE_WORK`, the annotation says `advancing` for **12** and `gap` for **4**
(`source_vs_source_disagreement`; ids `g074, g090, g000, g047, g061` are the gaps). Two label
sources that use different definitions of "no work" disagree on three quarters of the human
`NO_ACTIVE_WORK` windows. This is the most useful thing TASK.00(b) produced, and it is a caution
about the labels, not a result about the pipeline.

### 3.4 Does the shipped 26-feature score point the right way? [MEASURED] (`score_direction`)

Primary score: the harness LOCO out-of-fold probability (`common.loco_proba`: StandardScaler +
LogisticRegression C=0.3, `class_weight="balanced"`, leave-one-clip-out over the 7 clips — the same
function that reproduces the pod's 0.810 in `GRID.00`).

| Score | AUROC (advancing vs `no_relevant_work`) | 95 % CI, clip-grouped bootstrap (2000 draws, seed 0) | n / n_neg |
|---|---|---|---|
| harness LOCO, C=0.3 balanced | **0.6476** | **[0.4565, 0.8305]** | 110 / 5 |
| portal-head LOCO, C=1.0, max_iter 2000 | 0.6248 | [0.4783, 0.8089] | 110 / 5 |
| portal-head full fit (not out-of-fold, optimistic) | 0.7048 | [0.5625, 0.8475] | 110 / 5 |

Higher score does mean "more likely `advancing`" in direction, and this is the sign the pipeline is
supposed to have. But the interval includes 0.5 and there are **5 negative windows in 4 clips**, so
this is not an established direction. Per-clip AUROC of the primary score: `desk` 0.4815 (n = 29),
`ecu_full` 0.8587 (n = 48), `ecu_late` 0.6923 (n = 14), `organise` **undefined** (all 19 windows
are `advancing`, one class only). For reference on the same 110 windows, the gold-label AUROC of
the same score is 0.7759 — i.e. the derived L1 label is a noisier target than the human label, as
expected since it agrees with the human label only 75 % of the time.

**Answer to the question as asked:** the annotations parse and are free, but in this form they are
not usable as L1/L2 ground truth — the gap class is 5 windows, the L2 flag has one value, and the
two sources disagree on 12 of 16 human `NO_ACTIVE_WORK` windows. Registry verdict: **drop** (as a
substitute for human labels; the per-step segmentation stays a candidate auxiliary signal if a
denser gap definition or more annotated clips appear).

---

## 4. What was NOT done, and why

* **No training, no new inference, no API calls.** TASK.00 is a join over existing labels and
  cached scores; TASK.00b is a parse plus numpy/sklearn on the cached 150x26 feature matrix.
  Cost: 0 GPU, < 15 s CPU total.
* **TASK.01 (pilot annotation) not started** — it needs the builder to label.
* **TASK.02 / TASK.03 not started** — they need the paid vision-model quota and the builder's
  go-ahead. No Muse call was made.
* **TASK.04 not started** — it needs Awign's task definition.
* **No 30-s slice labels.** The brief's L1 design is a 30-second slice with three labels
  (`advancing | no_relevant_work | cannot_tell`); TASK.00b could only use the 10-second windows
  that already have both a gold label and cached features, so the derived L1 here is a 10-s label
  with two values. `cannot_tell` was never needed or produced.
* **No inter-rater evidence.** All human labels used here come from ONE labeller; the dataset
  annotations come from whoever produced the IndEgo release. No kappa between two humans exists,
  so nothing here bounds label noise.
* **No factory-work claim.** `NO_ACTIVE_WORK` is measured on Egocentric-10K; the step annotations
  are IndEgo demo-style assembly and logistics recorded for a research dataset, one worker and one
  task per clip. Neither transfers to factory work without testing it there.
* **`new22` (the hardened head) was not evaluated.** Out of scope here; the shipped `old26` was
  used because the brief asked for it.
* **The per-factory correlation is not a usable predictor**, and is reported only because the brief
  asked for it. 47 points, 11 informative, p = 0.17.

## 5. Open questions for Awign

1. **The task list for these clips.** We hold step lists for the IndEgo dataset tasks
   (`Assembly_ECU`, `Assembly_Table`, `Organize`, ...) but nothing for Egocentric-10K factory
   footage. What is the step list for the work in factory clips?
2. **What end state counts as done, per task?** The builder's draft is "picks an object from one
   place, puts it in another place, and does so accurately". Does Awign agree, and what counts as
   inaccurate (wrong object, wrong destination, a drop, a redo)?
3. **Is a 3-minute clip a whole task, part of one, or several repeats?** Our L2 rule ("reaches the
   last step") says `completed` for all 4 IndEgo clips we could test, including one that skips 10
   of 15 listed steps — does Awign's definition agree that this counts as completed?
4. **Do collectors repeat the same task across clips?** If yes, we need episode grouping before any
   cross-validation, or windows from one episode will leak across folds.
5. **Which partial states are billable** (farthest-step credit)? This decides whether L2 needs a
   "how far did it get" number rather than a binary.
6. **What does `NO_ACTIVE_WORK` mean operationally?** Our data has human `NO_ACTIVE_WORK` windows
   that the dataset's own step annotation places *inside* a step (12 of 16 on the 4 clips we could
   check). If both labels are correct, then "inside a step" and "doing relevant work" are different
   things, and we need Awign's definition of relevant activity to reconcile them.
7. **Is there a second labeller available?** Every human label in this repo comes from one person,
   so we cannot report inter-rater agreement, and the L1/L2 targets cannot be validated.

## 6. The one honest sentence the product may say

> On 160 human-labelled Egocentric-10K windows, the shipped 26-feature head flags 63 % of the
> windows a human called NO_ACTIVE_WORK as reject-or-review against 55 % of the rest, and accepts
> the other 37 % of them — so NO_ACTIVE_WORK is a weak, noisy correlate of the verdict, not a
> substitute for it; and on the four IndEgo clips that ship step annotations, "inside an annotated
> step" is not a usable stand-in for "relevant work is progressing", because it agrees with the
> human window labels only 75 % of the time and disagrees with them on 12 of the 16 windows the
> human called NO_ACTIVE_WORK.
