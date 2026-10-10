# How to run and log an experiment

This page is the protocol for every experiment in Jadal QA. It applies to a new idea, to a re-run, and to a result that is not what we hoped for. The worked example at the end is a real case of a result that changed a design decision.

## Principles

1. **Every experiment is logged.** Each one gets a registry line and a folder: `tools/exp.py new` at the start, `tools/exp.py finish` at the end. The log records:
   - the **reason** (why this experiment, what decision it informs);
   - the **outputs** (the results files, committed on the branch);
   - the **data type** (IndEgo, Egocentric-10K, Awign footage, synthetic) and the **label source** (human, generated, none).
2. **Regime-specific models and profiles.** A model or threshold is fitted for one regime. A profile tuned on one footage source is named for that source. Using it on another regime needs a new experiment, not a silent reuse.
3. **Negative results are kept.** A result that says "no" is logged with the verdict `drop` or `inconclusive`, its results file is committed, and the write-up says what it does not show. Do not delete or rerun a failed experiment until it passes.
4. **Fix the criterion before the run.** The hypothesis and the decision rule are written in the proposal. A result is judged against them, not against a rule chosen after seeing the numbers.
5. **Numbers come with their context.** Every reported rate has its `n`, its label source, its protocol (split, what was fitted on which data, how the threshold was chosen), a confidence interval, and the device.

## Steps

1. **Propose.** Open an issue with the experiment proposal template. Name the question, the criterion, the data, the regime and the negative-result plan.
2. **Register.** Pick an id such as `E1.01` and run:
   ```
   python tools/exp.py new --id E1.01 --track E1 --title "..." --why "..." --hypothesis "..." \
     --data "..." --regime "..." --method "..." --owner NAME
   ```
   This appends a line to `experiments/registry.jsonl` and creates `experiments/E1.01/README.md`.
3. **Run.** Write a results JSON with `n`, the label source, the protocol, the CI and the device. Commit it on the experiment branch. Do not commit frames or clips.
4. **Finish.** Record the verdict: `adopt`, `maybe`, `drop` or `inconclusive`. Keep the summary to what the results file shows:
   ```
   python tools/exp.py finish --id E1.01 --verdict drop --summary "..." --results path/to/results.json
   python tools/exp.py render
   ```
5. **Decide.** If the decision changes scope or the plan, write it in the decision log with the cost of being wrong.

## Verdicts

| Verdict | Meaning |
|---|---|
| `adopt` | The criterion was met on the stated regime, with the CI reported. |
| `maybe` | Met on part of the regime, or the CI is too wide to decide. Needs another experiment. |
| `drop` | The criterion was not met. The result stays in the log. |
| `inconclusive` | The experiment did not separate the options (for example, the labels could not support the test). |

## Worked example: E0.01, does skipping the start of a clip help?

**Question.** Could the first part of an IndEgo clip be skipped, because it is usually setup and not work? The idea would save decode time.

**Why it was run.** The first minutes of a clip were suspected to be setup, with more bad windows than the rest. If so, a skip rule would be cheap and remove mostly bad windows.

**Hypothesis and criterion (written before the run).** The first 120 s has a clearly higher bad-window rate than the rest of the clip. Skipping is supported only if that holds. Neighbouring windows must also agree more than chance, so the labels are not noise.

**Data and labels.** IndEgo clips (CC BY 4.0 per `docs/ATTRIBUTIONS.md`), with the human gold labels. Regime: IndEgo ECU-assembly footage. It is not Awign footage.

**Results (owner-supplied figures; the results JSON is not on this branch yet).**

- Neighbouring windows agree **73%** of the time, against **56%** expected by chance.
- The bad-window rate in the **first 120 s** is **0.19**. In the part **after** 120 s it is **0.42**.

**Verdict: `drop`.** The criterion was not met. On IndEgo the first 120 s is not the bad part of the clip; it is cleaner than the rest. Skipping it would discard the cleaner windows and keep the bad ones. Skipping the start is not supported on IndEgo. The negative result stays in the log.

**What it does not show.** It is one dataset, from a few wearers, with labels from one labeller. It does not show that the start is clean on other footage, and it does not measure the decode time a skip would save. Before this is cited, the results JSON with `n`, the per-window labels and the CI must be committed, and the registry line must point to it. Until then the figures above are not backed by a file on this branch.

**Registry line (the shape, after `finish`).**

```
{"id": "E0.01", "status": "done", "verdict": "drop", "summary": "...", "results": "<path to results JSON>", "decision": "do not skip the clip start on IndEgo"}
```

## Checklist before you ask for review

- [ ] The experiment is registered and finished with a verdict (`tools/exp.py`).
- [ ] The results file is committed, with `n`, label source, protocol, CI and device.
- [ ] The regime and the models or profiles used are named.
- [ ] A negative or inconclusive result is kept, not removed.
- [ ] No frames, clips or thumbnails are committed.
- [ ] `tools/oss_audit.sh` reports no secret pattern.
