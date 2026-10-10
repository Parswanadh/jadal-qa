# How this repo was built

Jadal QA was built during the hackathon by a small set of AI workers, with every change reviewed before it merged. This page explains that workflow and records what the reviews caught. The raw records are alongside it: [`decision-log.md`](decision-log.md) (the ledger of rulings and incidents) and [`reviews/`](reviews/) (every review and re-review, with diffs and probe files left out).

## Workflow

1. **Contract first.** [`docs/SPEC.md`](../SPEC.md) defines the data schemas, module interfaces and determinism rules. GitHub issues #1 to #9, #17 and #18 define the tasks.
2. **One worktree and one branch per module** (`feat/<module>`). Each worker owns a disjoint set of files. Protected files (`jadal/__init__.py`, `jadal/config.py`, `docs/SPEC.md`) are read-only for module workers. Changes to them are proposed in the report.
3. **Implementation.** opencode workers running on Muse Spark do the coding in their own worktree. Each worker is told which files it may touch.
4. **Review.** A Haiku reviewer reads the diff against `SPEC.md` and against code quality. It gives a verdict of SPEC PASS or FAIL and QUALITY OK or CHANGES NEEDED. The reviewer re-runs the tests in the worker's worktree and checks the report's claims against the output.
5. **Fix rounds, then scoped re-review.** Workers fix the findings. The re-review covers only the fix diff, and it checks each original finding by ID and looks for new breakage.
6. **Merge only when clean.** A branch merges when its re-review says every finding is addressed and there is no new breakage. Merges are merge commits, not squashes. Before merging, a trial merge of all open branches runs the full test suite.
7. **Backups.** A loop pushes committed work and snapshots every worktree every five minutes.
8. **Rulings.** Decisions that change scope or the plan are written into the decision log, with the cost of being wrong.

## What this process caught

Five real bugs, from the round-0 reviews. Each is cited by review file and finding ID.

| # | Bug | Found by | Status |
|---|---|---|---|
| 1 | **Certify threshold chosen by a loose scan, not fixed-sequence testing.** `pick_threshold` scanned every candidate and kept the loosest one that passed the bound. That is a multiple-comparisons selection, so the returned threshold did not carry the advertised 1-delta certificate. Fix: stop at the first failing candidate, strictest first, as the SPEC requires. | [`certify-review.md`](reviews/certify-review.md) C1 | Fixed in round 1 ([`certify-rereview.md`](reviews/certify-rereview.md): ADDRESSED). Merged in PR #14. |
| 2 | **Path traversal in `/voice`.** `reason_code` and `lang` went straight into a file path, so `/voice/secret/%2E%2E` served a file from above `voicebank/`. Fix: reject values that are not in `LANGS` or `REASON_CODES`. | [`service-review.md`](reviews/service-review.md) C2 | Fixed ([`service-rereview.md`](reviews/service-rereview.md) C2: ADDRESSED). Merged in PR #13. |
| 3 | **Test-payload backdoor reachable in production.** Any uploaded clip whose first bytes were a `JADALFAKE1` marker could set its own `blur_var`, `hands_ok` and `answer_conf`, and force an accept. The stubs ran whenever the real modules were absent, which was the case on that branch. Fix: remove payload parsing, so stub values come from the hash only, and gate the stubs behind `JADAL_USE_STUBS=1`. | [`service-review.md`](reviews/service-review.md) C3 | Fixed ([`service-rereview.md`](reviews/service-rereview.md) C3: ADDRESSED). Merged in PR #13. |
| 4 | **MediaPipe API bug hidden by a bare `except`.** `mp_core.BaseOptions` does not exist, and the resulting `AttributeError` was swallowed by `except Exception: return None`. With the face model configured, the code silently fell back to the skin-colour proxy. Fix: import `BaseOptions` from `mediapipe.tasks.python.core.base_options`, narrow the exception handling, and log failures. | [`signals-review.md`](reviews/signals-review.md) Critical 1 | Fixed on the branch ([`signals-rereview.md`](reviews/signals-rereview.md) C1: ADDRESSED). PR #12 is still open, and the re-review found stale report text to fix before merge. |
| 5 | **Eval harness: threshold chosen on the split it reports, and near-duplicates across splits.** (a) `calibration_report.py` picked the threshold on the same records it reported, and `--split test` restricted those records to test, so the threshold was chosen on test. (b) `dedup()` skipped every blocking bucket with more than 400 members. The 563 all-zero-hash frames went through, which left 71,343 cross-split near-duplicate pairs. | [`evalharness-review.md`](reviews/evalharness-review.md) Critical 2 (threshold) and Critical 1 (near-duplicates) | **Caught; fix in progress.** No re-review yet. PR #16 is open, so none of the eval numbers should be cited. |

## Open at the time of writing

- **Integration (PR #19, issue #17).** Round-0 review: SPEC PASS, QUALITY CHANGES NEEDED, with two Important findings (save outside the requeue guard; reverify without `ensure_determinism`). Fix round 1 was dispatched. See [`reviews/integration-review.md`](reviews/integration-review.md).
- **Voicebank tests (`main`).** Installing the `sarvamai` SDK broke two tests that assumed it was absent. One of them made a real API call with an empty key. A fix to make the tests network-free was dispatched.
- **Eval harness (PR #16), signals (PR #12).** See the table above.

## Lessons recorded in the ledger

- **Check tool activity, not the startup banner.** One worker sat dead for about 15 minutes after a model switch was blocked by a workspace privacy setting. The status check had passed on the banner alone. Workers are now verified by their tool calls.
- **Reviews re-run the tests.** The reviewers ran the test suites in the worker worktrees, and the reports' claims were checked against that output.
- **Reports drift.** Several re-reviews found stale report text after the code had been fixed (for example, signals `docs/reports/signals.md`, and certify's old test-count block). Report claims are checked before merge.

## Limits

- Reviewers ran on code, not on hardware or GPU. Pico firmware and the GPU determinism rules are not verified on device.
- Reviews judge claims against the code and the tests. They do not replace a human check of the evaluation labels, and the labels were not human-checked.
