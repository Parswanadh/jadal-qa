SPEC: FAIL, QUALITY: CHANGES NEEDED

PR #16 (feat/evalharness @ 9173fbb) for issue #8. Reviewed the diff vs main (13 files: eval/, tests/, docs/reports/). Nothing modified, merged, pushed or commented on GitHub. Only this report and the scratch files under reviews/ and the scratchpad were written.

## Verification performed
- `git diff main...origin/feat/evalharness --stat` saved to reviews/evalharness.stat. Worktree is clean and matches origin.
- Unit tests: `PYTHONNOUSERSITE=1 <env>/bin/python -m unittest discover -s tests` gives Ran 16 tests, OK. Matches the doc claim.
- Warm-cache full rerun into a scratch dir (31 s): splits.json is identical to the committed one apart from created_utc, and duplicates_removed.json is identical. The determinism claim holds for the warm-cache path.
- Manifest checks: 29,770 frame_ids, no frame_id in two splits, all removed frame_ids absent from every split, all kept_frame_id references present. Sizes 20828/4471/4471 (0.6996/0.1502/0.1502).
- Leakage check on the pHash cache (numpy, all kept frames): 71,343 pairs with Hamming <= 4 cross splits, and 87,423 more within a split.

## Critical
1. **Cross-split near-duplicate leakage (SPEC item 4 FAILS).** dedup() skips every blocking bucket with more than 400 members (eval/make_splits.py:356-358). 563 frames share pHash 0000000000000000, so the block keys for that value form a bucket over 400. The 68 skipped buckets are recorded only as `skipped_large_buckets: 68` in splits.json and are never removed or reported in the console. Result: all 563 all-zero-hash frames survive dedup (they are spread across train 402 / calibration 85 / test 76), and every one of the 71,343 cross-split pairs involves one of them. The guarantee in eval/make_splits.py:18-19, eval/README.md:55-58 and docs/reports/evalharness.md (dedup claim) is therefore false as written. Fix: for buckets above the cap, compare pairs exactly (or hash-group identical pHash values first, e.g. collapse exact-duplicate hash groups before blocking), and fail or warn loudly when any bucket is skipped. Add a test with more than 400 identical hashes that asserts they collapse to one frame and never span splits. Re-run and regenerate splits.json, duplicates_removed.json and the cache-derived counts.

2. **Threshold selected on the split it is reported on (statistical error).** calibration_report.py:333-366 (local_choose_threshold) picks the accept threshold from the same records it reports coverage and error on. In main() (lines 470-485) `--split test` restricts those same records to test, so the threshold is chosen on test. The committed eval/results/calibration.json has threshold 0.0 / coverage 1.0 "from": "local_min_accepted" on test, which is not a usable operating point. The spec's intent is threshold selection on the calibration split and reporting on test. Fix: add a `--select-split calibration` path (or require two splits) and report the chosen threshold's test error separately. Also, no error bound is computed, so the docs must not describe the threshold as certified (docs/reports/evalharness.md states this only as NOT DONE, which is acceptable).

## Important
3. **scan_metadata() and hash_dataset() materialise image bytes.** Both read COLUMNS, which includes `image` (make_splits.py:110 and :274), and call to_pylist() on 2048-row batches. Each batch therefore holds 2048 encoded images in Python objects. This contradicts "only --batch-size encoded rows are ever materialised" (eval/README.md:65). Fix: read the metadata columns without `image` in the metadata and cache-check passes (pf.iter_batches(columns=[...]) excluding "image"), and keep the image column only in hash_parquet.

4. **Committed demo artifacts sit in the results directory.** eval/results/calibration.json, calibration.png and predictions_demo.json are generated from noisy copies of the gemini auto-labels (predictions_demo.json frame probabilities are derived from the labels). The docs say so, but the artifacts carry the same names as real outputs (accuracy 0.9975, ECE 0.0742). Fix: move them under a clearly named path such as eval/results/demo/ or delete them, or rename the outputs with a demo prefix.

5. **Test gaps.** No test covers the large-bucket skip (see item 1), no test checks that a split assignment never places near-duplicates in two splits, and no test checks that the image column is absent from metadata reads. tests/test_eval_splits.py also has no end-to-end determinism test through the CLI (only assign_splits() determinism at lines 169-174).

6. **Docs overstate the stratification.** eval/README.md:62 says "exact per cell". make_splits.py:439-440 uses ceil() for test and calibration, so tiny cells are over-represented in test and calibration. Three cells (ego4d|0|yes, egocentric_10k|0|yes, epic_kitchens|0|yes) deviate from 70% train by more than 2 points. Fix: say "rounded up per cell" in the README and report the per-cell deviation.

## Minor
7. make_splits.py:108 has a confused comment ("take the larger key? no: ordered"). The two-key dict branch in _as_p_yes (calibration_report.py:108-111) takes values[1] in dict insertion order, which is arbitrary for keys other than yes/no. Use the explicit yes/no keys or raise.
8. calibration_report.py:176 is a hard-to-read chained expression. Simplify and test it.
9. calibration_report.py:161-164 and :150-151: a float label strictly between 0 and 1 is silently thresholded at 0.5. Either reject it or document it.
10. docs/reports/evalharness.md says the first pHash pass took about 163 s. That is not re-measured here, and was not verified. Fine to keep if the author ran it, but it should be marked as a single observed run.

## Checks that passed
- Labels: gemini-2.5-flash, not human-checked, stated in eval/README.md, make_splits.py docstring and the manifest's label_source field.
- Scope: the diff touches only eval/, tests/ and docs/reports/. No code outside the owned set changed, and no secrets were found in the PR diff. (A placeholder API-key string in docs/sarvam_reference.md exists on main, not on this PR.)
- Determinism: warm-cache rerun reproduces splits.json and duplicates_removed.json exactly.
- Memory: the pHash path downscales each image and closes it immediately (see item 3 for the metadata path).
- Calibration fallback: jadal.certify is optional, the fallback is documented, and the JSON records which source produced each number.
