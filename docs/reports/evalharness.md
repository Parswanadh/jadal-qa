# evalharness — module report (issue #8)

## What exists

- `eval/make_splits.py` — dataset report + stratified train/calibration/test
  manifests with pHash near-duplicate removal across splits.
- `eval/calibration_report.py` — accuracy-vs-coverage + ECE from a predictions
  JSON; uses `jadal.certify` when importable, else a local fallback (labelled
  in the output). Writes JSON + PNG.
- `eval/README.md` — usage, formats, label provenance.
- `tests/test_eval_splits.py` — 16 tests, all passing.
- `eval/results/` — `dataset_report.json`, `splits.json` (seed 1234),
  `duplicates_removed.json`, `phash_cache.json`, `calibration.json`,
  `calibration.png`, `predictions_demo.json` (synthetic demo predictions).

## How to run

```bash
cd /home/parshu/projects/jadal-qa-wt/evalharness
/home/parshu/envs/jadal/bin/python -m eval.make_splits --report-only
/home/parshu/envs/jadal/bin/python -m eval.make_splits
/home/parshu/envs/jadal/bin/python -m eval.calibration_report \
  --predictions eval/results/predictions_demo.json \
  --splits eval/results/splits.json --split test
/home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v
```

## Test output tail (real)

```
Ran 16 tests in 0.790s
OK
```

Full split run (warm pHash cache): `splits.json` (1.44 MB) in ~30-40 s.
First pHash pass over 30k frames: ~163 s at ~217 rows/s (4 threads).

Split report (seed 1234, hamming ≤ 4):

```
rows in dataset: 30000, kept after dedup: 29770 (removed 230 near-duplicates in 221 groups)
split            count  fraction  sources                                       hand_count          active_labor
train            20828    0.6996  ego4d:6987,egocen:6994,epic:6847              0:3189,1:5532,2:12107no:5093,yes:15735
calibration       4471    0.1502  ego4d:1499,egocen:1502,epic:1470              0:688,1:1187,2:2596 no:1095,yes:3376
test              4471    0.1502  ego4d:1499,egocen:1502,epic:1470              0:688,1:1187,2:2596 no:1095,yes:3376
```

Dataset label distributions per source:

```
source_dataset           hand_count=0  hand_count=1  hand_count=2  active_labor=yes  active_labor=no  rows
ego4d                    3267          3038          3695          5007              4993             10000
egocentric_10k           358           2008          7634          9166              834              10000
epic_kitchens            963           2932          6105          8504              1496             10000
```

Calibration end-to-end on synthetic demo predictions (noisy label copies,
seed 11, n=4471 test frames — a pipeline smoke test, NOT a model result):
accuracy 0.9975, ECE 0.0742, sources `local_fallback` (jadal.certify absent).

Determinism: two consecutive full runs produce byte-identical `splits.json`
apart from the `created_utc` field.

## Known gaps / NOT DONE

- No model baselines: `predictions_demo.json` is synthetic (noisy copies of
  the auto-labels). A real baseline producer is future work.
- The demo calibration numbers above are pipeline smoke-test outputs, not
  measurements of any model.
- `jadal/certify.py` does not exist on this branch, so ECE / risk-coverage /
  threshold numbers come from the local fallback (stated in the JSON).
- Labels are gemini-2.5-flash auto-labels, NOT human-checked (stated in
  `eval/README.md` and the manifests). All accuracy numbers are
  model-vs-model agreement until human labels exist.
- `eval/results/*.log` run logs are scratch (untracked); the JSON/PNG
  artifacts are committed.
- Contract proposal: none needed — `docs/SPEC.md` already covers
  `jadal/certify.py`; this module only consumes it optionally.

## Fix round 1 (review rulings C1, C2, I3–I6)

- **C1 — connected components, no cutoff, no leakage.** `dedup()` (with its
  >400 bucket skip) is replaced by `find_components()`: exact-duplicate
  hashes collapse in O(n) first, then multi-index blocking compares *every*
  candidate pair (`bucket_cutoff: None` in the report). Each component is
  assigned wholly to one split; the stratum is the majority
  (source, hand_count, active_labor) vote with deterministic tie-breaks.
  Per-stratum frame targets use largest-remainder 70/15/15; components go to
  the split with the largest remaining deficit. New `splits.json` (seed 1234):
  train 21001 / calibration 4501 / test 4498 (fractions 0.7/0.15/0.1499).
- **Leakage proof** (`eval/check_leakage.py`):
  `[check_leakage] 30000 frames, threshold=4: cross-split pairs at Hamming <= threshold: 0`
- **C2 — select on calibration, report on test.** `calibration_report` takes
  `--select-split` (default `calibration` with `--splits`) and
  `--report-split`, refuses with an error on any frame_id overlap, chooses
  the threshold with `jadal.certify.pick_threshold(target_err, delta)` on the
  selection set only, and reports the threshold's *observed* (not certified)
  error/coverage on the report set. Demo run on synthetic predictions:
  selection calibration n=4501 → threshold 0.5005 (upper_bound 0.0034 at
  delta=0.1), test n=4498 accuracy 0.9964, ECE 0.0712, sources
  `jadal.certify` throughout.
- **I3 — memory.** Metadata passes read only `META_COLUMNS` (no `image`
  bytes); image batches clamped to ≤ 64 rows. Peak RSS recorded in the
  manifest and console: full run peaked at **2687.1 MiB** (blocking
  pair-set dominates; box has ~15 GB).
- **I4 — demo names.** Committed demo artifacts renamed to
  `DEMO_synthetic_predictions.json`, `DEMO_synthetic_calibration.json/.png`
  (predictions file also carries `"synthetic_demo_data": true` inside).
- **I5/I6 — tests.** 26 eval tests: large-bucket collapse (>400 identical
  hashes → one component), component atomicity across disagreeing strata,
  planted-leak detector check, imageless-parquet metadata scan, CLI
  determinism (byte-identical manifests modulo `created_utc`; `peak_rss_mb`
  excluded), overlap refusal, largest-remainder exactness.
- **Test command + tail (real):**
  `cd /home/parshu/projects/jadal-qa-wt/evalharness && PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests`
  → `Ran 239 tests ... FAILED (failures=1, errors=1)` — all 26 eval tests
  pass; the 1 failure + 1 error are pre-existing `tests/test_voicebank.py`
  items (verified failing with my changes stashed; owned by another module,
  untouched).
- First-pass pHash timing (~163 s, single observed run, review minor #10).
  Review minors #7 (ambiguous dict order now refused), #8 (chained
  expression split), #9 (soft-label binarisation documented) fixed.
