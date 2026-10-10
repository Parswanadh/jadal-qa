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
