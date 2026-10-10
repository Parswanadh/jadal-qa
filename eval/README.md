# Jadal eval harness

Dataset splits + calibration curves for the ego10k evaluation set
(issue #8).

## Data

Three Parquet files in `/home/parshu/projects/awing/data/ego10k_eval/`
(read-only inputs, never copied into the repo):

| file | rows | source_dataset | hand_count | active_labor |
|---|---|---|---|---|
| `ego4d.parquet` | 10,000 | ego4d | 0: 3267, 1: 3038, 2: 3695 | no: 4993, yes: 5007 |
| `egocentric_10k.parquet` | 10,000 | egocentric_10k | 0: 358, 1: 2008, 2: 7634 | no: 834, yes: 9166 |
| `epic_kitchens.parquet` | 10,000 | epic_kitchens | 0: 963, 1: 2932, 2: 6105 | no: 1496, yes: 8504 |

Columns: `frame_id` (str, distinct across all files), `image` (struct of
PNG/JPEG `bytes` + `path`), `source_dataset`, `hand_count` (0/1/2),
`active_labor` ("yes"/"no").

> **Label provenance:** `hand_count` and `active_labor` were produced by
> **gemini-2.5-flash** auto-labelling and were **NOT human-checked**. Treat
> every accuracy / calibration number computed from them as a
> *model-vs-model agreement* measurement, not ground-truth performance.

## Tools

All commands run from the repo root with `/home/parshu/envs/jadal/bin/python`
(never `pip install`; no GPU; no network at import time).

```bash
# 1. dataset report only (metadata, no image decoding)
python -m eval.make_splits --report-only

# 2. full pipeline: pHash all frames -> dedup -> stratified manifests
python -m eval.make_splits [--seed 1234] [--dup-threshold 4] [--workers 4]

# 3. calibration report: threshold selected on calibration, reported on test
python -m eval.calibration_report --predictions eval/results/predictions.json \
    --splits eval/results/splits.json --select-split calibration --report-split test

# 4. leakage proof (must print 0 cross-split pairs)
python -m eval.check_leakage

# 5. unit tests (synthetic fixtures, no real data needed)
python -m unittest discover -s tests -v
```

### `eval/make_splits.py`

1. **Metadata scan** — per-file row counts + `hand_count` / `active_labor`
   distributions per `source_dataset` (printed + `eval/results/dataset_report.json`).
2. **pHash** — streams images in small batches (`--batch-size`, default 256),
   decodes with PIL, immediately converts to 64x64 grayscale, computes
   `imagehash.phash`, closes the image. A cache
   (`eval/results/phash_cache.json`) makes re-runs cheap; `--no-cache` to skip.
3. **Near-duplicate CONNECTED COMPONENTS** — pHash Hamming distance ≤
   `--dup-threshold` (default 4) groups frames via exact-hash collapse +
   multi-index blocking + union-find, with **no bucket-size cutoff** (a huge
   group of identical hashes is simply one component). Each component is
   assigned **wholly to one split**, so no near-duplicate pair can span two
   splits (`python -m eval.check_leakage` proves it: 0 cross-split pairs).
   Removed non-representative members go to
   `eval/results/duplicates_removed.json`.
4. **Stratified split** — components are grouped by stratum = majority
   `(source_dataset, hand_count, active_labor)` vote (ties break
   deterministically). Per-stratum frame targets come from largest-remainder
   70/15/15, and each whole component goes to the split with the largest
   remaining deficit, so per-stratum totals are exact up to the largest
   component in the stratum. Manifest + split report:
   `eval/results/splits.json` (includes `peak_rss_mb`).

Memory safety: metadata passes never request the `image` column; image reads
use batches of at most 64 rows, each image is downscaled to 64x64 grayscale
immediately and closed. Never prints image bytes or rows.

### `eval/calibration_report.py`

Input: JSON list (or `{"predictions": [...]}`) of
`{"frame_id", "probs", "label"}` where `probs` is a float, `[p_no, p_yes]`,
or `{"yes": p, "no": q}` / `{"0": .., "1": ..}` (a two-key dict without a
recognised positive key is refused, not guessed), and `label` is
`"yes"`/`"no"`/`0`/`1` (soft labels in (0, 1) are binarised at 0.5).
Unparseable records are counted as `skipped`.
`--write-example PATH` writes a schema example.

Operating convention: binary classifier with confidence `max(p, 1-p)`,
prediction `1 iff p >= 0.5`. The accept threshold is chosen on the
SELECTION set (default `--select-split calibration`) with
`jadal.certify.pick_threshold` (fixed-sequence testing, exact-binomial
certificate at `--delta`, default 0.1); accuracy, ECE and the
risk-coverage curve are computed on the REPORT set (`--report-split test`).
Selection and report frame_ids must be disjoint — the tool exits with an
error otherwise. The threshold's error/coverage on the report set is
reported as *observed*, never certified (the certificate lives on the
selection set). Uses `jadal.certify.ece` / `risk_coverage` /
`pick_threshold` when that module is importable, otherwise local
implementations of the same formulas; the JSON records which source
produced each number.

Outputs: `eval/results/calibration.json` and `eval/results/calibration.png`
(reliability diagram + accuracy/error vs coverage, matplotlib `Agg`).
Files prefixed `DEMO_synthetic_*` are synthetic demo data (noisy copies of
the auto-labels, for pipeline smoke tests only); real model outputs use the
unprefixed names (`predictions.json`, `calibration.json`, `calibration.png`).

## NOT DONE

- No model baselines yet: there is no `predictions.json` producer. The
  calibration report runs on caller-supplied predictions only.
- pHash cache invalidation is by recipe version only; re-run with
  `--no-cache` if the input parquet files change.
- `eval/check_leakage.py` checks pHash leakage only; semantic near-duplicates
  with very different hashes are out of scope.
