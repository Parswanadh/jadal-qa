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

# 3. calibration report for a predictions file (optionally test split only)
python -m eval.calibration_report --predictions eval/results/predictions.json
python -m eval.calibration_report --predictions eval/results/predictions.json \
    --splits eval/results/splits.json --split test

# 4. unit tests (synthetic fixtures, no real data needed)
python -m unittest discover -s tests -v
```

### `eval/make_splits.py`

1. **Metadata scan** — per-file row counts + `hand_count` / `active_labor`
   distributions per `source_dataset` (printed + `eval/results/dataset_report.json`).
2. **pHash** — streams images in small batches (`--batch-size`, default 256),
   decodes with PIL, immediately converts to 64x64 grayscale, computes
   `imagehash.phash`, closes the image. A cache
   (`eval/results/phash_cache.json`) makes re-runs cheap; `--no-cache` to skip.
3. **Near-duplicate removal** — pHash Hamming distance ≤ `--dup-threshold`
   (default 4) groups frames via blocking + union-find *before* splitting, so
   no near-duplicate pair can land in two different splits. The kept
   representative is the smallest `frame_id` (order-independent).
   Removed pairs go to `eval/results/duplicates_removed.json`.
4. **Stratified split** — deterministic seeded SHA-256 ordering inside each
   `(source_dataset, hand_count, active_labor)` cell, assigned 70/15/15 to
   `train` / `calibration` / `test`, exact per cell (tiny cells with < 4 rows
   stay whole in train). Manifest + split report: `eval/results/splits.json`.

Memory safety: only `--batch-size` encoded rows are ever materialised; decoded
images live only inside worker threads. Never prints image bytes or rows.

### `eval/calibration_report.py`

Input: JSON list (or `{"predictions": [...]}`) of
`{"frame_id", "probs", "label"}` where `probs` is a float, `[p_no, p_yes]`,
or `{"yes": p, "no": q}` / `{"0": .., "1": ..}`, and `label` is
`"yes"`/`"no"`/`0`/`1`. Unparseable records are counted as `skipped`.
`--write-example PATH` writes a schema example.

Operating convention: binary classifier with confidence `max(p, 1-p)`,
prediction `1 iff p >= 0.5`; the risk-coverage curve is "accept if confidence
≥ threshold". Uses `jadal.certify.ece` / `risk_coverage` / `pick_threshold`
when that module is importable, otherwise local implementations of the same
formulas; the JSON records which source produced each number
(`ece_source`, `risk_coverage_source` — currently `local_fallback` because
`jadal/certify.py` does not exist yet on this branch).

Outputs: `eval/results/calibration.json` (accuracy, ECE, full curve,
threshold pick for `--target-error`, default 0.05) and
`eval/results/calibration.png` (reliability diagram + accuracy/error vs
coverage, matplotlib `Agg`).

## NOT DONE

- No model baselines yet: there is no `predictions.json` producer. The
  calibration report runs on caller-supplied predictions only.
- `jadal/certify.py` does not exist yet, so all numbers come from the local
  fallback (clearly labelled in the output).
- pHash cache invalidation is by recipe version only; re-run with
  `--no-cache` if the input parquet files change.
