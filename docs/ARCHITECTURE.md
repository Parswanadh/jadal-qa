# Architecture

This page describes the Jadal QA pipeline as it is on `main` today, and says where each stage is still open. The contract every module codes against is [`SPEC.md`](SPEC.md). Where this page and `SPEC.md` disagree, `SPEC.md` governs.

Status labels used below:

- **real (merged)**: implemented and merged to `main`.
- **stub on main**: the stage exists on `main` as a deterministic stub that only runs when `JADAL_USE_STUBS=1`.
- **open PR #N**: implemented on a branch, not yet merged.
- **in progress #N** / **planned**: no merged code yet.

## Data flow

```
 head camera -> auto-sync -> ingestion bucket            (outside this repo)
                                  |
                                  | upload event: POST /ingest/event, or bucket watcher
                                  v
 +--------------- jadal service (service.py: FastAPI + SQLite) ----------------+
 |                                                                             |
 |  segment        split_clip()           segmenter.py    open PR #12          |
 |                                        (stub on main)                       |
 |  L0 signals     compute_signals()      signals.py      open PR #12          |
 |                                        (stub on main)                       |
 |  L1 decisions   Decider.decide()       decision.py     in progress #7       |
 |                                        (stub on main)                       |
 |  L2 certify     pick_threshold()       certify.py      real (merged #14)    |
 |  per-segment    make_verdict()         pipeline.py     real (merged #14)    |
 |  verdict                                                                    |
 |  receipt        make_receipt()         receipts.py     real (merged #14)    |
 |  batch rule     batch_rule(verdicts,k) certify.py      real (merged #14)    |
 |                                                                             |
 +-----------------------------------------------------------------------------+
        |                       |                              |
        v                       v                              v
  dashboard (web/)        voice (voicebank/,           Pico status
  real (merged #15)       GET /voice/...)              GET /pico/status
                          real (merged #11)            real (merged #10);
                                                       LED driven by batch
                                                       outcome: not built
```

Notes on the diagram:

- `pipeline.py` (orchestration) is real and merged. Only the L0 and L1 stages behind it are stubs on `main`. Without `JADAL_USE_STUBS=1` and without the real modules, the service refuses to ingest (HTTP 500), so a stub run cannot be mistaken for real output.
- The orchestration order in `pipeline.run_batch` is: per-segment verdict and receipt, then the batch decision over the batch's verdicts. `SPEC.md` draws the batch rule before the receipt. The batch decision is **not** part of any `receipt_sha256` in the current code. This difference is recorded here so it can be settled before submission.
- Open PR #19 (integration, issue #17) changes `pipeline.py` and `service.py`. It is not on `main` yet.

## Layers

### L0: signals (per segment, OpenCV, MediaPipe)

Input: the original segment file. Output: the 12 fields in `SPEC.md` (`decodable`, `blur_var`, brightness and exposure fractions, `motion_mean`, hand and face counts, `phash`, `dup_of`, `faces_blurred`).

- Signals are computed on the original frames. Any blurred output goes to a separate file under `data_runtime/blurred/`, and only its hash is recorded as `blurred_sha256`.
- Face and hand detection use the MediaPipe Tasks models from `models/` (git-ignored). Model paths come from `jadal/config.py`.
- Status: open PR #12 (`jadal/signals.py`, `jadal/segmenter.py`). On `main`, a stub stands in.

### L1: decisions (per segment)

Baseline: EmbeddingGemma 2 embeddings with small heads for `hands_ok`, `active_work` and the SOP step choice. Output: `p_yes` probabilities, step probabilities and `answer_conf`, plus `model_versions`.

- Status: in progress (issue #7). No decider code is on `main`. The Laya fusion (issue #9) is a stretch goal and is planned only.
- L1 determinism has not been measured, because no real decider is merged.

### L2: certify (calibration and batch rule)

`jadal/certify.py` (merged, PR #14):

- `fit_temperature` (temperature scaling), `ece` (expected calibration error).
- `exact_binomial_upper`: Clopper-Pearson upper bound on the error rate at `delta = 0.1`.
- `pick_threshold`: fixed-sequence testing from the strictest threshold downward, stopping at the first failure, with `min_accepted` (default 30).
- `risk_coverage`, and `batch_rule(verdicts, k)`, which rejects a batch when `rejected_count > k`.

Until calibration data exists, the answer threshold is a recorded placeholder (`STUB_THRESHOLD = 0.9`). The real threshold has to come from calibration on our own clips.

Per-segment verdicts are `accept`, `reject` or `review`. Hard reasons (`UNDECODABLE`, `HANDS_NOT_VISIBLE`, `BLUR`, `TOO_DARK`, `OVEREXPOSED`, `SHAKY`, `DUPLICATE`, `NO_ACTIVE_WORK`) reject. Soft reasons (`WRONG_STEP`, `LOW_CONFIDENCE`) send the segment to review.

## Receipts and re-verify

`jadal/receipts.py` (merged, PR #14):

- A receipt holds the segment hash, signals, decisions, verdict, and a config block (thresholds, pipeline version, versions).
- `receipt_sha256` is the SHA-256 of the canonical JSON of the receipt with `receipt_sha256`, `receipt_id` and `created_utc` removed. `receipt_id` is the first 16 hex characters of that hash.
- Canonical JSON uses sorted keys, separators `(",", ":")` and UTF-8. Floats are rounded to 4 decimals and `-0.0` is normalised. NaN and infinity are rejected.
- `verify_receipt(receipt, recompute_fn)` returns `{"match", "expected", "actual", "diff"}`. `diff` lists the differing field paths.
- **Re-verify** (`POST /receipts/{id}/reverify`) re-runs the pipeline on the stored segment, using the stored config block and the stored verdict threshold, and compares hashes. The segment is always re-read from its original bytes.

Known gaps:

- The `versions` block records the Python version and a config hash. Torch, model names and model weight SHA-256 are not yet written into receipts. This is parked for integration (issue #17).
- Re-verify does not compare the stored `versions` with the live implementations.
- Receipts created with the ASCII-escaped canonical JSON from the earlier stub code will not re-verify against the real module. This is documented in the service report.

## Determinism rules

These come from `SPEC.md` and apply to every module.

1. Same input plus same pinned weights and config gives the same output. No sampling, no temperature, no randomness without a fixed seed.
2. Every model call is per segment (batch size 1) with fixed frame sampling. A segment must give an identical `receipt_sha256` whether it is processed alone or inside a batch of five. For torch, `use_deterministic_algorithms(True, warn_only=True)`, `cudnn.benchmark = False` and fixed seeds are required.
3. Probabilities and other floats are rounded to 4 decimals before they enter a receipt. Dict keys and lists with no meaningful order are sorted.
4. Versions recorded: Python, torch, model names and weight SHA-256 (or size and name if hashing is too slow), and a pipeline config hash. Only the first two and the config hash are recorded today.
5. `segment_sha256` is the hash of the file the pipeline read (the original).

What the tests show: the batch-independence tests on `main` call pure functions repeatedly, and they do not run a real batch. The true "alone vs batch of five" test needs `run_batch` with real stages, which is part of integration (issue #17).

## Failure modes

| Failure | Behaviour today | Status |
|---|---|---|
| Clip path outside the bucket | Ingest returns 400 | fixed (service review C1) |
| Voice request with traversal in `lang` or `reason_code` | Returns 404 | fixed (service review C2) |
| Stub payloads that set signals or decisions | Ignored; stubs derive values from the hash only | fixed (service review C3) |
| Partly copied clip picked up by the watcher | Watcher waits for the same size and mtime on two polls; `dock_sim` copies to `.part` then renames | fixed (service review I2) |
| Stubs enabled by accident | Only with `JADAL_USE_STUBS=1`; otherwise HTTP 500 | on `main` |
| Segmentation or processing error on a clip | Clip is not marked ingested, so it can be retried | fixed (service review I3) |
| Failed batch | Its segments are dropped from the buffer and the clip stays marked ingested | open (service deferred minor) |
| Permanently failing file | Watcher retries every two polls, logging each time, with no backoff | open (service deferred minor) |
| Byte-identical clips from two workers or sites | Same `segment_id`; the second record's metadata is lost | open (signals deferred minor) |
| Face blur misses faces | The blur is a skin-colour proxy, so it can miss faces | open PR #12 (signals I3) |
| Pico port permissions after reflashing MicroPython | Device node is recreated with a restricted group, so the port must be re-granted | operational (ledger) |
| Voicebank tests in an environment with the `sarvamai` SDK installed | Two tests assume the SDK is absent and fail | open on `main` (see README test status) |

## What we deliberately do not claim

- **No accuracy numbers** until a file under `eval/results/` says so. On `main` there is none. PR #16 adds files there, and some of them are demo artifacts derived from noisy labels (see the process records).
- **Labels for Egocentric-10K-Evaluation come from gemini-2.5-flash and were not checked by a human.**
- **Thresholds are placeholders** (0.9) until calibrated on our own clips. No threshold on `main` is certified.
- **The batch rule is a fixed rule with no statistical guarantee** at n = 5 to 10.
- **The face blur is not a privacy guarantee.**
- **Batch-independence** is tested through pure-function proxies, not a real batch run.
- **Pico firmware is not verified on the device.** Hardware validation is pending.
- **The SOP is a labelled sample** (`sop/sop_garment_v1.md`), not a task specification taken from the footage.
- **Voice translations are machine drafts.** The Tamil header has a known typo, deliberately left for native-speaker review.
- **L1 determinism is not measured**, because no real decider is merged.
