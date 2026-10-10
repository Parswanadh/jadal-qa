# Architecture

This page is **descriptive**. It describes the Jadal QA pipeline as it is on `origin/main` (55b0654) on 2026-10-10, and the direction the build is moving in. It is not the build contract. [`SPEC.md`](SPEC.md) is the binding build contract that every merged module codes against. `SPEC.md` predates the direction below and has not been updated in this change, so where this page and `SPEC.md` disagree, `SPEC.md` governs. The whole-video, streaming, no-frame-sampling direction below is planned, pending a `SPEC.md` amendment. It is not the current contract.

Status labels used below:

- **merged (PR #N)**: implemented and merged to `main`.
- **open PR #N**: implemented on a branch, not yet merged.
- **planned**: no merged code yet.
- **not measured**: built or designed, but no measurement exists in a file on this branch.

## Environment

The pipeline runs in a Python 3.12 conda env with torch+cu130 (CUDA user-space wheels inside the env; the NVIDIA driver is not changed). Set the path to that env locally; it is not stored in this repo. Use `PYTHONNOUSERSITE=1` so user-site packages do not shadow the env.

Measurement hardware so far: an RTX 4070 laptop GPU and x86 laptop CPU cores (hardware specifications are not recorded in a file on this branch). There is no Raspberry Pi measurement yet.

## Direction (2026-10-10)

- **Edge-first.** Mentors at Awign said workers record their own video, a decentralised deployment is acceptable if it runs on a Raspberry Pi, rejection reasons can be generic, and accept/reject may be fully automated with no mandatory human. The Pi run is pending. x86 cores are a proxy.
- **Whole-video, per-pixel analysis of every frame.** Frames are not sampled. Stage 1 streams over every frame at reduced resolution and keeps running accumulators.
- **Streaming with early reject.** The worker is planned to analyse chunks while it records. The chunk size is not measured or backed by a file on this branch. Stage 0 is planned to reject dark, blocked, over-exposed or frozen footage early in the clip, with evidence. Its timing is not measured (see the Stage 0 note below).
- **Mathematics-first, deterministic decisions.** Decisions come from measured signals and a transparent rule list with exact thresholds. Logistic regression and boosted trees are comparison baselines only. Every reject carries its evidence (value, comparison, threshold). The first rule list tried rejected nearly everything and was dropped (unverified builder note, no backing file in this repo).
- **Signals.** Poisson-Gaussian noise model (SNR), H.264 bits per frame, frame difference against the noise floor, noise-normalised sharpness, residual optical flow after ego-motion removal, exposure clipping, histogram entropy, dHash. Heat-equation blur, colour temperature, MSCN and max-entropy were dropped.
- **Not in the pipeline.** A generic VLM (stopped as too probabilistic), Laya (text-only) and Clef (hosted). EmbeddingGemma 2 is an optional helper for hub search, de-duplication and frame heads.
- **RGB, not hyperspectral.** The cameras are RGB (3 bands). The analysis is per-pixel frequency and temporal-spectral analysis of RGB frames. It is not hyperspectral imaging.

## Data flow (merged code)

```
 head camera -> auto-sync -> ingestion bucket            (outside this repo)
                                  |
                                  | upload event: POST /ingest/event, or bucket watcher
                                  v
 +--------------- jadal service (service.py: FastAPI + SQLite) ----------------+
 |                                                                             |
 |  segment        split_clip()           segmenter.py    merged (PR #12)      |
 |  L0 signals     compute_signals()      signals.py      merged (PR #12)      |
 |  L1 decisions   Decider.decide()       decision.py     merged (PR #20)      |
 |  L2 certify     pick_threshold()       certify.py      merged (PR #14)      |
 |  per-segment    make_verdict()         pipeline.py     merged (PR #14)      |
 |  verdict                                                                    |
 |  receipt        make_receipt()         receipts.py     merged (PR #14)      |
 |  batch rule     batch_rule(verdicts,k) certify.py      merged (PR #14)      |
 |                                                                             |
 +-----------------------------------------------------------------------------+
        |                       |                              |
        v                       v                              v
  dashboard (web/)        voice (voicebank/,           Pico status
  merged (PR #15, #21)    GET /voice/...)              GET /pico/status
                          merged (PR #11)              merged (PR #10);
                                                       LED driven by batch
                                                       outcome: not built
```

Notes on the diagram:

- The orchestration order in `pipeline.run_batch` is: per-segment verdict and receipt, then the batch decision over the batch's verdicts. `SPEC.md` draws the batch rule before the receipt. The batch decision is **not** part of any `receipt_sha256` in the current code. Settle this before submission.
- The whole-video, streaming direction is not on `main`. The stages above are the merged scaffold it will replace or extend.

## Layers

### Stage 0 and stage 1 (direction; not yet on `main`)

- **Stage 0** (early in the clip, tiny resolution): exact rules that reject dark, blocked, over-exposed or frozen footage immediately, with evidence. Planned direction, pending `SPEC.md` amendment; any collapse-gate timing is an unverified builder note with no backing file or command in this repo. The gate concept is blind to semantic failures (no active work, hands not visible).
- **Stage 1** (whole video, every frame at reduced resolution, streaming accumulators): per-pixel spatial and temporal spectrum, noise model, motion after camera-motion removal. These feed the rule list and a score.
- **Stage 2** (uncertain windows only, optional): heavier tools (MediaPipe hands, EmbeddingGemma 2).
- Quality features are measured on RAW pixels. Activity features are measured on CLEANED video (denoised, normalised, motion-compensated). Each cleaning step has to earn its place in cross-validation.
- A label-noise audit flags suspicious labels. It never edits a human's label.

### L0: signals (merged, PR #12)

`jadal/signals.py` and `jadal/segmenter.py`. Output: the fields in `SPEC.md` (`decodable`, `blur_var`, brightness and exposure fractions, `motion_mean`, hand and face counts, `phash`, `dup_of`, `faces_blurred`). This is the merged scaffold. The per-pixel stages above replace it as the main signal path.

- Signals are computed on the original frames. Any blurred output goes to a separate file under `data_runtime/blurred/`, and only its hash is recorded as `blurred_sha256`.
- Face and hand detection use the MediaPipe Tasks models from `models/` (git-ignored). Model paths come from `jadal/config.py`.

### L1: decisions (merged, PR #20)

`jadal/decision.py`. Baseline: EmbeddingGemma 2 embeddings with small heads for `hands_ok`, `active_work` and the SOP step choice. Output: `p_yes` probabilities, step probabilities and `answer_conf`, plus `model_versions`.

- EmbeddingGemma 2 is an optional helper in this direction, not the core.
- The Laya fusion (issue #9) is not in the pipeline. It is a stretch idea only.
- L1 determinism is **not measured**.

### L2: certify (calibration and batch rule, merged, PR #14)

`jadal/certify.py`:

- `fit_temperature` (temperature scaling), `ece` (expected calibration error).
- `exact_binomial_upper`: Clopper-Pearson upper bound on the error rate, at the default `DEFAULT_DELTA` in `jadal/certify.py`.
- `pick_threshold`: fixed-sequence testing from the strictest threshold downward, stopping at the first failure, with `min_accepted` (default `DEFAULT_MIN_ACCEPTED` in `jadal/certify.py`).
- `risk_coverage`, and `batch_rule(verdicts, k)`, which rejects a batch when `rejected_count > k`.

**These bounds are machinery, not a delivered guarantee.** With 150 human labels from 7 clips, the multiple-testing correction needs about 86 clean windows (see `docs/reports/funnel.md`: support floor 86). Windows within a clip are correlated, so the usable independent sample is smaller than 150. No bound the current data can support is certified. Error rates are reported as observed, with n and label source.

Until calibration data exists, the answer threshold is a recorded placeholder (`STUB_THRESHOLD = 0.9` in `jadal/stubs.py`). No threshold on `main` is certified.

Per-segment verdicts are `accept`, `reject` or `review`. Hard reasons (`UNDECODABLE`, `HANDS_NOT_VISIBLE`, `BLUR`, `TOO_DARK`, `OVEREXPOSED`, `SHAKY`, `DUPLICATE`, `NO_ACTIVE_WORK`) reject. Soft reasons (`WRONG_STEP`, `LOW_CONFIDENCE`) send the segment to review.

## Receipts and re-verify

`jadal/receipts.py` (merged, PR #14):

- A receipt holds the segment hash, signals, decisions, verdict, and a config block (thresholds, pipeline version, versions).
- `receipt_sha256` is the SHA-256 of the canonical JSON of the receipt with `receipt_sha256`, `receipt_id` and `created_utc` removed. `receipt_id` is the first 16 hex characters of that hash.
- Canonical JSON uses sorted keys, separators `(",", ":")` and UTF-8. Floats are rounded to 4 decimals and `-0.0` is normalised. NaN and infinity are rejected.
- Receipts are **content-hashed, not signed.** An optional HMAC sidecar exists (`jadal/receipt_mac.py`, `scripts/verify_receipt.py`; see [`reports/verify.md`](reports/verify.md)). The MAC is never a field inside the hashed receipt.
- `verify_receipt(receipt, recompute_fn)` returns `{"match", "expected", "actual", "diff"}`. `diff` lists the differing field paths.
- **Re-verify** (`POST /receipts/{id}/reverify`) re-runs the pipeline on the stored segment, using the stored config block and the stored verdict threshold, and compares hashes. The segment is always re-read from its original bytes.

Known gaps:

- The `versions` block records the Python version and a config hash. Torch, model names and model weight SHA-256 are not yet written into receipts. This is parked for integration (issue #17).
- Re-verify does not compare the stored `versions` with the live implementations.
- Receipts created with the ASCII-escaped canonical JSON from the earlier stub code will not re-verify against the real module. This is documented in the service report.

## Determinism rules

These come from `SPEC.md` and apply to every merged module.

1. Same input plus same pinned weights and config gives the same output. No sampling, no temperature, no randomness without a fixed seed.
2. Each model call is per segment (batch size 1). `SPEC.md` specifies fixed frame sampling; the whole-video direction above would process frames without sampling, but that change is planned, pending a `SPEC.md` amendment — `SPEC.md` governs until then. A segment must give an identical `receipt_sha256` whether it is processed alone or inside a batch (`SPEC.md` specifies a batch of 5). For torch, `use_deterministic_algorithms(True, warn_only=True)`, `cudnn.benchmark = False` and fixed seeds are required.
3. Probabilities and other floats are rounded to 4 decimals before they enter a receipt. Dict keys and lists with no meaningful order are sorted.
4. Versions recorded: Python, torch, model names and weight SHA-256 (or size and name if hashing is too slow), and a pipeline config hash. Only the first two and the config hash are recorded today.
5. `segment_sha256` is the hash of the file the pipeline read (the original).

What is measured: the CPU feature path's canonical-JSON digest (`sha256(canonical-json(signals+route))`) was identical across repeated runs and across 1-core vs 4-core runs (see `reports/edge.md:81-83`). GPU numerics are not claimed identical across devices, and cross-architecture determinism is not claimed. The batch-independence tests on `main` call pure functions repeatedly and do not run a real batch. The real "alone vs batch" proof through the real modules is **pending**.

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
| Face blur misses faces | The blur is a skin-colour proxy, so it can miss faces | open (signals I3) |
| GPU decode starves the GPU | PoC-work note (unverified here): CPU-side JPEG decode was noted slower than GPU JPEG decode; a GPU lock was added in PoC work to share the GPU | not verified on this branch (PoC work; see `scripts/poc_demo.sh`) |
| Pico port permissions after reflashing MicroPython | Device node is recreated with a restricted group, so the port must be re-granted | operational (ledger) |
| Voicebank tests in an environment with the `sarvamai` SDK installed | Two tests assume the SDK is absent and fail; one also makes a real network call with an empty key | network-free test fix merged (PR #22) |

## What we deliberately do not claim

- **No certified error bounds.** The machinery exists; it does not produce a usable guarantee at n = 150 from 7 clips.
- **No accuracy claim.** The README records no accuracy numbers on this branch. The Egocentric-10K notes below are descriptive.
- **Egocentric-10K notes are descriptive.** Those windows have no human labels. The domain shift from the calibration footage (480p IndEgo, see `reports/edge.md:69`) is untested.
- **Labels for Egocentric-10K-Evaluation are AI-generated** (gemini-2.5-flash per ATTRIBUTIONS) and not human-checked.
- **Thresholds are placeholders** (`STUB_THRESHOLD = 0.9` in `jadal/stubs.py`) until calibrated on our own clips. No threshold on `main` is certified.
- **The batch rule is a fixed rule with no statistical guarantee** at small batch sizes.
- **Receipts are content-hashed, not signed.**
- **No Raspberry Pi compute measurement.** The Pico is a status light, verified on RP2040 only.
- **No hyperspectral imaging.** The cameras are RGB (3 bands).
- **No generic VLM in the pipeline,** and no Laya or Clef in the pipeline.
- **The face blur is not a privacy guarantee.**
- **Batch independence** is tested through pure-function proxies, not a real batch run.
- **Pico firmware is exercised on a real Raspberry Pi Pico (RP2040)** with MicroPython 1.29.0: heartbeat, LED, FLASH and PING/PONG verified over USB serial (`docs/reports/pico_hardware.md`). RP2350 is not tested.
- **The SOP is a labelled sample** (`sop/sop_garment_v1.md`), not a task specification taken from the footage.
- **Voice translations are machine drafts.** The Tamil header has a known typo, deliberately left for native-speaker review.
- **L1 determinism is not measured.**
