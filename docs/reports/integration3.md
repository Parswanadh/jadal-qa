# integration3 — baseline-v1 round 3 (issue #55): real modules end-to-end

Owner files touched: `jadal/pipeline.py`, `jadal/service.py`,
`scripts/demo.sh`, `scripts/gen_api_examples.py`, `docs/api_examples/*`,
`tests/test_integration.py`, `tests/test_service.py`, this report.
Not touched: `jadal/signals.py`, `jadal/decision*.py`, `jadal/funnel.py`,
`jadal/certify.py`, `jadal/receipts.py`, `jadal/config.py`,
`jadal/__init__.py`, `docs/SPEC.md`, `docs/BASELINE.md`, `README.md`,
`models/*` (weights are git-ignored; staged locally for the run).

## What changed and why

1. **`scripts/demo.sh` proves the REAL pipeline (B2).** `JADAL_USE_STUBS`
   set => refuse. Missing model files => clear message + exit 1 (never
   stubs silently). Flow: preflight (models + funnel-thresholds hash) ->
   GPU-locked service phase (dock 7 IndEgo clips -> real segmenter ->
   real L0/mediapipe -> real EG2 decider -> `funnel.route` -> batches ->
   receipts) -> verdict table, `receipt_problems()` empty on EVERY
   receipt (else exit 1), re-verify MATCH on every receipt via the API,
   tampered receipt FAILS `scripts/verify_receipt.py`, clean receipt
   passes, dashboard `/` serves, `/pico/status` answers -> GPU-locked
   determinism phase (10x identical, alone-vs-batch-of-5 identical,
   per-stage latency) -> offline assembly of
   `eval/results/integration3_demo.json` (no lock held).
2. **`config.versions` records the full B2 block** (`pipeline.impl_versions`):
   implementation names + last-commit git sha per file (signals, decision,
   segmenter, funnel, certify, receipts) + repo HEAD; `device` (cpu/cuda)
   + `gpu_name`; torch/opencv/mediapipe/transformers/numpy versions;
   `signals_backend`; face/hand task `name:sha256` tokens; EG2
   `model.safetensors` sha256 + decider head tokens (all weight hashes
   cached per process); python, platform, machine arch. Best-effort keys
   only: anything missing is an absent key, never an exception.
3. **Fail-closed stub/fallback detector** (`pipeline.receipt_problems`,
   unit-tested with crafted receipts): stub-named stages, non-real
   funnel, non-mediapipe L0 backend, decider `heads=missing` (0.5 silent
   defaults), 0 decoded frames on a decodable segment. The demo exits
   non-zero naming the offending receipts.
4. **Decider instance cache** (`pipeline._get_decider`, keyed by class
   object + decimals): the EG2 backbone (~1.4 G) loads once per process
   instead of once per batch/re-verify. Keyed by class object so injected
   test fakes never leak across tests.
5. **API examples regenerated** (`scripts/gen_api_examples.py`): the
   receipt fixture is now picked by smallest `segment_id` instead of
   `receipt_ids[0]` (which is receipt-hash order and reshuffles whenever
   the versions block gains keys); `tests/test_service.py` drops the new
   weight-dependent keys (`eg2_*`, `decider_*`) in the drift comparison.

## Results (every number from `eval/results/integration3_demo.json`)

FINAL 17:35 IST (demo run 3, after the ingest-guard fix):

- `n_clips=7` -> `n_batches=23` -> `n_receipts=113`,
  `verdict_counts={review: 113}`, `batch_decisions={ACCEPT: 23}`.
  All-review is the placeholder calibration (accept gate closed), not a
  failure; the batch rule (`reject if rejected_count > k`, k=1) accepted
  every batch with zero rejects.
- No-stub assertion: `receipt_problems()` empty on all 113 receipts.
- Re-verify: MATCH 113/113 via `POST /receipts/{id}/reverify`, all with
  `versions_match=True`. Tampered receipt FAILS `verify_receipt.py`
  (exit 1); clean receipt passes (exit 0).
- Determinism (GPU, `device=cuda`, demo run 3 determinism phase): 10
  repeated full runs on the 94.5 s clip (5 segments) => IDENTICAL
  receipt_sha256 per segment (`2c8e09ed8d2b2d99...`); alone (batch of
  1) vs in a batch of 5 => IDENTICAL. The same checks were proven
  independently on the CPU fallback earlier (`45e6611ade3bbbe5...`,
  re-verify 5/5, tamper exit 1) — the two hashes differ only in the
  recorded backbone dtype string (bfloat16 vs float32); each path is
  internally bit-identical, cross-device equality is NOT claimed.
- Latency (GPU run, steady-state, embedding cache warm): segmenter
  0.745 s/segment, signals 2.347 s/segment, decider 1.079 s/segment,
  route ~0.0 s/segment => 1051.0 segments/hour; dock-to-drained wall
  249.4 s for 7 clips on this x86 laptop (shared GPU, NOT a Pi).
  (CPU fallback measured separately: 1742.3 seg/hour cache-warm.)
- `device=cuda:NVIDIA GeForce RTX 4070 Laptop GPU`, `arch=x86_64`
  in every receipt (cross-architecture determinism NOT claimed).
- Suite: `Ran 564 tests in 212.654s -- OK (skipped=3)` (37.8 s
  unloaded); fresh clone: `Ran 564 tests in 63.523s -- OK (skipped=9)`
  with no models, and the demo refuses without models (exit 1).
- GPU spot check (under flock, `Decider(..., use_cache=False)`):
  same frame embedded twice fresh on CUDA => max abs diff `0.0`
  (bit-identical, per-frame batch size 1 as SPEC requires);
  `Decider.decide` twice => identical dicts. First encode 8.96 s
  (model load), second 0.10 s; decide 1.4 s/segment. Heads:
  `hands_ok=3d9b5c0b8da0ea9e,active_work=260657b5a9bb0dd8`
  (full-splits provenance), step prototypes present, frames_decoded 4.
- Fresh-clone quickstart (PROVEN 17:00 IST): `git clone
  https://github.com/Parswanadh/jadal-qa.git` + `git checkout
  feat/integration3`, suite green with NO models (9 skips, nothing
  fails); `bash scripts/demo.sh` without models prints the per-file
  MISSING list + fetch pointers and exits 1, never stubs silently.

Superseded interim notes: run 1 gave 115 receipts (114 review /
1 reject) before the key-spelling assertion bug (fixed); run 2 gave
116 receipts with 3 transient `segment_sha256` MISMATCHes (the ingest
race, fixed — see section below). Run 3 is the clean record.

- Command: `flock -w 3600 tools/gpu.lock bash scripts/demo.sh`
  (demo holds the lock only across the service phase and the
  determinism phase; JSON assembly is offline).

## Determinism incident: concurrent same-clip ingest tore segment bytes

Demo run 2 (full service phase, 7 clips -> 116 receipts) re-verified
113/116 MATCH and 3 MISMATCH, all with `diff == ['segment_sha256']`
(signals/decisions/verdicts identical: the bytes changed under a
pixel-identical decode, i.e. a rewrite). Evidence of concurrent
processing of the same bytes: 31 batches closed in ~2 min (three of
them 0.25 s apart — impossible sequentially at ~2 s/segment signals +
~1.4 s/segment decider), 116 receipts over 113 stored segments.
Mechanism: `handle_clip` marked a clip ingested only AFTER the
minutes-long `process_clip` returned, so the dock POST and the bucket
watcher (or two request threads) ran the segmenter concurrently into
the same segment files. Fixed in owned `jadal/service.py`: an
`_in_flight` set of real paths under the service lock; a concurrent
second ingest of the same path returns the duplicate shape instead of
reprocessing (regression test
`TestConcurrentIngest.test_concurrent_same_clip_processes_once`).
Demo run 3 (after the fix) is the clean B2/B3 record below.

Note on `versions_match=False` in run 2: the ONLY differing key was
`repo_git_sha` — interim report commits landed mid-run and moved HEAD.
The receipt hashes still MATCHED (versions are stored verbatim); the
flag correctly reported the stack moved. No commits land mid-run again.

## Defects in other modules (not fixed; minimal reproductions)

1. **`duration_s` never reaches `funnel.route`: every receipt carries
   `missing:duration_s` LOW_CONFIDENCE evidence.** `compute_signals`
   returns neither `duration_s` nor `segment_duration_s`, so
   `route`'s `need_float("duration_s")` always records a missing-field
   evidence entry (and `missing_l0=True` forces review: auto-accept is
   unreachable even after calibration). Observed on real receipts:
   `evidence[1] = {check: missing:duration_s, ...}` on all 115 demo
   receipts. This does not affect B3 determinism (the missing entry is
   itself deterministic), but the funnel worker must decide: either the
   L0 signals schema gains a duration field (SPEC change) or the
   `min_duration_s` rule reads it from the segment. Not fixed here:
   stuffing duration into the SPEC-fixed signals dict from the pipeline
   would break the schema (not owned on either side).
2. **`jadal/signals.py::backend_info` reports "mediapipe" from file
   presence alone, without checking the import.** Repro:
   `sys.modules['mediapipe'] = None; signals.backend_info()` =>
   `{'faces': 'mediapipe', ...}` even though no detector can construct.
   Consequence: `scripts/edge_check.py` mediapipe-blocked probe reports
   `fallback_works=False`. Integration impact: none for B2 (models are
   present here, detectors construct); a broken mediapipe install would
   surface as UNDECODABLE verdicts via `_compute_signals_safe`, never as
   silent proxy numbers. Owner: signals worker. Suggested fix (1 line):
   `backend_info` should try `import mediapipe` and report `proxy` on
   failure. Not applied (file not owned).
2. **`scripts/edge_check.py` FAILs with `runtime=['matplotlib']`.**
   No `jadal/*` module imports matplotlib; the import arrives via
   `mediapipe` itself (upstream dependency). The edge allowlist needs an
   auditor decision (allow transitive matplotlib vs lazy import in
   signals). Observed command/output:
   `PYTHONNOUSERSITE=1 .../bin/python scripts/edge_check.py` =>
   `FAIL: forbidden modules imported: import-time=[] runtime=['matplotlib']`
   plus the mediapipe-blocked probe SKIP above. Not owned, not fixed.
   B9 numbers that DID measure (same run): edge modules import in
   0.738 s; `compute_signals` probe 1.883 s/segment, peak RSS 252.9 MB,
   backend mediapipe/mediapipe; label `x86-laptop-proxy`
   ("NOT measured on a Raspberry Pi").

## Models / reproducibility (README patch proposed, not applied)

`models/` is git-ignored. The demo needs, and refuses to run without:
`models/blaze_face_short_range.tflite` and `models/hand_landmarker.task`
(from `https://storage.googleapis.com/mediapipe-models/...`, see
`models/README.md`), `models/decider_hands.npz`,
`models/decider_work.npz`, `models/decider_step_prototypes.npz`
(built by the decider worker: `eval/train_heads.py` on the fixed
splits), and the backbone at `/home/parshu/projects/awing/data/eg2`
(`model.safetensors`, EmbeddingGemma 2). Proposed README quickstart
addition (for the README owner): document these paths + the demo's
preflight failure as the absent-models UX. Fresh-clone test: TBD.

## NOT DONE / caveats

- Thresholds are placeholders (`configs/funnel_thresholds.json`,
  status `placeholder`): verdicts prove plumbing + determinism, not
  calibrated quality. Calibration is the funnel worker's lane.
- Determinism is claimed for this machine + pinned stack at batch size
  1 only (arch in every receipt).
- The demo holds the GPU lock across the whole service phase (the
  decider runs inside the service on demand); per-phase release between
  service and determinism sections. Other GPU jobs queue meanwhile.
- If CUDA OOMs under contention, the fallback is retry-later (the
  Decider currently requires CUDA: `Decider backbone requires CUDA`;
  a CPU path would be a decider-worker change, recorded here as not
  attempted).
