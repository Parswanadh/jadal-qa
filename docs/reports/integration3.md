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

STATUS 16:45 IST: the full demo is queued on the shared-GPU lock behind
four other workers' holds (frames10k, gpu-throughput, spectral-E, gpu-e2e;
this demo waited 14+ min at last check) and has not yet completed. Below
is what is PROVEN so far; the JSON + full B3 numbers land when the lock
frees, before 17:50. Nothing below is projected: each line says how it
was measured.

- Suite: `PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests`
  => `Ran 564 tests in 212.654s -- OK (skipped=3)` (no GPU, no network;
  slow wall time is shared-machine CPU contention; 37.8 s unloaded).
- Fresh-clone quickstart (goal 5, PROVEN 17:00 IST): `git clone
  https://github.com/Parswanadh/jadal-qa.git` + `git checkout
  feat/integration3` outside the repo, then the README commands:
  suite => `Ran 564 tests in 63.523s -- OK (skipped=9)` with NO models
  present (9 skips vs 3: the model-gated tests skip cleanly, nothing
  fails); `bash scripts/demo.sh` without models prints the per-file
  MISSING list + fetch pointers and exits 1 (verified `demo exit=1`),
  never stubs silently. models/ contents are git-ignored by design.
- Real-pipeline processing (demo run 1, killed by the no-stub assertion
  on a key-spelling bug of mine, since fixed): 7 clips -> 115 segments
  -> 115 receipts through real segmenter + real L0 (mediapipe) + real
  EG2 decider + `funnel.route`. After the fix, `receipt_problems()` is
  `[]` on every one of those 115 receipts (checked offline from the
  saved `data_runtime/demo_receipts.json` before run 2 wiped it).
  Verdict split: 114 review / 1 reject (placeholder calibration gates
  every accept to review; expected, not a failure).
- GPU spot check (under flock, `Decider(..., use_cache=False)`):
  same frame embedded twice fresh on CUDA => max abs diff `0.0`
  (bit-identical, per-frame batch size 1 as SPEC requires);
  `Decider.decide` twice => identical dicts. First encode 8.96 s
  (model load), second 0.10 s; decide 1.4 s/segment. Heads:
  `hands_ok=3d9b5c0b8da0ea9e,active_work=260657b5a9bb0dd8`
  (full-splits provenance), step prototypes present, frames_decoded 4.
- `device` recorded as `cuda:NVIDIA GeForce RTX 4070 Laptop GPU`;
  `arch: x86_64` (cross-architecture determinism explicitly NOT claimed).
- PENDING (demo run 2, same code + fixed assertion): re-verify MATCH on
  all receipts via the API, tampered receipt FAILS, 10x identical
  receipt_sha256 per segment, alone-vs-batch-of-5 identical,
  `eval/results/integration3_demo.json` with per-stage latency.
- B3 PROVEN ON CPU 17:15 IST (GPU lock still queued; house-rule CPU
  fallback, `device=cpu` recorded): `CUDA_VISIBLE_DEVICES=""` +
  `.scratch/cpu_determinism.py` (EG2 embedding cache warm from the GPU
  run, `_get_model` guarded to abort loudly on any cache miss — none
  occurred; torch confirms `cuda.is_available()==False`, so no GPU was
  touched and no lock was needed) on the shortest IndEgo clip
  (94.5 s -> 5 segments, real segmenter/signals/decider/funnel):
  10 repeated full runs => IDENTICAL receipt_sha256 on all 5 segments
  (hash `45e6611ade3bbbe5...`); alone (batch of 1) vs in a batch of 5
  => IDENTICAL; re-verify (pinned pipeline on the segment bytes) =>
  MATCH 5/5; tampered receipt => `verify_receipt.py` exit 1 (hash
  MISMATCH named), clean receipt => exit 0. Per-stage latency:
  signals 1.523 s/seg, decider 0.544 s/seg (cache-warm), 1742.3
  segments/hour steady-state on this x86 laptop. Source:
  `.scratch/cpu_det.json` (regenerable with the command above; its
  contents fold into `eval/results/integration3_demo.json` when the
  GPU demo completes). The remaining GPU-run claim is same-hashes-on
  -cuda; the CPU proof stands on its own with `device=cpu`.

- Command: `flock -w 3600 tools/gpu.lock bash scripts/demo.sh`
  (demo holds the lock only across the service phase and the
  determinism phase; JSON assembly is offline).

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
