# signals module report (issue #6: L0 signals + segmenter)

## What exists
- `jadal/segmenter.py` — `split_clip(path, seg_seconds, out_dir, worker_id="", site_id="", source_clip=None, config=None) -> list[Segment]`. OpenCV decode, frame-window cut, mp4v re-encode with deterministic naming (`<stem>-<sha8>-<NNNN>.mp4`, sha8 from source bytes so same-named clips never collide), per-segment sha256. Returns `[]` for undecodable clips. At most one writer open at a time, low memory.
- `jadal/signals.py` — `compute_signals(path, config=None, index=None) -> Signals` with exactly the 12 SPEC fields; `blur_faces(frame, config=None) -> (frame, n_found, n_blurred)`; `phash_segment(path) -> str` (64-bit hex of middle sampled frame); `find_duplicate(phash, index, threshold=10) -> segment_id|None`; `blur_and_hash_segment(path, config) -> {path, sha256, faces_found, faces_blurred}` which streams every frame, blurs face-classified regions, and writes a SEPARATE video under `config.blurred_dir`. Original segment is only read. `backend_info() -> dict` (active backend, model paths + sha256) and `backend_versions() -> dict` (integration hook for receipt `config.versions`: stable key order, mediapipe/opencv versions, model sha256s, backend names). `analysis_diagnostics(path, config)` (analysis scale, clip/sampled fps) kept OUT of Signals.
- Metrics: 16 evenly-spaced sample frames read in one sequential pass; blur_var = resolution- and brightness-normalised variance of Laplacian (fixed 480-px analysis width via INTER_AREA, never upscaled, divided by (mean/128)^2); brightness = gray mean; under/over-exposed = gray<20 / >=245 pixel Fractions; motion_mean = mean Farneback flow magnitude in px/sec at 480-px width (fixed 8x8 grid, annualised by effective sampled fps); hands via mediapipe HandLandmarker when its model resolves (num_hands=2, IMAGE mode, CPU), else colour proxy; faces via mediapipe FaceDetector when its model resolves, else colour proxy; all floats rounded to `config.float_decimals` (4). `dup_of` filled via near-dup lookup when an index (phash->segment_id) is passed, else None. CPU only, no torch import, no network, no GPU use. Decision thresholds live in `jadal.config.DEFAULT` (placeholders pending calibration); the module returns raw metrics and hardcodes no decision threshold.
- `tests/test_signals.py` — 24 unittests (incl. 5 mediapipe tests against the real models, normalisation tests, `backend_versions` test) with synthetic videos generated in-test (mp4v). Scratch dirs under `data_runtime/test_signals_*` (gitignored), never /tmp.

## How to run
- `/home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v` (repo root of worktree; after the main merge this also runs other modules' tests)

## Test output tail (2026-10-10, real run)
```
Ran 13 tests in 9.263s
OK
```
(historical first green run; see Fix round 1/2 sections for current runs)
All 13 pass: keys/types, determinism+4-decimal rounding, undecodable, blur responds, brightness/exposure, static-vs-moving motion, faces/hands ranges + fixtures, phash stability + dup lookup, blur_faces unit, blur+hash separate file (byte-deterministic), segmenter basic + determinism, batch-independence proxy (same segment 5x identical).

## Known gaps / NOT DONE
1. **mediapipe models are git-ignored**, so a fresh checkout has no models: faces/hands fall back to the deterministic OpenCV YCrCb skin-region proxy (approximate; cannot read hand pose or verify faces). With the real models (`/home/parshu/projects/jadal-qa/models/`, or `$JADAL_MODELS_DIR`) the mediapipe path is active and TESTED (5 tests, incl. a synthetic drawn face detected at score ~0.78). `backend_info()` always reports which backend is active.
2. The written blurred file is a best-effort privacy aid, NOT a guarantee (missed faces possible); only face-classified regions are blurred (not hands). Must not be presented as a guarantee on a dashboard.
3. Segmenter re-encodes with mp4v (byte-deterministic on this build, verified); no stream-copy/GOP-exact cutting, no audio in segments. Byte-identical clips from different workers/sites share segment ids (documented; deferred by reviewer).

## Fix round 1 (review of PR #12)
Merged `origin/main` (config gained `face_detector_task` / `hand_landmarker_task`
pointing at `models/`; models verified present at
`/home/parshu/projects/jadal-qa/models/`: `blaze_face_short_range.tflite`,
`hand_landmarker.task`). Fixes, all in owned files:
- C1: `BaseOptions` now imported from `mediapipe.tasks.python.core.base_options`;
  the bare `except` that hid the failure is removed -- a configured-but-broken
  model raises loudly (tested), and the colour proxy is used only when no model
  file resolves. Active backend inspectable via `backend_info() -> dict`
  (backend name, resolved paths, model sha256 for the versions record); never
  added to the SPEC Signals dict.
- C2: mediapipe `HandLandmarker` path implemented (`num_hands=2`, IMAGE mode,
  CPU only via the default XNNPACK CPU delegate, fixed thresholds) for
  `hands_visible_frac` / `hand_count_mode`; proxy kept as documented fallback.
- C3: model paths come from config; when the configured path is absent,
  `$JADAL_MODELS_DIR/<filename>` is also accepted (documented in the module
  docstring). Detectors are created once and cached at module level, not per frame.
- `faces_found` = faces detected across sampled frames; `faces_blurred` =
  faces actually blurred there (measured via `blur_faces`, no longer a copy).
  `blur_and_hash_segment` reports all-frame totals for the written artifact.
  The written blurred file is a best-effort privacy aid, NOT a guarantee
  (docstring + here); it must not be presented as one on a dashboard.
- `test_batch_independence_proxy` replaced with `test_repeat_after_other_segments`
  (same segment alone vs after other segments => identical dict, exercising the
  shared cached detectors). `find_duplicate` self-match semantics documented.
- Segmenter: one writer open at a time (released on segment change, loud
  RuntimeError if unopenable); dead `seg_for` removed; `segment_id` is now
  `<stem>-<sha8>-<NNNN>` with sha8 from the source file bytes (unique across
  same-named clips; tested); brightness averaged over all sampled frames;
  `_sample_indices` guards k<=1; blur output stem sanitized.
- Committed the previously uncommitted test rewrite after fixing its 2 failures.

Command: `cd /home/parshu/projects/jadal-qa-wt/signals && PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v 2>&1 | tail -30`
Real output tail (2026-10-10):
```
Ran 21 tests in 12.722s
OK
```
21/21 pass, 0 skipped. The 5 mediapipe tests ran against the real models
(incl. face detector returning a box, score ~0.78, on the synthetic drawn face;
hand counts valid + deterministic; broken-model path raises). Mediapipe C++
`__del__` shutdown noise ("Exception ignored") appears on stderr at process
exit; harmless, tests still OK.
Remaining proposal for the config owner (config is read-only for this module):
no L0 mediapipe confidence-threshold fields exist, so thresholds are fixed
module constants (`MP_FACE_MIN_DETECTION_CONFIDENCE`,
`MP_HAND_MIN_DETECTION_CONFIDENCE`, `MP_HAND_MIN_PRESENCE_CONFIDENCE`).

## IndEgo real-data run (fix round 2)
Clips read read-only from `/home/parshu/projects/awing/data/indego/` (never
copied into the repo; attribution: IndEgo, Fraunhofer IPK, CC BY 4.0,
https://huggingface.co/datasets/FraunhoferIPK/IndEgo). All 7 catalog clips
(480x480, 10 fps fisheye head-cam), `seg_seconds=20` (113 segments total),
per-segment results in `eval/results/indego_signals_<clip>.json` (5-36 KB each:
segment + signals + diagnostics with analysis scale / sampled fps). Segments
were written transiently under gitignored `data_runtime/indego_segs/`.
Hands% = share of segments with `hands_visible_frac > 0`.

| clip | nseg | blur_var min/med/p95 | brightness med | motion med (px/s) | hands% | faces tot |
|---|---|---|---|---|---|---|
| User_15_410_0411_2_480 (ECU) | 35 | 541.6 / 880.1 / 1349.3 | 70.6 | 5.22 | 100% | 187 |
| User_1_412_2010_3_480 (ECU late) | 11 | 605.2 / 1394.7 / 1671.1 | 61.7 | 6.37 | 100% | 40 |
| User_1_User_15_411_2910_2_1_480 (desk) | 21 | 225.7 / 355.6 / 480.5 | 86.5 | 4.71 | 57% | 99 |
| User_1_H_pal_transport2_Nar_480 | 5 | 527.1 / 645.0 / 854.7 | 90.4 | 6.25 | 80% | 7 |
| User_1_H_pal_transport_Nar_480 | 13 | 302.2 / 463.1 / 1803.9 | 73.5 | 4.91 | 62% | 45 |
| User_1_2110_Organisation_480 | 15 | 359.7 / 841.3 / 2702.1 | 65.4 | 5.01 | 80% | 44 |
| User_1_User_16_451_3101_Logistics_4_480 | 13 | 181.2 / 388.2 / 858.6 | 74.7 | 5.05 | 38% | 68 |

Full min/med/p95 per signal per clip (from the JSONs): under_exposed_frac medians
0.16-0.32 (fisheye borders are dark); over_exposed_frac ~0 everywhere (max p95
0.089 on the bright desk clip); hand_count_mode median 0-2 depending on clip;
faces_blurred == faces_found everywhere by construction.
Calibration input for the orchestrator: with placeholder `blur_var_min=60`, ZERO
of 113 segments would flag BLUR (observed min 181.2); placeholder
`motion_max=12` sits just above the observed p95s (6.6-11.7 px/s).

Frame inspection (mid-frame PNGs of the 3 lowest / 3 highest blur_var segments,
viewed; PNGs under gitignored `data_runtime/inspect/`): the ranking is sensible.
Lowest (181-238): flat white door, empty room floor/table, corridor doorway --
low-detail scenes, correctly low. Highest (2029-3907): dark pallet-jack scene
with strong metal edges (scores HIGH *despite* darkness -- the brightness
normalisation working as designed), cluttered workbench with tools/cables, busy
workshop with hands at work. Darkness alone does not suppress the score.
Face check: the max-faces segment (15/16 frames) shows a clearly visible real
woman's face (matches the catalog's 182 s note) -- true-positive streak. But a
dark pallet scene with no visible face scored faces=3, so false positives occur;
face precision on this footage is UNMEASURED (no labelled face frames).

## Fix round 2 (review of PR #12, rereview)
- Stale report lines fixed (mediapipe TESTED, current test count, current
  `<stem>-<sha8>-<NNNN>` naming).
- C3 hook: `backend_versions() -> dict` (stable key order: faces_backend,
  hands_backend, face/hand model_sha256, mediapipe_version, opencv_version) +
  test; the integration owner puts it into receipt `config.versions`.
- SHARPNESS FIX: `blur_var` is now resolution- and brightness-normalised
  (gray -> INTER_AREA resize to fixed 480-px width, never upscaled, scale
  recorded in diagnostics -> variance of Laplacian / (mean/128 + 1e-3)^2;
  the divisor cancels any global brightness gain since the Laplacian is linear
  in intensity). Motion is fps-aware px/sec at 480-px width (mean grid flow
  magnitude x effective sampled fps). Measured: same texture at 1920 vs 480 px
  -> ratio 1.000; x0.25-darkened copy -> ratio 1.026 with unchanged sharpness
  class vs `DEFAULT.blur_var_min`. No decision thresholds hardcoded: raw metrics
  only; tests use `DEFAULT.blur_var_min` as the class boundary.
- New tests: resolution invariance, brightness invariance (+class stability),
  `backend_versions` (key order, repeat equality, agreement with backend_info).

Command: `cd /home/parshu/projects/jadal-qa-wt/signals && PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v 2>&1 | tail -30`
Real outcome (2026-10-10): after the main merge, `discover -s tests` also runs
other modules' tests (test_certify, test_service, ...). The run reached 124
passed / 8 errors, then hung indefinitely in
`test_service.TestServiceAcceptance.test_websocket_pushes_batch_event`
(`ws.receive_json()` with no timeout; no batch event ever arrives). The 8
errors are all `test_service.TestPipeline.test_reason_codes_for_bad_signals`
(`ValueError: run_batch needs at least one segment`): those tests write fake
text bytes as `.mp4` and `pipeline.process_clip` calls the REAL `split_clip`,
which correctly returns [] for non-video input. Both are service/pipeline-side
integration debt (their files, not mine) -- reported, not fixed here.
Zero failures/errors in `test_signals`.
Module run (same env): `PYTHONNOUSERSITE=1 .../bin/python -m unittest tests.test_signals`
REAL tail: `Ran 24 tests in 15.584s / OK` (24/24, 0 skipped; incl. 5 mediapipe
tests on real models and the 3 new normalisation/versions tests).
