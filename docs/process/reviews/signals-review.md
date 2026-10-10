SPEC: FAIL, QUALITY: CHANGES NEEDED

Scope: branch feat/signals (PR #12, issue #6), reviewed from the committed tree (origin/feat/signals a2b2a76). The worker's live tree at jadal-qa-wt/signals has an UNCOMMITTED rewrite of tests/test_signals.py that fails (1 failure, 1 error, 17 tests). It is not part of the PR, but it must be committed or reverted before merge.

## Test run (committed branch)
Command: PYTHONNOUSERSITE=1 <env>/bin/python -m unittest discover -s tests -v (run on a git-archive export of origin/feat/signals; the export was deleted afterwards)
Result: Ran 13 tests in 8.760s, OK.
Worker tree (uncommitted test file): Ran 17 tests, FAILED (failures=1, errors=1). test_phash_and_duplicate: find_duplicate("f"*16, index) returns 'seg_far'. test_blur_and_hash_segment: AttributeError, TestBlurFaces has no _big_face.

## Spec compliance
1. split_clip: PASS. segmenter.py:112 fields match Segment; naming {stem}_seg{NNNN}. Extra optional kwargs are fine.
2. compute_signals returns exactly the 12 SPEC fields: PASS (signals.py:330-383). But faces_blurred is a copy of faces_found (signals.py:380), not a count of regions actually blurred. See Important 2.
3. Determinism: PASS in structure (fixed sampling, 4-decimal rounding, no RNG). Test coverage is weak (Minor 1).
4. OpenCV decode, no torch/torchcodec, no GPU: PASS.
5. Blur output in a separate file with its own sha256, original only read: PASS (signals.py:287-326). Signals computed on original frames: PASS.
6. pHash per segment + find_duplicate: PASS.
7. Synthetic in-test videos with real assertions: PARTIAL (Important 4; Minor 1).
mediapipe requirement (hands_visible_frac and hand_count_mode via mediapipe; faces via mediapipe): FAIL. Face adapter is broken (Critical 1). Hand adapter does not exist (Critical 2).

## Mediapipe verdict
The two downloaded models load and run correctly in <env> with the mediapipe tasks API (verified):
  from mediapipe.tasks.python import vision
  from mediapipe.tasks.python.core.base_options import BaseOptions
  vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=".../blaze_face_short_range.tflite")))
  vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=".../hand_landmarker.task")))
Both load and detect() runs on a blank frame. The structure is partly pluggable (env var for the face path, colour fallback kept), but the implementation is not wired correctly. Required changes are Critical 1-3.

## Critical
1. signals.py:215 (and 225). `mp_core.BaseOptions` does not exist: `mediapipe.tasks.python.core` has no BaseOptions attribute (verified). The AttributeError is swallowed by the bare `except Exception: return None` at signals.py:225, so with JADAL_FACE_DETECTOR_TASK set the code silently falls back to the skin proxy. Verified: `_mp_face_boxes` returns None with the env var set to the real model. Fix: `from mediapipe.tasks.python.core.base_options import BaseOptions` and `BaseOptions(model_asset_path=model)`. Narrow the except and log the exception, or re-raise when a model path is configured.
2. signals.py:16-17 (docstring) and docs/reports/signals.md:21 claim a hand adapter driven by JADAL_HAND_LANDMARKER_TASK. No such code exists (the env var appears only in docs and docstring). The per-frame hand count (signals.py:361-364) is the skin-blob proxy, which contradicts the SPEC (hands via mediapipe) and the report. Fix: add `_mp_hand_counts(bgr) -> Optional[int]` using vision.HandLandmarker (num_hands=2, min_hand_detection_confidence=0.5, RunningMode.IMAGE) and use it in compute_signals in place of the blob count. Delete `_mp_available` (signals.py:198, dead). Keep the proxy as fallback.
3. Model paths should come from config, not env vars alone. Config is read-only for module agents, so propose in jadal/config.py (ROOT is the repo root, RUNTIME = ROOT / "data_runtime"):
     face_detector_task: Path = field(default_factory=lambda: ROOT / "models" / "blaze_face_short_range.tflite")
     hand_landmarker_task: Path = field(default_factory=lambda: ROOT / "models" / "hand_landmarker.task")
   Pass config into _detect_faces and the hand adapter instead of reading os.environ. Record the model sha256 in the versions dict (SPEC: "Record versions ... weight sha256").

## Important
1. signals.py:202-226 and 231: `_mp_face_boxes` creates a new FaceDetector on every call, and blur_and_hash_segment calls it per frame (signals.py:314 loop). Create the detector once per model path and reuse it. Otherwise model load cost dominates runtime.
2. signals.py:380 `"faces_blurred": int(faces_found)`: a copy, not a measurement, and compute_signals does not blur. Also faces_found sums detections across the 16 sampled frames (signals.py:361), while blur_and_hash_segment sums across all frames (signals.py:314), so the two counts are not comparable. Either compute faces_blurred via blur_faces on the sampled frames, or document the definition.
3. Privacy: blur_and_hash_segment (signals.py:287-326) blurs only colour-proxy "face" blobs (classification at signals.py:93-97). The YCrCb range is fixed for light skin tones and size/aspect heuristics miss most real faces. The output is not a reliable face blur, but docs/reports/signals.md (line 5) presents it as one. Until Critical 1 is fixed, the report must say the blurred output is NOT a privacy guarantee, and the dashboard must not expose it.
4. tests/test_signals.py:207-212 `test_batch_independence_proxy` calls the same deterministic function 5 times on one file. It does not test batching, which is the SPEC requirement (one segment alone vs in a batch of 5, identical receipt_sha256). Relabel it as a repeat test; the real batch test belongs to pipeline/receipts.
5. segmenter.py:77-98 keeps ALL segment VideoWriters open until the loop ends (writers[seg] created at line 84, released at line 98). For a long clip (10 min, 20 s segments = 30 open encoders) this holds every encoder at once. docs/reports/signals.md (line 4) claims "one VideoWriter per segment, low memory". Fix: release the previous writer when seg changes and drop the dict. Check vw.isOpened() and fail loudly if not.
6. segmenter.py:58 and 112: segment_id = `{stem}_seg{NNNN}` is NOT unique across clips. Two clips with the same filename from different workers or sites produce the same segment_id and overwrite each other's files in out_dir (segmenter.py:106 then hashes another clip's file). Fix: include a short hash of source_clip in the stem, deterministically.

## Minor
1. tests/test_signals.py:76 `self.assertEqual(s := f"{a[k]}", s[: s.index(".") + 5] if "." in s else s)` compares a string with its own truncation. It detects unrounded floats but is unreadable. Replace with `self.assertEqual(a[k], round(a[k], 4))`.
2. signals.py:121 `_sample_indices` divides by (k-1); k=1 raises ZeroDivisionError. Guard `if k <= 1`.
3. signals.py:351 brightness uses only grays[0] while blur, exposure and motion use all sampled frames. The module docstring says aggregated metrics use 16 frames. Average it or document it.
4. segmenter.py:79 and 93: `seg_for` is written but never read (dead code). Remove it.
5. signals.py:198-199 `_mp_available` is dead code (see Critical 2).
6. signals.py:296 `stem = Path(path).stem` is not sanitized (segmenter.py:58 does sanitize). Output path is service-generated, so risk is low, but make it consistent.
7. signals.py:249-250 blur_faces clips boxes to the frame correctly; keep that.
8. find_duplicate (signals.py:269) does not exclude the segment's own entry. If the pipeline inserts a segment's phash before lookup, every segment matches itself. Also phash is used as a dict key, so two segments with the same phash silently overwrite each other in the index. Document or key by segment_id.
9. docs/reports/signals.md (line 21) says the mediapipe path is untested because no model is available. The models are available (verified), so after Critical 1-3 this becomes a measured result; the report needs a run output.
10. The PR branch has no record of how the worker's live tree (uncommitted test rewrite) relates to the PR. Commit or revert before merge.

## Not checked
- imagehash and Pillow are imported (signals.py:28-37) but I did not find them in any requirements or pyproject file in the worktree. Confirm they are pinned before merge.
- data_runtime/ is in .gitignore (confirmed). Tests write to data_runtime/test_tmp, not /tmp (confirmed).
- No network calls at import time, no secrets (confirmed).
- No edits to jadal/config.py, jadal/__init__.py, or docs/SPEC.md in the branch (confirmed by diff stat).
- No GitHub comments, merges, pushes, or code edits were made. The scratch export used for testing was deleted after the run.

## Required before merge (in order)
1. Fix Critical 1 and 2; add the config fields from Critical 3.
2. Re-run tests with the real model paths set; add a test asserting the mediapipe face path returns a box on a synthetic face (skip only if the model file is absent, never silently pass).
3. Fix Important 5 and 6.
4. Commit or revert the worker's uncommitted tests/test_signals.py.
