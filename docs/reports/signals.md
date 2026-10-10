# signals module report (issue #6: L0 signals + segmenter)

## What exists
- `jadal/segmenter.py` — `split_clip(path, seg_seconds, out_dir, worker_id="", site_id="", source_clip=None, config=None) -> list[Segment]`. OpenCV decode, frame-window cut, mp4v re-encode with deterministic naming (`{stem}_seg{NNNN}.mp4`), per-segment sha256. Returns `[]` for undecodable clips. Streaming (one VideoWriter per segment), low memory.
- `jadal/signals.py` — `compute_signals(path, config=None, index=None) -> Signals` with exactly the 12 SPEC fields; `blur_faces(frame) -> (frame, n_found, n_blurred)`; `phash_segment(path) -> str` (64-bit hex of middle sampled frame); `find_duplicate(phash, index, threshold=10) -> segment_id|None`; `blur_and_hash_segment(path, config) -> {path, sha256, faces_found, faces_blurred}` which streams every frame, blurs face-classified regions, and writes a SEPARATE video under `config.blurred_dir`. Original segment is only read.
- Metrics: 16 evenly-spaced sample frames read in one sequential pass; blur = variance of Laplacian (mean over sampled frames); brightness = gray mean; under/over-exposed = gray<20 / >=245 pixel Fractions; motion = Farneback optical flow mean on a fixed 8x8 grid between consecutive sampled frames; all floats rounded to `config.float_decimals` (4). `dup_of` filled via near-dup lookup when an index (phash->segment_id) is passed, else None. CPU only, no torch import, no network, no GPU use.
- `tests/test_signals.py` — 13 unittests with synthetic videos generated in-test (mp4v). Scratch dirs under `data_runtime/test_tmp` (gitignored), never /tmp.

## How to run
- `/home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v`

## Test output tail (2026-10-10, real run)
```
Ran 13 tests in 9.263s
OK
```
All 13 pass: keys/types, determinism+4-decimal rounding, undecodable, blur responds, brightness/exposure, static-vs-moving motion, faces/hands ranges + fixtures, phash stability + dup lookup, blur_faces unit, blur+hash separate file (byte-deterministic), segmenter basic + determinism, batch-independence proxy (same segment 5x identical).

## Known gaps / NOT DONE
1. **mediapipe Hands / FaceDetector NOT wired as default.** This env ships mediapipe 1.1.0 with only the `tasks` API and NO offline `.task` model files (verified: no .tflite/.task in site-packages; cv2 5.0.0 bundles NO Haar cascade XMLs; FaceDetectorYN needs a downloaded ONNX). Downloads are disallowed. So faces/hands counts use a deterministic offline OpenCV YCrCb skin-region proxy that partitions blobs by size/aspect (`_skin_blobs`, `_detect_faces`). The colour proxy cannot read hand pose or verify faces — counts are approximate.
2. Guarded mediapipe adapters exist (`_mp_face_boxes`, env `JADAL_FACE_DETECTOR_TASK` / `JADAL_HAND_LANDMARKER_TASK`): if valid model paths are supplied they are used for face boxes; otherwise the colour fallback runs. The mediapipe path is UNTESTED (no model available) — flagged in the module docstring.
3. Only face-classified regions are blurred in the blurred output (hand skin regions are not). Blurred dir/files land under `data_runtime/blurred` (gitignored).
4. Segmenter re-encodes with mp4v (byte-deterministic on this build, verified); no stream-copy/GOP-exact cutting, no audio in segments.
