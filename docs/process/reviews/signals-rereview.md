ALL ADDRESSED: no, NEW BREAKAGE: none

# Re-review of PR #12 (feat/signals), fix round 1

Scope: fix diff 0abab2b..origin/feat/signals (saved at reviews/signals-fix1.diff). Worktree HEAD = origin/feat/signals = e78a9e6, clean status. No code, GitHub, or merge actions taken.

## Tests (run in worker tree)
`PYTHONNOUSERSITE=1 <env>/bin/python -m unittest discover -s tests -v`
Result: Ran 21 tests, OK, 0 skipped, exit 0. The mediapipe tests ran (not skipped) against the real models. Mediapipe `__del__` shutdown noise on stderr, as the report says. No `-s web` run: the worktree has no web/ directory.

## Report accuracy (docs/reports/signals.md, "Fix round 1")
The appended section matches the run: 21 tests, OK, 0 skipped. The timing differs (12.7s claimed, 13.3s on my run), which is expected. Stale lines outside the new section still contradict the fix:
- line 21: "The mediapipe path is UNTESTED (no model available)".
- line 17: "All 13 pass".
- line 4: old `{stem}_seg{NNNN}.mp4` naming.

## Original findings
- C1 (BaseOptions): ADDRESSED. signals.py:253 and :271 import from `mediapipe.tasks.python.core.base_options`. The bare except is gone. `test_face_detector_finds_drawn_face` passes.
- C2 (hand adapter): ADDRESSED. `_mp_hand_count` (signals.py ~312) uses HandLandmarker with num_hands=2 and IMAGE mode. `_count_hands` is used in compute_signals. `_mp_available` is removed. Proxy kept as fallback.
- C3 (config paths): ADDRESSED for paths. `_face_model_path` / `_hand_model_path` read config.py:50-51. Env access is only the `JADAL_MODELS_DIR` fallback the worker was told to add. Detectors and config are passed through. PARTIAL for the sha256: `backend_info()` exposes the sha256 (signals.py ~353), but no versions dict exists in this tree and nothing writes it.
- Important 1 (create once): ADDRESSED. Module-level caches `_FACE_DETECTORS` / `_HAND_LANDMARKERS` (signals.py:245-246, cached at :264 and :284).
- Important 2 (faces_blurred): ADDRESSED. compute_signals counts via `blur_faces` (signals.py:494-499). Definitions are in the docstring.
- Important 3 (privacy): ADDRESSED. Not-a-guarantee wording is in the signals.py docstring (:30-33), in blur_and_hash_segment (:452), and in the report fix section.
- Important 4 (batch test): ADDRESSED. `test_repeat_after_other_segments` replaces the proxy test and passes.
- Important 5 (open writers): ADDRESSED. One writer at a time, released on segment change (segmenter.py ~:120-135), `isOpened()` check raises (segmenter.py:99).
- Important 6 (segment_id collisions): ADDRESSED. `<stem>-<sha8>-<NNNN>` with sha8 from source bytes. `test_split_ids_unique_across_same_named_clips` passes.
- Minors 1-8, 10: ADDRESSED (rounding assert, k<=1 guard, brightness averaged, seg_for and _mp_available removed, stem sanitized, find_duplicate documented, worktree now clean).
- Minor 9 (report still says mediapipe untested): NOT ADDRESSED (see stale lines above).

## New Critical/Important breakage
None found.

## Deferred minors
1. compute_signals runs a full blur_faces (copy and blur) on each sampled frame just to count faces. Detection-only would be cheaper.
2. segment_id depends only on source bytes. Byte-identical clips from different workers or sites get the same id and file, and the second record's source_clip/site metadata is lost.
3. Stale report lines (21, 17, 4) listed above.
4. Test file hard-codes <repo>/models as MAIN_MODELS, which is not portable.
5. imagehash/Pillow pins still unverified (carried over from the original review).
6. Versions-dict wiring for the model sha256 (Critical 3 remainder).
