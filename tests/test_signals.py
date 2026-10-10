"""Unittests for jadal.signals (L0) and jadal.segmenter (issue #6).

Test videos are generated synthetically with OpenCV. All scratch output is kept
under <repo>/data_runtime/ (git-ignored) -- never /tmp, per docs/SPEC.md.

Detection backends: face/hand tests are backend-aware. The mediapipe tests run
when the model files resolve (config paths or $JADAL_MODELS_DIR, else the main
repo models dir below) and SKIP only when no model file exists -- they never
silently pass. The colour-proxy fallback is unit-tested directly and always runs.
"""
from __future__ import annotations

import hashlib
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from jadal import segmenter, signals
from jadal.config import DEFAULT

ROOT = Path(__file__).resolve().parents[1]
MAIN_MODELS = Path("/home/parshu/projects/jadal-qa/models")
W, H, FPS, N = 320, 240, 30, 90
SKIN = (90, 140, 220)          # BGR inside the YCrCb skin range
SIGNAL_KEYS = {
    "decodable", "blur_var", "brightness_mean", "under_exposed_frac", "over_exposed_frac",
    "motion_mean", "hands_visible_frac", "hand_count_mode", "faces_found", "faces_blurred",
    "phash", "dup_of",
}
SEG_KEYS = {
    "segment_id", "source_clip", "worker_id", "site_id",
    "start_s", "end_s", "path", "sha256",
}


def _bg() -> np.ndarray:
    f = np.full((H, W, 3), 128, np.uint8)
    f[:, :, 0] = 200  # bluish, NOT in the skin range
    return f


def _write_video(path: Path, frame_fn) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert vw.isOpened(), "VideoWriter failed to open"
    try:
        for i in range(N):
            vw.write(frame_fn(i))
    finally:
        vw.release()


def _sharp(i: int) -> np.ndarray:
    f = _bg()
    f[0:24, 0:24] = np.random.default_rng(i).integers(0, 255, (24, 24, 3)).astype(np.uint8)
    return f


def _moving_skin(i: int) -> np.ndarray:
    f = _sharp(i)
    cv2.circle(f, (min(W - 1, 30 + i * 2), 120), 18, SKIN, -1)  # hand-sized skin blob
    return f


def _big_face(i: int) -> np.ndarray:
    f = _sharp(i)
    cv2.circle(f, (160, 120), 30, SKIN, -1)  # face-sized round skin blob (proxy only)
    return f


def _drawn_face(i: int) -> np.ndarray:
    """Synthetic face detected by BOTH the mediapipe face model and the proxy."""
    f = np.full((H, W, 3), 60, np.uint8)
    cv2.ellipse(f, (160, 125), (55, 70), 0, 0, 360, (96, 148, 228), -1)
    cv2.ellipse(f, (138, 105), (10, 7), 0, 0, 360, (30, 30, 30), -1)
    cv2.ellipse(f, (182, 105), (10, 7), 0, 0, 360, (30, 30, 30), -1)
    cv2.ellipse(f, (160, 165), (16, 8), 0, 0, 180, (40, 20, 20), -1)
    return f


def _hand_bar(i: int) -> np.ndarray:
    f = _bg()
    cv2.rectangle(f, (40, 150), (170, 190), SKIN, -1)  # wide -> not face-like
    return f


def _model_paths():
    """(face_model, hand_model) or (None, None) when no model file resolves."""
    face = MAIN_MODELS / "blaze_face_short_range.tflite"
    hand = MAIN_MODELS / "hand_landmarker.task"
    if face.is_file() and hand.is_file():
        return face, hand
    return None, None


class SignalsBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(dir=str(ROOT / "data_runtime"), prefix="test_signals_"))
        self.cfg = replace(
            DEFAULT,
            blurred_dir=self.tmp / "blurred",
            segments_dir=self.tmp / "segments",
        )
        face, hand = _model_paths()
        if face is not None and hand is not None:
            self.cfg = replace(self.cfg, face_detector_task=face, hand_landmarker_task=hand)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _path(self, name: str) -> Path:
        return self.tmp / name

    def _mp4(self, name: str, frame_fn) -> Path:
        p = self._path(name)
        _write_video(p, frame_fn)
        return p


class TestSignals(SignalsBase):
    def test_signals_contract(self) -> None:
        s = signals.compute_signals(self._mp4("contract.mp4", _moving_skin), self.cfg)
        self.assertEqual(set(s), SIGNAL_KEYS)
        self.assertIsInstance(s["decodable"], bool)
        self.assertTrue(s["decodable"])
        self.assertIsInstance(s["hand_count_mode"], int)
        self.assertIn(s["hand_count_mode"], (0, 1, 2))
        self.assertEqual(s["faces_blurred"], s["faces_found"])
        self.assertGreaterEqual(s["faces_found"], 0)
        self.assertGreaterEqual(s["hands_visible_frac"], 0.0)
        self.assertLessEqual(s["hands_visible_frac"], 1.0)
        for k in ("blur_var", "brightness_mean", "under_exposed_frac", "over_exposed_frac",
                  "motion_mean", "hands_visible_frac"):
            v = s[k]
            self.assertIsInstance(v, float)
            self.assertEqual(v, round(v, 4), f"{k} not rounded to 4 decimals: {v}")
        self.assertIsNone(s["dup_of"])

    def test_determinism(self) -> None:
        p = self._mp4("det.mp4", _moving_skin)
        self.assertEqual(signals.compute_signals(p, self.cfg), signals.compute_signals(p, self.cfg))

    def test_repeat_after_other_segments(self) -> None:
        """Same segment processed alone vs after other segments => identical dict.

        Exercises the shared cached detectors: any cross-call state leak would
        show up here. (The alone-vs-batch-of-5 receipt test belongs to
        pipeline/receipts; this is the L0-level determinism guarantee.)
        """
        a = self._mp4("rep_a.mp4", _drawn_face)
        b = self._mp4("rep_b.mp4", _moving_skin)
        c = self._mp4("rep_c.mp4", _sharp)
        alone = signals.compute_signals(a, self.cfg)
        signals.compute_signals(b, self.cfg)
        signals.compute_signals(c, self.cfg)
        signals.blur_and_hash_segment(b, self.cfg)
        self.assertEqual(signals.compute_signals(a, self.cfg), alone)

    def test_undecodable(self) -> None:
        bad = self._path("bad.mp4")
        bad.write_text("this is not a video")
        s = signals.compute_signals(bad, self.cfg)
        self.assertFalse(s["decodable"])
        self.assertEqual(s["phash"], "")
        self.assertEqual(s["blur_var"], 0.0)
        self.assertEqual(s["motion_mean"], 0.0)
        self.assertEqual(s["hand_count_mode"], 0)
        self.assertEqual(signals.phash_segment(bad), "")

    def test_blur_var_responds(self) -> None:
        sharp = signals.compute_signals(self._mp4("sharp.mp4", _sharp), self.cfg)["blur_var"]

        def blurred(i: int) -> np.ndarray:
            return cv2.GaussianBlur(_sharp(i), (17, 17), 0)

        soft = signals.compute_signals(self._mp4("soft.mp4", blurred), self.cfg)["blur_var"]
        self.assertGreater(sharp, 50.0)
        self.assertLess(soft, sharp / 3.0)

    def test_blur_var_resolution_invariant(self) -> None:
        rng = np.random.default_rng(7)
        big = rng.integers(0, 255, (1440, 1920, 3)).astype(np.uint8)
        small = cv2.resize(big, (480, 360), interpolation=cv2.INTER_AREA)
        vb, sb = signals._sharpness(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY))
        vs, ss = signals._sharpness(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        self.assertEqual((sb, ss), (0.25, 1.0))
        self.assertAlmostEqual(vb / vs, 1.0, delta=0.10)

    def test_blur_var_brightness_invariant(self) -> None:
        bright = signals.compute_signals(self._mp4("bright.mp4", _sharp), self.cfg)["blur_var"]

        def darkened(i: int) -> np.ndarray:
            return (_sharp(i).astype(np.float32) * 0.25).astype(np.uint8)

        dark = signals.compute_signals(self._mp4("darksharp.mp4", darkened), self.cfg)["blur_var"]
        self.assertAlmostEqual(dark / bright, 1.0, delta=0.15)
        thr = DEFAULT.blur_var_min
        self.assertEqual(bright > thr, dark > thr, "dark copy must not change sharpness class")

    def test_backend_versions(self) -> None:
        v1 = signals.backend_versions(self.cfg)
        self.assertEqual(list(v1.keys()), ["faces_backend", "hands_backend",
                                           "face_model_sha256", "hand_model_sha256",
                                           "mediapipe_version", "opencv_version"])
        self.assertEqual(v1, signals.backend_versions(self.cfg))
        bi = signals.backend_info(self.cfg)
        self.assertEqual(v1["faces_backend"], bi["faces"])
        self.assertEqual(v1["hands_backend"], bi["hands"])
        self.assertEqual(v1["face_model_sha256"], bi["face_model_sha256"])
        self.assertEqual(v1["hand_model_sha256"], bi["hand_model_sha256"])
        self.assertIsInstance(v1["opencv_version"], str)
        self.assertTrue(v1["opencv_version"])
        self.assertTrue(v1["mediapipe_version"] is None
                        or isinstance(v1["mediapipe_version"], str))

    def test_brightness_and_exposure(self) -> None:
        dark = signals.compute_signals(self._mp4("dark.mp4", lambda i: np.full((H, W, 3), 10, np.uint8)), self.cfg)
        over = signals.compute_signals(self._mp4("over.mp4", lambda i: np.full((H, W, 3), 250, np.uint8)), self.cfg)
        mid = signals.compute_signals(self._mp4("mid.mp4", lambda i: np.full((H, W, 3), 128, np.uint8)), self.cfg)
        self.assertGreater(dark["under_exposed_frac"], 0.9)
        self.assertLess(dark["brightness_mean"], 20)
        self.assertGreater(over["over_exposed_frac"], 0.9)
        self.assertLess(mid["under_exposed_frac"], 0.05)
        self.assertLess(mid["over_exposed_frac"], 0.05)

    def test_motion_static_vs_moving(self) -> None:
        static = signals.compute_signals(self._mp4("static.mp4", lambda i: _bg()), self.cfg)["motion_mean"]
        moving = signals.compute_signals(self._mp4("moving.mp4", _moving_skin), self.cfg)["motion_mean"]
        self.assertLess(static, 0.01)
        self.assertGreater(moving, static)

    def test_phash_and_duplicate(self) -> None:
        p = self._mp4("phash.mp4", _moving_skin)
        ph = signals.phash_segment(p)
        self.assertEqual(len(ph), 16)
        self.assertTrue(all(c in "0123456789abcdef" for c in ph))
        self.assertEqual(ph, signals.phash_segment(p))

        self.assertEqual(signals.find_duplicate(ph, {ph: "seg_reference"}), "seg_reference")
        self.assertIsNone(signals.find_duplicate("f" * 16, {"0" * 16: "seg_far"}))
        self.assertIsNone(signals.find_duplicate(ph, {}))

        with_index = signals.compute_signals(p, self.cfg, index={ph: "seg_reference"})
        self.assertEqual(with_index["dup_of"], "seg_reference")

    def test_clean_video_invariants(self) -> None:
        s = signals.compute_signals(self._mp4("clean.mp4", lambda i: _bg()), self.cfg)
        self.assertEqual(s["hands_visible_frac"], 0.0)
        self.assertEqual(s["hand_count_mode"], 0)
        self.assertEqual(s["faces_found"], 0)
        self.assertEqual(s["faces_blurred"], 0)

    def test_proxy_counts_directly(self) -> None:
        """Colour-proxy fallback logic, independent of which backend is active."""
        self.assertEqual(signals._proxy_hand_count(_hand_bar(0)), 1)
        self.assertEqual(signals._proxy_hand_count(_bg()), 0)
        self.assertGreaterEqual(len(signals._proxy_face_boxes(_big_face(0))), 1)
        self.assertEqual(signals._proxy_face_boxes(_bg()), [])
        blobs = signals._skin_blobs(_drawn_face(0), signals._area_thresholds(_drawn_face(0)))
        self.assertGreaterEqual(sum(1 for b in blobs if b["face"]), 1)


class TestMediapipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        face, hand = _model_paths()
        if face is None or hand is None:
            raise unittest.SkipTest("mediapipe model files absent")
        from dataclasses import replace as _replace

        cls.cfg = _replace(DEFAULT, face_detector_task=face, hand_landmarker_task=hand)

    def test_backend_info_reports_mediapipe(self) -> None:
        bi = signals.backend_info(self.cfg)
        self.assertEqual(bi["faces"], "mediapipe")
        self.assertEqual(bi["hands"], "mediapipe")
        self.assertTrue(bi["face_model"].endswith("blaze_face_short_range.tflite"))
        self.assertTrue(bi["hand_model"].endswith("hand_landmarker.task"))
        self.assertEqual(len(bi["face_model_sha256"]), 64)
        self.assertEqual(len(bi["hand_model_sha256"]), 64)

    def test_face_detector_finds_drawn_face(self) -> None:
        boxes = signals._mp_face_boxes(_drawn_face(0), signals._face_model_path(self.cfg))
        self.assertIsNotNone(boxes)
        self.assertGreaterEqual(len(boxes), 1)
        x, y, w, h = boxes[0]
        self.assertGreater(w, 20)
        self.assertGreater(h, 20)

    def test_hand_landmarker_counts_valid(self) -> None:
        for frame in (_bg(), _drawn_face(0), _hand_bar(0)):
            n = signals._mp_hand_count(frame, signals._hand_model_path(self.cfg))
            self.assertIsNotNone(n)
            self.assertIn(n, (0, 1, 2))
        n1 = signals._mp_hand_count(_bg(), signals._hand_model_path(self.cfg))
        n2 = signals._mp_hand_count(_bg(), signals._hand_model_path(self.cfg))
        self.assertEqual(n1, n2)

    def test_end_to_end_drawn_face_video(self) -> None:
        root = ROOT / "data_runtime"
        tmp = Path(tempfile.mkdtemp(dir=str(root), prefix="test_mp_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        p = tmp / "mpface.mp4"
        _write_video(p, _drawn_face)
        s1 = signals.compute_signals(p, self.cfg)
        self.assertGreaterEqual(s1["faces_found"], 1)
        self.assertEqual(s1["faces_blurred"], s1["faces_found"])
        self.assertEqual(signals.compute_signals(p, self.cfg), s1)

    def test_broken_model_raises_not_silent(self) -> None:
        with self.assertRaises(Exception):
            signals._get_face_detector(Path(self.cfg.blurred_dir) / "no-such-model.tflite")


class TestBlurFaces(SignalsBase):
    def test_blur_faces_drawn_face(self) -> None:
        frame = _drawn_face(0)
        out, n_found, n_blurred = signals.blur_faces(frame, self.cfg)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(n_found, n_blurred)
        self.assertGreaterEqual(n_found, 1)
        self.assertFalse(np.array_equal(out, frame), "frame should be altered")

    def test_blur_faces_no_faces(self) -> None:
        frame = _bg()
        out, n_found, n_blurred = signals.blur_faces(frame, self.cfg)
        self.assertEqual((n_found, n_blurred), (0, 0))
        self.assertTrue(np.array_equal(out, frame))

    def test_blur_and_hash_segment(self) -> None:
        p = self._mp4("blurhv.mp4", _drawn_face)
        res = signals.blur_and_hash_segment(p, self.cfg)
        out_path = Path(res["path"])
        self.assertEqual(out_path.parent, self.cfg.blurred_dir)
        self.assertTrue(out_path.exists())
        self.assertGreater(out_path.stat().st_size, 0)
        self.assertEqual(len(res["sha256"]), 64)
        self.assertEqual(res["faces_found"], res["faces_blurred"])
        self.assertGreater(res["faces_found"], 0)
        again = signals.blur_and_hash_segment(p, self.cfg)
        self.assertEqual(res["sha256"], again["sha256"], "blurred output must be byte-deterministic")
        self.assertTrue(Path(p).exists(), "original segment must not be modified")


class TestSegmenter(SignalsBase):
    def _expected_ids(self, src: Path, n: int) -> list[str]:
        sha8 = hashlib.sha256(src.read_bytes()).hexdigest()[:8]
        return [f"{src.stem}-{sha8}-{i:04d}" for i in range(n)]

    def test_split_basic(self) -> None:
        p = self._mp4("clip.mp4", _sharp)
        segs = segmenter.split_clip(p, 1.0, str(self.tmp / "segs"),
                                    worker_id="w1", site_id="s1", config=self.cfg)
        self.assertEqual(len(segs), 3)
        self.assertEqual([s["segment_id"] for s in segs], self._expected_ids(p, 3))
        for seg in segs:
            self.assertEqual(set(seg), SEG_KEYS)
            self.assertEqual(seg["worker_id"], "w1")
            self.assertEqual(seg["site_id"], "s1")
            self.assertEqual(len(seg["sha256"]), 64)
            self.assertTrue(Path(seg["path"]).exists())
            self.assertTrue(Path(seg["path"]).name.startswith("clip-"))
        self.assertEqual(segs[0]["start_s"], 0.0)
        self.assertEqual(segs[0]["end_s"], 1.0)
        self.assertGreater(segs[2]["end_s"], 2.0)
        self.assertTrue(signals.compute_signals(segs[1]["path"], self.cfg)["decodable"])

    def test_split_ids_unique_across_same_named_clips(self) -> None:
        d1 = self.tmp / "w1"
        d2 = self.tmp / "w2"
        d1.mkdir()
        d2.mkdir()
        p1 = d1 / "clip.mp4"
        p2 = d2 / "clip.mp4"
        _write_video(p1, _sharp)
        _write_video(p2, _moving_skin)
        out = self.tmp / "segs_dup"
        a = segmenter.split_clip(p1, 1.0, str(out), config=self.cfg)
        b = segmenter.split_clip(p2, 1.0, str(out), config=self.cfg)
        ids_a = {s["segment_id"] for s in a}
        ids_b = {s["segment_id"] for s in b}
        self.assertTrue(ids_a.isdisjoint(ids_b), "same-named clips must not share segment ids")
        self.assertEqual(len({s["path"] for s in a} | {s["path"] for s in b}), len(a) + len(b))

    def test_split_deterministic(self) -> None:
        p = self._mp4("clipd.mp4", _sharp)
        a = segmenter.split_clip(p, 1.0, str(self.tmp / "segs_a"), config=self.cfg)
        b = segmenter.split_clip(p, 1.0, str(self.tmp / "segs_b"), config=self.cfg)
        self.assertEqual([s["sha256"] for s in a], [s["sha256"] for s in b])
        self.assertEqual([s["segment_id"] for s in a], [s["segment_id"] for s in b])
        self.assertEqual(segmenter.split_clip(str(self.tmp / "nope.mp4"), 1.0,
                                              str(self.tmp / "sz"), config=self.cfg), [])
