"""Unit tests for jadal/segmenter.py and jadal/signals.py (synthetic videos generated in-test)."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from jadal import segmenter, signals
from jadal.config import DEFAULT

H, W, FPS = 240, 320, 30
SKIN = (90, 140, 220)  # BGR, YCrCb-checked to sit in the skin range
GRAY_BG = (128, 128, 128)

SIGNALS_KEYS = ["decodable", "blur_var", "brightness_mean", "under_exposed_frac",
                "over_exposed_frac", "motion_mean", "hands_visible_frac",
                "hand_count_mode", "faces_found", "faces_blurred", "phash", "dup_of"]
SEGMENT_KEYS = ["segment_id", "source_clip", "worker_id", "site_id",
                "start_s", "end_s", "path", "sha256"]


class SignalsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.path.join(os.path.dirname(__file__), "..", "data_runtime", "test_tmp")
        os.makedirs(root, exist_ok=True)
        cls.tmp = tempfile.mkdtemp(prefix="t_", dir=root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _video(self, name, frames_fn, n=60, size=(W, H)):
        path = os.path.join(self.tmp, name)
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
        self.assertTrue(vw.isOpened())
        for i in range(n):
            vw.write(frames_fn(i))
        vw.release()
        return path

    def _gray(self, i, v=128):
        return np.full((H, W, 3), v, np.uint8)

    def _textured_mover(self, i):
        a = np.full((H, W, 3), 128, np.uint8)
        a[:, :, 0] = 200
        a[0:20, 0:20] = np.random.default_rng(i).integers(0, 255, (20, 20, 3)).astype(np.uint8)
        cv2.circle(a, (min(W - 1, 30 + i * 2), 120), 18, SKIN, -1)
        return a

    def test_signals_keys_and_types(self):
        p = self._video("keys.mp4", self._textured_mover)
        s = signals.compute_signals(p, DEFAULT)
        self.assertEqual(sorted(s.keys()), sorted(SIGNALS_KEYS))
        self.assertTrue(s["decodable"])
        for k in ("blur_var", "brightness_mean", "under_exposed_frac", "over_exposed_frac",
                  "motion_mean", "hands_visible_frac"):
            self.assertIsInstance(s[k], float, k)
        self.assertIsInstance(s["hand_count_mode"], int)
        self.assertIsInstance(s["faces_found"], int)
        self.assertIsInstance(s["faces_blurred"], int)
        self.assertIsInstance(s["phash"], str)

    def test_deterministic_and_rounded(self):
        p = self._video("det.mp4", self._textured_mover)
        a = signals.compute_signals(p, DEFAULT)
        b = signals.compute_signals(p, DEFAULT)
        self.assertEqual(a, b)
        for k in ("blur_var", "brightness_mean", "under_exposed_frac", "over_exposed_frac",
                  "motion_mean", "hands_visible_frac"):
            self.assertEqual(s := f"{a[k]}", s[: s.index(".") + 5] if "." in s else s)

    def test_undecodable(self):
        bad = os.path.join(self.tmp, "bad.mp4")
        with open(bad, "w") as fh:
            fh.write("not a video")
        s = signals.compute_signals(bad, DEFAULT)
        self.assertFalse(s["decodable"])
        self.assertEqual(s["phash"], "")
        self.assertIsNone(s["dup_of"])
        self.assertEqual(s["faces_found"], 0)
        self.assertEqual(s["faces_blurred"], 0)

    def test_blur_var_responds(self):
        sharp = self._video("sharp.mp4", self._textured_mover)
        flat = self._video("flat.mp4", self._gray)
        self.assertGreater(signals.compute_signals(sharp, DEFAULT)["blur_var"],
                           signals.compute_signals(flat, DEFAULT)["blur_var"])

    def test_brightness_and_exposure(self):
        dark = self._video("dark.mp4", lambda i: self._gray(i, 10))
        sd = signals.compute_signals(dark, DEFAULT)
        self.assertAlmostEqual(sd["brightness_mean"], 10, delta=6)
        self.assertGreater(sd["under_exposed_frac"], 0.9)
        over = self._video("over.mp4", lambda i: self._gray(i, 250))
        so = signals.compute_signals(over, DEFAULT)
        self.assertAlmostEqual(so["brightness_mean"], 250, delta=6)
        self.assertGreater(so["over_exposed_frac"], 0.9)
        mid = self._video("mid.mp4", lambda i: self._gray(i, 128))
        sm = signals.compute_signals(mid, DEFAULT)
        self.assertLess(sm["under_exposed_frac"], 0.05)
        self.assertLess(sm["over_exposed_frac"], 0.05)

    def test_motion_static_vs_moving(self):
        static = self._video("st.mp4", lambda i: self._gray(i, 128))
        mover = self._video("mv.mp4", self._textured_mover)
        ms = signals.compute_signals(static, DEFAULT)["motion_mean"]
        mm = signals.compute_signals(mover, DEFAULT)["motion_mean"]
        self.assertAlmostEqual(ms, 0.0, delta=1e-3)
        self.assertGreater(mm, ms)

    def test_faces_hands_ranges_and_face_fixture(self):
        nodetect = self._video("nh.mp4", lambda i: self._gray(i, 128))
        s = signals.compute_signals(nodetect, DEFAULT)
        self.assertEqual(s["hands_visible_frac"], 0.0)
        self.assertEqual(s["hand_count_mode"], 0)
        self.assertEqual(s["faces_found"], 0)
        self.assertEqual(s["faces_blurred"], 0)

        def hand_bar(i):
            a = np.full((H, W, 3), 128, np.uint8)
            cv2.rectangle(a, (100, 100), (220, 140), SKIN, -1)  # wide: aspect>face band
            return a
        sh = signals.compute_signals(self._video("hands.mp4", hand_bar), DEFAULT)
        self.assertEqual(sh["hands_visible_frac"], 1.0)
        self.assertIn(sh["hand_count_mode"], (1, 2))

        def face_blob(i):
            a = np.full((H, W, 3), 128, np.uint8)
            cv2.circle(a, (160, 120), 30, SKIN, -1)  # large round blob -> face-like
            return a
        sf = signals.compute_signals(self._video("face.mp4", face_blob), DEFAULT)
        self.assertGreaterEqual(sf["faces_found"], 1)
        self.assertEqual(sf["faces_blurred"], sf["faces_found"])

    def test_phash_stable_and_dup_lookup(self):
        p = self._video("ph.mp4", self._textured_mover)
        ph1 = signals.phash_segment(p)
        ph2 = signals.phash_segment(p)
        self.assertEqual(ph1, ph2)
        self.assertEqual(len(ph1), 16)
        index = {ph1: "seg_A", "0" * 16: "seg_B"}
        self.assertEqual(signals.find_duplicate(ph1, index), "seg_A")
        self.assertEqual(signals.compute_signals(p, DEFAULT, index)["dup_of"], "seg_A")
        self.assertIsNone(signals.find_duplicate("f" * 16, index))
        self.assertIsNone(signals.find_duplicate(ph1, {}))
        self.assertIsNone(signals.compute_signals(p, DEFAULT)["dup_of"])

    def test_blur_faces_unit(self):
        frame = np.full((H, W, 3), 128, np.uint8)
        cv2.circle(frame, (160, 120), 30, SKIN, -1)
        out, n_found, n_blurred = signals.blur_faces(frame)
        self.assertGreaterEqual(n_found, 1)
        self.assertEqual(n_blurred, n_found)
        self.assertEqual(out.shape, frame.shape)
        # blurred region must differ from the flat skin circle
        self.assertGreater(float(np.abs(out.astype(int) - frame.astype(int)).sum()), 0)

    def test_blur_and_hash_separate_file(self):
        p = self._video("bh.mp4", self._textured_mover)
        out_dir = os.path.join(self.tmp, "blurred")
        cfg = DEFAULT
        old = cfg.blurred_dir
        cfg.blurred_dir = type(old)(out_dir)
        try:
            r1 = signals.blur_and_hash_segment(p, cfg)
            r2 = signals.blur_and_hash_segment(p, cfg)
        finally:
            cfg.blurred_dir = old
        self.assertTrue(os.path.exists(r1["path"]))
        self.assertNotEqual(r1["path"], p)
        self.assertEqual(len(r1["sha256"]), 64)
        self.assertEqual(r1["sha256"], r2["sha256"])
        self.assertEqual(r1["faces_found"], r1["faces_blurred"])
        self.assertTrue(cv2.VideoCapture(r1["path"]).isOpened())

    def test_segmenter_basic(self):
        p = self._video("clip.mp4", self._textured_mover, n=90)
        out = os.path.join(self.tmp, "segs")
        segs = segmenter.split_clip(p, 1.0, out, worker_id="w1", site_id="s1")
        self.assertEqual(len(segs), 3)
        for i, s in enumerate(segs):
            self.assertEqual(sorted(s.keys()), sorted(SEGMENT_KEYS))
            self.assertEqual(s["segment_id"], f"clip_seg{i:04d}")
            self.assertEqual(s["worker_id"], "w1")
            self.assertEqual(s["site_id"], "s1")
            self.assertTrue(os.path.exists(s["path"]))
            self.assertEqual(len(s["sha256"]), 64)
        self.assertEqual(segs[0]["start_s"], 0.0)
        self.assertEqual(segs[0]["end_s"], 1.0)
        self.assertGreater(segs[2]["end_s"], 2.0)
        self.assertTrue(signals.compute_signals(segs[1]["path"], DEFAULT)["decodable"])

    def test_segmenter_deterministic(self):
        p = self._video("clip2.mp4", self._textured_mover, n=75)
        a = segmenter.split_clip(p, 1.0, os.path.join(self.tmp, "sa"))
        b = segmenter.split_clip(p, 1.0, os.path.join(self.tmp, "sb"))
        self.assertEqual([x["sha256"] for x in a], [x["sha256"] for x in b])
        self.assertEqual(segmenter.split_clip(os.path.join(self.tmp, "nope.mp4"), 1.0,
                                              os.path.join(self.tmp, "sz")), [])

    def test_batch_independence_proxy(self):
        # L0 answer for one segment must be identical whether run alone or 5 times over.
        p = self._video("batch.mp4", self._textured_mover)
        alone = signals.compute_signals(p, DEFAULT)
        for _ in range(5):
            self.assertEqual(signals.compute_signals(p, DEFAULT), alone)


if __name__ == "__main__":
    unittest.main()
