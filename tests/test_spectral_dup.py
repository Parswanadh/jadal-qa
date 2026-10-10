"""Synthetic-clip tests for jadal.spectral.dup (no data/network deps, <30s)."""

import os
import tempfile
import unittest

import cv2
import numpy as np

from jadal.spectral.dup import signature, similarity

W, H, FPS, DUR = 48, 48, 10, 2
N = FPS * DUR


def _write(path, frames):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    try:
        for f in frames:
            vw.write(np.ascontiguousarray(f))
    finally:
        vw.release()


def _base_frames():
    xs = np.arange(W, dtype=np.float64)
    ys = np.arange(H, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    out = []
    for i in range(N):
        v = np.mod(2.0 * xx + yy + 8.0 * i, 256.0).astype(np.uint8)
        out.append(cv2.merge([v, v, v]))
    return out


def _bright_frames(base):
    return [np.clip(b.astype(np.int16) + 30, 0, 255).astype(np.uint8) for b in base]


def _different_frames():
    xs = np.arange(W, dtype=np.float64)
    xx = np.tile(xs, (H, 1))
    out = []
    for i in range(N):
        v = (128.0 + 120.0 * np.sin(2.0 * np.pi * (3.0 * xx / W + 2.0 * i / N))).astype(np.uint8)
        out.append(cv2.merge([v, v, v]))
    return out


def _flipped_frames(base):
    return [cv2.flip(b, 1) for b in base]


class TestSpectralDup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        d = cls._td.name
        cls.base_p = os.path.join(d, "base.mp4")
        cls.bright_p = os.path.join(d, "bright.mp4")
        cls.diff_p = os.path.join(d, "diff.mp4")
        cls.flip_p = os.path.join(d, "flip.mp4")
        base = _base_frames()
        _write(cls.base_p, base)
        _write(cls.bright_p, _bright_frames(base))
        _write(cls.diff_p, _different_frames())
        _write(cls.flip_p, _flipped_frames(base))
        cls.sig_base = signature(cls.base_p, 0.0, float(DUR))
        cls.sig_bright = signature(cls.bright_p, 0.0, float(DUR))
        cls.sig_diff = signature(cls.diff_p, 0.0, float(DUR))
        cls.sig_flip = signature(cls.flip_p, 0.0, float(DUR))
        cls.sim_bright = similarity(cls.sig_base, cls.sig_bright)
        cls.sim_diff = similarity(cls.sig_base, cls.sig_diff)
        cls.sim_flip = similarity(cls.sig_base, cls.sig_flip)
        print(f"\nsim(base, bright)={cls.sim_bright:.4f} "
              f"sim(base, diff)={cls.sim_diff:.4f} "
              f"sim(base, flip)={cls.sim_flip:.4f}")

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_signature_keys_and_types(self):
        s = self.sig_base
        self.assertEqual(set(s), {"dct64", "dct32", "temp", "nframes", "fps"})
        self.assertRegex(s["dct64"], r"^[0-9a-f]{16}$")
        self.assertRegex(s["dct32"], r"^[0-9a-f]{16}$")
        self.assertEqual(len(s["temp"]), 32)
        self.assertTrue(all(isinstance(v, float) for v in s["temp"]))
        self.assertEqual(s["nframes"], N)
        self.assertAlmostEqual(s["fps"], FPS, delta=1.0)

    def test_brightness_copy_high(self):
        self.assertGreater(self.sim_bright, 0.85)

    def test_different_pattern_low(self):
        self.assertLess(self.sim_diff, 0.85)

    def test_flipped_lower_than_brightness_copy(self):
        self.assertLess(self.sim_flip, self.sim_bright)

    def test_determinism(self):
        again = signature(self.base_p, 0.0, float(DUR))
        self.assertEqual(again, self.sig_base)

    def test_missing_file_raises(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            signature(os.path.join(self._td.name, "nope.mp4"), 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
