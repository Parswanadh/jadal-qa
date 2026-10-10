"""Focused unit tests for arm B spectral features (synthetic only, <30 s)."""

import os
import tempfile
import unittest

import cv2
import numpy as np

from jadal.spectral.features import (ALL_FEATURES, FAMILIES, compressed_stats,
                                     decode_pipe, extract, FrameAccumulator)
from jadal.spectral.stream import SpectralStream


def _synth_frames(n=30, size=64, seed=0, frozen=False, dark=False):
    rng = np.random.default_rng(seed)
    base = rng.random((size, size, 3))
    frames = []
    for i in range(n):
        if frozen:
            f = base
        else:
            shift = (i * 2) % size
            f = np.roll(base, shift, axis=1) * (0.8 + 0.2 * np.sin(i))
        if dark:
            f = f * 0.05
        frames.append((np.clip(f, 0, 1) * 255).astype(np.uint8))
    return frames


class TestAccumulator(unittest.TestCase):
    def test_feature_keys_complete(self):
        acc = FrameAccumulator(32, window=40)
        for f in _synth_frames(n=10, size=32):
            acc.update(f.astype(np.float64) / 255.0)
        feats = acc.running_features()
        fam_keys = [k for fam, v in FAMILIES.items() for k in v
                    if fam != "compressed"]
        for k in fam_keys:
            self.assertIn(k, feats, k)
            self.assertTrue(np.isfinite(feats[k]), k)
        self.assertEqual(len(ALL_FEATURES), 40)

    def test_frozen_detected(self):
        acc = FrameAccumulator(32, window=40)
        for f in _synth_frames(n=10, size=32, frozen=True):
            acc.update(f.astype(np.float64) / 255.0)
        feats = acc.running_features()
        self.assertGreater(feats["temp_frozen_frac"], 0.9)
        acc2 = FrameAccumulator(32, window=40)
        for f in _synth_frames(n=10, size=32, frozen=False):
            acc2.update(f.astype(np.float64) / 255.0)
        feats2 = acc2.running_features()
        self.assertLess(feats2["temp_frozen_frac"], 0.5)
        self.assertGreater(feats2["motion_energy"], feats["motion_energy"])

    def test_dark_detected(self):
        acc = FrameAccumulator(32, window=40)
        for f in _synth_frames(n=10, size=32, dark=True):
            acc.update(f.astype(np.float64) / 255.0)
        feats = acc.running_features()
        self.assertLess(feats["brightness"], 0.1)
        self.assertGreater(feats["under_frac"], 0.5)

    def test_determinism(self):
        outs = []
        for _ in range(2):
            acc = FrameAccumulator(32, window=40)
            for f in _synth_frames(n=10, size=32, seed=7):
                acc.update(f.astype(np.float64) / 255.0)
            outs.append(acc.running_features())
        for k in outs[0]:
            self.assertEqual(outs[0][k], outs[1][k], k)

    def test_stream_chunked_running_vector(self):
        import jadal.spectral.stream as sm
        frames = _synth_frames(n=20, size=48, seed=5)
        bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
        # monkeypatch a small-tier stream via TIERS-independent path:
        st = sm.SpectralStream(tier="T0")
        v1 = st.feed(bgr[:10])
        v2 = st.feed(bgr[10:])
        self.assertEqual(st.n_frames, 20)
        for k in v2:
            self.assertTrue(np.isfinite(v2[k]), k)
        # running vector changes as more evidence arrives
        self.assertFalse(all(v1[k] == v2[k] for k in v1))


class TestIOPipe(unittest.TestCase):
    """End-to-end I/O: real mp4 through ffmpeg/ffprobe (Important-4)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.mp4 = os.path.join(cls.tmp.name, "io.mp4")
        frames = _synth_frames(n=20, size=64, seed=11)
        vw = cv2.VideoWriter(cls.mp4, cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (64, 64))
        for f in frames:
            vw.write(cv2.cvtColor(np.ascontiguousarray(f),
                                  cv2.COLOR_RGB2BGR))
        vw.release()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_extract_40_finite_features(self):
        feats = extract(self.mp4, 0.0, 2.0, tier="T0")
        self.assertEqual(set(feats), set(ALL_FEATURES))
        for k, v in feats.items():
            self.assertTrue(np.isfinite(v), k)

    def test_compressed_stats_nonzero_on_valid_file(self):
        cs = compressed_stats(self.mp4, 0.0, 2.0)
        self.assertGreater(cs["i_size"], 0.0)
        self.assertGreater(cs["p_count_frac"], 0.0)
        frames, mv = decode_pipe(self.mp4, 0.0, 2.0, 64)
        self.assertEqual(frames.shape[0], 20)
        self.assertGreaterEqual(mv, 0.0)

    def test_missing_file_raises_not_silent_zeros(self):
        bad = os.path.join(self.tmp.name, "nope.mp4")
        with self.assertRaises((FileNotFoundError, ValueError)):
            extract(bad, 0.0, 2.0, tier="T0")
        with self.assertRaises((FileNotFoundError, ValueError)):
            decode_pipe(bad, 0.0, 2.0, 64)


if __name__ == "__main__":
    unittest.main()
