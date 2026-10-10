"""Focused unit tests for arm B spectral features (synthetic only, <30 s)."""

import os
import unittest

import cv2
import numpy as np

from jadal.spectral.features import ALL_FEATURES, FAMILIES, FrameAccumulator
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


if __name__ == "__main__":
    unittest.main()
