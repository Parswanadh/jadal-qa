"""Per-pixel package tests: synthetic arrays only, no GPU, no data files (<30 s)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jadal.perpixel import series as S  # noqa: E402


class TestSeries(unittest.TestCase):
    def test_channels_and_finite(self):
        rng = np.random.RandomState(0)
        frames = np.clip(rng.randn(24, 45, 80, 3) * 0.1 + 0.5, 0, 1)
        M = S.per_frame_series(frames)
        self.assertEqual(M.shape, (24, 8))
        self.assertTrue(np.all(np.isfinite(M)))
        self.assertEqual(M[0, 2], 0.0)  # first diff_e is 0
        self.assertEqual(M[0, 4], 0.0)  # first shift is 0

    def test_frozen_series_descriptors(self):
        M = np.ones((40, 8)) * 0.5
        M[:, 2] = 0.0
        d = S.describe_series(M, 10.0)
        self.assertGreater(d["still_run_frac"], 0.99)
        self.assertEqual(d["motion_frac_above"], 0.0)
        for v in d.values():
            self.assertTrue(np.isfinite(v), v)

    def test_bursty_series(self):
        M = np.ones((60, 8)) * 0.4
        M[::10, 1] = 5.0  # bursts in lap_var
        d = S.describe_series(M, 10.0)
        self.assertGreater(d["lap_var_burst"], 1.0)
        self.assertGreaterEqual(d["lap_var_ac_peak"], 0.0)

    def test_determinism(self):
        rng = np.random.RandomState(7)
        frames = np.clip(rng.randn(16, 45, 80, 3) * 0.2 + 0.4, 0, 1)
        a = S.per_frame_series(frames)
        b = S.per_frame_series(frames.copy())
        np.testing.assert_array_equal(a, b)
        d1 = S.describe_series(a, 8.0)
        d2 = S.describe_series(b, 8.0)
        self.assertEqual(d1, d2)


class TestDupSim(unittest.TestCase):
    def test_identical_is_one(self):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from eval.perpixel import detectors as D
        sig = {"dct64": "0" * 16, "dct32": "f" * 16,
               "temp": [0.1] * 32}
        self.assertAlmostEqual(D.dup_sim(sig, dict(sig)), 1.0)


if __name__ == "__main__":
    unittest.main()
