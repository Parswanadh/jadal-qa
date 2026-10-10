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


class TestTheoryPhysics(unittest.TestCase):
    def test_theory_quantise_finite(self):
        from jadal.perpixel import theory as T
        d = {"sigma_immer": 0.01, "frozen_logp": float("-inf"),
             "blur_norm": float("nan"), "p05": 0.12345}
        q = T.quantise(d)
        self.assertTrue(all(isinstance(v, int) for v in q.values()))
        # 4-decimal fixed point: p05 -> 1234 or 1235
        self.assertIn(q["p05"], (1234, 1235))

    def test_theory_extract_keys(self):
        from jadal.perpixel import theory as T
        rng = np.random.RandomState(3)
        fr = np.clip(rng.randn(20, 45, 80, 3) * 0.1 + 0.5, 0, 1)
        d = T.extract_theory(fr, 10.0)
        self.assertEqual(sorted(d), sorted(T.THEORY_FEATURES))
        self.assertTrue(all(np.isfinite(v) for v in d.values()))

    def test_noise_estimators_agree(self):
        from jadal.perpixel import theory as T
        rng = np.random.RandomState(4)
        g = np.clip(0.5 + rng.randn(90, 160) * 0.03, 0, 1)
        si, sw = T.immerkaer_sigma(g), T.wavelet_mad_sigma(g)
        self.assertGreater(si, 0)
        # both estimate ~0.03 within a factor of 3
        self.assertLess(max(si, sw) / (min(si, sw) + 1e-12), 3.0)

    def test_rule_hits_monotone(self):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from eval.perpixel import theory_clean as TC
        from jadal.perpixel import theory as T
        qm = np.full((2, len(T.THEORY_FEATURES)), 0.5)
        qm[0, T.THEORY_FEATURES.index("p05")] = 0.01
        qm[1, T.THEORY_FEATURES.index("p05")] = 0.20
        h = TC.hits(qm, None, "dark", "p05", "<", 0.1)
        self.assertEqual(list(h), [True, False])

    def test_physics_extract_keys(self):
        from jadal.perpixel import physics as P
        rng = np.random.RandomState(5)
        fr = np.clip(rng.randn(24, 45, 80, 3) * 0.08 + 0.5, 0, 1)
        d = P.extract_physics(fr, 10.0, "/nonexistent.mp4", 0, 2.4)
        for k in P.PHYSICS_FEATURES:
            self.assertIn(k, d)
        self.assertGreaterEqual(d["clip_frac"], 0.0)

    def test_dhash_hamming(self):
        from jadal.perpixel import physics as P
        rng = np.random.RandomState(6)
        g = np.clip(0.5 + rng.randn(90, 160) * 0.05, 0, 1)
        self.assertEqual(P._ham(P.dhash(g), P.dhash(g.copy())), 0)


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
