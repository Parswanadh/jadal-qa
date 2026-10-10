"""Tests for jadal.lab.early_reject (synthetic series only, no video files)."""
import unittest

from jadal.lab import early_reject as er


class TestCollapseGate(unittest.TestCase):
    def test_healthy_motion_never_fires(self):
        g = er.CollapseGate()
        # steady healthy motion around calibration mean: no fire
        for _ in range(50):
            self.assertFalse(g.update(mad=10.0, dark_frac=0.0))
        self.assertFalse(g.fired)
        v = g.verdict("w-test")
        self.assertEqual(v["window"], "w-test")
        self.assertEqual(len(v["hash"]), 64)

    def test_frozen_stream_fires_fast(self):
        g = er.CollapseGate()
        fired_at = None
        for i in range(50):
            if g.update(mad=0.2, dark_frac=0.0):
                fired_at = i
                break
        self.assertIsNotNone(fired_at)
        self.assertLessEqual(fired_at / g.fps, 4.0)  # well within the window
        self.assertEqual(g.fire_cause, "cusum")

    def test_dark_fast_path(self):
        g = er.CollapseGate()
        for _ in range(er.DEFAULT_DARK_MIN_FRAMES):
            g.update(mad=9.0, dark_frac=0.95)
        self.assertTrue(g.fired)
        self.assertEqual(g.fire_cause, "dark")

    def test_fire_is_sticky_and_reset_clears(self):
        g = er.CollapseGate()
        g.update(mad=0.0, dark_frac=1.0)
        for _ in range(er.DEFAULT_DARK_MIN_FRAMES):
            g.update(mad=0.0, dark_frac=1.0)
        self.assertTrue(g.update(mad=50.0, dark_frac=0.0))  # sticky
        g.reset()
        self.assertFalse(g.fired)
        self.assertFalse(g.update(mad=10.0, dark_frac=0.0))

    def test_verdict_hash_deterministic(self):
        g1, g2 = er.CollapseGate(), er.CollapseGate()
        for m in (10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1):
            g1.update(mad=m)
            g2.update(mad=m)
        self.assertEqual(g1.verdict("w"), g2.verdict("w"))

    def test_nominal_arl0_sane(self):
        # k=0.5,h=5 -> ARL0 ~= 285 frames (Siegmund); just pin the formula
        self.assertAlmostEqual(er.CollapseGate().nominal_arl0_frames(),
                               284.8, places=1)


class TestSprtGate(unittest.TestCase):
    def test_boundaries_match_wald(self):
        import math
        g = er.SprtGate()
        self.assertAlmostEqual(g.a, math.log(0.10 / 0.95), places=9)
        self.assertAlmostEqual(g.b, math.log(0.90 / 0.05), places=9)

    def test_collapse_fires_healthy_does_not(self):
        g = er.SprtGate()
        for _ in range(50):
            g.update(mad=10.2)
        self.assertFalse(g.fired)
        g.reset()
        fired = any(g.update(mad=0.5) for _ in range(50))
        self.assertTrue(fired)


if __name__ == "__main__":
    unittest.main()
