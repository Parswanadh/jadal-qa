"""Tests for jadal.lab.perm_entropy (synthetic series only)."""
import math
import unittest

from jadal.lab import perm_entropy as pe


class TestPermutationEntropy(unittest.TestCase):
    def test_constant_series_is_zero(self):
        self.assertEqual(pe.permutation_entropy([5.0] * 49, m=4), 0.0)

    def test_monotone_series_is_zero(self):
        # strictly increasing -> single ordinal pattern -> PE 0
        self.assertEqual(
            pe.permutation_entropy([float(i) for i in range(49)], m=3), 0.0)

    def test_noise_is_high_periodic_is_low(self):
        import random
        rng = random.Random(0)
        noise = [rng.gauss(0, 1) for _ in range(200)]
        periodic = [math.sin(2 * math.pi * i / 8) for i in range(200)]
        self.assertGreater(pe.permutation_entropy(noise, m=4),
                           pe.permutation_entropy(periodic, m=4))

    def test_ties_deterministic(self):
        s = [1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 0.5] * 8
        self.assertEqual(pe.permutation_entropy(s, m=3),
                         pe.permutation_entropy(list(s), m=3))

    def test_too_short_is_nan(self):
        self.assertTrue(math.isnan(pe.permutation_entropy([1.0, 2.0], m=4)))

    def test_loop_score_shape(self):
        out = pe.pe_loop_score([1.0] * 49)
        self.assertEqual(out["pe"], 0.0)
        self.assertTrue(out["low_pe"])
        out2 = pe.pe_loop_score([float(i % 7) for i in range(49)])
        self.assertIn("pe", out2)


if __name__ == "__main__":
    unittest.main()
