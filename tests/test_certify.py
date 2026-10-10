"""Tests for jadal/certify.py (L2: calibration, exact bounds, batch rule)."""
from __future__ import annotations

import unittest

from jadal import certify


class ExactBinomialUpperTest(unittest.TestCase):
    """One-sided Clopper-Pearson upper bound on the error rate."""

    def test_zero_errors_n_100_delta_point_one_is_about_0_0228(self):
        # known value: 1 - 0.1 ** (1/100) = 0.022763
        self.assertAlmostEqual(certify.exact_binomial_upper(0, 100, 0.1), 0.0228, places=4)

    def test_zero_errors_uses_rule_of_three(self):
        # 1 - delta ** (1/n)
        for n, delta in [(1, 0.1), (10, 0.1), (10, 0.05), (250, 0.1)]:
            expected = 1.0 - delta ** (1.0 / n)
            self.assertAlmostEqual(
                certify.exact_binomial_upper(0, n, delta), expected, places=10
            )

    def test_known_values_against_clopper_pearson(self):
        cases = [
            (0, 1, 0.1, 0.9),
            (0, 10, 0.1, 0.205672),
            (1, 10, 0.1, 0.336848),
            (5, 100, 0.1, 0.090771),
            (50, 100, 0.1, 0.568583),
            (3, 20, 0.05, 0.343664),
        ]
        for errors, n, delta, expected in cases:
            with self.subTest(errors=errors, n=n, delta=delta):
                self.assertAlmostEqual(
                    certify.exact_binomial_upper(errors, n, delta), expected, places=5
                )

    def test_all_errors_gives_one(self):
        self.assertEqual(certify.exact_binomial_upper(10, 10, 0.1), 1.0)

    def test_bound_strictly_exceeds_the_empirical_rate(self):
        previous = -1.0
        for errors in range(0, 41):
            bound = certify.exact_binomial_upper(errors, 40, 0.1)
            self.assertGreaterEqual(bound, previous)
            self.assertLessEqual(bound, 1.0)
            if errors < 40:
                self.assertGreater(bound, errors / 40.0)
            else:
                self.assertEqual(bound, 1.0)
            previous = bound

    def test_bound_shrinks_when_delta_grows(self):
        # delta is the error allowed on the bound itself: bigger delta, smaller bound
        looser = certify.exact_binomial_upper(2, 30, 0.2)
        stricter = certify.exact_binomial_upper(2, 30, 0.01)
        self.assertLess(looser, stricter)

    def test_rejects_bad_input(self):
        for bad in [(-1, 10, 0.1), (11, 10, 0.1), (1, 0, 0.1), (1, -3, 0.1)]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    certify.exact_binomial_upper(*bad)
        for bad_delta in [0.0, 1.0, -0.1, 1.5]:
            with self.subTest(delta=bad_delta):
                with self.assertRaises(ValueError):
                    certify.exact_binomial_upper(0, 10, bad_delta)


if __name__ == "__main__":
    unittest.main()
