"""Tests for jadal.lab.honest_bounds (pure arithmetic, no data)."""
import unittest

from jadal.lab import honest_bounds as hb


class TestHonestBounds(unittest.TestCase):
    def test_cp_covers_standard_cases(self):
        ci = hb.cp_interval(33, 100)
        self.assertLess(ci["lo"], 0.33)
        self.assertGreater(ci["hi"], 0.33)
        self.assertTrue(ci["exact"])  # scipy present in jadal env
        # textbook: 0/59 upper bound ~= 5% one-sided; two-sided 0/59:
        ci0 = hb.cp_interval(0, 59)
        self.assertEqual(ci0["lo"], 0.0)
        self.assertAlmostEqual(ci0["hi"], 0.0605, places=3)

    def test_cp_invalid(self):
        with self.assertRaises(ValueError):
            hb.cp_interval(1, 0)
        with self.assertRaises(ValueError):
            hb.cp_interval(5, 4)

    def test_zero_failure_arithmetic(self):
        # the 86-vs-84 check: (5%, 95%, Bonferroni m=4) -> 86
        self.assertEqual(hb.required_n_zero_failure(0.05, 0.05, 4), 86)
        self.assertEqual(hb.required_n_zero_failure(0.05, 0.05, 1), 59)

    def test_kish_deff(self):
        # direction-4 numbers: ICC=0.583 over mean cluster ~14.3 -> ~8.75
        deff = hb.kish_deff(0.583, 100 / 7)
        self.assertAlmostEqual(deff, 8.75, places=1)
        self.assertAlmostEqual(hb.effective_n(100, deff), 100 / deff)

    def test_dashboard_line_honest(self):
        line = hb.dashboard_line("false-reject", 33, 100,
                                 icc=0.583, mean_cluster_size=100 / 7)
        self.assertIn("33/100", line)
        self.assertIn("effective n", line)
        self.assertIn("understates", line)


if __name__ == "__main__":
    unittest.main()
