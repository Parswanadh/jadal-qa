"""Tests for E3 temporal-policy artefacts.

Two groups:
  * TestE3: checks on the committed JSONs (no sibling data needed).
  * TestSimulation: synthetic-input unit tests of the simulation logic in run_e3.py
    (no files read, no sibling data, runs in well under 5 s).
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(HERE, "..", "eval", "exp_temporal")
sys.path.insert(0, os.path.abspath(EXP_DIR))

import run_e3  # noqa: E402  (module-level code is path setup and RNG only; no file IO)
from write_e305 import TOLERANCE  # noqa: E402

D = os.path.join(HERE, "..", "eval", "results", "exp_temporal")


def load(n):
    with open(os.path.join(D, n)) as f:
        return json.load(f)


class TestE3(unittest.TestCase):
    def test_e301_present_and_consistent(self):
        r = load("E3.01_policies.json")
        self.assertEqual(r["n_windows"], 150)
        self.assertEqual(r["n_clips"], 7)
        pp = r["per_policy"]
        # cost decreases with sparser grids
        self.assertGreater(pp["every-2"]["mean_frac"], pp["every-4"]["mean_frac"])
        self.assertGreater(pp["every-4"]["mean_frac"], pp["every-8"]["mean_frac"])
        # error grows with sparser grids (label oracle)
        self.assertLess(pp["every-2"]["mean_err_L"], pp["every-4"]["mean_err_L"])
        self.assertLess(pp["every-4"]["mean_err_L"], pp["every-8"]["mean_err_L"])
        # full coverage has zero sampling error
        self.assertEqual(pp["full"]["mean_err_L"], 0.0)
        for v in r["cusum_loco"]["folds"].values():
            self.assertIn("test_err_M", v)

    def test_e302_thin_but_valid(self):
        r = load("E3.02_ood_position.json")
        self.assertEqual(r["n_windows"], 60)
        self.assertEqual(r["n_bad"], 9)
        self.assertGreaterEqual(r["mw_raw_p"], 0.0)
        self.assertLessEqual(r["mw_raw_p"], 1.0)
        # logistic-skip is recorded explicitly, never silently dropped
        self.assertIn("logreg_skipped", r)
        self.assertIsInstance(r["logreg_skipped"], bool)
        if r["logreg_skipped"]:
            self.assertNotIn("logreg_coef_posnorm", r)
        else:
            self.assertIn("logreg_coef_posnorm", r)

    def test_e303_curve_saturates(self):
        r = load("E3.03_early_exit.json")
        c = {x["seconds"]: x["auroc"] for x in r["curve"]}
        self.assertGreater(c[4.0], 0.83)
        # 4 s within 0.02 of the best prefix
        self.assertGreaterEqual(c[4.0], max(c.values()) - 0.02)
        # 2 s clearly worse than 4 s
        self.assertLess(c[2.0], c[4.0] - 0.03)

    def test_e304_arithmetic(self):
        r = load("E3.04_compute.json")
        self.assertAlmostEqual(r["corpus_hours"], 4.404, places=2)
        for row in r["per_policy"]:
            self.assertAlmostEqual(row["speedup"], 1.0 / row["window_frac"], places=1)

    def test_e305_decision_exists(self):
        r = load("E3.05_decision.json")
        # chosen policy by name
        self.assertEqual(r["adopted_policy"], "every-2")
        self.assertIn("every-2", r["strict_tolerance"]["pass_list"])
        # tolerance value, and it matches the constant used to write the decision
        self.assertEqual(r["strict_tolerance"]["tolerance"], 0.06)
        self.assertEqual(r["strict_tolerance"]["tolerance"], TOLERANCE)
        # the CI crosses the tolerance, and the selection is labelled post hoc
        lo, hi = r["strict_tolerance"]["every2_errL_ci"]
        self.assertLess(lo, TOLERANCE)
        self.assertGreater(hi, TOLERANCE)
        self.assertIn("post hoc", r["strict_tolerance"]["selection"])
        self.assertEqual(r["status"], "working policy on IndEgo, not validated")


class TestSimulation(unittest.TestCase):
    """Synthetic inputs only; each expected value is traced by hand in the comment."""

    def test_cusum_never_fires_on_flat_scores(self):
        # constant P(good)=0.9: (mean - p - delta) = -0.1 always, cum stays 0, so every-4 grid
        self.assertEqual(run_e3.pol_cusum(np.full(20, 0.9)), [0, 4, 8, 12, 16])

    def test_cusum_forced_alarm_densifies(self):
        # scores: 0.9 x4 then 0.0. Hand trace (k0=4, delta=0.1, h=0.5, max_run=3):
        #   i=0 cum 0; i=4 cum 0.35; i=8 cum 0.55 > 0.5 -> alarm, probe 9,10,11 (dense),
        #   then sparse again: 15, 19.
        s = np.array([0.9] * 4 + [0.0] * 16)
        self.assertEqual(run_e3.pol_cusum(s, k0=4, delta=0.10, h=0.50, max_run=3),
                         [0, 4, 8, 9, 10, 11, 15, 19])

    def test_skip_first_M(self):
        starts = np.array([0.0, 15.0, 30.0, 45.0])
        self.assertEqual(run_e3.pol_skip_first_M(starts, 20), [2, 3])
        # M beyond every start falls back to the last window
        self.assertEqual(run_e3.pol_skip_first_M(starts, 100), [3])

    def test_first_bad_run(self):
        self.assertEqual(run_e3.first_bad_run(np.array([0, 0, 1, 1, 0])), 2)
        self.assertEqual(run_e3.first_bad_run(np.array([0, 0, 0])), None)
        self.assertEqual(run_e3.first_bad_run(np.array([1, 0, 1])), 0)

    def _clip(self):
        # 10 windows, 2 bad -> truth_frac 0.2 -> accept at THRESH 0.30
        y_bad = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
        p_good = np.where(y_bad == 1, 0.2, 0.8)
        return y_bad, p_good

    def test_eval_policy_clip_full_coverage(self):
        y_bad, p_good = self._clip()
        r = run_e3.eval_policy_clip(y_bad, p_good, list(range(10)))
        self.assertAlmostEqual(r["truth_frac"], 0.2)
        self.assertEqual(r["truth_verdict"], "accept")
        self.assertAlmostEqual(r["err_L"], 0.0)
        self.assertEqual(r["agree_L"], 1)
        self.assertEqual(r["miss_L"], 0)
        self.assertEqual(r["delay_L"], 0.0)
        self.assertEqual(r["frac"], 1.0)

    def test_eval_policy_clip_wrong_verdict(self):
        y_bad, p_good = self._clip()
        # probing only bad windows: est_L = 1.0 -> reject, truth accept
        r = run_e3.eval_policy_clip(y_bad, p_good, [2, 3])
        self.assertEqual(r["ver_L"], "reject")
        self.assertEqual(r["agree_L"], 0)
        self.assertAlmostEqual(r["err_L"], 0.8)
        self.assertAlmostEqual(r["frac"], 0.2)

    def test_eval_policy_clip_missed_run(self):
        y_bad, p_good = self._clip()
        # probes 0,1,5,6 never touch the bad run (2,3): miss, infinite delay, est_L = 0
        r = run_e3.eval_policy_clip(y_bad, p_good, [0, 1, 5, 6])
        self.assertEqual(r["miss_L"], 1)
        self.assertEqual(r["delay_L"], float("inf"))
        self.assertEqual(r["est_L"], 0.0)
        self.assertAlmostEqual(r["err_L"], 0.2)
        self.assertEqual(r["agree_L"], 1)

    def test_eval_policy_clip_all_good_clip(self):
        y_bad = np.zeros(6, dtype=int)
        r = run_e3.eval_policy_clip(y_bad, np.full(6, 0.9), [0, 2, 4])
        self.assertEqual((r["delay_L"], r["miss_L"], r["delay_M"], r["miss_M"]), (0.0, 0, 0.0, 0))

    def test_prefix_embed_excludes_nan_padding(self):
        F = np.full((2, 4, 3), np.nan, dtype=np.float32)
        F[0, :2] = [[1, 0, 0], [3, 0, 0]]          # window 0 has 2 valid frames
        F[1, :4] = [[0, 1, 0]] * 4                 # window 1 has 4 valid frames
        counts = np.array([2, 4])
        E = run_e3.prefix_embed(F, counts, 3)      # window 0 capped at its 2 valid frames
        self.assertFalse(np.isnan(E).any())
        np.testing.assert_allclose(E[0], [1.0, 0.0, 0.0])   # mean [2,0,0] normalised
        np.testing.assert_allclose(E[1], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(np.linalg.norm(E, axis=1), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
