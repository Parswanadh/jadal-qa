"""Synthetic-input tests for the E1 fps / frame-selection helpers.

Covers the subsampling index rule {10:1,5:2,3.33:3,2:5,1:10}, the keyframe
selector (fallback must be explicit, off-grid I-frames must change the set),
ffprobe output parsing, and the motion/random selectors. No video, GPU or
network is used.

Run (repo root): PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests
"""

import os
import sys
import unittest

import numpy as np

_EXP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "eval", "exp_fps")
sys.path.insert(0, _EXP)

from fps_lib import FPS_STEPS, NATIVE_FPS, SPATIAL_FEATURES, subsample_idx  # noqa: E402
import run_fps  # noqa: E402


class SubsampleRuleTests(unittest.TestCase):
    def test_step_table(self):
        self.assertEqual(FPS_STEPS, {10: 1, 5: 2, 3.33: 3, 2: 5, 1: 10})

    def test_lengths_for_100_native_frames(self):
        expect = {10: 100, 5: 50, 3.33: 34, 2: 20, 1: 10}
        for fps, step in FPS_STEPS.items():
            self.assertEqual(len(subsample_idx(100, step)), expect[fps], fps)

    def test_indices_are_arange(self):
        for step in FPS_STEPS.values():
            np.testing.assert_array_equal(subsample_idx(100, step),
                                          np.arange(0, 100, step))

    def test_effective_fps_matches_key(self):
        for fps, step in FPS_STEPS.items():
            self.assertAlmostEqual(NATIVE_FPS / step, fps, delta=0.01)

    def test_short_window(self):
        np.testing.assert_array_equal(subsample_idx(7, 5), np.array([0, 5]))

    def test_spatial_list_is_26_minus_dt_dependent(self):
        dt_dep = {"temp_b0", "temp_b1", "temp_b2", "temp_b3", "frozen_frac",
                  "flicker_idx", "motion_energy", "rppg_pulse"}
        self.assertEqual(len(SPATIAL_FEATURES), 18)
        self.assertTrue(dt_dep.isdisjoint(SPATIAL_FEATURES))


class KeyframeSelectTests(unittest.TestCase):
    def test_too_few_iframes_is_explicit_fallback(self):
        idx, n, fb = run_fps.keyframe_select([5.0, 7.3], 5.0, 100, budget=20)
        self.assertIsNone(idx)
        self.assertEqual(n, 2)
        self.assertTrue(fb)

    def test_offgrid_iframes_give_non_uniform_set(self):
        start = 5.0
        # 24 I-frames at 0.4 s spacing, offset by 0.05 s (off the 10 fps grid);
        # all land inside the 10 s window at distinct decoded indices
        times = [start + 0.05 + 0.4 * k for k in range(24)]
        idx, n, fb = run_fps.keyframe_select(times, start, 100, budget=20)
        self.assertFalse(fb)
        self.assertEqual(n, 24)
        self.assertEqual(len(idx), 20)
        self.assertEqual(len(set(idx.tolist())), 20)
        pos = {int(round((t - start) * NATIVE_FPS)) for t in times}
        self.assertTrue(set(idx.tolist()) <= pos)
        uniform20 = subsample_idx(100, 5)
        self.assertFalse(np.array_equal(idx, uniform20))

    def test_iframes_outside_window_are_dropped(self):
        idx, n, fb = run_fps.keyframe_select([-1.0, 50.0], 5.0, 100, budget=1)
        self.assertEqual(n, 0)
        self.assertTrue(fb)


class ParseFfprobeTests(unittest.TestCase):
    def test_keeps_iframes_inside_window_only(self):
        out = "-1.0,I\n4.6,I\n5.0,P\n9.9,I\n10.0,I\n12.0,I\n"
        ts = run_fps.parse_ffprobe_iframes(out, 0.0, win_s=10.0)
        self.assertEqual(ts, [4.6, 9.9])

    def test_malformed_timestamp_raises(self):
        with self.assertRaises(ValueError):
            run_fps.parse_ffprobe_iframes("N/A,I\n", 0.0)


class MotionRandomSelectTests(unittest.TestCase):
    def test_motion_select_picks_high_energy_diffs(self):
        T = 100
        luma = np.zeros((T, 2, 2))
        for i in (3, 40, 77):  # jump between frame i and i+1
            luma[i + 1:] += 1.0
        keep = run_fps.motion_select(luma, budget=20)
        self.assertEqual(len(keep), 20)
        self.assertEqual(len(set(keep.tolist())), 20)
        self.assertEqual(keep[0], 0)
        self.assertTrue(np.all(np.diff(keep) > 0))
        self.assertTrue({4, 41, 78} <= set(keep.tolist()))
        self.assertTrue(keep.max() < T)

    def test_random_select_seeded(self):
        a = run_fps.random_select(100, 20, seed=0)
        b = run_fps.random_select(100, 20, seed=0)
        c = run_fps.random_select(100, 20, seed=1)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        self.assertEqual(len(set(a.tolist())), 20)
        self.assertTrue(np.all(np.diff(a) > 0))
        self.assertTrue(a.min() >= 0 and a.max() < 100)


if __name__ == "__main__":
    unittest.main()
