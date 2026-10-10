#!/usr/bin/env python3
"""M4: the standard harness must never silently mis-align a regenerated shard.

`eval/standard/run_all.py` stores OOD feature vectors in the shard's own name
order; before `predict_proba` they are reordered into `FEATURE_NAMES` order (or
an error is raised).  These tests pin that pure function.
"""
import unittest

from eval.standard.run_all import align_features


class TestAlignFeatures(unittest.TestCase):
    def test_identity_order_is_returned_unchanged(self):
        names = ["a", "b", "c"]
        self.assertEqual(align_features([1, 2, 3], names, names), [1, 2, 3])

    def test_reordered_shard_is_reordered_into_target_order(self):
        self.assertEqual(align_features([30, 10, 20], ["c", "a", "b"], ["a", "b", "c"]),
                         [10, 20, 30])

    def test_missing_feature_raises(self):
        with self.assertRaises(KeyError):
            align_features([1, 2], ["a", "b"], ["a", "b", "c"])

    def test_extra_shard_features_are_dropped_and_target_order_kept(self):
        self.assertEqual(align_features([9, 1, 2, 8], ["z", "a", "b", "y"], ["a", "b"]),
                         [1, 2])


if __name__ == "__main__":
    unittest.main()
