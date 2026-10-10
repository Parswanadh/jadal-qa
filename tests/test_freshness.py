"""Hardware-free tests for jadal.freshness (synthetic generator only)."""
import random
import unittest

import numpy as np

from jadal import freshness as F


class TestNonce(unittest.TestCase):
    def test_issue_shape_and_determinism(self):
        rng = random.Random(7)
        n = F.issue_nonce(rng=rng)
        self.assertEqual(len(n.hex), 8)
        self.assertEqual(n.value, int(n.hex, 16))
        n2 = F.issue_nonce(rng=random.Random(7))
        self.assertEqual(n.hex, n2.hex)

    def test_hash_hides_value(self):
        n = F.issue_nonce(rng=random.Random(1))
        h = F.nonce_hash(n)
        self.assertEqual(len(h), 64)
        self.assertNotIn(n.hex, h)

    def test_bits_msb_first(self):
        self.assertEqual(F.bits_of(0x80000000)[:4], [1, 0, 0, 0])
        self.assertEqual(len(F.bits_of(0x12345678)), 32)


class TestVerifySynthetic(unittest.TestCase):
    def test_correct_nonce_fresh_with_right_lag(self):
        nonce = F.issue_nonce(rng=random.Random(11))
        frames, ts, start = F.synthesize_frames(nonce, seed=3, noise_sigma=5.0)
        res = F.verify_clip(frames, nonce, {"timestamps": ts})
        self.assertEqual(res["status"], "fresh", res)
        self.assertGreaterEqual(res["bits_matched"], 30)
        # lag should be near the embedded pattern start (pre_s=1.0s => ~1000ms)
        self.assertAlmostEqual(res["lag_ms"], start / 30.0 * 1000.0, delta=150.0)
        self.assertGreaterEqual(res["z"], 4.0)

    def test_wrong_nonce_stale(self):
        nonce = F.issue_nonce(rng=random.Random(21))
        other = F.issue_nonce(rng=random.Random(22))
        frames, ts, _ = F.synthesize_frames(nonce, seed=3, noise_sigma=5.0)
        res = F.verify_clip(frames, other, {"timestamps": ts})
        self.assertEqual(res["status"], "stale", res)

    def test_no_led_inconclusive(self):
        nonce = F.issue_nonce(rng=random.Random(31))
        frames, ts, _ = F.synthesize_frames(nonce, seed=5, on=20.0, off=20.0,
                                            noise_sigma=2.0)
        res = F.verify_clip(frames, nonce, {"timestamps": ts})
        self.assertEqual(res["status"], "inconclusive", res)

    def test_determinism(self):
        nonce = F.issue_nonce(rng=random.Random(41))
        frames, ts, _ = F.synthesize_frames(nonce, seed=9, noise_sigma=8.0)
        r1 = F.verify_clip(frames, nonce, {"timestamps": ts})
        r2 = F.verify_clip(frames, nonce, {"timestamps": ts})
        self.assertEqual(r1, r2)

    def test_receipt_field_shape(self):
        nonce = F.issue_nonce(rng=random.Random(51))
        frames, ts, _ = F.synthesize_frames(nonce, seed=3, noise_sigma=5.0)
        res = F.verify_clip(frames, nonce, {"timestamps": ts})
        self.assertEqual(set(res), {"nonce_hash", "status", "lag_ms", "z",
                                    "bits_matched", "bits_total"})
        self.assertEqual(res["bits_total"], 32)
        self.assertEqual(len(res["nonce_hash"]), 64)


if __name__ == "__main__":
    unittest.main()
