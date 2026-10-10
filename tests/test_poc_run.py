"""PoC unit tests: cascade logic + receipt hashing. <30s, no GPU/video/network."""
import hashlib
import json
import unittest

from jadal.poc import cascade as C
from jadal.poc import receipts as R


class TestS0(unittest.TestCase):
    def test_each_rule_fires(self):
        base = {"mean": 0.5, "p5": 0.3, "p95": 0.8, "spatial_var": 0.01,
                "temp_e": 0.01}
        cases = [("TOO_DARK", "mean", 0.01), ("BLACK", "p95", 0.01),
                 ("LENS_BLOCKED", "spatial_var", 1e-7),
                 ("OVEREXPOSED", "p5", 0.9), ("FROZEN", "temp_e", 1e-6)]
        for code, key, val in cases:
            s0 = dict(base)
            s0[key] = val
            rule, ev = C.s0_decide(s0)
            self.assertEqual(rule, code)
            self.assertEqual(ev["op"], ">" if code == "OVEREXPOSED" else "<")

    def test_clean_passes(self):
        s0 = {"mean": 0.5, "p5": 0.3, "p95": 0.8, "spatial_var": 0.01,
              "temp_e": 0.01}
        rule, _ = C.s0_decide(s0)
        self.assertIsNone(rule)

    def test_missing_keys_skipped(self):
        rule, _ = C.s0_decide({})
        self.assertIsNone(rule)


class FakeHead:
    """Deterministic stub: p = 0.9 if luma_mean high else 0.1."""

    def predict_proba(self, X):
        import numpy as np
        X = np.asarray(X)
        i = C.FEATURE_NAMES.index("luma_mean")
        p1 = (X[:, i] > 0.2).astype(float) * 0.8 + 0.1
        return np.stack([1 - p1, p1], axis=1)


class FakeScaler:
    def transform(self, X):
        return X


def feats(mean):
    f = {n: 0.0 for n in C.FEATURE_NAMES}
    f["luma_mean"] = mean
    return f


class TestDecide(unittest.TestCase):
    S0CLEAN = {"mean": 0.5, "p5": 0.3, "p95": 0.8, "spatial_var": 0.01,
               "temp_e": 0.01}

    def test_s0_reject(self):
        v, stage, reasons, score = C.decide_window(
            {"mean": 0.01, "p5": 0.0, "p95": 0.02, "spatial_var": 0.01,
             "temp_e": 0.01}, feats(0.5), FakeHead(), FakeScaler())
        self.assertEqual((v, stage), ("reject", "S0"))
        self.assertIsNone(score)

    def test_s1_accept_reject_review(self):
        v, s, _, score = C.decide_window(self.S0CLEAN, feats(0.9),
                                         FakeHead(), FakeScaler())
        self.assertEqual((v, s), ("accept", "S1"))
        self.assertGreaterEqual(score, C.THRESHOLD)
        v, s, _, _ = C.decide_window(self.S0CLEAN, feats(0.01),
                                     FakeHead(), FakeScaler())
        self.assertEqual((v, s), ("reject", "S1"))

    def test_review_bucket_is_s2(self):
        class Mid:
            def predict_proba(self, X):
                import numpy as np
                return np.tile([0.45, 0.55], (len(X), 1))
        v, s, reasons, _ = C.decide_window(self.S0CLEAN, feats(0.5),
                                           Mid(), FakeScaler())
        self.assertEqual((v, s), ("review", "S2"))
        self.assertEqual(reasons[0]["code"], "UNCERTAIN")


class TestReceipts(unittest.TestCase):
    def test_stable_and_sensitive(self):
        kw = dict(window_id="f001_w001_c1_t0000", verdict="accept",
                  reasons=[{"code": "S1_USABLE"}], feats_digest="ab12",
                  versions={"code_sha": "x"})
        h1 = R.receipt_for(**kw)
        self.assertEqual(h1, R.receipt_for(**kw))
        self.assertEqual(len(h1), 64)
        kw2 = dict(kw, verdict="reject")
        self.assertNotEqual(h1, R.receipt_for(**kw2))

    def test_canonical_key_order(self):
        a = R.sha256_hex({"b": 1, "a": 2})
        b = R.sha256_hex({"a": 2, "b": 1})
        self.assertEqual(a, b)

    def test_features_digest_rounding(self):
        d1 = R.features_digest({"x": 0.123456})
        d2 = R.features_digest({"x": 0.123454})
        self.assertEqual(d1, d2)  # same at 4dp


if __name__ == "__main__":
    unittest.main()
