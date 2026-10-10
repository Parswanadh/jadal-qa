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

    def test_frame_bytes_raises(self):
        # I1: fail-loud — missing file and bad window raise, never NaN dicts.
        from jadal.perpixel import physics as P
        with self.assertRaises(FileNotFoundError):
            P.frame_bytes("/nonexistent_pr72.mp4", 0, 10.0)
        with self.assertRaises(FileNotFoundError):
            P.extract_physics(
                __import__("numpy").zeros((8, 45, 80, 3)),
                10.0, "/nonexistent_pr72.mp4", 0, 1.0)

    def test_fit_length_no_tiling(self):
        # I4: short series are zero-padded, never tiled.
        from jadal.perpixel import series as S
        import numpy as np
        a = np.arange(24 * 8, dtype=float).reshape(24, 8)
        b = S.fit_length(a, 100)
        self.assertEqual(b.shape, (100, 8))
        np.testing.assert_array_equal(b[24:], np.zeros((76, 8)))
        np.testing.assert_array_equal(b[:24], a)
        c = S.fit_length(np.ones((150, 8)), 100)
        self.assertEqual(c.shape, (100, 8))

    def test_dhash_hamming(self):
        from jadal.perpixel import physics as P
        rng = np.random.RandomState(6)
        g = np.clip(0.5 + rng.randn(90, 160) * 0.05, 0, 1)
        self.assertEqual(P._ham(P.dhash(g), P.dhash(g.copy())), 0)


class TestScorerContract(unittest.TestCase):
    """I7: scorer decisions/digest/config on a tiny synthetic clip."""

    # M3: anchor to repo root (not cwd).
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @classmethod
    def setUpClass(cls):
        import subprocess
        import tempfile
        cls.tmp = tempfile.mkdtemp(prefix="perpixel_test_")
        cls.clip = os.path.join(cls.tmp, "tiny.mp4")
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-f", "lavfi",
             "-i", "testsrc=size=64x48:rate=5:duration=3",
             "-pix_fmt", "yuv420p", "-c:v", "mpeg4", cls.clip],
            capture_output=True, timeout=120)
        if p.returncode != 0 or not os.path.exists(cls.clip):
            raise unittest.SkipTest("ffmpeg testsrc encode unavailable")

    def test_config_thresholds_present(self):
        import json
        cfg = json.load(open(os.path.join(self.ROOT,
                                          "configs/perpixel_v1.json")))
        self.assertEqual(
            sorted(r["name"] for r in cfg["quality_stage"]["rules"]),
            ["blur", "lowent", "ten"])
        for r in cfg["quality_stage"]["rules"]:
            self.assertIn("thr_int", r)
            self.assertIsInstance(r["thr_int"], int)
        self.assertIn("loop_tripwire", cfg)
        self.assertTrue(cfg["activity"]["weights_sha256"])
        self.assertTrue(cfg["activity"]["weights"].endswith(".json"))

    def test_digest_repeat_identical(self):
        from jadal.perpixel import scorer as SC
        a = SC.score(self.clip, 0, 2.0)
        b = SC.score(self.clip, 0, 2.0)
        self.assertEqual(a["features_digest"], b["features_digest"])
        self.assertEqual(len(a["features_digest"]), 64)
        self.assertEqual(a["decision"], b["decision"])
        for k in ("value", "op", "threshold"):
            self.assertIn(k, a["reasons"]["lowent"])

    def test_mv_excluded_from_digest(self):
        import json
        cfg = json.load(open(os.path.join(self.ROOT,
                                          "configs/perpixel_v1.json")))
        self.assertIn("B_mv_per_frame", cfg.get("digest_exclude", []))

    def test_duplicate_wired(self):
        from jadal.perpixel import scorer as SC
        a = SC.score(self.clip, 0, 2.0)
        self.assertIn("loop_tripwire", a["reasons"])
        self.assertIn("hit", a["reasons"]["loop_tripwire"])
        # I2/I3: duplicate.hit == loop hit and carries the GOVERNING ints,
        # never the offline corpus threshold as "threshold".
        dup = a["reasons"]["duplicate"]
        self.assertEqual(dup["hit"], a["reasons"]["loop_tripwire"]["hit"])
        self.assertEqual(dup["via"], "loop_tripwire")
        self.assertIn("dhash_max", dup)
        self.assertIn("still_min", dup)
        self.assertNotIn("threshold", dup)
        self.assertIn("offline_corpus_threshold", dup)

    def test_config_path_selects_net(self):
        # I1: score(config_path=alt) uses the ALT config's model+threshold.
        # Tampered weights copy MUST raise (old code loaded the default net
        # and would silently score with the wrong model).
        import copy
        import hashlib
        import json
        import shutil
        import tempfile
        from jadal.perpixel import scorer as SC
        base = json.load(open(os.path.join(
            self.ROOT, "configs/perpixel_v1.json")))
        wsrc = os.path.join(self.ROOT, base["activity"]["weights"])
        tmp = tempfile.mkdtemp(prefix="perpixel_cfg_")
        try:
            wtamper = os.path.join(tmp, "tampered.json")
            shutil.copy(wsrc, wtamper)
            tj = json.load(open(wtamper))
            tj["threshold"] = 0.9999
            json.dump(tj, open(wtamper, "w"))
            alt = copy.deepcopy(base)
            alt["activity"]["weights"] = wtamper
            alt_path = os.path.join(tmp, "alt.json")
            json.dump(alt, open(alt_path, "w"))
            with self.assertRaises(ValueError):
                SC.score(self.clip, 0, 2.0, config_path=alt_path)
            # untampered copy at an alt path scores fine (same sha)
            wok = os.path.join(tmp, "ok.json")
            shutil.copy(wsrc, wok)
            alt2 = copy.deepcopy(base)
            alt2["activity"]["weights"] = wok
            alt2["activity"]["threshold"] = 0.9999
            alt2_path = os.path.join(tmp, "alt2.json")
            json.dump(alt2, open(alt2_path, "w"))
            b = SC.score(self.clip, 0, 2.0, config_path=alt2_path)
            self.assertEqual(b["reasons"]["activity"]["threshold"], 0.9999)
            self.assertTrue(
                hashlib.sha256(open(wok, "rb").read()).hexdigest()
                == base["activity"]["weights_sha256"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


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
