"""Hardware-free tests for jadal.freshness (synthetic generator only)."""
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

from jadal import freshness as F


def _clip(nonce, **kw):
    kw.setdefault("seed", 3)
    kw.setdefault("noise_sigma", 5.0)
    frames, ts, start = F.synthesize_frames(nonce, **kw)
    return frames, ts, start


class TestNonce(unittest.TestCase):
    def test_issue_shape_and_determinism(self):
        n = F.issue_nonce(rng=random.Random(7))
        self.assertEqual(len(n.hex), 8)
        self.assertEqual(n.value, int(n.hex, 16))
        self.assertEqual(n.hex, F.issue_nonce(rng=random.Random(7)).hex)

    def test_hash_is_commitment_not_hiding(self):
        # Binding: deterministic sha256 of the 8-hex value. NOT hiding: the
        # preimage is 32 bits and enumerable (~2.1M hashes/s single core, so
        # secrecy holds only inside the validity window). This test pins the
        # construction, not secrecy.
        n = F.issue_nonce(rng=random.Random(1))
        self.assertEqual(F.nonce_hash(n),
                         hashlib.sha256(n.hex.encode()).hexdigest())
        self.assertEqual(len(F.nonce_hash(n)), 64)

    def test_bits_msb_first(self):
        self.assertEqual(F.bits_of(0x80000000)[:4], [1, 0, 0, 0])
        self.assertEqual(len(F.bits_of(0x12345678)), 32)

    def test_expired_method(self):
        n = F.issue_nonce(rng=random.Random(2), validity_s=30.0)
        self.assertFalse(n.is_expired(now=n.issued_at + 10.0))
        self.assertTrue(n.is_expired(now=n.issued_at + 31.0))
        self.assertFalse(n.expired)  # just issued


class TestVerifySynthetic(unittest.TestCase):
    CFG = {"null_n": 300}

    def _cfg(self, ts, extra=None):
        d = dict(self.CFG)
        d["timestamps"] = ts
        if extra:
            d.update(extra)
        return d

    def test_correct_nonce_fresh_with_right_lag(self):
        nonce = F.issue_nonce(rng=random.Random(11))
        frames, ts, start = _clip(nonce)
        res = F.verify_clip(frames, nonce, self._cfg(ts))
        self.assertEqual(res["status"], "fresh", res)
        self.assertEqual(res["bits_matched"], 32)
        self.assertEqual(res["reason"], "ok")
        self.assertAlmostEqual(res["lag_ms"], start / 30.0 * 1000.0, delta=150.0)
        self.assertGreaterEqual(res["z"], 3.0)
        self.assertAlmostEqual(res["fps_measured"], 30.0, delta=0.5)

    def test_wrong_nonce_stale(self):
        nonce = F.issue_nonce(rng=random.Random(21))
        other = F.issue_nonce(rng=random.Random(22))
        frames, ts, _ = _clip(nonce)
        res = F.verify_clip(frames, other, self._cfg(ts))
        self.assertEqual(res["status"], "stale", res)
        self.assertEqual(res["reason"], "mismatch")

    def test_low_entropy_nonce_not_inflated(self):
        # 0x0000FFFF: long runs, still decodable; z must stay on the same
        # scale (no 1/std(pattern) inflation — null uses identical Pearson).
        nonce = F.Nonce(value=0x0000FFFF, hex="0000ffff",
                        issued_at=1_700_000_000.0)
        frames, ts, _ = _clip(nonce)
        res = F.verify_clip(frames, nonce, self._cfg(ts))
        self.assertEqual(res["status"], "fresh", res)
        self.assertEqual(res["bits_matched"], 32)
        self.assertLess(res["z"], 15.0, res)  # same scale as balanced nonces
        other = F.issue_nonce(rng=random.Random(23))
        res2 = F.verify_clip(frames, other, self._cfg(ts))
        self.assertEqual(res2["status"], "stale", res2)

    def test_no_led_inconclusive(self):
        nonce = F.issue_nonce(rng=random.Random(31))
        frames, ts, _ = _clip(nonce, on=20.0, off=20.0, noise_sigma=2.0)
        res = F.verify_clip(frames, nonce, self._cfg(ts))
        self.assertEqual(res["status"], "inconclusive", res)
        self.assertEqual(res["reason"], "no-signal")

    def test_fps_25_and_15(self):
        for fps in (25.0, 15.0):
            nonce = F.issue_nonce(rng=random.Random(60 + int(fps)))
            frames, ts, _ = _clip(nonce, fps=fps)
            res = F.verify_clip(frames, nonce, self._cfg(ts))
            self.assertEqual(res["status"], "fresh", (fps, res))
            self.assertEqual(res["bits_matched"], 32)
            self.assertAlmostEqual(res["lag_ms"], 1000.0, delta=200.0)
            self.assertAlmostEqual(res["fps_measured"], fps, delta=0.5)

    def test_container_timestamps_path(self):
        # Written mp4 at 25 fps, verified with NO explicit timestamps: the
        # container timing must carry the decode.
        nonce = F.issue_nonce(rng=random.Random(71))
        frames, _, _ = _clip(nonce, fps=25.0)
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "c25.mp4")
            F.write_video(frames, p, fps=25.0)
            res = F.verify_clip(p, nonce, dict(self.CFG))
        self.assertEqual(res["status"], "fresh", res)
        self.assertAlmostEqual(res["fps_measured"], 25.0, delta=1.0)

    def test_expiry_enforced(self):
        nonce = F.Nonce(value=0x12345678, hex="12345678",
                        issued_at=1_700_000_000.0, validity_s=30.0)
        frames, ts, _ = _clip(nonce)
        dur = ts[-1] - ts[0]
        # Clip fully inside the window -> decided on signal.
        ok = F.verify_clip(frames, nonce, self._cfg(
            ts, {"clip_start_wall": 1_700_000_000.0 + 5.0}))
        self.assertEqual(ok["status"], "fresh", ok)
        # Clip starts after the window closed -> stale/expired, no decode.
        old = F.verify_clip(frames, nonce, self._cfg(
            ts, {"clip_start_wall": 1_700_000_000.0 + 31.0}))
        self.assertEqual(old["status"], "stale", old)
        self.assertEqual(old["reason"], "expired", old)
        self.assertIsNone(old["z"])
        # Clip ends before the nonce was issued -> stale/expired.
        early = F.verify_clip(frames, nonce, self._cfg(
            ts, {"clip_start_wall": 1_700_000_000.0 - dur - 1.0}))
        self.assertEqual((early["status"], early["reason"]),
                         ("stale", "expired"), early)

    def test_no_nonce_value_in_artifacts(self):
        nonce = F.issue_nonce(rng=random.Random(81))
        frames, ts, _ = _clip(nonce)
        with tempfile.TemporaryDirectory() as d:
            meta = {"fps": 30.0, "frame_times": ts,
                    "capture_start_wall": 1_700_000_000.0,
                    "nonce_hash": F.nonce_hash(nonce)}
            saved = F.save_recording(frames, meta, d, F.nonce_hash(nonce),
                                     stamp="20261010_000000")
            base = Path(saved["path"]).name
            self.assertNotIn(nonce.hex, base)
            self.assertIn(F.nonce_hash(nonce)[:16], base)
            side = Path(saved["ts_path"]).read_text()
            self.assertNotIn(nonce.hex, side)
            res = F.verify_clip(saved["path"], nonce, dict(self.CFG))
            for v in res.values():
                if isinstance(v, str):
                    self.assertNotIn(nonce.hex, v)

    def test_determinism(self):
        nonce = F.issue_nonce(rng=random.Random(41))
        frames, ts, _ = _clip(nonce, noise_sigma=8.0)
        r1 = F.verify_clip(frames, nonce, self._cfg(ts))
        r2 = F.verify_clip(frames, nonce, self._cfg(ts))
        self.assertEqual(r1, r2)

    def test_receipt_field_shape(self):
        nonce = F.issue_nonce(rng=random.Random(51))
        frames, ts, _ = _clip(nonce)
        res = F.verify_clip(frames, nonce, self._cfg(ts))
        self.assertEqual(set(res), {"nonce_hash", "status", "lag_ms", "z",
                                    "bits_matched", "bits_total", "reason",
                                    "fps_measured", "fps_source"})
        self.assertEqual(res["bits_total"], 32)
        self.assertEqual(len(res["nonce_hash"]), 64)


if __name__ == "__main__":
    unittest.main()
