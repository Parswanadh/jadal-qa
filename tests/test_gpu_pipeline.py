"""GPU-pipeline tests: decode helpers, S0 rules, determinism digest, math.

No GPU, no network, no git-ignored data (< 30 s). Test clips are generated
on the fly with ffmpeg lavfi testsrc.
"""
import json
import os
import subprocess
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.path.join(REPO, ".scratch")
os.makedirs(SCRATCH, exist_ok=True)

from eval.gpu.decode import clip_duration, decode_window, safe_offset  # noqa: E402
from eval.gpu.determinism import canon, digest  # noqa: E402
from eval.gpu.stream import S0_RULES, s0_probe_eval  # noqa: E402
from eval.gpu.throughput import windows_per_hour  # noqa: E402


def gen_clip(path, dur=2, w=160, h=90, fps=10):
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin", "-y",
           "-f", "lavfi", "-i", f"testsrc=duration={dur}:size={w}x{h}:rate={fps}",
           "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, check=True, timeout=60)


class TestDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(dir=SCRATCH)
        cls.clip = os.path.join(cls.d, "t.mp4")
        try:
            gen_clip(cls.clip)
        except Exception as e:
            raise unittest.SkipTest(f"ffmpeg lavfi unavailable: {e}")

    def test_duration(self):
        self.assertAlmostEqual(clip_duration(self.clip), 2.0, delta=0.2)

    def test_safe_offset_in_range(self):
        for i in range(5):
            o = safe_offset(self.clip, i, 1.0)
            self.assertGreaterEqual(o, 0)
            self.assertLess(o + 1.0, clip_duration(self.clip) + 0.01)

    def test_decode_rgb_shape(self):
        arr, fps = decode_window(self.clip, 0.0, 1.0, 160, 90, "rgb24")
        self.assertEqual(arr.shape[1:], (90, 160, 3))
        self.assertTrue((arr >= 0).all() and (arr <= 1).all())
        self.assertAlmostEqual(fps, 10.0, delta=1.0)

    def test_decode_gray_shape(self):
        arr, _ = decode_window(self.clip, 0.0, 1.0, 80, 45, "gray")
        self.assertEqual(arr.shape[1:], (45, 80))


class TestS0Rules(unittest.TestCase):
    def test_dark_fires(self):
        s0 = {"mean": 0.01, "p5": 0.005, "p95": 0.02, "spatial_var": 1e-4,
              "temp_e": 0.01}
        hit = s0_probe_eval(s0)
        self.assertIsNotNone(hit)
        self.assertIn(hit[0], ("black", "dark"))

    def test_clean_passes(self):
        s0 = {"mean": 0.45, "p5": 0.2, "p95": 0.8, "spatial_var": 1e-2,
              "temp_e": 0.05}
        self.assertIsNone(s0_probe_eval(s0))

    def test_frozen_fires(self):
        s0 = {"mean": 0.45, "p5": 0.2, "p95": 0.8, "spatial_var": 1e-2,
              "temp_e": 0.0}
        hit = s0_probe_eval(s0)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], "frozen")

    def test_rules_have_evidence_shape(self):
        for name, key, op, thr in S0_RULES:
            self.assertIn(op, ("<", ">"))
            self.assertIsInstance(thr, float)


class TestDeterminism(unittest.TestCase):
    def test_canon_stable(self):
        f = {"b": 0.123456, "a": 1.0, "fp_hash": "ab12"}
        self.assertEqual(digest(f), digest(dict(reversed(list(f.items())))))

    def test_digest_sensitive(self):
        self.assertNotEqual(digest({"x": 0.1234}), digest({"x": 0.1235}))

    def test_canon_json_sorted(self):
        q = json.loads(canon({"z": 1.0, "a": 2.0}))
        self.assertEqual(list(q.keys()), ["a", "z"])


class TestMath(unittest.TestCase):
    def test_hours_per_hour(self):
        self.assertAlmostEqual(windows_per_hour(2.0), 20.0)  # 10-s windows
        self.assertAlmostEqual(windows_per_hour(0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
