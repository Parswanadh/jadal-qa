"""Tests for the PoC stream replay (issue #64). Fast: no GPU, no network,
no real video (decode is monkeypatched with synthetic frames)."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jadal.pico_link import FakePicoSerial  # noqa: E402
from jadal.poc_stream import replay as R  # noqa: E402


def gray(v, n=4):
    return [np.full((45, 80), v, np.uint8) for _ in range(n)]


def bgr(v, n=4):
    return [np.full((45, 80, 3), v, np.uint8) for _ in range(n)]


class TestS0(unittest.TestCase):
    def test_black_fires(self):
        ev = R.s0_fire(R.window_stats(gray(0)), R.S0_RULES_FALLBACK)
        self.assertIsNotNone(ev)
        self.assertEqual(ev["rule"], "black")

    def test_overexposed_fires(self):
        rng = np.random.default_rng(2)
        frames = [(np.full((45, 80), 235, np.float64)
                   + rng.integers(-25, 25, (45, 80))).clip(0, 255).astype(np.uint8)
                  for _ in range(4)]
        ev = R.s0_fire(R.window_stats(frames), R.S0_RULES_FALLBACK)
        self.assertIsNotNone(ev)
        self.assertEqual(ev["rule"], "overexposed")

    def test_mid_grey_passes(self):
        rng = np.random.default_rng(0)
        frames = [(np.full((45, 80), 110, np.float64)
                   + rng.integers(-8, 8, (45, 80))).clip(0, 255).astype(np.uint8)
                  for _ in range(4)]
        self.assertIsNone(R.s0_fire(R.window_stats(frames), R.S0_RULES_FALLBACK))

    def test_frozen_fires(self):
        self.assertIsNotNone(
            R.s0_fire(R.window_stats(gray(110, n=6)), R.S0_RULES_FALLBACK))

    def test_single_frame_not_frozen(self):
        st = R.window_stats(gray(110, n=1))
        self.assertGreater(st["temp_e"], 0.1)


class TestDegrade(unittest.TestCase):
    def test_blackout(self):
        out, _ = R.apply_degrade(bgr(120), "blackout")
        self.assertTrue(all(f.max() == 0 for f in out))

    def test_freeze_repeats_first(self):
        frames = bgr(10) + bgr(200)
        out, _ = R.apply_degrade(frames, "freeze")
        self.assertTrue(all((f == out[0]).all() for f in out))

    def test_blur_reduces_sharpness(self):
        import cv2
        rng = np.random.default_rng(1)
        sharp = [rng.integers(0, 255, (45, 80, 3)).astype(np.uint8)]
        out, _ = R.apply_degrade(sharp, "blur")
        l0 = cv2.Laplacian(cv2.cvtColor(sharp[0], cv2.COLOR_BGR2GRAY),
                           cv2.CV_64F).var()
        l1 = cv2.Laplacian(cv2.cvtColor(out[0], cv2.COLOR_BGR2GRAY),
                           cv2.CV_64F).var()
        self.assertLess(l1, l0)


class TestPico(unittest.TestCase):
    def test_fake_serial_led(self):
        fake = FakePicoSerial()
        self.assertTrue(R.send_pico("early_reject", lock_path=os.path.join(
            tempfile.gettempdir(), "poc_test_pico.lock"), serial_obj=fake))
        self.assertIn("LED R", fake.pico.led_commands)

    def test_headless_logs_not_connected(self):
        self.assertFalse(R.send_pico("accept", lock_path=os.path.join(
            tempfile.gettempdir(), "poc_test_pico2.lock")))


class TestReplayDriver(unittest.TestCase):
    def _run(self, degrade="none", degrade_at=None, max_chunks=6):
        real_decode = R.decode_chunk

        def fake_decode(clip, start_s, chunk_s, **kw):
            rng = np.random.default_rng(int(start_s * 10) + 7)
            base = 110 + int(20 * np.sin(start_s))
            frames = []
            for _ in range(4):
                f = (np.full((90, 160, 3), base, np.float64)
                     + rng.integers(-30, 30, (90, 160, 3))).clip(0, 255)
                frames.append(f.astype(np.uint8))
            return frames, "cpu"

        R.decode_chunk = fake_decode
        R.make_spectral_stream = lambda: (None, "test-stub")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "events.jsonl")
                summary = R.replay("fake.mp4", out, speed=1000.0,
                                   max_chunks=max_chunks, degrade=degrade,
                                   degrade_at=degrade_at)
                evs = [json.loads(l) for l in open(out)]
            return summary, evs
        finally:
            R.decode_chunk = real_decode

    def test_event_schema_and_accept(self):
        summary, evs = self._run()
        self.assertEqual(len(evs), 6)
        for e in evs:
            for k in ("t", "window_id", "chunk", "running_score",
                      "decision", "reason"):
                self.assertIn(k, e)
            self.assertIn(e["decision"], ("continue", "early_reject", "accept"))
        self.assertIn("accept", [e["decision"] for e in evs])

    def test_blackout_rejects_fast(self):
        summary, evs = self._run(degrade="blackout", degrade_at=4.0)
        self.assertIsNotNone(summary["time_to_reject_s"])
        self.assertLessEqual(summary["time_to_reject_s"], 4.0)
        self.assertEqual(evs[2]["decision"], "early_reject")


if __name__ == "__main__":
    unittest.main()
