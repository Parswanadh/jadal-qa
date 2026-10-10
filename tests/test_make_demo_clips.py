"""Tests for scripts/make_demo_clips.py (issue #38).

Builds a tiny synthetic source video in a temp dir (never touches the real
IndEgo clips), runs the generator's build_source_set twice, and asserts:
every defect variant is produced, the manifest entries match the files,
sha256s are stable across two runs, and garbage/truncated fail to decode
with cv2 while good ones decode. Fast (<30 s) and hang-free (no waits;
one subprocess call carries an explicit timeout).
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "make_demo_clips.py")


def load_module():
    spec = importlib.util.spec_from_file_location("make_demo_clips", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mdc = load_module()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def decodable(path):
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return False
        ok, frame = cap.read()
        return bool(ok) and frame is not None
    finally:
        cap.release()


def read_all_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def make_synthetic_source(path, n_frames=20, w=64, h=64, fps=10.0):
    """Deterministic moving-square clip so blur/dark/bright/frozen differ."""
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened(), "VideoWriter failed in test setup"
    try:
        for i in range(n_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.rectangle(frame, (i % (w - 16), 8), ((i % (w - 16)) + 16, 24), (200, 120, 60), -1)
            cv2.circle(frame, (w // 2, h // 2 + (i % 9) - 4), 6, (60, 200, 120), -1)
            vw.write(frame)
    finally:
        vw.release()


class TestMakeDemoClips(unittest.TestCase):
    def test_all_variants_manifest_determinism_and_decodability(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.mp4")
            make_synthetic_source(src)
            out1 = os.path.join(tmp, "out1")
            out2 = os.path.join(tmp, "out2")
            os.makedirs(out1)
            os.makedirs(out2)

            e1 = mdc.build_source_set(src, "synthetic/src.mp4", "t", 0.0, 2.0, out1, 7, 0)
            e2 = mdc.build_source_set(src, "synthetic/src.mp4", "t", 0.0, 2.0, out2, 7, 0)

            # Every defect variant is produced.
            variants = sorted(e["variant"] for e in e1)
            self.assertEqual(
                variants,
                sorted(
                    [
                        "good",
                        "blurred",
                        "too_dark",
                        "overexposed",
                        "frozen",
                        "truncated",
                        "garbage",
                        "reupload_duplicate",
                    ]
                ),
            )
            for e in e1:
                self.assertTrue(os.path.isfile(os.path.join(out1, e["file"])), e["file"])
                # Manifest matches the files; labels are synthetic-by-construction.
                self.assertEqual(e["label"], "SYNTHETIC-DEFECT")
                self.assertEqual(e["sha256"], sha256_file(os.path.join(out1, e["file"])))

            # sha256s stable across two runs for the deterministic ones (all 8).
            h1 = {e["file"]: e["sha256"] for e in e1}
            h2 = {e["file"]: e["sha256"] for e in e2}
            self.assertEqual(h1, h2)

            # Re-upload duplicate is byte-identical to good.
            with open(os.path.join(out1, "t_good.mp4"), "rb") as f:
                good = f.read()
            with open(os.path.join(out1, "t_reupload_duplicate.mp4"), "rb") as f:
                dup = f.read()
            self.assertEqual(good, dup)

            # Good-family decodes; garbage/truncated fail.
            for v in ("good", "blurred", "too_dark", "overexposed", "frozen", "reupload_duplicate"):
                self.assertTrue(decodable(os.path.join(out1, f"t_{v}.mp4")), v)
            self.assertFalse(decodable(os.path.join(out1, "t_garbage.mp4")))
            self.assertFalse(decodable(os.path.join(out1, "t_truncated.mp4")))

            # Defects actually change the pixels: frozen frames are all equal,
            # dark is dimmer and bright is brighter than good (decoded means).
            frozen = read_all_frames(os.path.join(out1, "t_frozen.mp4"))
            self.assertGreaterEqual(len(frozen), 2)
            for f in frozen[1:]:
                np.testing.assert_array_equal(frozen[0], f)
            mean = lambda p: float(np.mean(read_all_frames(p)[0].astype(float)))
            self.assertLess(mean(os.path.join(out1, "t_too_dark.mp4")), mean(os.path.join(out1, "t_good.mp4")))
            self.assertGreater(
                mean(os.path.join(out1, "t_overexposed.mp4")), mean(os.path.join(out1, "t_good.mp4"))
            )

    def test_list_flag(self):
        proc = subprocess.run(
            [sys.executable, SCRIPT, "--list"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0)
        lines = proc.stdout.strip().splitlines()
        self.assertEqual(len(lines), 16)  # 2 sources x 8 variants
        self.assertIn("ecu_good.mp4", lines)
        self.assertIn("logistics_garbage.mp4", lines)


if __name__ == "__main__":
    unittest.main()
