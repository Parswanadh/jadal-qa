"""Tests for scripts/make_demo_clips.py (issue #38).

Builds tiny synthetic source videos in a temp dir (never touches the real
IndEgo clips) and asserts: every defect variant is produced with the right
defect code and label (SYNTHETIC-GOOD only for the clean copy), the manifest
entries match the files, byte-ops are sha-stable across runs while re-encoded
mp4s match by decoded pixels (NOT byte-hash: encoder builds differ), the
duplicate is byte-identical to good, wrong_fps really is 15 fps, garbage /
truncated fail to decode while the rest decode, --out outside data_runtime/
is refused, and read_window resamples odd-fps sources to 10 fps.
Fast (<30 s) and hang-free (no waits; subprocess calls carry timeouts).
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

ALL_VARIANTS = [
    "good",
    "blurred",
    "too_dark",
    "overexposed",
    "frozen",
    "wrong_fps",
    "truncated",
    "garbage",
    "reupload_duplicate",
]
REENCODED = ("good", "blurred", "too_dark", "overexposed", "frozen", "wrong_fps")
BYTE_OPS = ("truncated", "garbage", "reupload_duplicate")


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
    def test_all_variants_labels_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.mp4")
            make_synthetic_source(src)
            out1 = os.path.join(tmp, "out1")
            out2 = os.path.join(tmp, "out2")
            os.makedirs(out1)
            os.makedirs(out2)

            (e1, size1) = mdc.build_source_set(src, "synthetic/src.mp4", "t", 0.0, 2.0, out1, 7, 0)
            (e2, size2) = mdc.build_source_set(src, "synthetic/src.mp4", "t", 0.0, 2.0, out2, 7, 0)
            self.assertEqual(size1, (64, 64))
            self.assertEqual(size2, (64, 64))

            # Every defect variant is produced.
            self.assertEqual(sorted(e["variant"] for e in e1), sorted(ALL_VARIANTS))
            by_name = {e["file"]: e for e in e1}
            for e in e1:
                self.assertTrue(os.path.isfile(os.path.join(out1, e["file"])), e["file"])
                self.assertEqual(e["sha256"], sha256_file(os.path.join(out1, e["file"])))

            # Labels: only the clean copy is SYNTHETIC-GOOD; the duplicate
            # carries defect "duplicate" pointing at the good file + its sha.
            self.assertEqual(by_name["t_good.mp4"]["label"], "SYNTHETIC-GOOD")
            self.assertEqual(by_name["t_good.mp4"]["defect"], "none")
            dup = by_name["t_reupload_duplicate.mp4"]
            self.assertEqual(dup["label"], "SYNTHETIC-DEFECT")
            self.assertEqual(dup["defect"], "duplicate")
            self.assertEqual(dup["parameters"]["copy_of"], "t_good.mp4")
            self.assertEqual(dup["parameters"]["good_sha256"], by_name["t_good.mp4"]["sha256"])
            for v in ("blurred", "too_dark", "overexposed", "frozen", "wrong_fps", "truncated", "garbage"):
                self.assertEqual(by_name[f"t_{v}.mp4"]["label"], "SYNTHETIC-DEFECT", v)
            self.assertEqual(by_name["t_wrong_fps.mp4"]["defect"], "wrong_fps")
            self.assertEqual(by_name["t_wrong_fps.mp4"]["fps"], 15.0)
            self.assertEqual(by_name["t_good.mp4"]["actual_duration_s"], 2.0)
            self.assertAlmostEqual(by_name["t_wrong_fps.mp4"]["actual_duration_s"], 20 / 15.0, places=2)

            # Byte-stability across runs ONLY for pure byte ops (M6: re-encoded
            # mp4s are not guaranteed stable across OpenCV builds).
            h1 = {e["file"]: e["sha256"] for e in e1}
            h2 = {e["file"]: e["sha256"] for e in e2}
            for v in BYTE_OPS:
                self.assertEqual(h1[f"t_{v}.mp4"], h2[f"t_{v}.mp4"], v)

            # Re-encoded variants match across runs by decoded pixels instead.
            for v in REENCODED:
                f1 = read_all_frames(os.path.join(out1, f"t_{v}.mp4"))
                f2 = read_all_frames(os.path.join(out2, f"t_{v}.mp4"))
                self.assertEqual(len(f1), len(f2), v)
                self.assertGreater(len(f1), 0, v)
                diff = float(np.mean(np.abs(f1[0].astype(float) - f2[0].astype(float))))
                self.assertLess(diff, 5.0, v)

            # Manifest is deterministic modulo file hashes: compare with
            # sha256 fields redacted, so this holds on any OpenCV build
            # (full-bytes equality would bake in same-build encoder bytes).
            mdc.write_manifest(out1, 7, e1, size1[0], size1[1])
            mdc.write_manifest(out2, 7, e2, size2[0], size2[1])
            with open(os.path.join(out1, "manifest.json")) as f:
                j1 = json.load(f)
            with open(os.path.join(out2, "manifest.json")) as f:
                j2 = json.load(f)
            for entry in j1["files"] + j2["files"]:
                entry["sha256"] = "redacted"
                entry.get("parameters", {}).pop("good_sha256", None)
            self.assertEqual(j1, j2)

            # Re-upload duplicate is byte-identical to good.
            with open(os.path.join(out1, "t_good.mp4"), "rb") as f:
                good = f.read()
            with open(os.path.join(out1, "t_reupload_duplicate.mp4"), "rb") as f:
                dup_bytes = f.read()
            self.assertEqual(good, dup_bytes)

            # Decodable set decodes (incl. wrong_fps at ~15 fps); garbage /
            # truncated fail.
            for v in ("good", "blurred", "too_dark", "overexposed", "frozen", "reupload_duplicate"):
                self.assertTrue(decodable(os.path.join(out1, f"t_{v}.mp4")), v)
            self.assertTrue(decodable(os.path.join(out1, "t_wrong_fps.mp4")))
            cap = cv2.VideoCapture(os.path.join(out1, "t_wrong_fps.mp4"))
            try:
                self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 15.0, places=1)
                self.assertEqual(len(read_all_frames(os.path.join(out1, "t_wrong_fps.mp4"))), 20)
            finally:
                cap.release()
            self.assertFalse(decodable(os.path.join(out1, "t_garbage.mp4")))
            self.assertFalse(decodable(os.path.join(out1, "t_truncated.mp4")))

            # Defects change pixels: frozen frames near-equal (lossy noise
            # only; closeness, not exact equality), dark dimmer, bright
            # brighter than good (decoded means).
            frozen = read_all_frames(os.path.join(out1, "t_frozen.mp4"))
            self.assertGreaterEqual(len(frozen), 2)
            for f in frozen[1:]:
                self.assertLess(float(np.mean(np.abs(f.astype(float) - frozen[0].astype(float)))), 10.0)
            mean = lambda p: float(np.mean(read_all_frames(p)[0].astype(float)))
            self.assertLess(mean(os.path.join(out1, "t_too_dark.mp4")), mean(os.path.join(out1, "t_good.mp4")))
            self.assertGreater(
                mean(os.path.join(out1, "t_overexposed.mp4")), mean(os.path.join(out1, "t_good.mp4"))
            )

    def test_read_window_resamples_to_fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src5.mp4")
            make_synthetic_source(src, n_frames=10, fps=5.0)
            frames, _start, size, src_fps = mdc.read_window(src, 0.0, 2.0)
            self.assertAlmostEqual(src_fps, 5.0, places=1)
            self.assertEqual(len(frames), 20)  # resampled 5 fps -> 10 fps
            self.assertEqual(size, (64, 64))

    def test_short_read_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "short.mp4")
            make_synthetic_source(src, n_frames=5, fps=5.0)
            with self.assertRaises(RuntimeError):
                mdc.read_window(src, 0.0, 2.0)

    def test_out_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            outside = os.path.join(tmp, "elsewhere")
            proc = subprocess.run(
                [sys.executable, SCRIPT, "--out", outside],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 3)
            self.assertFalse(os.path.exists(os.path.join(outside, "manifest.json")))

    def test_list_flag(self):
        proc = subprocess.run(
            [sys.executable, SCRIPT, "--list"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0)
        lines = proc.stdout.strip().splitlines()
        self.assertEqual(len(lines), 18)  # 2 sources x 9 variants
        self.assertIn("ecu_good.mp4", lines)
        self.assertIn("ecu_wrong_fps.mp4", lines)
        self.assertIn("logistics_garbage.mp4", lines)


if __name__ == "__main__":
    unittest.main()
