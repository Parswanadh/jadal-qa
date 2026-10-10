"""Tests for tools/exp.py. The tool reads JADAL_EXP_DIR at import time, so it runs as a subprocess."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "tools" / "exp.py"


class ExpRegistryTest(unittest.TestCase):
    def run_exp(self, env, *args):
        return subprocess.run([sys.executable, str(EXP), *args], env=env,
                              capture_output=True, text=True, timeout=30)

    def test_new_finish_render_in_temp_dir(self):
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, JADAL_EXP_DIR=tmp)
            r = self.run_exp(env, "new", "--id", "T0.01", "--track", "T0", "--title", "unit test",
                             "--why", "test", "--hypothesis", "h", "--data", "none",
                             "--regime", "n/a", "--method", "m", "--owner", "test")
            self.assertEqual(r.returncode, 0, r.stderr)
            r = self.run_exp(env, "finish", "--id", "T0.01", "--verdict", "inconclusive",
                             "--summary", "smoke test", "--results", "none")
            self.assertEqual(r.returncode, 0, r.stderr)
            r = self.run_exp(env, "list", "--track", "T0")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("T0.01", r.stdout)
            r = self.run_exp(env, "render")
            self.assertEqual(r.returncode, 0, r.stderr)

            reg = Path(tmp, "registry.jsonl").read_text().splitlines()
            self.assertEqual(len(reg), 2)
            first, second = (json.loads(x) for x in reg)
            self.assertEqual(first["status"], "running")
            self.assertEqual(second["verdict"], "inconclusive")
            self.assertTrue(Path(tmp, "T0.01", "README.md").exists())
            self.assertTrue(Path(tmp, "EXPERIMENTS.md").exists())
        self.assertLess(time.monotonic() - t0, 5.0)

    def test_unknown_id_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, JADAL_EXP_DIR=tmp)
            r = self.run_exp(env, "finish", "--id", "X9.99", "--verdict", "drop", "--summary", "s")
            self.assertNotEqual(r.returncode, 0)

    def test_repo_registry_is_untouched(self):
        # The committed registry must stay empty: tests never write to it.
        self.assertEqual((REPO / "experiments" / "registry.jsonl").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
