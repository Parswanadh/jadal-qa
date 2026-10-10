"""Unit tests for scripts/baseline_report.py against a fixture directory.

Fast and hermetic: no test-suite runs, no service, no network. The script's
check functions all take a repo-root path, so fixtures stand in for main.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "baseline_report.py"


def load_script():
    spec = importlib.util.spec_from_file_location("baseline_report", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


br = load_script()


class FixtureMixin:
    def make_root(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "eval" / "results").mkdir(parents=True)
        (root / "jadal").mkdir()
        (root / "web").mkdir()
        (root / "docs" / "screens").mkdir(parents=True)
        return root

    def write(self, root, rel, obj, as_json=True):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if as_json:
            p.write_text(json.dumps(obj), encoding="utf-8")
        else:
            p.write_text(obj, encoding="utf-8")
        return p


class TestB4Numbers(FixtureMixin, unittest.TestCase):
    def test_missing_artifacts_pending_never_invented(self):
        root = self.make_root()
        self.write(root, "eval/results/splits.json",
                   {"label_source": "gemini-2.5-flash auto-labels",
                    "sizes": {"calibration": 1, "test": 2}, "seed": 1234})
        self.write(root, "eval/results/dataset_report.json",
                   {"distinct_frame_ids": 10, "total_rows": 30})
        res = br.check_b4(root)
        self.assertEqual(res["status"], "PENDING")
        self.assertIn("decider_baseline.json", res["evidence"])
        nums = res["numbers"]
        self.assertEqual(
            nums["dataset"]["split_sizes"]["source"],
            "eval/results/splits.json")
        self.assertEqual(
            nums["dataset"]["split_sizes"]["label_source"],
            "gemini-2.5-flash auto-labels")
        # no invented numbers: empty slots stay empty
        self.assertEqual(nums["decider"], {})
        self.assertEqual(nums["funnel"], {})
        self.assertEqual(nums["latency"], {})

    def test_all_artifacts_pass(self):
        root = self.make_root()
        self.write(root, "eval/results/splits.json",
                   {"label_source": "gemini", "sizes": {"test": 5}})
        self.write(root, "eval/results/dataset_report.json",
                   {"distinct_frame_ids": 5, "total_rows": 5})
        self.write(root, "eval/results/decider_baseline.json",
                   {"accuracy": 0.9, "ece": 0.05, "certified_bound": 0.1,
                    "n": 100, "label_source": "gemini"})
        self.write(root, "eval/results/funnel_calibration.json",
                   {"split": "test", "n": 50, "label_source": "gemini"})
        self.write(root, "eval/results/bench_eg2.json",
                   {"segments_per_hour": 100.0, "n": 7})
        res = br.check_b4(root)
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["numbers"]["decider"]["accuracy"]["value"], 0.9)
        self.assertEqual(res["numbers"]["decider"]["accuracy"]["n"], 100)
        self.assertEqual(
            res["numbers"]["latency"]["segments_per_hour"]["source"],
            "eval/results/bench_eg2.json")

    def test_synthetic_demo_excluded(self):
        root = self.make_root()
        self.write(root, "eval/results/DEMO_synthetic_calibration.json",
                   {"accuracy": 0.99, "n": 4498})
        res = br.check_b4(root)
        self.assertEqual(res["status"], "PENDING")
        self.assertNotIn("0.99", json.dumps(res["numbers"]))
        self.assertIn("EXCLUDED", str(res.get("found")))


class TestB5Funnel(FixtureMixin, unittest.TestCase):
    def test_no_funnel_fail(self):
        root = self.make_root()
        self.write(root, "jadal/pipeline.py", "x = 1\n", as_json=False)
        self.assertEqual(br.check_b5(root)["status"], "FAIL")

    def test_funnel_pass(self):
        root = self.make_root()
        self.write(root, "jadal/pipeline.py",
                   "from jadal import funnel\n", as_json=False)
        self.assertEqual(br.check_b5(root)["status"], "PASS")


class TestB3Parse(unittest.TestCase):
    DEMO = ("determinism: same clip 10x -> identical abc...\n"
            "determinism: alone vs batch-of-5 -> IDENTICAL receipt_sha256\n"
            "all receipts reverify MATCH\n")

    def test_full_demo_pass(self):
        self.assertEqual(br.check_b3(None, self.DEMO)["status"], "PASS")

    def test_no_demo_pending(self):
        self.assertEqual(br.check_b3(None, None)["status"], "PENDING")

    def test_partial_demo_fail(self):
        self.assertEqual(
            br.check_b3(None, "all receipts reverify MATCH\n")["status"],
            "FAIL")


class TestB2Probe(FixtureMixin, unittest.TestCase):
    def test_no_real_modules_pending(self):
        root = self.make_root()
        res = br.check_b2(root)
        self.assertEqual(res["status"], "PENDING")
        self.assertIn("not importable", res["evidence"])


class TestB8Tag(FixtureMixin, unittest.TestCase):
    def git(self, root, *args):
        env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1",
                   GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                   GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
        return subprocess.run(["git", *args], cwd=str(root), env=env,
                              capture_output=True, text=True)

    def test_missing_tag_fail(self):
        root = self.make_root()
        self.assertEqual(self.git(root, "init", "-q").returncode, 0)
        (root / "f.txt").write_text("x")
        self.assertEqual(self.git(root, "add", ".").returncode, 0)
        self.assertEqual(self.git(root, "commit", "-qm", "x").returncode, 0)
        res = br.check_b8(root, clean_clone=False)
        self.assertEqual(res["status"], "FAIL")
        self.assertIn("baseline-v1", res["evidence"])


class TestB6Git(FixtureMixin, unittest.TestCase):
    def test_non_git_pending(self):
        root = self.make_root()
        (root / "docs" / "screens" / "a.png").write_bytes(b"fake")
        self.assertEqual(br.check_b6(root)["status"], "PENDING")

    def test_no_screens_pending(self):
        root = self.make_root()
        self.assertEqual(br.check_b6(root)["status"], "PENDING")


class TestOutputs(FixtureMixin, unittest.TestCase):
    def test_writers_shape(self):
        root = self.make_root()
        results = {cid: {"status": "PENDING", "evidence": f"{cid} tbd"}
                   for cid, _ in br.CHECKS}
        results["B5"] = {"status": "FAIL", "evidence": "no funnel"}
        numbers = {"decider": {}, "funnel": {}, "latency": {}, "dataset": {}}
        sp, jp, sha, stamp = br.write_outputs(root, results, numbers)
        self.assertTrue(sp.exists() and jp.exists())
        text = sp.read_text(encoding="utf-8")
        for cid, _ in br.CHECKS:
            self.assertIn(cid, text)
        payload = json.loads(jp.read_text(encoding="utf-8"))
        self.assertIn("generated_utc", payload)
        self.assertIn("git_sha", payload)
        self.assertEqual(set(payload["checks"]), {c for c, _ in br.CHECKS})
        self.assertEqual(payload["checks"]["B5"]["status"], "FAIL")
        self.assertEqual(set(payload["numbers"]),
                         {"decider", "funnel", "latency", "dataset"})


class TestNoSignalsWrapper(unittest.TestCase):
    def test_wrapper_blocks_real_stages(self):
        prelude = br.NO_SIGNALS_WRAPPER.split("start_dir")[0]
        probe = prelude + "import jadal.signals\n"
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                           text=True, cwd=str(REPO),
                           env={**os.environ, "PYTHONNOUSERSITE": "1"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("blocked for baseline B1 no-signals run",
                      r.stderr + r.stdout)

    def test_wrapper_still_runs_suite(self):
        # the wrapper must be valid unittest-driving code, not just a blocker
        self.assertIn("unittest.TestLoader().discover",
                      br.NO_SIGNALS_WRAPPER)
        self.assertIn("wasSuccessful", br.NO_SIGNALS_WRAPPER)


if __name__ == "__main__":
    unittest.main()
