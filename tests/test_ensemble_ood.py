"""Regression tests for the OOD label loader (real labelling-app schema)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval", "ensemble"))
from eval_ood import load_labels_file


def app_schema(entries):
    return {"history": [], "latest": entries}


class TestOODLoader(unittest.TestCase):
    def test_real_dict_schema(self):
        d = app_schema({
            "q000_s_x": {"ts": "2026-10-10T00:00:00+00:00", "segment_id": "q000_s_x",
                         "verdict": "good", "reason": None, "note": ""},
            "q001_s_y": {"ts": "2026-10-10T00:00:01+00:00", "segment_id": "q001_s_y",
                         "verdict": "bad", "reason": "NO_ACTIVE_WORK", "note": ""},
            "q002_s_z": {"segment_id": "q002_s_z", "verdict": "bad", "reason": "SHAKY"},
        })
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(d, f)
            path = f.name
        try:
            out = load_labels_file(path)
        finally:
            os.unlink(path)
        self.assertEqual(out, {"q000_s_x": 1, "q001_s_y": 0, "q002_s_z": 0})

    def test_gold_and_list_schemas(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"items": [{"id": "g1", "label": 1}, {"id": "g0", "label": 0}]}, f)
            p1 = f.name
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump([{"segment_id": "s1", "verdict": "good"}], f)
            p2 = f.name
        try:
            self.assertEqual(load_labels_file(p1), {"g1": 1, "g0": 0})
            self.assertEqual(load_labels_file(p2), {"s1": 1})
        finally:
            os.unlink(p1)
            os.unlink(p2)

    def test_real_labels_file_loads_60(self):
        path = "/home/parshu/projects/jadal-qa-wt/ood_labels/labels.json"
        if not os.path.exists(path):
            self.skipTest("live labels file absent")
        out = load_labels_file(path)
        self.assertEqual(len(out), 60)
        self.assertEqual(sum(out.values()), 51)

    def test_eval_ood_exploratory_runs(self):
        r = subprocess.run(
            [sys.executable, "eval/ensemble/eval_ood.py", "--labels",
             "/home/parshu/projects/jadal-qa-wt/ood_labels/labels.json",
             "--exploratory", "--boot", "50"],
            capture_output=True, text=True, cwd="/home/parshu/projects/jadal-qa-wt/ensemble-views",
            env={**os.environ, "OMP_NUM_THREADS": "2", "PYTHONNOUSERSITE": "1",
                 "PYTHONPATH": "/home/parshu/projects/jadal-qa-wt/ensemble-views"})
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("EXPLORATORY", r.stdout)
        self.assertIn("OOD AUROC", r.stdout)


if __name__ == "__main__":
    unittest.main()
