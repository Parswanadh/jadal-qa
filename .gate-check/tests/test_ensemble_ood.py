"""Regression tests for the OOD label loader (real labelling-app schema)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "eval", "ensemble"))
from eval_ood import check_args, intersect_labels, load_labels_file  # noqa: E402

# Live OOD labels: env JADAL_OOD_LABELS, default ../ood_labels/labels.json (sibling of repo root).
LIVE_LABELS = os.environ.get("JADAL_OOD_LABELS",
                             os.path.normpath(os.path.join(ROOT, "..", "ood_labels", "labels.json")))


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
        if not os.path.exists(LIVE_LABELS):
            self.skipTest("live labels file absent (set JADAL_OOD_LABELS)")
        out = load_labels_file(LIVE_LABELS)
        self.assertEqual(len(out), 60)
        self.assertEqual(sum(out.values()), 51)

    def test_eval_ood_exploratory_runs(self):
        if not os.path.exists(LIVE_LABELS):
            self.skipTest("live labels file absent (set JADAL_OOD_LABELS)")
        r = subprocess.run(
            [sys.executable, "eval/ensemble/eval_ood.py", "--labels", LIVE_LABELS,
             "--exploratory", "--boot", "50"],
            capture_output=True, text=True, cwd=ROOT,
            env={**os.environ, "OMP_NUM_THREADS": "2", "PYTHONNOUSERSITE": "1",
                 "PYTHONPATH": ROOT})
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("EXPLORATORY", r.stdout)
        self.assertIn("OOD AUROC", r.stdout)
        self.assertIn("dropped 0 labelled ids", r.stdout)


class TestGateInputs(unittest.TestCase):
    def test_intersect_drops_unscored_before_gate(self):
        labels = {"a": 1, "b": 0, "c": 1, "z": 0}
        scored = {"a": {}, "b": {}, "c": {}}
        kept, dropped = intersect_labels(labels, scored)
        self.assertEqual(kept, {"a": 1, "b": 0, "c": 1})
        self.assertEqual(dropped, ["z"])

    def test_gate_counts_only_scored_labels(self):
        # 10 good + 10 bad labelled, but 3 bad have no cached score: the gate must see 10/7.
        labels = {f"g{i}": 1 for i in range(10)}
        labels.update({f"b{i}": 0 for i in range(10)})
        scored = {f"g{i}": {} for i in range(10)}
        scored.update({f"b{i}": {} for i in range(7)})
        kept, dropped = intersect_labels(labels, scored)
        self.assertEqual(len(dropped), 3)
        n0 = sum(1 for v in kept.values() if v == 0)
        self.assertEqual(n0, 7)
        self.assertLess(n0, 10)  # the pre-fix gate counted all 10 bad labels and passed

    def test_labels2_without_scores2_refused(self):
        class A:
            labels2 = "x.json"
            scores2 = None
        self.assertIsNotNone(check_args(A()))
        A.scores2 = "y.json"
        self.assertIsNone(check_args(A()))
        A.labels2 = None
        A.scores2 = None
        self.assertIsNone(check_args(A()))

    def test_cli_refuses_labels2_without_scores2(self):
        r = subprocess.run(
            [sys.executable, "eval/ensemble/eval_ood.py", "--labels", LIVE_LABELS,
             "--labels2", LIVE_LABELS, "--exploratory"],
            capture_output=True, text=True, cwd=ROOT,
            env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": ROOT})
        self.assertEqual(r.returncode, 2)
        self.assertIn("--labels2 requires --scores2", r.stdout)
        self.assertNotIn("OOD AUROC", r.stdout)


class TestOODHeads(unittest.TestCase):
    def test_committed_head_results(self):
        path = os.path.join(ROOT, "eval", "results", "ensemble", "ood_heads.json")
        if not os.path.exists(path):
            self.skipTest("ood_heads.json absent")
        with open(path) as f:
            d = json.load(f)
        self.assertEqual((d["n"], d["n_bad"]), (60, 9))
        self.assertAlmostEqual(d["heads"]["p_eg2"]["auroc"], 0.586, places=3)
        self.assertAlmostEqual(d["heads"]["p_pixel"]["auroc"], 0.342, places=3)
        self.assertAlmostEqual(d["heads"]["usable_prob"]["auroc"], 0.415, places=3)


if __name__ == "__main__":
    unittest.main()
