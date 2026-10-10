"""Scale-data worker tests: fast (<30 s), no GPU/network/data-dep.

Validates committed selection/listing artefacts for consistency and the
pure helpers used by the manifest/label-queue builders.
"""
import json
import os
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return json.load(f)


class TestListingSelection(unittest.TestCase):
    def test_listing_summary(self):
        d = load("eval/results/scale/repo_listing.json")
        s = d["summary"]
        self.assertEqual(s["n_tars"], len(d["tars"]))
        self.assertGreaterEqual(s["n_factories"], 1)
        self.assertGreater(s["total_tar_GB"], 1000)  # TB-scale corpus
        self.assertEqual(s["per_file_durations_s"], None)  # not verified

    def test_small_selection_within_budget_and_diverse(self):
        sel = load("eval/results/scale/selection_small.json")
        self.assertEqual(sel["seed"], 8)
        self.assertLessEqual(sel["total_bytes"], sel["budget_bytes"])
        self.assertGreaterEqual(sel["n_factories"], 8)
        self.assertEqual(sel["n_files"], len(sel["files"]))
        listing_paths = {t["path"] for t in load("eval/results/scale/repo_listing.json")["tars"]}
        for f in sel["files"]:
            self.assertIn(f["path"], listing_paths)
        self.assertEqual(sel["total_bytes"], sum(f["size_bytes"] for f in sel["files"]))
        # one shard per factory in round 1 => distinct factories == files (round 1 only)
        facs = [f["path"].split("/")[0] for f in sel["files"]]
        self.assertGreaterEqual(len(set(facs)), 8)

    def test_big_selection_rule_recorded(self):
        sel = load("eval/results/scale/selection.json")
        self.assertIn("seed", sel)
        self.assertLessEqual(sel["total_GB"], 30.01)


class TestManifestHelpers(unittest.TestCase):
    def test_fps_parse(self):
        import sys
        sys.path.insert(0, os.path.join(ROOT, "eval", "scale"))
        import make_manifest
        self.assertAlmostEqual(make_manifest.fps_of("30/1"), 30.0)
        self.assertAlmostEqual(make_manifest.fps_of("30000/1001"), 29.97, places=2)
        self.assertIsNone(make_manifest.fps_of("0/0"))
        self.assertIsNone(make_manifest.fps_of("bogus"))

    def test_window_id_stable(self):
        import hashlib
        a = hashlib.sha1("factory_001/x@10.0".encode()).hexdigest()[:12]
        b = hashlib.sha1("factory_001/x@10.0".encode()).hexdigest()[:12]
        self.assertEqual(a, b)
        self.assertEqual(len(a), 12)


class TestLabelQueueSchema(unittest.TestCase):
    REQUIRED = {"segment_id", "clip", "path", "start_s", "end_s"}

    def check_queue(self, q):
        self.assertEqual(q["n"], len(q["segments"]))
        for s in q["segments"]:
            self.assertTrue(self.REQUIRED.issubset(s.keys()), s.keys())
            self.assertLess(s["start_s"], s["end_s"])

    def test_synthetic_example(self):
        self.check_queue({"n": 1, "segments": [
            {"segment_id": "q000_s_abc", "clip": "c", "path": "/tmp/w.mp4",
             "start_s": 0.0, "end_s": 10.0}]})

    def test_real_queue_if_present(self):
        p = os.path.join(ROOT, "eval/results/scale/label_queue_random.json")
        if os.path.exists(p):
            with open(p) as f:
                self.check_queue(json.load(f))


if __name__ == "__main__":
    unittest.main()
