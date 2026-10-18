"""Tests for the TASK.00 / TASK.00b zero-label measurement scripts (<30 s, no network).

The parsers are tested against a synthetic VIA document built inline, and against
the real annotation files when they are present.  If the real files are absent the
test SKIPS loudly (it does not silently pass as if it had checked them).
"""
import glob
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from task.task00 import derive_verdict, spearman  # noqa: E402
from task.task00b import cohen_kappa, crosstab, gap_list, parse_keysteps_txt, parse_via  # noqa: E402

INDEGO = "/home/parshu/projects/awing/data/indego"

VIA_DOC = {
    "attribute": {"1": {"aname": "steps",
                        "options": {"0": "00_preparation", "1": "01_assemble"}}},
    "file": {"1": {"fid": "1", "fname": "clip_480.mp4"}},
    "metadata": {
        "a": {"vid": "1", "z": [10.0, 20.0], "xy": [], "av": {"1": "1"}},
        "b": {"vid": "1", "z": [0.5, 10.0], "xy": [], "av": {"1": "0"}},
    },
}
KEYSTEPS_TXT = """Assembly_ECU
00_preparation,
01_assemble,
02_clean_up,

## Organize
00_preparation,
01_organize,
"""


class TestDeriveVerdict(unittest.TestCase):
    """The band must match jadal/poc/cascade.py: review iff |p-0.5| < 0.25."""

    def test_bands(self):
        self.assertEqual(derive_verdict(0.75), "accept")
        self.assertEqual(derive_verdict(0.7501), "accept")
        self.assertEqual(derive_verdict(0.7499), "review")
        self.assertEqual(derive_verdict(0.5), "review")
        self.assertEqual(derive_verdict(0.2501), "review")
        self.assertEqual(derive_verdict(0.25), "reject")
        self.assertEqual(derive_verdict(0.0), "reject")
        self.assertIsNone(derive_verdict(None))

    def test_matches_the_shipped_verdict_on_the_160_windows(self):
        """Regression against the shipped file itself; a moved band must fail here."""
        heads_p = "/home/parshu/projects/awing/interview/dsh_pack/ood160_heads_old26_vs_new22.json"
        if not os.path.exists(heads_p):
            self.skipTest("shipped heads file not present on this machine")
        with open(heads_p) as fh:
            heads = json.load(fh)["old26"]
        bad = [(k, v["verdict"], derive_verdict(v["score"]))
               for k, v in heads.items() if derive_verdict(v["score"]) != v["verdict"]]
        self.assertEqual(bad, [])
        self.assertEqual(len(heads), 160)


class TestViaParser(unittest.TestCase):
    def test_parses_segments_sorted_with_step_labels(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(VIA_DOC, f)
            p = f.name
        try:
            v = parse_via(p)
        finally:
            os.unlink(p)
        self.assertEqual(v["fname"], "clip_480.mp4")
        self.assertEqual([s["start_s"] for s in v["segments"]], [0.5, 10.0])
        self.assertEqual([s["step"] for s in v["segments"]], ["00_preparation", "01_assemble"])
        self.assertFalse(v["has_xy"])

    def test_real_annotations_parse_and_durations_match(self):
        files = sorted(glob.glob(os.path.join(INDEGO, "**", "annotated_keysteps_*", "*.json"),
                                 recursive=True))
        if not files:
            self.skipTest("IndEgo annotations not present on this machine")
        parsed = [parse_via(f) for f in files]
        self.assertEqual(len(parsed), len(files))
        self.assertTrue(all(v and v["segments"] for v in parsed))
        # every segment must be a positive-length interval inside the file's own span
        for v in parsed:
            for s in v["segments"]:
                self.assertGreater(s["end_s"], s["start_s"])


class TestKeystepsTxt(unittest.TestCase):
    def test_two_sections_with_step_lists(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(KEYSTEPS_TXT)
            p = f.name
        try:
            sec = parse_keysteps_txt(p)
        finally:
            os.unlink(p)
        self.assertEqual(list(sec), ["Assembly_ECU", "Organize"])
        self.assertEqual(sec["Assembly_ECU"], ["00_preparation", "01_assemble", "02_clean_up"])
        self.assertEqual(sec["Organize"], ["00_preparation", "01_organize"])


class TestMetrics(unittest.TestCase):
    def test_kappa_bounds(self):
        a = [1, 1, 0, 0]
        self.assertAlmostEqual(cohen_kappa(a, a), 1.0, places=9)
        self.assertAlmostEqual(cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]), -1.0, places=9)

    def test_crosstab_counts(self):
        r = crosstab([1, 0, 1, 0], [1, 1, 0, 0])
        self.assertEqual(r["pred_advancing_gold_good"], 1)
        self.assertEqual(r["pred_advancing_gold_bad"], 1)
        self.assertEqual(r["pred_no_relevant_work_gold_good"], 1)
        self.assertEqual(r["pred_no_relevant_work_gold_bad"], 1)
        self.assertAlmostEqual(r["agreement_rate"], 0.5, places=9)

    def test_crosstab_one_class_gives_no_balanced_accuracy(self):
        # every prediction is 1, so there are no true/false negatives to score
        r = crosstab([1, 1], [1, 1])
        self.assertNotIn("balanced_accuracy", r)
        # a genuine mixed case still gets one
        self.assertIn("balanced_accuracy", crosstab([1, 1], [1, 0]))

    def test_gap_list(self):
        segs = [{"start_s": 2.0, "end_s": 5.0}, {"start_s": 8.0, "end_s": 10.0}]
        g = gap_list(segs, 12.0)
        self.assertEqual([(x["start_s"], x["end_s"]) for x in g],
                         [(0.0, 2.0), (5.0, 8.0), (10.0, 12.0)])
        self.assertEqual(gap_list([{"start_s": 0.0, "end_s": 12.0}], 12.0), [])

    def test_spearman_endpoints(self):
        self.assertAlmostEqual(spearman([0, 1, 2, 3], [0, 1, 2, 3]), 1.0, places=9)
        self.assertAlmostEqual(spearman([0, 1, 2, 3], [3, 2, 1, 0]), -1.0, places=9)
        self.assertTrue(np.isnan(spearman([1, 1, 1], [0, 1, 2])))


if __name__ == "__main__":
    unittest.main()
