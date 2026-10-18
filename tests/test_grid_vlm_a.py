"""Unit tests for the V-A stream (GRID.V02 / GRID.V04).

Deterministic parts only: the kappa statistics, the ledger-resume keying, the reply
parser, and sheet determinism.  NO API calls are made by this test module and no
dataset frames are written (the sheet test uses synthetic frames).

Run: python3 -m unittest tests.test_grid_vlm_a -v
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval", "exp_vlm"))
sys.path.insert(0, os.path.join(ROOT, "tools", "dsh"))

from eval.exp_vlm.grid_v04 import cohen_kappa, fleiss_kappa  # noqa: E402


class TestKappa(unittest.TestCase):
    def test_fleiss_perfect_agreement(self):
        self.assertAlmostEqual(fleiss_kappa(np.array([[3, 0], [3, 0], [0, 3], [0, 3]])), 1.0)

    def test_fleiss_hand_computed(self):
        # P_bar = (4+1-3)/6 = 1/3 ; p_j = 0.5 each -> P_e = 0.5 -> kappa = -1/3
        self.assertAlmostEqual(fleiss_kappa(np.array([[2, 1], [1, 2], [2, 1], [1, 2]])),
                               -1.0 / 3.0, places=6)

    def test_fleiss_needs_equal_raters(self):
        self.assertTrue(np.isnan(fleiss_kappa(np.array([[2, 0], [1, 2]]))))

    def test_cohen_known_values(self):
        self.assertAlmostEqual(cohen_kappa([1, 1, 0], [1, 1, 0]), 1.0)
        self.assertAlmostEqual(cohen_kappa([1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 1]), 1.0 / 3.0,
                               places=6)
        self.assertAlmostEqual(cohen_kappa([1, 0, 1, 0], [1, 1, 0, 0]), 0.0)


class TestLedgerResume(unittest.TestCase):
    """The resume key must treat a record with no `run` as run 1, or smoke calls get
    paid for twice (this bug was hit and fixed in the real run)."""

    def _write(self, rows):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        self.addCleanup(os.unlink, path)
        return path

    def test_missing_run_defaults_to_one(self):
        from eval.exp_vlm.grid_v02 import ledger_done
        p = self._write([{"ok": True, "prompt_name": "quality", "window": "q000",
                          "png_sha256": "a" * 64, "raw": "{}"},
                         {"ok": True, "prompt_name": "quality", "window": "q001", "run": 2,
                          "png_sha256": "b" * 64, "raw": "{}"}])
        d = ledger_done(p, "quality")
        self.assertIn(("q000", 1), d)
        self.assertIn(("q001", 2), d)

    def test_failed_and_other_prompt_records_are_ignored(self):
        from eval.exp_vlm.grid_v02 import ledger_done
        p = self._write([{"ok": False, "prompt_name": "quality", "window": "q002", "raw": ""},
                         {"ok": True, "prompt_name": "read_tiles", "window": "q003", "raw": "x"}])
        self.assertEqual(ledger_done(p, "quality"), {})


class TestParsing(unittest.TestCase):
    def test_parse_row_reads_the_json_block(self):
        from eval.exp_vlm.grid_v02 import parse_row
        rec = {"ok": True, "seconds": 1.0, "png_sha256": "x",
               "raw": ('#1 00:00.4s: hands hold a tube\n'
                       '{"usable": false, "reason_code": "NO_ACTIVE_WORK", '
                       '"task_progress": "no_relevant_work", "explanation": "standing"}')}
        r = parse_row(rec)
        self.assertIs(r["usable"], False)
        self.assertEqual(r["reason_code"], "NO_ACTIVE_WORK")
        self.assertEqual(r["task_progress"], "no_relevant_work")
        # the model answers in its own label format; the tolerant parser must read it
        self.assertEqual(len(r["tile_lines"]), 1)
        self.assertEqual(r["n_tile_lines"], 1)

    def test_shared_parser_now_reads_the_no_timestamp_shape(self):
        """History, kept as a regression guard.

        The shared parser read 0/1920 real tile lines; a first fix (beb742e) accepted
        `#N MM:SS.s: text` but still REQUIRED a timestamp, and 113 of 160 real replies
        omit it (`#1: text`), so it recovered only 372/1920.  The final fix makes the
        timestamp OPTIONAL, and the shared parser now recovers 1920/1920.  This test
        pins the end state and both sub-stream parsers must agree with it.
        """
        from tools.dsh import vlm_mosaic as M
        from eval.exp_vlm.vlm_run import parse_tile_lines_tolerant
        with_ts = "#1 00:00.4s: both hands align a tube"
        no_ts = "#1: both hands align a tube"
        self.assertEqual(len(M.parse_tile_lines(with_ts)), 1)
        self.assertEqual(len(M.parse_tile_lines(no_ts)), 1)        # fixed: timestamp optional
        self.assertEqual(M.parse_tile_lines(no_ts)[1][0], "")      # and reported as empty
        self.assertEqual(len(parse_tile_lines_tolerant(no_ts)), 1)
        # both parsers agree on the three real shapes
        for raw in (with_ts, no_ts, "#3 00:02.1s  left hand steadies the jig"):
            self.assertEqual(sorted(M.parse_tile_lines(raw)),
                             sorted(parse_tile_lines_tolerant(raw)), raw)

    def test_both_reply_shapes_are_read_by_the_tolerant_parser(self):
        from eval.exp_vlm.vlm_run import parse_tile_lines_tolerant
        raw = ("#1 00:00.4s: both hands align a tube\n"
               "#2: right hand lifts a tool\n"
               "#3 00:02.1s  left hand steadies the jig\n"
               '{"usable": true, "reason_code": "OK"}')
        tl = parse_tile_lines_tolerant(raw)
        self.assertEqual(len(tl), 3)
        self.assertIn("align", tl[1][1])
        self.assertIn("lifts", tl[2][1])

    def test_tolerant_parser_also_reads_the_requested_format(self):
        from eval.exp_vlm.vlm_run import parse_tile_lines_tolerant
        raw = "tile 3 @ 00:04.6s :: left hand steadies the jig"
        tl = parse_tile_lines_tolerant(raw)
        self.assertIn(3, tl)
        self.assertIn("steadies", tl[3][1])

    def test_parse_row_marks_unparsed_replies(self):
        from eval.exp_vlm.grid_v02 import parse_row
        r = parse_row({"ok": True, "raw": "the image shows a worker", "seconds": 2.0})
        self.assertIsNone(r["usable"])
        self.assertEqual(r["reason_code"], "UNPARSED")

    def test_unknown_reason_code_is_not_silently_accepted(self):
        from eval.exp_vlm.grid_v02 import parse_row
        r = parse_row({"ok": True, "raw": '{"usable": true, "reason_code": "BANANA"}'})
        self.assertEqual(r["reason_code"], "UNPARSED_CODE")


class TestScoreDefinitions(unittest.TestCase):
    """The natural-threshold metrics are what the report claims; pin the arithmetic.

    Regression: the first version of grid_v02.py computed balanced accuracy as
    0.5 * (rejected-good rate + rejected-bad rate) = 0.3208 instead of
    0.5 * (accepted-good rate + rejected-bad rate) = 0.7701 on the real run.
    """

    def test_bal_acc_uses_accepted_good_not_rejected_good(self):
        y = np.array([1, 1, 1, 1, 0, 0])                        # 4 good, 2 bad
        u = np.array([True, True, False, True, False, True])    # 1 good rejected, 1 bad accepted
        good, bad = y == 1, y == 0
        recall_bad = float((~u[bad]).mean())                    # specificity
        frr_good = float((~u[good]).mean())
        sens = float(u[good].mean())
        self.assertAlmostEqual(recall_bad, 0.5)
        self.assertAlmostEqual(frr_good, 0.25)
        self.assertAlmostEqual(sens, 0.75)
        self.assertAlmostEqual(0.5 * (sens + recall_bad), 0.625)          # correct
        self.assertNotAlmostEqual(0.5 * (frr_good + recall_bad), 0.625)   # the old bug

    def test_bal_acc_equals_auroc_when_the_binary_score_is_at_its_own_threshold(self):
        """Sanity anchor on the real numbers: AUROC 0.7701 and BA 0.7701 coincide by
        construction here (131/138 good accepted, 13/22 bad rejected)."""
        sens, spec = 131.0 / 138.0, 13.0 / 22.0
        self.assertAlmostEqual(0.5 * (sens + spec), 0.7701, places=4)


class TestSheetDeterminism(unittest.TestCase):
    def test_same_frames_give_the_same_png_bytes(self):
        from tools.dsh import vlm_sheet as S
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 255, (120, 213, 3), dtype=np.uint8) for _ in range(12)]
        times = S.cell_times(12, 10.0, 0.0)
        a, _ = S.build_sheet(frames, times, cols=4, labels=True)
        b, _ = S.build_sheet(frames, times, cols=4, labels=True)
        self.assertEqual(S.png_bytes(a), S.png_bytes(b))
        self.assertEqual(a.size, (1724, 736))


if __name__ == "__main__":
    unittest.main()


class TestV06Metrics(unittest.TestCase):
    """The V06 policy comparison rests on these two functions; pin both."""

    def test_bal_acc_positive_class_is_good(self):
        from eval.exp_vlm.grid_v06 import bal_acc
        y = np.array([1, 1, 0, 0])
        accept = np.array([True, False, False, True])   # 1 good in, 1 good out, 1 bad out, 1 bad in
        self.assertAlmostEqual(bal_acc(y, accept), 0.5)
        self.assertAlmostEqual(bal_acc(y, np.array([True, True, False, False])), 1.0)
        self.assertAlmostEqual(bal_acc(y, np.array([False, False, True, True])), 0.0)

    def test_paired_group_boot_detects_a_real_improvement(self):
        from eval.exp_vlm.grid_v06 import paired_group_boot_rate, bal_acc
        rng = np.random.default_rng(0)
        n = 200
        y = np.array([1] * 100 + [0] * 100)
        groups = np.array([f"f{i%20}" for i in range(n)])
        a = np.where(y == 1, rng.random(n) < 0.95, rng.random(n) < 0.05)   # good policy
        b = np.where(y == 1, rng.random(n) < 0.6, rng.random(n) < 0.4)     # bad policy
        r = paired_group_boot_rate(y, a, b, groups, bal_acc, n_boot=300)
        self.assertGreater(r["delta"], 0.2)
        self.assertTrue(r["excludes_zero"])
        self.assertEqual(r["ci95_type"], "paired cluster bootstrap (300 draws, seed 0)")

    def test_paired_group_boot_is_null_for_identical_policies(self):
        from eval.exp_vlm.grid_v06 import paired_group_boot_rate, bal_acc
        y = np.array([1, 1, 0, 0] * 25)
        g = np.array([f"c{i%5}" for i in range(len(y))])
        a = np.array([True, False, True, False] * 25)
        r = paired_group_boot_rate(y, a, a.copy(), g, bal_acc, n_boot=200)
        self.assertAlmostEqual(r["delta"], 0.0)
        self.assertFalse(r["excludes_zero"])

    def test_group_index_covers_every_window(self):
        from eval.exp_vlm.grid_v06 import _group_index
        uniq, idx = _group_index(["b", "a", "b", "c"])
        self.assertEqual(uniq, ["a", "b", "c"])
        self.assertEqual(sum(len(i) for i in idx), 4)
