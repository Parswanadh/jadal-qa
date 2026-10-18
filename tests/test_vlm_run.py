"""Tests for the GRID.V run helpers (sub-stream V-B).

Pure functions only: no API call is made, no key is touched, no sheet is built from
dataset frames.  The budget guard test is the important one -- it is the mechanism
that stops a runaway run from exceeding the pre-registered call cap.

Run: python3 -m unittest tests.test_vlm_run -v
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tools", "dsh")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval.exp_vlm import vlm_run as R  # noqa: E402


class TestTimestamps(unittest.TestCase):
    def test_fmt_ts_matches_the_sheet_label_format(self):
        self.assertEqual(R.fmt_ts(0.4), "00:00.4")
        self.assertEqual(R.fmt_ts(5.4), "00:05.4")
        self.assertEqual(R.fmt_ts(63.25), "01:03.2")
        self.assertEqual(R.fmt_ts(600.0), "10:00.0")

    def test_norm_ts_accepts_what_the_model_actually_returns(self):
        # the smoke reply used "tile 1 @ 00:00.4s :: ..."
        self.assertEqual(R.norm_ts("00:00.4s"), "00:00.4")
        self.assertEqual(R.norm_ts("00:00.4"), "00:00.4")
        self.assertEqual(R.norm_ts(" 01:03.2s "), "01:03.2")
        self.assertEqual(R.norm_ts("0:05.4s"), "00:05.4")

    def test_norm_ts_rejects_junk(self):
        for junk in ("", "abc", "12", "not a time"):
            self.assertEqual(R.norm_ts(junk), "")


class TestReadTilesMetrics(unittest.TestCase):
    def setUp(self):
        # sidecar exactly as tools/dsh/vlm_sheet.build_sheet emits it
        self.side = [{"index": i, "slot": i, "row": i // 4, "col": i % 4,
                      "t_s": round((i + 0.5) * 10.0 / 12, 3),
                      "label": f"#{i+1}  {R.fmt_ts((i + 0.5) * 10.0 / 12)}",
                      "x": 0, "y": 0, "w": 426, "h": 240} for i in range(12)]

    def test_perfect_reply_scores_one(self):
        reply = {i + 1: (R.fmt_ts((i + 0.5) * 10.0 / 12) + "s", "x") for i in range(12)}
        m = R.read_tiles_metrics(reply, self.side)
        self.assertEqual(m["n_returned_lines"], 12)
        self.assertEqual(m["tile_coverage_rate"], 1.0)
        self.assertEqual(m["exact_match_rate_per_sheet_tile"], 1.0)
        self.assertEqual(m["exact_match_rate_per_returned_line"], 1.0)

    def test_wrong_timestamp_is_not_an_exact_match(self):
        reply = {i + 1: (R.fmt_ts((i + 0.5) * 10.0 / 12) + "s", "x") for i in range(12)}
        reply[3] = ("00:09.9s", "x")               # right tile, wrong time
        m = R.read_tiles_metrics(reply, self.side)
        self.assertEqual(m["n_number_ok_timestamp_wrong"], 1)
        self.assertEqual(m["n_exact_number_and_timestamp"], 11)
        self.assertEqual(m["tile_coverage_rate"], 1.0)

    def test_missing_tiles_lower_coverage_not_exactness(self):
        reply = {i + 1: (R.fmt_ts((i + 0.5) * 10.0 / 12) + "s", "x") for i in range(6)}
        m = R.read_tiles_metrics(reply, self.side)
        self.assertEqual(m["tile_coverage_rate"], 0.5)
        self.assertEqual(m["exact_match_rate_per_sheet_tile"], 0.5)
        self.assertEqual(m["exact_match_rate_per_returned_line"], 1.0)

    def test_out_of_range_tile_number_is_counted(self):
        reply = {99: ("00:01.0s", "x")}
        m = R.read_tiles_metrics(reply, self.side)
        self.assertEqual(m["n_out_of_range_lines"], 1)
        self.assertEqual(m["tile_coverage_rate"], 0.0)

    def test_expected_tiles_is_one_based(self):
        exp = R.expected_tiles(self.side)
        self.assertEqual(sorted(exp), list(range(1, 13)))
        self.assertEqual(exp[1], R.fmt_ts(self.side[0]["t_s"]))


class TestQualityScores(unittest.TestCase):
    def test_maps_the_three_progress_levels(self):
        for prog, want in (("advancing", 1.0), ("cannot_tell", 0.5),
                           ("no_relevant_work", 0.0)):
            rec = {"quality": (True, "OK", prog, "x")}
            u, p = R.quality_scores(rec)
            self.assertEqual(u, 1.0)
            self.assertEqual(p, want)

    def test_usable_false_is_zero(self):
        u, p = R.quality_scores({"quality": (False, "NO_ACTIVE_WORK", "no_relevant_work", "x")})
        self.assertEqual(u, 0.0)
        self.assertEqual(p, 0.0)

    def test_unparsed_reply_is_none_not_a_silent_zero(self):
        self.assertEqual(R.quality_scores({"quality": None}), (None, None))
        self.assertEqual(R.quality_scores({}), (None, None))


class TestBudgetGuard(unittest.TestCase):
    """The guard must REFUSE rather than overspend the pre-registered cap."""

    def _ledger(self, n, arm="a"):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        with open(path, "w") as f:
            for i in range(n):
                f.write(json.dumps({"window": f"w{i}", "arm": arm, "ok": True}) + "\n")
        return path

    def test_passes_below_cap(self):
        p = self._ledger(5)
        try:
            self.assertEqual(R.guard(p, 10, 1)["substream_total"], 5)
        finally:
            os.unlink(p)

    def test_refuses_at_cap(self):
        p = self._ledger(122)
        try:
            with self.assertRaises(RuntimeError) as ctx:
                R.guard(p, 122, 1)
            self.assertIn("BUDGET STOP", str(ctx.exception))
        finally:
            os.unlink(p)

    def test_refuses_a_batch_that_would_overshoot(self):
        p = self._ledger(120)
        try:
            with self.assertRaises(RuntimeError):
                R.guard(p, 122, 3)
            self.assertEqual(R.guard(p, 122, 2)["substream_total"], 120)
        finally:
            os.unlink(p)

    def test_arms_filter_keeps_experiments_separable(self):
        """A shared ledger must not let V01's calls eat V03's cap.

        This is the exact bug that stopped the first V03 run at 122 total calls
        (20 V01 + 102 V03) when V03's own cap was 122: per-experiment accounting
        has to filter by arm.
        """
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        with open(path, "w") as f:
            for i in range(20):                       # V01's 20 calls
                f.write(json.dumps({"window": f"v{i}", "arm": "v01_read_tiles", "ok": True}) + "\n")
            for i in range(102):                      # V03 so far
                f.write(json.dumps({"window": f"w{i}", "arm": "a_no_timestamps", "ok": True}) + "\n")
        try:
            # counting everything would refuse (122 >= 122)
            with self.assertRaises(RuntimeError):
                R.guard(path, 122, 1)
            # counting V03 only leaves exactly 20 calls, which must be allowed
            got = R.guard(path, 122, 20, arms={"a_no_timestamps", "e_json_only_prompt"})
            self.assertEqual(got["experiment"], 102)
            self.assertEqual(got["substream_total"], 122)
            # ...but not 21
            with self.assertRaises(RuntimeError):
                R.guard(path, 122, 21, arms={"a_no_timestamps", "e_json_only_prompt"})
        finally:
            os.unlink(path)

    def test_substream_total_cap_is_enforced(self):
        p = self._ledger(R.SUBSTREAM_TOTAL_CAP, arm="x")
        try:
            with self.assertRaises(RuntimeError) as ctx:
                R.guard(p, 9999, 1)
            self.assertIn("sub-stream total", str(ctx.exception))
        finally:
            os.unlink(p)


class TestResume(unittest.TestCase):
    def test_only_ok_records_resume(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        with open(path, "w") as f:
            f.write(json.dumps({"window": "q002", "arm": "a", "ok": True}) + "\n")
            f.write(json.dumps({"window": "q005", "arm": "a", "ok": False}) + "\n")
            f.write(json.dumps({"window": "q005", "arm": "b", "ok": True}) + "\n")
        try:
            done = R.done_pairs(path)
            self.assertIn(("q002", "a"), done)
            self.assertIn(("q005", "b"), done)
            self.assertNotIn(("q005", "a"), done)
        finally:
            os.unlink(path)


class TestFrozenPrompts(unittest.TestCase):
    """The PREREG pins the prompt hashes; a drifting prompt must fail loudly."""

    def test_prompt_hashes_match_the_prereg(self):
        import vlm_mosaic as VM
        for name, sha in R.PREREG["prompt_sha256"].items():
            self.assertIn(name, VM.PROMPTS, f"{name} missing from PROMPTS")
            self.assertEqual(VM.PROMPT_SHA256[name], sha,
                             f"prompt {name} changed after the PREREG was frozen")

    def test_prereg_windows_and_caps_are_unchanged(self):
        self.assertEqual(len(R.PREREG["windows"]["V01_20"]), 20)
        self.assertEqual(len(R.PREREG["windows"]["V03_20_stratified"]), 20)
        # NOTE: the frozen PREREG sets V01's budget to 20 calls (the task brief said
        # "cap 160" -- a discrepancy recorded in docs/reports/grid_vlm_b.md).  The run
        # made exactly 20 V01 calls, so it is inside BOTH readings.
        self.assertEqual(R.PREREG["call_budget"]["V01"], 20)
        self.assertEqual(R.PREREG["call_budget"]["V02"], 160)
        self.assertEqual(R.PREREG["call_budget"]["V03"], 122)
        # the V03 plan must consume exactly its cap: 6 arms x 20 windows + 2 nulls
        self.assertEqual(len(R.V03_ARMS) * len(R.PREREG["windows"]["V03_20_stratified"]) + 2,
                         R.PREREG["call_budget"]["V03"])


if __name__ == "__main__":
    unittest.main()
