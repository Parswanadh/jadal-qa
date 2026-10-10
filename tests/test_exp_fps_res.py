"""Tests for H3/E1.06 fps x resolution cross. Fast, no network, no GPU."""
import json
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "eval", "results", "exp_fps_res")
RES = os.path.join(D, "E1.06.json")

EXPECTED_CELLS = {f"f{t}_r{r}" for t in ("5", "3p33", "2", "1") for r in (480, 360, 240)}


class TestFpsRes(unittest.TestCase):
    def test_merged_json_exists_and_grid_complete(self):
        self.assertTrue(os.path.exists(RES), f"missing {RES} (run merge_fps_res.py)")
        res = json.load(open(RES))
        cells = {r["cell"] for r in res["rows"]}
        self.assertEqual(cells, EXPECTED_CELLS, f"grid incomplete: {sorted(cells)}")

    def test_row_schema_and_honesty_fields(self):
        res = json.load(open(RES))
        for r in res["rows"]:
            for k in ("cell", "fps", "resolution_px", "n", "auroc", "ci95",
                      "delta_vs_baseline", "bytes_per_window", "total_ms_per_window",
                      "frames_per_window"):
                self.assertIn(k, r, f"{r.get('cell')}: missing {k}")
            self.assertEqual(r["n"], 150, f"{r['cell']}: n != 150")
            lo, hi = r["ci95"]
            self.assertLess(lo, r["auroc"] - 1e-9, f"{r['cell']}: CI lo >= auroc")
            self.assertGreater(hi, r["auroc"] + 1e-9, f"{r['cell']}: CI hi <= auroc")
            self.assertGreater(r["bytes_per_window"], 0)
        for k in ("n", "label_source", "protocol", "device", "command", "git_sha"):
            self.assertIn(k, res, f"top-level missing {k}")
        self.assertEqual(res["n"], 150)

    def test_cheapest_claim_matches_table(self):
        res = json.load(open(RES))
        base = res["baseline"]["auroc"]
        ok = [r for r in res["rows"] if r["auroc"] - base >= -0.02]
        if ok:
            cheapest = min(ok, key=lambda r: (r["bytes_per_window"], r["total_ms_per_window"]))
            self.assertEqual(res["cheapest_within_budget"], cheapest["cell"])
        else:
            self.assertIsNone(res["cheapest_within_budget"])

    def test_shards_present(self):
        for c in EXPECTED_CELLS:
            p = os.path.join(D, f"_feats_{c}.json")
            self.assertTrue(os.path.exists(p), f"missing shard {p}")


if __name__ == "__main__":
    unittest.main()
