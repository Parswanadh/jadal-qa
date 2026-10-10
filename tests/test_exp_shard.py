"""S1.01 tests — deterministic shard plan + shard-invariant aggregation.

No GPU, no network, < 30 s: pure helpers over synthetic rows.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "eval", "exp_shard"))
sys.path.insert(0, os.path.join(REPO, "eval", "exp_rescore"))

import common  # noqa: E402


def _items(spec):
    """spec: {clip: n_windows} -> canonical item list."""
    out = []
    for clip, n in spec.items():
        for j in range(n):
            out.append({"clip": clip, "start_s": j * 10.0, "end_s": j * 10.0 + 10.0,
                        "window_id": f"{clip}#{j}", "id": f"{clip}#{j}",
                        "factory": clip.split("/")[0], "worker": "w"})
    return sorted(out, key=lambda i: (i["clip"], i["start_s"], i["window_id"]))


SPEC = {"c/1/a": 119, "c/1/b": 119, "c/2/a": 52, "c/2/b": 17, "c/3/a": 17,
        "c/3/b": 17, "c/4/a": 17, "c/4/b": 17}


def _row(wid, lat):
    return {"window_id": wid, "verdict": "review", "score": 0.5, "clip": wid,
            "latency_ms": {"total": lat}, "stage_wall": {"decode": [lat, lat]},
            "gpu": {"eg2_vram_peak_mib": 1.0},
            "contract": {"window_id": wid, "verdict": "review", "score": 0.5,
                         "features": {"a": 1.0}, "features_digest": "d" * 64,
                         "latency_ms": {"total": lat}, "receipt_sha256": "x" * 64,
                         "versions": {"recipe": "v"}}}


class TestPlan(unittest.TestCase):
    def test_deterministic_and_balanced(self):
        items = _items(SPEC)
        p1 = common.make_plan(items, 4)
        p2 = common.make_plan(items, 4)
        self.assertEqual(p1, p2)
        self.assertEqual(sum(p1["loads_windows"]), len(items))
        self.assertEqual(len(p1["assignment"]), len(SPEC))
        # LPT balance: max-min load <= the single largest clip (119)
        self.assertLessEqual(max(p1["loads_windows"]) - min(p1["loads_windows"]), 119)

    def test_every_clip_assigned_once(self):
        items = _items(SPEC)
        for k in (1, 2, 4, 7):
            plan = common.make_plan(items, k)
            self.assertEqual(sorted(plan["assignment"]), sorted(SPEC))
            self.assertTrue(all(0 <= s < k for s in plan["assignment"].values()))

    def test_select_items(self):
        items = _items(SPEC)
        s = common.select_items(items, "smallest3")
        self.assertEqual(len({i["clip"] for i in s}), 3)
        # smallest clips all have 17 windows
        self.assertEqual(len(s), 51)
        self.assertEqual(common.select_items(items, "all"), items)


class TestDigest(unittest.TestCase):
    def test_timings_do_not_affect_digest(self):
        a = _row("w1", 100.0)
        b = _row("w1", 999.0)
        self.assertEqual(common.row_digest(a), common.row_digest(b))

    def test_shard_invariance(self):
        rows = [_row(f"w{i}", i) for i in range(20)]
        full = common.combined_digest(rows)
        # the combined digest does not depend on how rows are partitioned
        for k in (1, 2, 4, 7):
            parts = {i: [] for i in range(k)}
            for idx, r in enumerate(rows):
                parts[idx % k].append(r)
            union = [r for i in range(k) for r in parts[i]]
            self.assertEqual(common.combined_digest(union), full)
        # order-independence
        self.assertEqual(common.combined_digest(list(reversed(rows))), full)

    def test_per_shard_integrity(self):
        rows = [_row(f"w{i}", i) for i in range(10)]
        a, b = rows[:4], rows[4:]
        self.assertEqual(common.combined_digest(a + b),
                         common.combined_digest(a + b))
        # a corrupted row changes only its shard's digest and the combined digest
        da, d = common.combined_digest(a), common.combined_digest(a + b)
        a2 = [dict(r) for r in a]
        a2[0]["contract"] = dict(a2[0]["contract"], score=0.9)
        self.assertNotEqual(common.combined_digest(a2), da)
        self.assertNotEqual(common.combined_digest(a2 + b), d)

    def test_content_change_is_detected(self):
        rows = [_row("w1", 1), _row("w2", 2)]
        d = common.combined_digest(rows)
        rows[0]["contract"]["features"] = {"a": 2.0}
        self.assertNotEqual(common.combined_digest(rows), d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
