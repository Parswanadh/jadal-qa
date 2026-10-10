"""Hardware-free tests for the video hub (GitHub issue #27).

No model, no GPU, no network: a deterministic fake embedder (sha256-seeded
unit vectors) stands in for HubEmbedder via Index(embed_fn=...).

Covers: build/save/load round trip, index_hash stability, filters,
tie-breaking by segment_id, cluster logic incl. one giant cluster,
results_hash determinism, thumb path-traversal, filter injection.
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jadal.hub import Index  # noqa: E402
from jadal.hub_api import create_hub_router  # noqa: E402

DIM = 16


def fake_vec(text: str, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:16], 16))
    v = rng.standard_normal(dim).astype(np.float64)
    return (v / np.linalg.norm(v)).astype(np.float32)


def fake_embed_fn(q: str) -> np.ndarray:
    return fake_vec("q:" + q)


def make_index(n: int = 6) -> Index:
    idx = Index(dim=DIM, embed_fn=fake_embed_fn)
    verdicts = ["accept", "reject", "review"]
    for i in range(n):
        sid = f"seg_{i:03d}"
        idx.add(sid, fake_vec("seg:" + sid), {
            "clip": "clipA" if i % 2 == 0 else "clipB",
            "start_s": float(i * 10), "end_s": float(i * 10 + 10),
            "worker_id": "w1" if i < n // 2 else "w2",
            "site_id": "s1", "verdict": verdicts[i % 3],
            "reasons": ["BLUR"] if i % 3 == 0 else [],
            "step": "attach_hard_disk" if i % 2 == 0 else None,
            "path": f"/tmp/{sid}.mp4"})
    return idx


class TestRoundTrip(unittest.TestCase):
    def test_save_load_round_trip_and_hash(self):
        idx = make_index()
        h1 = idx.index_hash()
        with tempfile.TemporaryDirectory() as d:
            idx.save(d)
            idx2 = Index.load(d, embed_fn=fake_embed_fn)
        self.assertEqual(idx2.index_hash(), h1)
        self.assertEqual(len(idx2), len(idx))
        r1 = idx.search_text("hello", k=3)
        r2 = idx2.search_text("hello", k=3)
        self.assertEqual(r1, r2)

    def test_rebuild_same_order_same_hash(self):
        a, b = make_index(), Index(dim=DIM)
        # Add in reverse order: canonical order is sorted by id anyway.
        for sid in sorted(a._vecs.keys(), reverse=True):
            b.add(sid, a._vecs[sid].astype(np.float32), a._meta[sid])
        self.assertEqual(a.index_hash(), b.index_hash())


class TestFilters(unittest.TestCase):
    def test_site_worker_verdict_reason_clip_step_time(self):
        idx = make_index(6)
        all_ids = {h["segment_id"] for h in
                   idx.search_by_vector(fake_vec("x"), k=50)["hits"]}
        self.assertEqual(len(all_ids), 6)
        w1 = idx.search_by_vector(fake_vec("x"), k=50, filters={"worker": "w1"})
        self.assertTrue(all_ids and len(w1["hits"]) == 3)
        v = idx.search_by_vector(fake_vec("x"), k=50, filters={"verdict": "accept"})
        self.assertEqual(len(v["hits"]), 2)
        r = idx.search_by_vector(fake_vec("x"), k=50, filters={"reason": "BLUR"})
        self.assertEqual(len(r["hits"]), 2)
        c = idx.search_by_vector(fake_vec("x"), k=50, filters={"clip": "clipB"})
        self.assertEqual(len(c["hits"]), 3)
        s = idx.search_by_vector(fake_vec("x"), k=50, filters={"step": "attach_hard_disk"})
        self.assertEqual(len(s["hits"]), 3)
        t = idx.search_by_vector(fake_vec("x"), k=50, filters={"t0": 15.0, "t1": 25.0})
        # Overlap semantics: seg_001 [10,20) and seg_002 [20,30) overlap [15,25).
        self.assertEqual({h["segment_id"] for h in t["hits"]}, {"seg_001", "seg_002"})

    def test_filter_injection_is_literal(self):
        idx = make_index(4)
        # A malicious filter value must match literally (no regex/glob), so it
        # matches nothing rather than everything.
        res = idx.search_by_vector(fake_vec("x"), k=50, filters={"site": "s1.*"})
        self.assertEqual(res["hits"], [])
        res2 = idx.search_by_vector(fake_vec("x"), k=50,
                                    filters={"reason": "BLUR' OR '1'='1"})
        self.assertEqual(res2["hits"], [])


class TestTieBreaking(unittest.TestCase):
    def test_ties_break_by_segment_id(self):
        idx = Index(dim=DIM)
        v = fake_vec("same")
        for sid in ["seg_b", "seg_a", "seg_c"]:
            idx.add(sid, v, {"clip": "c", "start_s": 0.0, "end_s": 1.0,
                             "worker_id": "w", "site_id": "s",
                             "verdict": "review", "reasons": [],
                             "step": None, "path": None})
        hits = idx.search_by_vector(v, k=3)["hits"]
        self.assertEqual([h["segment_id"] for h in hits],
                         ["seg_a", "seg_b", "seg_c"])


class TestResultsHash(unittest.TestCase):
    def test_same_query_twice_identical(self):
        idx = make_index()
        r1 = idx.search_text("attach hard disk", k=4)
        r2 = idx.search_text("attach hard disk", k=4)
        self.assertEqual(r1["results_hash"], r2["results_hash"])
        self.assertEqual(r1["hits"], r2["hits"])


class TestClusters(unittest.TestCase):
    def test_two_cliques(self):
        idx = Index(dim=DIM)
        a, b = fake_vec("clique-a"), fake_vec("clique-b")
        for i in range(3):
            idx.add(f"a_{i}", a, {"clip": "c", "start_s": 0.0,
                                  "end_s": 1.0, "worker_id": "w",
                                  "site_id": "s", "verdict": "review",
                                  "reasons": [], "step": None, "path": None})
        for i in range(2):
            idx.add(f"b_{i}", b, {"clip": "c", "start_s": 0.0,
                                  "end_s": 1.0, "worker_id": "w",
                                  "site_id": "s", "verdict": "review",
                                  "reasons": [], "step": None, "path": None})
        out = idx.dedup_clusters(threshold=0.99)
        self.assertEqual(out["n_clusters"], 2)
        reps = sorted(c["representative"] for c in out["clusters"])
        self.assertEqual(reps, ["a_0", "b_0"])
        self.assertAlmostEqual(out["dup_fraction"], (5 - 2) / 5)

    def test_one_giant_cluster(self):
        idx = Index(dim=DIM)
        v = fake_vec("giant")
        for i in range(20):
            idx.add(f"g_{i:02d}", v, {"clip": "c", "start_s": 0.0,
                                      "end_s": 1.0, "worker_id": "w",
                                      "site_id": "s", "verdict": "review",
                                      "reasons": [], "step": None, "path": None})
        out = idx.dedup_clusters(threshold=0.99)
        self.assertEqual(out["n_clusters"], 1)
        self.assertEqual(out["clusters"][0]["representative"], "g_00")
        self.assertEqual(len(out["clusters"][0]["members"]), 20)


class TestAPI(unittest.TestCase):
    def test_search_similar_stats_clusters(self):
        from fastapi import FastAPI
        idx = make_index()
        app = FastAPI()
        app.include_router(create_hub_router(idx, fake_embed_fn))
        tc = TestClient(app)
        r = tc.get("/hub/search", params={"q": "hard disk", "k": 2})
        self.assertEqual(r.status_code, 200)
        self.assertIn("results_hash", r.json())
        r = tc.get("/hub/similar", params={"segment_id": "seg_000", "k": 2})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(all(h["segment_id"] != "seg_000" for h in r.json()["hits"]))
        r = tc.get("/hub/similar", params={"segment_id": "nope"})
        self.assertEqual(r.status_code, 404)
        r = tc.get("/hub/clusters", params={"threshold": 0.99})
        self.assertEqual(r.status_code, 200)
        r = tc.get("/hub/stats")
        self.assertEqual(r.json()["n_segments"], 6)

    def test_thumb_path_traversal(self):
        from fastapi import FastAPI
        idx = make_index(2)
        app = FastAPI()
        with tempfile.TemporaryDirectory() as d:
            app.include_router(create_hub_router(idx, fake_embed_fn,
                                                 thumb_cache_dir=d))
            tc = TestClient(app)
            for evil in ["..%2F..%2Fetc%2Fpasswd", "..", "seg_000%2F..%2Fseg_001",
                         "unknown_id"]:
                r = tc.get(f"/hub/thumb/{evil}.jpg")
                self.assertIn(r.status_code, (404, 422), evil)
            # Known id but missing file -> 404, never leaks the path.
            r = tc.get("/hub/thumb/seg_000.jpg")
            self.assertEqual(r.status_code, 404)
            self.assertNotIn("/tmp", r.text)


if __name__ == "__main__":
    unittest.main()
