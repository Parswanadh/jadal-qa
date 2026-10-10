"""Hardware-free tests for the video hub (GitHub issue #27, fix round 1).

No model, no GPU, no network by default: a deterministic fake embedder
(sha256-seeded unit vectors) stands in for HubEmbedder via
Index(embed_fn=...). GPU-gated tests (real embedder) run only with
JADAL_TEST_GPU=1, under `flock tools/gpu.lock` (see docs/reports/hub.md).

Scratch dirs live under data_runtime/hub/.test_tmp (git-ignored): SPEC
forbids writing RESULTS to /tmp, so even test scratch stays in the repo
runtime dir.

Covers: build/save/load round trip + hash verification on load, index_hash
stability, filters, tie-breaking by segment_id (incl. top-k boundary ties),
SPEC batch-invariance (fake always; real embedder when GPU-gated), cluster
logic incl. one giant cluster + N cap (413), results_hash determinism,
thumb path-traversal + '#'-ids, filter injection, device construction,
frame-sampling errors, no-pickle save format.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jadal.hub import ClusterLimitError, Index  # noqa: E402
from jadal.hub_api import create_hub_router, valid_segment_id  # noqa: E402

DIM = 16

# Repo-local scratch (git-ignored via data_runtime/); see module docstring.
SCRATCH = ROOT / "data_runtime" / "hub" / ".test_tmp"


def scratch_dir(name: str) -> Path:
    d = SCRATCH / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


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
        d = scratch_dir("roundtrip")
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
        for sid in sorted(a.ids_sorted(), reverse=True):
            b.add(sid, a.get_vector(sid), a.get_meta(sid))
        self.assertEqual(a.index_hash(), b.index_hash())

    def test_load_verifies_hash(self):
        from jadal.hub import Index as I
        idx = make_index(3)
        d = scratch_dir("verify")
        idx.save(d)
        # Tamper with the stored hash -> load must refuse.
        (d / "index_hash.txt").write_text("0" * 64)
        with self.assertRaises(ValueError):
            I.load(d)
        # Restore hash but corrupt the matrix -> load must refuse.
        idx.save(d)
        z = dict(np.load(d / "index.npz", allow_pickle=False))
        mat = np.asarray(z["matrix"]).copy()
        mat[0, 0] = np.float16(mat[0, 0] + 1.0)
        np.savez_compressed(d / "index.npz", ids=np.asarray(z["ids"]),
                            matrix=mat, dim=np.asarray(z["dim"]))
        with self.assertRaises(ValueError):
            I.load(d)

    def test_load_needs_no_pickle(self):
        idx = make_index(2)
        d = scratch_dir("nopickle")
        idx.save(d)
        # allow_pickle=False must suffice (ids are plain unicode).
        z = np.load(d / "index.npz", allow_pickle=False)
        self.assertEqual(len(z["ids"]), 2)


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

    def test_topk_boundary_ties_resolve_by_id(self):
        # 100 identical vectors, k=5: exact top-k must be the 5 smallest ids
        # (argpartition alone would return an arbitrary 5).
        idx = Index(dim=DIM)
        v = fake_vec("tie100")
        for i in range(100):
            sid = f"t_{i:03d}"
            idx.add(sid, v, {"clip": "c", "start_s": 0.0, "end_s": 1.0,
                             "worker_id": "w", "site_id": "s",
                             "verdict": "review", "reasons": [],
                             "step": None, "path": None})
        hits = idx.search_by_vector(v, k=5)["hits"]
        self.assertEqual([h["segment_id"] for h in hits],
                         [f"t_{i:03d}" for i in range(5)])

    def test_torch_backend_matches_numpy(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        idx = make_index(30)
        q = fake_vec("backend-q")
        a = idx.search_by_vector(q, k=7)["hits"]
        b = idx.search_by_vector(q, k=7, use_torch=True)["hits"]
        self.assertEqual([h["segment_id"] for h in a],
                         [h["segment_id"] for h in b])


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
        d = scratch_dir("thumbs")
        app.include_router(create_hub_router(idx, fake_embed_fn,
                                             thumb_cache_dir=d))
        tc = TestClient(app)
        for evil in ["..%2F..%2Fetc%2Fpasswd", "..", "seg_000%2F..%2Fseg_001",
                     "..%2Fseg_000", "%2e%2e", "unknown_id"]:
            r = tc.get(f"/hub/thumb/{evil}.jpg")
            self.assertIn(r.status_code, (404, 422), evil)
        # Known id but missing file -> 404, never leaks the path.
        r = tc.get("/hub/thumb/seg_000.jpg")
        self.assertEqual(r.status_code, 404)
        self.assertNotIn("/tmp", r.text)

    def test_clusters_cap_returns_413(self):
        from fastapi import FastAPI
        idx = Index(dim=DIM, max_clusters_n=4)
        for i in range(5):
            idx.add(f"c_{i}", fake_vec(f"cap{i}"),
                    {"clip": "c", "start_s": 0.0, "end_s": 1.0,
                     "worker_id": "w", "site_id": "s", "verdict": "review",
                     "reasons": [], "step": None, "path": None})
        with self.assertRaises(ClusterLimitError):
            idx.dedup_clusters(threshold=0.99)
        app = FastAPI()
        app.include_router(create_hub_router(idx, fake_embed_fn))
        tc = TestClient(app)
        r = tc.get("/hub/clusters", params={"threshold": 0.99})
        self.assertEqual(r.status_code, 413)


def _write_tiny_mp4(path: Path, frames: int = 8, size: int = 48) -> Path:
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                        10.0, (size, size))
    rng = np.random.default_rng(7)
    for _ in range(frames):
        w.write(rng.integers(0, 255, (size, size, 3)).astype(np.uint8))
    w.release()
    return path


class TestIdGrammar(unittest.TestCase):
    EVAL_ID = "User_1_412_2010_3_480#0003"

    def test_eval_ids_match_one_grammar(self):
        self.assertTrue(valid_segment_id(self.EVAL_ID))
        self.assertTrue(valid_segment_id("seg_000"))
        for bad in ["../x", "a/b", "a%b", "..", "", ".x", "-x"]:
            self.assertFalse(valid_segment_id(bad), bad)

    def test_similar_and_thumb_with_hash_id(self):
        from fastapi import FastAPI
        d = scratch_dir("hashid")
        clip = _write_tiny_mp4(d / "tiny.mp4")
        idx = Index(dim=DIM, embed_fn=fake_embed_fn)
        idx.add(self.EVAL_ID, fake_vec("hashseg"),
                {"clip": "tiny", "start_s": 0.0, "end_s": 0.5,
                 "worker_id": "w", "site_id": "s", "verdict": "review",
                 "reasons": [], "step": None, "path": str(clip)})
        idx.add("other_1", fake_vec("other"),
                {"clip": "tiny", "start_s": 0.0, "end_s": 0.5,
                 "worker_id": "w", "site_id": "s", "verdict": "review",
                 "reasons": [], "step": None, "path": str(clip)})
        app = FastAPI()
        app.include_router(create_hub_router(idx, fake_embed_fn,
                                             thumb_cache_dir=d / "th"))
        tc = TestClient(app)
        enc = self.EVAL_ID.replace("#", "%23")
        r = tc.get("/hub/similar", params={"segment_id": self.EVAL_ID, "k": 2})
        self.assertEqual(r.status_code, 200)
        r = tc.get(f"/hub/thumb/{enc}.jpg")
        self.assertEqual(r.status_code, 200, r.text[:200])
        self.assertEqual(r.headers["content-type"], "image/jpeg")
        # Traversal via encoded slash still blocked.
        r = tc.get("/hub/thumb/..%2Ftiny.jpg")
        self.assertIn(r.status_code, (404, 422))


class FakeSegmentEmbedder:
    """Stateless fake with the HubEmbedder segment interface.

    embed_sequence processes 5 segments 'together' the way the hub always
    does (one model call per segment, batch size 1, in a loop) — the SPEC
    batch-invariance shape: alone vs inside a batch of 5 must be identical.
    """

    def __init__(self, dim: int = DIM):
        self.dim = dim

    def embed_segment(self, path: str, start_s: float, end_s: float,
                      n_frames: int = 8) -> np.ndarray:
        return fake_vec(f"{path}|{start_s:.3f}|{end_s:.3f}|{n_frames}|{self.dim}",
                        self.dim)

    def embed_sequence(self, specs: list) -> list[np.ndarray]:
        return [self.embed_segment(*s) for s in specs]


def _meta(path="c", start=0.0, end=1.0) -> dict:
    return {"clip": path, "start_s": start, "end_s": end, "worker_id": "w",
            "site_id": "s", "verdict": "review", "reasons": [], "step": None,
            "path": path}


class TestBatchInvariance(unittest.TestCase):
    """SPEC determinism: one segment alone vs inside a batch of 5 gives
    IDENTICAL index vectors and hashes (hardware-free fake version)."""

    def test_alone_vs_in_batch_of_five_identical(self):
        emb = FakeSegmentEmbedder()
        specs = [(f"clip{i}.mp4", float(i * 10), float(i * 10 + 10)) for i in range(5)]
        alone = emb.embed_segment(*specs[2])
        batch = emb.embed_sequence(specs)
        for v in batch:
            self.assertEqual(v.shape, (DIM,))
        np.testing.assert_array_equal(alone, batch[2])
        ia, ib = Index(dim=DIM), Index(dim=DIM)
        ia.add("target", alone, _meta())
        for s, v in zip(specs, batch):
            ib.add(f"seg_{s[0]}", v, _meta())
        tgt = ib.search_by_vector(alone, k=5)["hits"][0]
        # float16 storage quantises the self-match to 0.9999 at 4 dp.
        self.assertGreaterEqual(tgt["score"], 0.999)
        ic = Index(dim=DIM)
        ic.add("target", batch[2], _meta())
        self.assertEqual(ia.index_hash(), ic.index_hash())


GPU_CLIP = ("/home/parshu/projects/awing/data/indego/1_Assembly/mp4_480/"
            "User_1_412_2010_3_480.mp4")


@unittest.skipUnless(os.environ.get("JADAL_TEST_GPU") == "1",
                     "needs JADAL_TEST_GPU=1 + GPU (run under flock)")
class TestBatchInvarianceGPU(unittest.TestCase):
    """SPEC batch-invariance on the REAL embedder + evidence file."""

    def test_real_embedder_alone_vs_batch_of_five(self):
        if not Path(GPU_CLIP).is_file():
            self.skipTest("IndEgo clip not present")
        import json as _json
        from jadal.hub_embed import HubEmbedder
        emb = HubEmbedder(dim=256, n_frames=4, cache_dir=None)
        specs = [(GPU_CLIP, 10.0 + i * 30.0, 20.0 + i * 30.0) for i in range(5)]
        alone = emb.embed_segment(*specs[2])
        batch = [emb.embed_segment(*s) for s in specs]  # hub's batch-of-5 shape
        diffs = [float(np.max(np.abs(alone - v))) if j == 2
                 else None for j, v in enumerate(batch)]
        self.assertEqual(diffs[2], 0.0)
        ia, ic = Index(dim=256), Index(dim=256)
        ia.add("target", alone, _meta())
        ic.add("target", batch[2], _meta())
        self.assertEqual(ia.index_hash(), ic.index_hash())
        ev = {"segment": os.path.basename(GPU_CLIP) + " [70,80]s N=4",
              "max_abs_diff_alone_vs_in_batch_of_5": diffs[2],
              "index_hash_alone": ia.index_hash(),
              "index_hash_batched": ic.index_hash(),
              "route": emb.last_route, "runtime_dtype": emb.runtime_dtype}
        out = ROOT / "eval" / "results" / "hub_batch_invariance.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json.dumps(ev, indent=2, sort_keys=True))
        print(f"\nwrote {out}")


class TestDeviceConstruction(unittest.TestCase):
    def test_device_goes_top_level(self):
        from unittest import mock
        from jadal import hub_embed as he
        seen: dict = {}

        class FakeST:
            def __init__(self, *a, **k):
                seen["args"], seen["kwargs"] = a, k
                import torch
                self._p = torch.nn.Parameter(torch.zeros(1))

            def parameters(self):
                yield self._p

        for dev in ("cpu", "cuda"):
            seen.clear()
            with mock.patch("sentence_transformers.SentenceTransformer", FakeST):
                emb = he.HubEmbedder(device=dev)
                emb._ensure_model()
            self.assertEqual(seen["kwargs"].get("device"), dev)
            self.assertIn("torch_dtype", seen["kwargs"].get("model_kwargs", {}))
            self.assertNotIn("device", seen["kwargs"].get("model_kwargs", {}))


class TestSampleFramesErrors(unittest.TestCase):
    def test_missing_and_bad_interval(self):
        from jadal.hub_embed import sample_frames
        with self.assertRaises(FileNotFoundError):
            sample_frames("/nonexistent/clip.mp4", 0.0, 1.0, 4)
        d = scratch_dir("frames")
        clip = _write_tiny_mp4(d / "tiny.mp4")  # 8 frames @10fps = 0.8 s
        with self.assertRaises(ValueError):
            sample_frames(clip, 0.0, 5.0, 4)  # end_s beyond duration
        with self.assertRaises(ValueError):
            sample_frames(clip, -1.0, 0.5, 4)
        with self.assertRaises(ValueError):
            sample_frames(clip, 0.5, 0.5, 4)
        # No silent substitution: repeated reads give identical bytes.
        a = sample_frames(clip, 0.0, 0.8, 4)
        b = sample_frames(clip, 0.0, 0.8, 4)
        self.assertEqual([np.asarray(f).tobytes() for f in a],
                         [np.asarray(f).tobytes() for f in b])


if __name__ == "__main__":
    unittest.main()
