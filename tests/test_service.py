"""Acceptance tests for the service module (GitHub issue #3, fix round 1).

Contract under test: docs/SPEC.md. Covers the FastAPI service (jadal/service.py),
the batch pipeline (jadal/pipeline.py), SQLite persistence (jadal/store.py) and the
flag-gated deterministic stubs (jadal/stubs.py, active only with JADAL_USE_STUBS=1).

Security rules under test: ingest and the watcher only accept paths inside
cfg.bucket_dir; /voice allowlists reason_code/lang; stubs never read file
contents (exact stub values come from a programmatic, env-gated override
registry, never from clip bytes).
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jadal import certify as certify_mod  # noqa: E402
from jadal import receipts as receipts_mod  # noqa: E402
from jadal import stubs  # noqa: E402
from jadal.config import Config  # noqa: E402
from jadal import pipeline  # noqa: E402
from jadal.pipeline import thresholds_snapshot  # noqa: E402
from jadal import store as store_mod  # noqa: E402
from jadal.service import create_app  # noqa: E402

PASS_SIGNALS = {
    "decodable": True, "blur_var": 120.0, "brightness_mean": 90.0,
    "under_exposed_frac": 0.0, "over_exposed_frac": 0.02, "motion_mean": 2.0,
    "hands_visible_frac": 0.8, "hand_count_mode": 2, "faces_found": 0,
    "faces_blurred": 0, "dup_of": None,
}
PASS_DECISIONS = {
    "hands_ok": {"p_yes": 0.97},
    "active_work": {"p_yes": 0.93},
    "step": {"choice": "wash", "probs": {"wash": 0.93}, "answer_conf": 0.93},
}


def dict_merge(base: dict, override: dict | None) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = dict_merge(out[k], v)
        else:
            out[k] = v
    return out


class StubsEnvMixin:
    """Enable JADAL_USE_STUBS=1 for the test, restore afterwards."""

    def _stubs_on(self) -> None:
        self._old_flag = os.environ.get("JADAL_USE_STUBS")
        os.environ["JADAL_USE_STUBS"] = "1"

    def _stubs_off(self) -> None:
        if self._old_flag is None:
            os.environ.pop("JADAL_USE_STUBS", None)
        else:
            os.environ["JADAL_USE_STUBS"] = self._old_flag
        stubs.clear_test_overrides()


class BucketCase(StubsEnvMixin, unittest.TestCase):
    """Temp runtime dir; fake clips are written INSIDE the bucket (C1)."""

    def setUp(self) -> None:
        import tempfile
        self._stubs_on()
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.cfg = Config(
            batch_size=5, batch_reject_k=1, segment_seconds=20.0, seed=1234,
            bucket_dir=tmp / "bucket",
            segments_dir=tmp / "segments",
            blurred_dir=tmp / "blurred",
            db_path=tmp / "jadal.db",
            web_dir=tmp / "web",
            voicebank_dir=tmp / "voicebank",
        )
        self.cfg.bucket_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._stubs_off()
        self._tmp.cleanup()

    # -- helpers -----------------------------------------------------------
    def fake_clip(self, name: str, signals: dict | None = None,
                  decisions: dict | None = None) -> str:
        """Write deterministic bytes into the bucket; exact stub values go
        through the programmatic override registry, never the file."""
        blob = (f"clip:{name}:salt-v1\n".encode() * 64)[:4096]
        path = self.cfg.bucket_dir / name
        path.write_bytes(blob)
        sig = dict_merge(PASS_SIGNALS, signals)
        dec = dict_merge(PASS_DECISIONS, decisions)
        stubs.set_test_override(path, signals=sig, decisions=dec)
        return str(path)

    def raw_clip(self, name: str, blob: bytes) -> str:
        path = self.cfg.bucket_dir / name
        path.write_bytes(blob)
        return str(path)


class TestStubsAndStore(BucketCase):
    """Unit-level contract checks for the stubs and the SQLite store."""

    def test_stub_signals_are_deterministic(self) -> None:
        p = self.cfg.bucket_dir / "a.mp4"
        p.write_bytes(b"same-bytes")
        s1 = stubs.stub_compute_signals(str(p), self.cfg)
        s2 = stubs.stub_compute_signals(str(p), self.cfg)
        self.assertEqual(s1, s2)
        for key in ["decodable", "blur_var", "brightness_mean", "hands_visible_frac",
                    "phash", "dup_of", "faces_found"]:
            self.assertIn(key, s1)
        self.assertEqual(s1["blur_var"], round(s1["blur_var"], 4))

    def test_stub_ignores_file_contents(self) -> None:
        # even a file CLAIMING to be a payload is treated as ordinary bytes
        evil = b"JADALFAKE1\n" + json.dumps(
            {"signals": {"blur_var": 1.0, "decodable": True}}).encode()
        a = self.cfg.bucket_dir / "evil_a.mp4"
        b = self.cfg.bucket_dir / "evil_b.mp4"
        a.write_bytes(evil + b"A")
        b.write_bytes(evil + b"B")
        sa = stubs.stub_compute_signals(str(a), self.cfg)
        sb = stubs.stub_compute_signals(str(b), self.cfg)
        self.assertNotEqual(sa["blur_var"], 1.0)
        self.assertNotEqual(sa, sb)  # derived from bytes, not content

    def test_stubs_require_flag(self) -> None:
        import importlib.util
        if (importlib.util.find_spec("jadal.signals") is not None
                or importlib.util.find_spec("jadal.decision") is not None
                or importlib.util.find_spec("jadal.segmenter") is not None):
            self.skipTest("real L0/L1 modules present")
        with mock.patch.dict(os.environ):
            os.environ.pop("JADAL_USE_STUBS", None)
            seg = {"segment_id": "s", "site_id": "x", "path": "/nope",
                   "sha256": "ab"}
            with self.assertRaises(RuntimeError):
                pipeline.run_batch([seg], self.cfg)

    def test_test_overrides_require_flag(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("JADAL_USE_STUBS", None)
            with self.assertRaises(RuntimeError):
                stubs.set_test_override("/x", signals={})

    def test_real_batch_rule(self) -> None:
        acc = {"verdict": "accept"}

        def v(x): return {"verdict": x}
        self.assertEqual(certify_mod.batch_rule([acc, acc, v("reject")], 1), "accept")
        self.assertEqual(certify_mod.batch_rule([acc, v("reject"), v("reject")], 1), "reject")
        self.assertEqual(certify_mod.batch_rule([v("reject")] * 3, 3), "accept")
        self.assertEqual(certify_mod.batch_rule([v("reject")] * 4, 3), "reject")

    def test_real_receipt_hash_ignores_id_and_time(self) -> None:
        base = {
            "receipt_id": "aaa", "created_utc": "2020-01-01T00:00:00Z",
            "segment_sha256": "ff", "signals": {"decodable": True},
            "verdict": {"verdict": "accept"},
            "config": {"thresholds": {}, "pipeline_version": "x", "versions": {}},
        }
        other = dict(base, receipt_id="bbb", created_utc="2030-01-01T00:00:00Z")
        self.assertEqual(receipts_mod.receipt_hash(base),
                         receipts_mod.receipt_hash(other))

    def test_store_roundtrip(self) -> None:
        store_mod.init_db(self.cfg.db_path)
        st = store_mod.Store(self.cfg.db_path)
        batch = {
            "batch_id": "b-1", "site_id": "s", "segment_ids": ["seg-1"],
            "verdicts": [{"segment_id": "seg-1", "verdict": "accept", "reasons": []}],
            "batch_decision": "accept", "rule": "reject if rejected_count > k",
            "k": 1, "n": 1, "created_utc": "now",
        }
        receipt = {
            "receipt_id": "r-1", "segment_sha256": "ab", "receipt_sha256": "cd",
            "created_utc": "now",
            "signals": {}, "decisions": {},
            "verdict": {"segment_id": "seg-1", "verdict": "accept", "reasons": []},
            "config": {},
        }
        segment = {
            "segment_id": "seg-1", "source_clip": "clip.mp4", "worker_id": "w",
            "site_id": "s", "start_s": 0.0, "end_s": 20.0, "path": "/x/clip.mp4",
            "sha256": "ab", "created_utc": "now",
        }
        st.save_batch(batch, [receipt], [segment])
        self.assertEqual(st.get_batch("b-1")["batch_decision"], "accept")
        self.assertEqual(st.get_receipt("r-1")["receipt_sha256"], "cd")
        self.assertEqual(len(st.list_batches()), 1)
        self.assertEqual(st.get_segment("seg-1")["path"], "/x/clip.mp4")
        self.assertIsNone(st.get_batch("missing"))
        self.assertIsNone(st.get_receipt("missing"))


class TestPipeline(BucketCase):
    """Pipeline behaviour: verdict reasons, batch rule wiring, determinism."""

    def segments_for(self, *paths: str):
        return [pipeline.process_clip(p, "w", "s", self.cfg) for p in paths]

    def flat_segments(self, *paths: str):
        return [s for group in self.segments_for(*paths) for s in group]

    def test_clean_clip_is_accepted_with_no_reasons(self) -> None:
        segs = self.flat_segments(self.fake_clip("ok.mp4"))
        batch, receipts = pipeline.run_batch(segs, self.cfg)
        self.assertEqual(batch["n"], 1)
        self.assertEqual(batch["verdicts"][0]["verdict"], "accept")
        self.assertEqual(batch["verdicts"][0]["reasons"], [])
        self.assertEqual(batch["batch_decision"], "accept")
        self.assertEqual(receipts[0]["verdict"]["verdict"], "accept")

    def test_reason_codes_for_bad_signals(self) -> None:
        cases = {
            "UNDECODABLE": {"signals": {"decodable": False}},
            "BLUR": {"signals": {"blur_var": 5.0}},
            "TOO_DARK": {"signals": {"brightness_mean": 3.0}},
            "OVEREXPOSED": {"signals": {"over_exposed_frac": 0.9}},
            "SHAKY": {"signals": {"motion_mean": 50.0}},
            "HANDS_NOT_VISIBLE": {"signals": {"hands_visible_frac": 0.1}},
            "NO_ACTIVE_WORK": {"decisions": {"active_work": {"p_yes": 0.2}}},
            "DUPLICATE": {"signals": {"dup_of": "seg-other"}},
        }
        for code, payload in cases.items():
            with self.subTest(code=code):
                kw = {"signals": payload.get("signals"),
                      "decisions": payload.get("decisions")}
                segs = self.flat_segments(self.fake_clip(f"bad_{code}.mp4", **kw))
                batch, _ = pipeline.run_batch(segs, self.cfg)
                self.assertEqual(batch["verdicts"][0]["verdict"], "reject", code)
                self.assertIn(code, batch["verdicts"][0]["reasons"], code)

    def test_low_confidence_is_review(self) -> None:
        segs = self.flat_segments(self.fake_clip(
            "low.mp4", decisions={"step": {"choice": "wash", "answer_conf": 0.2}}))
        batch, _ = pipeline.run_batch(segs, self.cfg)
        self.assertEqual(batch["verdicts"][0]["verdict"], "review")
        self.assertIn("LOW_CONFIDENCE", batch["verdicts"][0]["reasons"])

    def test_duplicate_by_phash_within_batch(self) -> None:
        p1 = self.fake_clip("dup_a.mp4", signals={"phash": "samephash1"})
        p2 = self.fake_clip("dup_b.mp4", signals={"phash": "samephash1"})
        segs = self.flat_segments(p1, p2)
        batch, _ = pipeline.run_batch(segs, self.cfg)
        by_id = {v["segment_id"]: v for v in batch["verdicts"]}
        first_id, second_id = [s["segment_id"] for s in segs]
        self.assertEqual(by_id[first_id]["reasons"], [])
        self.assertEqual(by_id[second_id]["verdict"], "reject")
        self.assertIn("DUPLICATE", by_id[second_id]["reasons"])

    def test_batch_rule_two_rejects_beats_k1(self) -> None:
        paths = [
            self.fake_clip("k_a.mp4", signals={"blur_var": 2.0}),
            self.fake_clip("k_b.mp4", signals={"motion_mean": 40.0}),
        ]
        paths += [self.fake_clip(f"k_{i}.mp4") for i in range(3)]
        segs = self.flat_segments(*paths)
        batch, _ = pipeline.run_batch(segs, self.cfg)
        rejected = [v for v in batch["verdicts"] if v["verdict"] == "reject"]
        self.assertEqual(len(rejected), 2)
        self.assertEqual(batch["k"], 1)
        self.assertEqual(batch["batch_decision"], "reject")

    def test_one_reject_pass_with_k1(self) -> None:
        paths = [self.fake_clip("one_bad.mp4", signals={"blur_var": 1.0})]
        paths += [self.fake_clip(f"one_ok_{i}.mp4") for i in range(4)]
        segs = self.flat_segments(*paths)
        batch, _ = pipeline.run_batch(segs, self.cfg)
        rejected = [v for v in batch["verdicts"] if v["verdict"] == "reject"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(batch["batch_decision"], "accept")

    def test_determinism_single_vs_batch_of_five(self) -> None:
        paths = [self.fake_clip(f"det_{i}.mp4") for i in range(5)]
        segs = self.flat_segments(*paths)
        target = segs[2]
        _, solo_receipts = pipeline.run_batch([target], self.cfg)
        _, receipts = pipeline.run_batch(segs, self.cfg)
        self.assertEqual(len(solo_receipts), 1)
        solo_hash = solo_receipts[0]["receipt_sha256"]
        batch_hash = [r["receipt_sha256"] for r in receipts
                      if r["verdict"]["segment_id"] == target["segment_id"]][0]
        self.assertEqual(solo_hash, batch_hash)
        _, receipts2 = pipeline.run_batch(segs, self.cfg)
        self.assertEqual([r["receipt_sha256"] for r in receipts],
                         [r["receipt_sha256"] for r in receipts2])

    def test_receipt_hash_matches_spec_definition(self) -> None:
        segs = self.flat_segments(self.fake_clip("spec.mp4"))
        _, receipts = pipeline.run_batch(segs, self.cfg)
        receipt = dict(receipts[0])
        expected = receipt["receipt_sha256"]
        payload = {k: v for k, v in receipt.items()
                   if k not in ("receipt_sha256", "receipt_id", "created_utc")}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")
        import hashlib
        self.assertEqual(hashlib.sha256(blob).hexdigest(), expected)
        self.assertEqual(receipt["receipt_id"], expected[:16])

    def test_thresholds_snapshot_covers_l0_and_l2(self) -> None:
        snap = thresholds_snapshot(self.cfg)
        for key in ["blur_var_min", "brightness_min", "over_exposed_frac_max",
                    "motion_max", "hands_visible_frac_min", "answer_threshold",
                    "target_error", "delta", "min_accepted"]:
            self.assertIn(key, snap)


class TestServiceAcceptance(BucketCase):
    """Issue #3 acceptance criteria, end to end through the HTTP API."""

    def setUp(self) -> None:
        super().setUp()
        store_mod.init_db(self.cfg.db_path)
        # large max-age: batches here fire on N only (no timeout races)
        self.app = create_app(self.cfg, batch_max_age_s=30.0, watch_bucket=False)
        self._store = store_mod.Store(self.cfg.db_path)

    def client(self, **kw) -> TestClient:
        if "app" in kw:
            return TestClient(kw["app"])
        return TestClient(self.app)

    def ingest(self, client: TestClient, *paths: str, site: str = "site-1",
               worker: str = "w-1"):
        last = None
        for p in paths:
            r = client.post("/ingest/event", json={
                "path": p, "worker_id": worker, "site_id": site,
            })
            self.assertEqual(r.status_code, 200, r.text)
            last = r.json()
        return last

    def ingest_n(self, client: TestClient, n: int, site: str = "site-1"):
        paths = [self.fake_clip(f"clip_{site}_{i}.mp4") for i in range(n)]
        return self.ingest(client, *paths, site=site)

    def test_health(self) -> None:
        with self.client() as c:
            r = c.get("/health")
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertEqual(body["status"], "ok")
            self.assertIn("pipeline_version", body)

    def test_ingest_5_segments_creates_batch_with_5_verdicts_and_receipts(self) -> None:
        with self.client() as c:
            last = self.ingest_n(c, 5)
            self.assertIsNotNone(last["batch"], "batch should fire at N=5")
            batch = last["batch"]
            self.assertEqual(batch["n"], 5)
            self.assertEqual(len(batch["verdicts"]), 5)
            self.assertEqual(len(batch["segment_ids"]), 5)
            self.assertEqual(batch["batch_decision"], "accept")
            for v in batch["verdicts"]:
                self.assertEqual(v["verdict"], "accept")
            r = c.get(f"/batches/{batch['batch_id']}")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["n"], 5)
            receipts = self._store.get_receipts_for_batch(batch["batch_id"])
            self.assertEqual(len(receipts), 5)

    def test_batch_rule_applies_k(self) -> None:
        with self.client() as c:
            bad1 = self.fake_clip("kb1.mp4", signals={"blur_var": 2.0})
            bad2 = self.fake_clip("kb2.mp4", signals={"motion_mean": 44.0})
            oks = [self.fake_clip(f"k{i}.mp4") for i in range(3)]
            last = self.ingest(c, bad1, *oks[:1], bad2, *oks[1:], site="krule")
            batch = last["batch"]
            self.assertEqual(batch["k"], 1)
            rejected = [v for v in batch["verdicts"] if v["verdict"] == "reject"]
            self.assertEqual(len(rejected), 2)
            self.assertEqual(batch["batch_decision"], "reject")

    def test_ingest_outside_bucket_is_400(self) -> None:
        import tempfile
        with self.client() as c:
            with tempfile.TemporaryDirectory() as outside:
                evil = Path(outside) / "evil.mp4"
                evil.write_bytes(b"not in the bucket")
                r = c.post("/ingest/event", json={
                    "path": str(evil), "worker_id": "w", "site_id": "s"})
                self.assertEqual(r.status_code, 400)
                self.assertEqual(c.get("/batches").json()["batches"], [])

    def test_ingest_rejects_missing_clip(self) -> None:
        with self.client() as c:
            r = c.post("/ingest/event", json={
                "path": str(self.cfg.bucket_dir / "nope.mp4"),
                "worker_id": "w", "site_id": "s"})
            self.assertEqual(r.status_code, 400)

    def test_failed_clip_is_not_poisoned_and_can_retry(self) -> None:
        from jadal import service as service_mod
        path = self.fake_clip("flaky.mp4")
        real = pipeline.process_clip
        calls = {"n": 0}

        def flaky(p, w, s, cfg):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return real(p, w, s, cfg)

        with self.client() as c:
            with mock.patch.object(service_mod.pipeline, "process_clip",
                                   side_effect=flaky):
                r = c.post("/ingest/event", json={
                    "path": path, "worker_id": "w", "site_id": "retry"})
            self.assertEqual(r.status_code, 500)
            self.assertNotIn("boom", r.text)  # no internals in the body
            r = c.post("/ingest/event", json={
                "path": path, "worker_id": "w", "site_id": "retry"})
            self.assertEqual(r.status_code, 200)
            self.assertFalse(r.json()["duplicate"])

    def test_batch_never_exceeds_n(self) -> None:
        app = create_app(self.cfg, batch_max_age_s=30.0, watch_bucket=False)
        with self.client(app=app) as c:
            paths = [self.fake_clip(f"big_{i}.mp4") for i in range(7)]
            responses = []
            for p in paths:
                r = c.post("/ingest/event", json={
                    "path": p, "worker_id": "w", "site_id": "big"})
                self.assertEqual(r.status_code, 200, r.text)
                responses.append(r.json())
            self.assertIsNone(responses[3]["batch"])
            self.assertIsNotNone(responses[4]["batch"])
            self.assertEqual(responses[4]["batch"]["n"], 5)
            self.assertIsNone(responses[6]["batch"])
            batches = c.get("/batches").json()["batches"]
            self.assertEqual(len(batches), 1)
            self.assertEqual(c.get("/health").json()["pending_segments"], 2)

    def test_reverify_returns_match_true(self) -> None:
        with self.client() as c:
            last = self.ingest_n(c, 5)
            receipts = self._store.get_receipts_for_batch(last["batch"]["batch_id"])
            self.assertEqual(len(receipts), 5)
            for rec in receipts:
                r = c.post(f"/receipts/{rec['receipt_id']}/reverify")
                self.assertEqual(r.status_code, 200, r.text)
                body = r.json()
                self.assertTrue(body["match"])
                self.assertEqual(body["diff"], [])
                self.assertEqual(body["expected"], body["actual"])
                self.assertEqual(body["actual"], rec["receipt_sha256"])

    def test_reverify_replays_stored_config(self) -> None:
        with self.client() as c:
            last = self.ingest_n(c, 5)
            receipts = self._store.get_receipts_for_batch(last["batch"]["batch_id"])
            # drift the LIVE config after the receipts were written
            self.cfg.blur_var_min = 9999.0
            self.cfg.brightness_min = 9999.0
            self.cfg.motion_max = 0.0
            for rec in receipts:
                r = c.post(f"/receipts/{rec['receipt_id']}/reverify")
                self.assertEqual(r.status_code, 200, r.text)
                self.assertTrue(r.json()["match"],
                                "stored config must be replayed, not live config")

    def test_reverify_mismatch_when_clip_changed(self) -> None:
        with self.client() as c:
            last = self.ingest_n(c, 5)
            receipts = self._store.get_receipts_for_batch(last["batch"]["batch_id"])
            rid = receipts[0]["receipt_id"]
            rec = self._store.get_receipt(rid)
            seg = self._store.get_segment(rec["verdict"]["segment_id"])
            Path(seg["path"]).write_bytes(b"tampered-bytes")
            stubs.clear_test_overrides()
            r = c.post(f"/receipts/{rid}/reverify")
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertFalse(body["match"])
            self.assertNotEqual(body["expected"], body["actual"])
            self.assertTrue(any("sha256" in p for p in body["diff"]))

    def test_get_batches_and_receipts(self) -> None:
        with self.client() as c:
            last = self.ingest_n(c, 5)
            batch_id = last["batch"]["batch_id"]
            r = c.get("/batches")
            self.assertEqual(r.status_code, 200)
            summaries = r.json()["batches"]
            self.assertEqual([b["batch_id"] for b in summaries], [batch_id])
            r = c.get(f"/batches/{batch_id}")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(len(r.json()["verdicts"]), 5)
            for rec in self._store.get_receipts_for_batch(batch_id):
                r = c.get(f"/receipts/{rec['receipt_id']}")
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["receipt_id"], rec["receipt_id"])

    def test_batch_timeout_trigger(self) -> None:
        app = create_app(self.cfg, batch_max_age_s=0.05, watch_bucket=False)
        with self.client(app=app) as c:
            self.ingest_n(c, 2, site="slow")
            deadline = time.time() + 3.0
            batches = []
            while time.time() < deadline:
                batches = [b for b in c.get("/batches").json()["batches"]
                           if b["site_id"] == "slow"]
                if batches:
                    break
                time.sleep(0.05)
            self.assertEqual(len(batches), 1)
            self.assertEqual(batches[0]["n"], 2)
            self.assertEqual(batches[0]["site_id"], "slow")

    def test_websocket_pushes_batch_event(self) -> None:
        with self.client() as c:
            with c.websocket_connect("/ws") as ws:
                self.ingest_n(c, 5, site="wsgen")
                event = None
                for _ in range(5):
                    msg = ws.receive_json()
                    if msg.get("type") == "batch":
                        event = msg
                        break
                self.assertIsNotNone(event, "expected a batch event on /ws")
                self.assertEqual(event["n"], 5)

    def test_unknown_ids_are_404(self) -> None:
        with self.client() as c:
            self.assertEqual(c.get("/batches/missing").status_code, 404)
            self.assertEqual(c.get("/receipts/missing").status_code, 404)
            self.assertEqual(c.post("/receipts/missing/reverify").status_code, 404)

    def test_pico_status_shape(self) -> None:
        with self.client() as c:
            r = c.get("/pico/status")
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertIsInstance(body.get("connected"), bool)
            self.assertTrue("port" in body or "reason" in body)

    def test_voice_endpoint_serves_and_404s(self) -> None:
        wav = self.cfg.voicebank_dir / "en-IN" / "BLUR.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"RIFFfake")
        with self.client() as c:
            r = c.get("/voice/BLUR/en-IN")
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.content.startswith(b"RIFF"))

    def test_voice_rejects_traversal_and_unknowns(self) -> None:
        secret = self.cfg.voicebank_dir.parent / "secret.wav"
        secret.write_bytes(b"RIFFsecret")
        with self.client() as c:
            self.assertEqual(c.get("/voice/secret/%2E%2E").status_code, 404)
            self.assertEqual(c.get("/voice/../en-IN").status_code, 404)
            self.assertEqual(c.get("/voice/NOPE/en-IN").status_code, 404)
            self.assertEqual(c.get("/voice/BLUR/xx-YY").status_code, 404)

    def test_stats_and_sop_endpoints_work(self) -> None:
        with self.client() as c:
            r = c.get("/stats/risk_coverage")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json(), [])
            r = c.get("/stats/sop_gaps")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json(), [])
            r = c.get("/sop/search", params={"q": "wash", "lang": "en-IN"})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json(), {"results": []})


class TestWatcher(BucketCase):
    """Bucket watcher: ignores .part files, waits for stable size, ingests."""

    def test_watcher_ingests_dropped_files_without_posts(self) -> None:
        app = create_app(self.cfg, batch_max_age_s=0.2, watch_bucket=True,
                         watch_poll_s=0.05)
        for i in range(2):
            blob = f"watch:{i}:salt\n".encode() * 64
            (self.cfg.bucket_dir / f"w_{i}.mp4").write_bytes(blob)
            stubs.set_test_override(self.cfg.bucket_dir / f"w_{i}.mp4",
                                    signals=dict(PASS_SIGNALS),
                                    decisions=dict(PASS_DECISIONS))
        with TestClient(app) as c:
            deadline = time.time() + 5.0
            batches = []
            while time.time() < deadline:
                batches = c.get("/batches").json()["batches"]
                if batches:
                    break
                time.sleep(0.05)
            self.assertEqual(len(batches), 1)
            self.assertEqual(batches[0]["n"], 2)

    def test_watcher_ignores_part_files(self) -> None:
        app = create_app(self.cfg, batch_max_age_s=0.2, watch_bucket=True,
                         watch_poll_s=0.05)
        (self.cfg.bucket_dir / ".x.mp4.part").write_bytes(b"staging")
        (self.cfg.bucket_dir / "y.mp4.part").write_bytes(b"staging")
        with TestClient(app) as c:
            time.sleep(0.6)
            self.assertEqual(c.get("/batches").json()["batches"], [])


class TestDockSim(unittest.TestCase):
    def test_dock_sim_module_imports_cleanly(self) -> None:
        path = ROOT / "sim" / "dock_sim.py"
        spec = importlib.util.spec_from_file_location("dock_sim_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertTrue(callable(mod.main))

    def test_dock_sim_copies_atomically_and_posts(self) -> None:
        import tempfile
        path = ROOT / "sim" / "dock_sim.py"
        spec = importlib.util.spec_from_file_location("dock_sim_run", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            bucket = Path(tmp) / "bucket"
            src.mkdir()
            bucket.mkdir()
            (src / "a.mp4").write_bytes(b"clip-bytes-a")
            (src / "b.mp4").write_bytes(b"clip-bytes-b")
            rc = mod.main(["--clips", str(src), "--url", "http://127.0.0.1:9",
                           "--delay", "0", "--bucket", str(bucket)])
            self.assertEqual(rc, 0)
            self.assertEqual(sorted(p.name for p in bucket.iterdir()),
                             ["a.mp4", "b.mp4"])
            leftovers = [p.name for p in bucket.iterdir()
                         if p.suffix == ".part" or p.name.startswith(".")]
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
