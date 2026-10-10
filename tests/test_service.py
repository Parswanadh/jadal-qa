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
import threading
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


def recv_json_with_timeout(ws, timeout: float = 10.0):
    """Receive one websocket JSON message with a bounded wait.

    Starlette's TestClient receive has no timeout knob, so the wait runs in a
    daemon thread: when nothing arrives within ``timeout`` this raises
    ``TimeoutError`` with a clear message instead of hanging forever.
    """
    box: dict = {}

    def _recv() -> None:
        try:
            box["msg"] = ws.receive_json()
        except Exception as exc:  # noqa: BLE001 -- ferried to the caller
            box["err"] = exc

    worker = threading.Thread(target=_recv, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(
            f"no websocket message arrived within {timeout:.1f}s: "
            "the expected batch event was never pushed")
    if "err" in box:
        raise box["err"]
    return box.get("msg")


class BucketCase(StubsEnvMixin, unittest.TestCase):
    """Temp runtime dir; fake clips are written INSIDE the bucket (C1).

    Every test here runs on injected stub callables (plus JADAL_USE_STUBS=1),
    so results never depend on whether the real jadal.signals / jadal.decision
    / jadal.segmenter modules happen to be importable.
    """

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
        self._stage_patches = [
            mock.patch.object(pipeline, "_compute_signals_fn",
                              new=lambda: stubs.stub_compute_signals),
            mock.patch.object(pipeline, "_decider_class",
                              new=lambda: stubs.StubDecider),
            mock.patch.object(pipeline, "_split_clip_fn",
                              new=lambda: stubs.stub_split_clip),
        ]
        for patcher in self._stage_patches:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(getattr(self, "_stage_patches", [])):
            patcher.stop()
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
        # the class-level stub injection must be lifted for this check: it
        # proves the bare resolvers refuse to run without the flag.
        for patcher in self._stage_patches:
            patcher.stop()
        try:
            with mock.patch.dict(os.environ):
                os.environ.pop("JADAL_USE_STUBS", None)
                seg = {"segment_id": "s", "site_id": "x", "path": "/nope",
                       "sha256": "ab"}
                with self.assertRaises(RuntimeError):
                    pipeline.run_batch([seg], self.cfg)
        finally:
            for patcher in self._stage_patches:
                patcher.start()

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
        # Inline fallback truth (unit level): a clean clip accepts cleanly.
        segs = self.flat_segments(self.fake_clip("ok.mp4"))
        signals = stubs.stub_compute_signals(segs[0]["path"], self.cfg)
        decisions = stubs.StubDecider(self.cfg).decide(segs[0]["path"], signals)
        verdict = pipeline.make_verdict(
            segs[0], signals, decisions,
            pipeline.thresholds_snapshot(self.cfg), 4)
        self.assertEqual(verdict["verdict"], "accept")
        self.assertEqual(verdict["reasons"], [])

    def test_clean_clip_routes_to_review_until_calibrated(self) -> None:
        # Merged truth: the real funnel decides run_batch verdicts, and with
        # placeholder (uncalibrated) thresholds a clean, confident clip goes
        # to review -- never an accept -- with evidence attached.
        segs = self.flat_segments(self.fake_clip("ok2.mp4"))
        batch, receipts = pipeline.run_batch(segs, self.cfg)
        self.assertEqual(batch["n"], 1)
        verdict = batch["verdicts"][0]
        self.assertEqual(verdict["verdict"], "review")
        self.assertIn("evidence", verdict)
        self.assertTrue(verdict["evidence"])
        self.assertEqual(receipts[0]["verdict"]["verdict"], "review")

    def test_reason_codes_for_bad_signals(self) -> None:
        # L0 hard rules reject; L1 model outputs only demote to review
        # (funnel contract, enforced by the merged jadal.funnel.route).
        reject_cases = {
            "UNDECODABLE": {"signals": {"decodable": False}},
            "BLUR": {"signals": {"blur_var": 5.0}},
            "TOO_DARK": {"signals": {"brightness_mean": 3.0}},
            "OVEREXPOSED": {"signals": {"over_exposed_frac": 0.9}},
            "SHAKY": {"signals": {"motion_mean": 50.0}},
            "HANDS_NOT_VISIBLE": {"signals": {"hands_visible_frac": 0.1}},
            "DUPLICATE": {"signals": {"dup_of": "seg-other"}},
        }
        for code, payload in reject_cases.items():
            with self.subTest(code=code):
                kw = {"signals": payload.get("signals"),
                      "decisions": payload.get("decisions")}
                segs = self.flat_segments(self.fake_clip(f"bad_{code}.mp4", **kw))
                batch, _ = pipeline.run_batch(segs, self.cfg)
                self.assertEqual(batch["verdicts"][0]["verdict"], "reject", code)
                self.assertIn(code, batch["verdicts"][0]["reasons"], code)
        review_cases = {
            "NO_ACTIVE_WORK": {"decisions": {"active_work": {"p_yes": 0.2}}},
        }
        for code, payload in review_cases.items():
            with self.subTest(code=code):
                kw = {"signals": payload.get("signals"),
                      "decisions": payload.get("decisions")}
                segs = self.flat_segments(self.fake_clip(f"soft_{code}.mp4", **kw))
                batch, _ = pipeline.run_batch(segs, self.cfg)
                self.assertEqual(batch["verdicts"][0]["verdict"], "review", code)
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
        self.assertNotIn("DUPLICATE", by_id[first_id]["reasons"])
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
            # uncalibrated funnel: clean clips review, so nothing rejects and
            # the fixed batch rule still accepts the batch.
            for v in batch["verdicts"]:
                self.assertEqual(v["verdict"], "review")
                self.assertIn("evidence", v)
            self.assertEqual(batch["batch_decision"], "accept")
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
                for _ in range(6):
                    msg = recv_json_with_timeout(ws, timeout=5.0)
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
            # main now ships eval/results/funnel_calibration_synthetic.json, so
            # the default report surfaces its real (synthetic-labelled) heads;
            # the honest-empty case is covered with a temp dir below.
            body = r.json()
            self.assertIsInstance(body, list)
            self.assertTrue(body, "expected funnel heads from the shipped file")
            by_head = {e["head"]: e for e in body}
            self.assertIn("accept", by_head)
            self.assertEqual(by_head["accept"]["label_source"], "synthetic")
            r = c.get("/stats/sop_gaps")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json(), [])
            r = c.get("/sop/search", params={"q": "wash", "lang": "en-IN"})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json(), {"results": []})

    def test_risk_coverage_empty_without_measured_data(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"JADAL_EVAL_RESULTS_DIR": tmp}):
                app = create_app(self.cfg, batch_max_age_s=30.0,
                                 watch_bucket=False)
                with TestClient(app) as c:
                    r = c.get("/stats/risk_coverage")
                    self.assertEqual(r.status_code, 200)
                    self.assertEqual(r.json(), [])


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


class TestStubIndependence(BucketCase):
    """The suite behaves identically with and without jadal.signals importable.

    A fake jadal.signals/decison/segmenter stack whose entry points RAISE on
    any call is installed: with the class-level stub injection active, no
    verdict may change and the fake decoder must never run.
    """

    def _install_raising_stack(self):
        import sys
        import types
        from jadal import pipeline as _pipeline_mod
        calls = {"signals": 0, "segmenter": 0}
        fake_sig = types.ModuleType("jadal.signals")

        def _boom(path, config=None):
            calls["signals"] += 1
            raise AssertionError("real decoder must not run under forced stubs")

        fake_sig.compute_signals = _boom
        fake_dec = types.ModuleType("jadal.decision")
        fake_dec.Decider = stubs.StubDecider
        fake_seg = types.ModuleType("jadal.segmenter")

        def _seg_boom(path, seconds, out_dir):
            calls["segmenter"] += 1
            raise AssertionError("real segmenter must not run under forced stubs")

        fake_seg.split_clip = _seg_boom
        names = {"jadal.signals": fake_sig, "jadal.decision": fake_dec,
                 "jadal.segmenter": fake_seg}
        old_modules = {name: sys.modules.get(name) for name in names}
        parent = sys.modules.get("jadal")
        old_attrs = {}
        if parent is not None:
            for name, mod in names.items():
                attr = name.rpartition(".")[2]
                if hasattr(parent, attr):
                    old_attrs[attr] = getattr(parent, attr)
                setattr(parent, attr, mod)
        for name, mod in names.items():
            sys.modules[name] = mod

        def _restore():
            for name in names:
                if old_modules[name] is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old_modules[name]
            if parent is not None:
                for name in names:
                    attr = name.rpartition(".")[2]
                    if attr in old_attrs:
                        setattr(parent, attr, old_attrs[attr])
                    else:
                        try:
                            delattr(parent, attr)
                        except AttributeError:
                            pass

        self.addCleanup(_restore)
        return calls

    def _ingest_five(self, client, prefix, site):
        paths = [self.fake_clip(f"{prefix}_{i}.mp4") for i in range(5)]
        last = None
        for path in paths:
            response = client.post("/ingest/event", json={
                "path": path, "worker_id": "w", "site_id": site})
            self.assertEqual(response.status_code, 200, response.text)
            last = response.json()
        self.assertIsNotNone(last["batch"])
        return [(v["verdict"], tuple(v["reasons"]))
                for v in last["batch"]["verdicts"]]

    def test_verdicts_identical_with_and_without_fake_signals(self) -> None:
        store_mod.init_db(self.cfg.db_path)
        app = create_app(self.cfg, batch_max_age_s=30.0, watch_bucket=False)
        with TestClient(app) as client:
            plain = self._ingest_five(client, "plain", "indep-a")
            calls = self._install_raising_stack()
            faked = self._ingest_five(client, "faked", "indep-b")
        self.assertEqual(plain, faked)
        self.assertEqual(calls, {"signals": 0, "segmenter": 0})

    def test_verdicts_identical_with_signals_blocked(self) -> None:
        # Stronger: with jadal.signals made unimportable, the suite behaves
        # exactly as with it importable (stub injection, never the decoder).
        store_mod.init_db(self.cfg.db_path)
        app = create_app(self.cfg, batch_max_age_s=30.0, watch_bucket=False)
        with TestClient(app) as client:
            plain = self._ingest_five(client, "blk_plain", "blk-a")
            with stubs.blocked_imports("jadal.signals"):
                blocked = self._ingest_five(client, "blk_faked", "blk-b")
        self.assertEqual(plain, blocked)

    def test_blocked_imports_clear_stale_parent_attrs(self) -> None:
        # Regression: `from jadal import signals` resolves through the parent
        # package attribute, so blocking must clear it too -- otherwise a
        # previously imported real module leaks back in.
        import sys
        import jadal
        real = sys.modules.get("jadal.signals")
        try:
            sys.modules["jadal.signals"] = real
            setattr(jadal, "signals", real)
            with stubs.blocked_imports("jadal.signals"):
                with self.assertRaises(ImportError):
                    __import__("jadal.signals", fromlist=["compute_signals"])
                with self.assertRaises(ImportError):
                    __import__("jadal", fromlist=["signals"])
        finally:
            if real is None:
                sys.modules.pop("jadal.signals", None)
                try:
                    delattr(jadal, "signals")
                except AttributeError:
                    pass
            else:
                sys.modules["jadal.signals"] = real
                setattr(jadal, "signals", real)


class TestUndecodablePath(BucketCase):
    """Decoder failure -> UNDECODABLE reject + normal batch, never a hang."""

    def _client(self):
        store_mod.init_db(self.cfg.db_path)
        return TestClient(create_app(self.cfg, batch_max_age_s=30.0,
                                     watch_bucket=False))

    def test_decoder_raise_yields_undecodable_reject_batch(self) -> None:
        def _moov(path, config=None):
            raise ValueError("moov atom not found")

        with self._client() as client:
            paths = [self.fake_clip(f"garbage_{i}.mp4") for i in range(5)]
            with mock.patch.object(pipeline, "_compute_signals_fn",
                                   new=lambda: _moov):
                last = None
                for path in paths:
                    response = client.post("/ingest/event", json={
                        "path": path, "worker_id": "w", "site_id": "undec"})
                    self.assertEqual(response.status_code, 200, response.text)
                    last = response.json()
            batch = last["batch"]
            self.assertIsNotNone(batch, "batch must close normally")
            self.assertEqual(batch["n"], 5)
            for verdict in batch["verdicts"]:
                self.assertEqual(verdict["verdict"], "reject")
                self.assertIn("UNDECODABLE", verdict["reasons"])

    def test_transient_io_retries_are_capped_then_raise(self) -> None:
        calls = {"n": 0}

        def _io(path, config=None):
            calls["n"] += 1
            raise OSError("disk hiccup")

        with mock.patch.object(pipeline, "_compute_signals_fn",
                               new=lambda: _io):
            segs = pipeline.process_clip(self.fake_clip("io.mp4"),
                                         "w", "s", self.cfg)
            with self.assertRaises(OSError):
                pipeline.run_batch(segs, self.cfg)
        self.assertEqual(calls["n"], pipeline._SIGNAL_IO_RETRIES)

    def test_segmenter_failure_yields_placeholder_segment(self) -> None:
        def _split(path, seconds, out_dir):
            raise ValueError("moov atom not found")

        with mock.patch.object(pipeline, "_split_clip_fn", new=lambda: _split):
            segs = pipeline.process_clip(self.fake_clip("novideo.mp4"),
                                         "w", "s", self.cfg)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["start_s"], 0.0)
        self.assertEqual(segs[0]["site_id"], "s")
        batch, _ = pipeline.run_batch(segs, self.cfg)
        self.assertEqual(batch["n"], 1)

    def test_segmenter_oserror_propagates_for_retry(self) -> None:
        def _split(path, seconds, out_dir):
            raise OSError("transient disk error")

        with mock.patch.object(pipeline, "_split_clip_fn", new=lambda: _split):
            with self.assertRaises(OSError):
                pipeline.process_clip(self.fake_clip("io2.mp4"),
                                      "w", "s", self.cfg)


class TestFunnelSeam(BucketCase):
    """The pipeline calls jadal.funnel.route when importable, else inline."""

    def _install_fake_funnel(self, route_fn):
        import sys
        import types
        import jadal
        mod = types.ModuleType("jadal.funnel")
        mod.route = route_fn
        old = sys.modules.get("jadal.funnel")
        old_attr = getattr(jadal, "funnel", None)
        had_attr = hasattr(jadal, "funnel")
        sys.modules["jadal.funnel"] = mod
        setattr(jadal, "funnel", mod)

        def _restore():
            if old is None:
                sys.modules.pop("jadal.funnel", None)
            else:
                sys.modules["jadal.funnel"] = old
            if had_attr:
                setattr(jadal, "funnel", old_attr)
            else:
                try:
                    delattr(jadal, "funnel")
                except AttributeError:
                    pass

        self.addCleanup(_restore)
        return mod

    def test_fake_funnel_decides_with_evidence_in_hash(self) -> None:
        seen: dict = {}

        def _route(signals, decisions, thresholds):
            seen["keys"] = (isinstance(signals, dict),
                            isinstance(decisions, dict),
                            isinstance(thresholds, dict))
            return {
                "verdict": "reject",
                "reasons": ["BLUR"],
                "evidence": [{"check": "blur_var", "value": 5.0,
                              "op": ">=", "threshold": 60.0, "code": "BLUR"}],
                "answer_conf": 0.2,
                "queue_priority": 0.9,
            }

        self._install_fake_funnel(_route)
        segs = pipeline.process_clip(self.fake_clip("f.mp4"), "w", "s",
                                     self.cfg)
        batch, receipts = pipeline.run_batch(segs, self.cfg)
        self.assertEqual(seen["keys"], (True, True, True))
        verdict = batch["verdicts"][0]
        self.assertEqual(verdict["verdict"], "reject")
        self.assertEqual(verdict["reasons"], ["BLUR"])
        self.assertEqual(verdict["evidence"],
                         [{"check": "blur_var", "value": 5.0,
                           "op": ">=", "threshold": 60.0, "code": "BLUR"}])
        # evidence is included in the hash only when present: strip it and
        # the digest must change.
        stripped = dict(receipts[0])
        stripped["verdict"] = {k: v for k, v in verdict.items()
                               if k != "evidence"}
        self.assertNotEqual(receipts_mod.receipt_hash(receipts[0]),
                            receipts_mod.receipt_hash(stripped))
        result = pipeline.reverify_receipt(receipts[0], dict(segs[0]),
                                           self.cfg)
        self.assertTrue(result["match"])

    def test_inline_path_has_no_evidence_key(self) -> None:
        if importlib.util.find_spec("jadal.funnel") is not None:
            self.skipTest("real jadal.funnel present")
        self.assertIsNone(pipeline._funnel_route_fn())
        segs = pipeline.process_clip(self.fake_clip("g.mp4"), "w", "s",
                                     self.cfg)
        batch, _ = pipeline.run_batch(segs, self.cfg)
        self.assertEqual(batch["verdicts"][0]["verdict"], "accept")
        self.assertNotIn("evidence", batch["verdicts"][0])

    def test_funnel_route_exception_propagates(self) -> None:
        # A crashing router is a bug, not a measurement: never swallowed.
        def _raising(signals, decisions, thresholds):
            raise RuntimeError("router bug")

        self._install_fake_funnel(_raising)
        segs = pipeline.process_clip(self.fake_clip("rb.mp4"), "w", "s",
                                     self.cfg)
        with self.assertRaises(RuntimeError):
            pipeline.run_batch(segs, self.cfg)


class TestRiskCoverageHeads(BucketCase):
    """Per-head /stats/risk_coverage entries from real measurement files."""

    def _results_dir(self, files: dict) -> Path:
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        for name, payload in files.items():
            (out / name).write_text(json.dumps(payload), encoding="utf-8")
        return out

    def _decider_fixture(self) -> dict:
        return {"heads": {
            "hands_ok": {
                "calibration": {
                    "n": 300,
                    "threshold_pick": {"threshold": None, "accepted": 300,
                                       "errors": 34, "upper_bound": 0.1404,
                                       "coverage": 1.0},
                },
                "test": {"n": 300},
                "provenance": "DEV_ONLY",
                "label_caveat": "labels by gemini-2.5-flash, not human-checked",
            },
            "step": {
                "calibration": {
                    "n": 200,
                    "threshold_pick": {"threshold": 0.9, "accepted": 150,
                                       "errors": 2, "upper_bound": 0.04,
                                       "coverage": 0.75},
                },
                "test": {"n": 200},
                "label_caveat": "human gold labels v1",
            },
        }}

    def _funnel_fixture(self) -> dict:
        return {
            "accept": {
                "threshold": 0.4,
                "calib_certified": {"accepted": 364, "errors": 8,
                                    "upper_bound": 0.0355, "coverage": 0.6067},
                "n_calib": 600,
            },
            "meta": {
                "n_calib": 600, "n_heldout": 600,
                "provenance": "synthetic-fixtures",
                "targets": {"target_alpha": 0.05, "target_beta": 0.05,
                            "delta": 0.1, "min_accepted": 30},
            },
            "reject_rules": {},
        }

    def test_decider_heads_null_means_not_certified(self) -> None:
        out = self._results_dir({"decider_baseline.json": self._decider_fixture()})
        entries = pipeline.risk_coverage_report(config=self.cfg, results_dir=out)
        by_head = {e["head"]: e for e in entries}
        hands = by_head["hands_ok"]
        self.assertIn("chosen_threshold", hands)
        self.assertIsNone(hands["chosen_threshold"])  # measured, NOT certified
        self.assertEqual(hands["upper_bound"], 0.1404)
        self.assertEqual(hands["label_source"], "gemini")
        self.assertEqual(hands["n_calibration"], 300)
        self.assertEqual(hands["n_test"], 300)
        step = by_head["step"]
        self.assertEqual(step["chosen_threshold"], 0.9)
        self.assertEqual(step["label_source"], "human_gold")

    def test_funnel_accept_entry(self) -> None:
        out = self._results_dir({"funnel_calibration.json": self._funnel_fixture()})
        entries = pipeline.risk_coverage_report(config=self.cfg, results_dir=out)
        by_head = {e["head"]: e for e in entries}
        accept = by_head["accept"]
        self.assertEqual(accept["chosen_threshold"], 0.4)
        self.assertEqual(accept["upper_bound"], 0.0355)
        self.assertEqual(accept["target_err"], 0.05)
        self.assertEqual(accept["delta"], 0.1)
        self.assertEqual(accept["n_calibration"], 600)
        self.assertEqual(accept["n_test"], 600)
        self.assertEqual(accept["label_source"], "synthetic")

    def test_head_provenance_beats_file_provenance_with_conflict(self) -> None:
        out = self._results_dir({"decider_baseline.json": {"heads": {
            "step": {
                "calibration": {
                    "n": 200,
                    "threshold_pick": {"threshold": 0.9, "accepted": 150,
                                       "errors": 2, "upper_bound": 0.04,
                                       "coverage": 0.75},
                },
                "test": {"n": 200},
                # head says human gold, file says gemini: report the conflict,
                # never silently override either side.
                "label_caveat": "human gold labels v1",
            },
        }, "provenance": "labels by gemini-2.5-flash"}})
        entries = pipeline.risk_coverage_report(config=self.cfg, results_dir=out)
        by_head = {e["head"]: e for e in entries}
        self.assertEqual(by_head["step"]["label_source"],
                         "conflict:human_gold|gemini")

    def test_label_source_omitted_without_provenance(self) -> None:
        out = self._results_dir({"decider_baseline.json": {"heads": {
            "x": {"calibration": {"n": 10,
                                  "threshold_pick": {"threshold": None,
                                                     "upper_bound": None}}},
        }}})
        entries = pipeline.risk_coverage_report(config=self.cfg, results_dir=out)
        self.assertEqual(len(entries), 1)
        self.assertNotIn("label_source", entries[0])
        # measured but uncertified: null threshold, null bound.
        self.assertIsNone(entries[0]["chosen_threshold"])
        self.assertIsNone(entries[0]["upper_bound"])

    def test_store_head_omits_n_test(self) -> None:
        # The SQLite store holds no test set: n_test must be absent, never 0.
        points = [{"conf": 0.9 - i * 0.01, "correct": 1} for i in range(40)]
        entries = pipeline.risk_coverage_report(config=self.cfg,
                                                results_dir="/nonexistent",
                                                store_points=points)
        self.assertEqual(len(entries), 1)
        answer = entries[0]
        self.assertEqual(answer["head"], "answer")
        self.assertEqual(answer["n_calibration"], 40)
        self.assertNotIn("n_test", answer)

    def test_empty_without_measured_data(self) -> None:
        out = self._results_dir({})
        self.assertEqual(
            pipeline.risk_coverage_report(config=self.cfg, results_dir=out), [])
        bad = self._results_dir({"decider_baseline.json": {"heads": "nope"},
                                 "funnel_calibration.json": [1, 2]})
        self.assertEqual(
            pipeline.risk_coverage_report(config=self.cfg, results_dir=bad), [])

    def test_endpoint_serves_report(self) -> None:
        store_mod.init_db(self.cfg.db_path)
        out = self._results_dir({"funnel_calibration.json": self._funnel_fixture()})
        with mock.patch.dict(os.environ, {"JADAL_EVAL_RESULTS_DIR": str(out)}):
            app = create_app(self.cfg, batch_max_age_s=30.0, watch_bucket=False)
            with TestClient(app) as client:
                response = client.get("/stats/risk_coverage")
        self.assertEqual(response.status_code, 200)
        by_head = {e["head"]: e for e in response.json()}
        self.assertEqual(by_head["accept"]["chosen_threshold"], 0.4)
        self.assertEqual(by_head["accept"]["label_source"], "synthetic")

    def test_all_committed_examples_match_live_responses(self) -> None:
        # docs/api_examples/*.json regenerates via scripts/gen_api_examples.py
        # (stub mode, fixed clip bytes); this rebuilds the same dict live and
        # fails on ANY drift, so docs/API.md quotes real output.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gen_api_examples",
            ROOT / "scripts" / "gen_api_examples.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        live = mod.build_examples()
        examples = ROOT / "docs" / "api_examples"
        self.assertEqual(sorted(live), sorted(
            p.stem for p in examples.glob("*.json")
            if p.stem not in ("decider_baseline", "funnel_calibration")))
        for name, payload in live.items():
            with self.subTest(endpoint=name):
                if name == "pico_status":
                    # Live hardware answers heartbeats on /dev/ttyACM0, so
                    # values vary run to run: pin the shape, not the values.
                    self.assertEqual(set(payload),
                                     {"connected", "port", "last_hb_age_s",
                                      "last_hb_ticks_ms", "last_hb_led"})
                    self.assertIsInstance(payload["connected"], bool)
                    continue
                expected = json.loads(
                    (examples / f"{name}.json").read_text(encoding="utf-8"))
                self.assertEqual(payload, expected)


class TestHubMount(BucketCase):
    """The optional /hub router serves under /hub exactly once, either way."""

    def _app_with_fake_hub(self, prefix):
        import sys
        import types
        from fastapi import APIRouter
        mod = types.ModuleType("jadal.hub_api")
        router = APIRouter() if prefix is None else APIRouter(prefix=prefix)

        @router.get("/ping")
        def ping() -> dict:
            return {"pong": True}

        mod.router = router
        old = sys.modules.get("jadal.hub_api")
        sys.modules["jadal.hub_api"] = mod
        try:
            return create_app(self.cfg, batch_max_age_s=30.0, watch_bucket=False)
        finally:
            if old is None:
                sys.modules.pop("jadal.hub_api", None)
            else:
                sys.modules["jadal.hub_api"] = old

    def test_hub_absent_by_default(self) -> None:
        if importlib.util.find_spec("jadal.hub_api") is not None:
            self.skipTest("real jadal.hub_api present")
        app = create_app(self.cfg, batch_max_age_s=30.0, watch_bucket=False)
        paths = [route.path for route in app.routes
                 if hasattr(route, "path")]
        self.assertFalse(any(p.startswith("/hub") for p in paths))

    def test_prefixed_router_serves_under_hub_once(self) -> None:
        app = self._app_with_fake_hub(prefix="/hub")
        with TestClient(app) as client:
            self.assertEqual(client.get("/hub/ping").status_code, 200)
            self.assertEqual(client.get("/hub/ping").json(), {"pong": True})
            self.assertEqual(client.get("/ping").status_code, 404)

    def test_unprefixed_router_is_mounted_under_hub(self) -> None:
        app = self._app_with_fake_hub(prefix=None)
        with TestClient(app) as client:
            self.assertEqual(client.get("/hub/ping").status_code, 200)
            self.assertEqual(client.get("/hub/ping").json(), {"pong": True})
            self.assertEqual(client.get("/ping").status_code, 404)


class TestWsTimeout(BucketCase):
    """A missing batch event fails fast with a clear message, never hangs."""

    def test_silent_server_fails_fast(self) -> None:
        store_mod.init_db(self.cfg.db_path)
        app = create_app(self.cfg, batch_max_age_s=30.0, watch_bucket=False)
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                hello = recv_json_with_timeout(ws, timeout=5.0)
                self.assertEqual(hello.get("type"), "hello")
                with self.assertRaises(TimeoutError) as ctx:
                    recv_json_with_timeout(ws, timeout=1.0)
                self.assertIn("within 1.0s", str(ctx.exception))


class TestFlushCap(BucketCase):
    """A failing batch-timeout flush is capped, recorded, then parked."""

    def test_failing_flush_parks_after_cap(self) -> None:
        from jadal import service as service_mod
        store_mod.init_db(self.cfg.db_path)
        app = create_app(self.cfg, batch_max_age_s=30.0, watch_bucket=False)
        svc = app.state.service
        now = {"t": 1000.0}
        svc._clock = lambda: now["t"]
        paths = [self.fake_clip(f"flushcap_{i}.mp4") for i in range(2)]
        segs = [s for p in paths
                for s in pipeline.process_clip(p, "w", "s", self.cfg)]
        for seg in segs:
            svc._pending.setdefault("cap", []).append(
                {"segment": seg, "ts": now["t"]})

        def _full(*args, **kwargs):
            raise RuntimeError("disk full")

        with mock.patch.object(store_mod.Store, "save_batch", _full):
            with TestClient(app) as client:
                for _ in range(service_mod.FLUSH_MAX_FAILURES - 1):
                    now["t"] += 60.0
                    svc._tick()
                    self.assertEqual(svc.pending_count(), 2)
                    self.assertEqual(svc.failed_count(), 0)
                    gate = svc._flush_backoff.get("cap")
                    self.assertIsNotNone(gate)
                # one more tick reaches the cap: segments park as failed.
                now["t"] += 60.0
                svc._tick()
                self.assertEqual(svc.pending_count(), 0)
                self.assertEqual(svc.failed_count(), 2)
                self.assertNotIn("cap", svc._flush_backoff)
                # every failure was recorded and is exposed...
                errors = client.get("/ingest/errors").json()["errors"]
                self.assertIn("batch:cap", errors)
                self.assertIn("gave up", errors["batch:cap"])
                # ...and the parked segments are visible in the API.
                failed = client.get("/batches/failed").json()["failed"]
                self.assertEqual(len(failed), 2)
                self.assertTrue(all(f["site_id"] == "cap" for f in failed))
                self.assertTrue(all("disk full" in f["error"] for f in failed))
                self.assertEqual(
                    client.get("/health").json()["failed_segments"], 2)
                # no infinite loop: further ticks change nothing.
                now["t"] += 600.0
                svc._tick()
                self.assertEqual(svc.failed_count(), 2)

    def test_flush_backoff_schedule(self) -> None:
        from jadal import service as service_mod
        store_mod.init_db(self.cfg.db_path)
        app = create_app(self.cfg, batch_max_age_s=30.0, watch_bucket=False)
        svc = app.state.service
        now = {"t": 2000.0}
        svc._clock = lambda: now["t"]
        svc._pending["sched"] = [{"segment": {"segment_id": "s-1"},
                                  "ts": now["t"]}]

        def _boom(*args, **kwargs):
            raise RuntimeError("transient")

        with mock.patch.object(service_mod.pipeline, "run_batch",
                               side_effect=_boom):
            waits = []
            for _ in range(service_mod.FLUSH_MAX_FAILURES - 1):
                now["t"] += 60.0
                svc._tick()
                gate = svc._flush_backoff.get("sched")
                self.assertIsNotNone(gate)
                waits.append(round(gate["next_ts"] - now["t"], 6))
        self.assertEqual(waits, [1.0, 2.0, 4.0, 8.0, 16.0])

    def test_recovered_site_clears_flush_error(self) -> None:
        from jadal import service as service_mod
        store_mod.init_db(self.cfg.db_path)
        app = create_app(self.cfg, batch_max_age_s=30.0, watch_bucket=False)
        svc = app.state.service
        now = {"t": 3000.0}
        svc._clock = lambda: now["t"]
        paths = [self.fake_clip(f"heal_{i}.mp4") for i in range(2)]
        segs = [s for p in paths
                for s in pipeline.process_clip(p, "w", "s", self.cfg)]
        for seg in segs:
            svc._pending.setdefault("heal", []).append(
                {"segment": seg, "ts": now["t"]})
        calls = {"n": 0}
        real_save = store_mod.Store.save_batch

        def _flaky(self, batch, receipts, segments):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("disk full once")
            return real_save(self, batch, receipts, segments)

        with mock.patch.object(store_mod.Store, "save_batch", _flaky):
            with TestClient(app) as client:
                now["t"] += 60.0
                svc._tick()
                self.assertIn("batch:heal", svc._errors)
                now["t"] += 60.0
                svc._tick()
                self.assertNotIn("batch:heal", svc._errors)
                self.assertEqual(
                    client.get("/ingest/errors").json()["errors"], {})


class TestThresholdsFailClosed(BucketCase):
    """A tampered thresholds file fails closed, never silently degrades."""

    def _point_funnel_at(self, path) -> None:
        from jadal import funnel as funnel_mod
        old = funnel_mod.DEFAULT_THRESHOLDS_PATH
        funnel_mod.DEFAULT_THRESHOLDS_PATH = path
        self.addCleanup(setattr, funnel_mod, "DEFAULT_THRESHOLDS_PATH", old)

    def _tampered_file(self) -> str:
        from jadal import funnel as funnel_mod
        good = json.loads(Path(funnel_mod.DEFAULT_THRESHOLDS_PATH
                               ).read_text(encoding="utf-8"))
        bad = dict(good)
        bad["accept_conf"] = 0.0  # edited without rehashing
        path = self.cfg.bucket_dir / "tampered_thresholds.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        return str(path)

    def test_tampered_file_raises_thresholds_error(self) -> None:
        self._point_funnel_at(self._tampered_file())
        with self.assertRaises(pipeline.ThresholdsError):
            pipeline.thresholds_snapshot(self.cfg)

    def test_missing_file_falls_back_to_config_only(self) -> None:
        self._point_funnel_at(str(self.cfg.bucket_dir / "no-such-file.json"))
        snap = pipeline.thresholds_snapshot(self.cfg)
        self.assertNotIn("calibration", snap)
        self.assertNotIn("motion_min", snap)
        self.assertNotIn("funnel_thresholds_sha256", snap)
        self.assertIn("answer_threshold", snap)

    def test_valid_file_enriches_snapshot(self) -> None:
        snap = pipeline.thresholds_snapshot(self.cfg)
        self.assertIn("calibration", snap)
        self.assertIn("motion_min", snap)
        self.assertIn("funnel_thresholds_sha256", snap)

    def test_batch_fails_closed_to_review_with_mismatch_evidence(self) -> None:
        self._point_funnel_at(self._tampered_file())
        store_mod.init_db(self.cfg.db_path)
        app = create_app(self.cfg, batch_max_age_s=30.0, watch_bucket=False)
        with TestClient(app) as client:
            paths = [self.fake_clip(f"closed_{i}.mp4") for i in range(5)]
            last = None
            for path in paths:
                response = client.post("/ingest/event", json={
                    "path": path, "worker_id": "w", "site_id": "closed"})
                self.assertEqual(response.status_code, 200, response.text)
                last = response.json()
            batch = last["batch"]
            self.assertIsNotNone(batch, "fail-closed batch must still close")
            self.assertIn("thresholds_error", batch)
            for verdict in batch["verdicts"]:
                self.assertNotEqual(verdict["verdict"], "accept")
                self.assertEqual(verdict["verdict"], "review")
                checks = [e["check"] for e in verdict.get("evidence", [])]
                self.assertIn("thresholds", checks)
            errors = client.get("/ingest/errors").json()["errors"]
            self.assertIn("batch:closed", errors)
            self.assertIn("ThresholdsError", errors["batch:closed"])


class TestWatcherCap(BucketCase):
    """Watcher retries are capped: a poison file is parked, not spun on."""

    def test_poison_is_parked_after_max_attempts(self) -> None:
        from jadal import service as service_mod
        app = create_app(self.cfg, batch_max_age_s=60.0, watch_bucket=True,
                         watch_poll_s=0.05)
        svc = app.state.service
        bad = self.cfg.bucket_dir / "poison2.mp4"
        bad.write_bytes(b"poison-bytes")

        def _down(path, worker_id, site_id, config):
            raise OSError("disk down")

        with mock.patch.object(service_mod.pipeline, "process_clip",
                               side_effect=_down):
            svc._watch_backoff["poison2.mp4"] = {
                "attempts": service_mod.WATCH_MAX_ATTEMPTS - 1,
                "next_ts": 0.0,
            }
            svc._watch_bucket_tick()  # records the stability fingerprint
            svc._watch_bucket_tick()  # ingest attempt -> cap reached -> parked
        self.assertIn("poison2.mp4", svc._done_files)
        self.assertNotIn("poison2.mp4", svc._watch_backoff)
        key = str(Path(os.path.realpath(bad)))
        self.assertIn("gave up", svc._errors.get(key, ""))


if __name__ == "__main__":
    unittest.main()
