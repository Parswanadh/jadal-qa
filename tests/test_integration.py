"""Integration tests owned by the integration worker (GitHub issue #17).

Covers the integration-owned behaviour in jadal/pipeline.py and
jadal/service.py: version provenance in receipts, reverify version
comparison, failure-safe batch flush, watcher retry-with-backoff +
GET /ingest/errors, real-data /stats endpoints, and the SPEC determinism
rule (same clip 10x identical; alone vs inside a batch of 5).

Runs on stubs (JADAL_USE_STUBS=1); the real jadal.signals / jadal.decision
modules are not in this worktree yet, so backend-specific assertions are
gated on importability and skipped otherwise.
"""
from __future__ import annotations

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

from jadal import pipeline  # noqa: E402
from jadal import stubs  # noqa: E402
from jadal.config import Config  # noqa: E402
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


def _merge(base: dict, override: dict | None) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class IntegrationCase(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        self._old_flag = os.environ.get("JADAL_USE_STUBS")
        os.environ["JADAL_USE_STUBS"] = "1"
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
        if self._old_flag is None:
            os.environ.pop("JADAL_USE_STUBS", None)
        else:
            os.environ["JADAL_USE_STUBS"] = self._old_flag
        stubs.clear_test_overrides()
        self._tmp.cleanup()

    def fake_clip(self, name: str, signals=None, decisions=None) -> str:
        blob = (f"iclip:{name}\n".encode() * 64)[:4096]
        path = self.cfg.bucket_dir / name
        path.write_bytes(blob)
        stubs.set_test_override(
            path,
            signals=_merge(PASS_SIGNALS, signals),
            decisions=_merge(PASS_DECISIONS, decisions),
        )
        return str(path)


class TestVersionProvenance(IntegrationCase):
    def test_impl_versions_names_stubs_until_real_modules_merge(self) -> None:
        versions = pipeline.impl_versions(self.cfg)
        self.assertEqual(versions["certify"], "jadal.certify")
        self.assertEqual(versions["receipts"], "jadal.receipts")
        self.assertIn("python", versions)
        try:
            __import__("jadal.signals", fromlist=["compute_signals"])
            real_signals = True
        except ImportError:
            real_signals = False
        if real_signals:
            self.assertEqual(versions["signals"], "jadal.signals")
        else:
            self.assertEqual(versions["signals"], "stubs.signals")
            self.assertEqual(versions["decision"], "stubs.decision")
            self.assertEqual(versions["segmenter"], "stubs.segmenter")

    def test_weight_token_hashes_real_files_and_skips_missing(self) -> None:
        self.assertIsNone(pipeline._weight_token(None))
        self.assertIsNone(pipeline._weight_token(self.cfg.bucket_dir / "nope.bin"))
        model = self.cfg.bucket_dir / "hand_landmarker.task"
        model.write_bytes(b"fake-weights-1234")
        token = pipeline._weight_token(model)
        assert token is not None
        self.assertTrue(token.startswith("hand_landmarker.task:sha256:"))
        import hashlib
        self.assertTrue(token.endswith(hashlib.sha256(b"fake-weights-1234").hexdigest()))

    def test_receipt_versions_carry_pipeline_version(self) -> None:
        segs = pipeline.process_clip(self.fake_clip("v.mp4"), "w", "s", self.cfg)
        _, receipts = pipeline.run_batch(segs, self.cfg)
        block = receipts[0]["config"]
        self.assertEqual(block["pipeline_version"], self.cfg.pipeline_version)
        self.assertIn("python", block["versions"])

    def test_compare_versions_match_and_drift(self) -> None:
        stored = {"versions": {"a": "1", "b": "2"}}
        match, diff = pipeline.compare_versions(stored, {"a": "1", "b": "2"})
        self.assertTrue(match)
        self.assertEqual(diff, [])
        match, diff = pipeline.compare_versions(stored, {"a": "1", "b": "CHANGED"})
        self.assertFalse(match)
        self.assertEqual(diff, ["b"])
        match, diff = pipeline.compare_versions({}, {"a": "1"})
        self.assertFalse(match)


class TestDeterminism(IntegrationCase):
    def test_same_clip_10_runs_identical_hash(self) -> None:
        path = self.fake_clip("det10.mp4")
        hashes = set()
        for _ in range(10):
            segs = pipeline.process_clip(path, "w", "s", self.cfg)
            _, receipts = pipeline.run_batch(segs, self.cfg)
            hashes.add(receipts[0]["receipt_sha256"])
        self.assertEqual(len(hashes), 1)

    def test_solo_vs_batch_of_five_identical(self) -> None:
        paths = [self.fake_clip(f"batch5_{i}.mp4") for i in range(5)]
        segs = [s for p in paths for s in pipeline.process_clip(p, "w", "s", self.cfg)]
        target = segs[2]
        _, solo = pipeline.run_batch([target], self.cfg)
        _, batched = pipeline.run_batch(segs, self.cfg)
        by_id = {r["verdict"]["segment_id"]: r for r in batched}
        self.assertEqual(solo[0]["receipt_sha256"],
                         by_id[target["segment_id"]]["receipt_sha256"])


class TestFailureSafety(IntegrationCase):
    def _client(self, **kw):
        store_mod.init_db(self.cfg.db_path)
        return TestClient(create_app(self.cfg, batch_max_age_s=30.0,
                                     watch_bucket=False, **kw))

    def test_failed_batch_keeps_segments_for_retry(self) -> None:
        from jadal import service as service_mod
        paths = [self.fake_clip(f"keep_{i}.mp4") for i in range(5)]
        real = pipeline.run_batch
        calls = {"n": 0}

        def flaky(segments, config):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient boom")
            return real(segments, config)

        with self._client() as c:
            with mock.patch.object(service_mod.pipeline, "run_batch",
                                   side_effect=flaky):
                for i, p in enumerate(paths):
                    r = c.post("/ingest/event", json={
                        "path": p, "worker_id": "w", "site_id": "keep"})
                    if i < 4:
                        self.assertEqual(r.status_code, 200, r.text)
                        self.assertIsNone(r.json()["batch"])
                    else:
                        # 5th ingest triggers the failing flush -> 500, and the
                        # segments must stay queued instead of being dropped.
                        self.assertEqual(r.status_code, 500, r.text)
            # segments must still be pending, and a retry must close the batch.
            self.assertEqual(c.get("/health").json()["pending_segments"], 5)
            # one more ingest retriggers the flush (mock now succeeds).
            extra = self.fake_clip("keep_extra.mp4")
            r = c.post("/ingest/event", json={
                "path": extra, "worker_id": "w", "site_id": "keep"})
            self.assertEqual(r.status_code, 200, r.text)
            batch = r.json()["batch"]
            self.assertIsNotNone(batch)
            self.assertEqual(batch["n"], 5)
            batches = c.get("/batches").json()["batches"]
            self.assertEqual(len(batches), 1)

    def test_failed_save_requeues_segments_for_retry(self) -> None:
        """A SQLite write failure must not drop the batch (review Important #1)."""
        paths = [self.fake_clip(f"save_{i}.mp4") for i in range(5)]
        real_save = store_mod.Store.save_batch
        calls = {"n": 0}

        def flaky_save(self, batch, receipts, segments):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("disk full")
            return real_save(self, batch, receipts, segments)

        store_mod.init_db(self.cfg.db_path)
        app = create_app(self.cfg, batch_max_age_s=30.0, watch_bucket=False)
        with mock.patch.object(store_mod.Store, "save_batch", flaky_save):
            with TestClient(app) as c:
                for i, p in enumerate(paths):
                    r = c.post("/ingest/event", json={
                        "path": p, "worker_id": "w", "site_id": "save"})
                    if i < 4:
                        self.assertEqual(r.status_code, 200, r.text)
                        self.assertIsNone(r.json()["batch"])
                    else:
                        # 5th ingest runs the batch but the SAVE fails -> 500,
                        # and the segments must stay queued instead of lost.
                        self.assertEqual(r.status_code, 500, r.text)
                self.assertEqual(c.get("/health").json()["pending_segments"], 5)
                self.assertEqual(c.get("/batches").json()["batches"], [])
                # one more ingest retriggers the flush; the save now succeeds
                # and the ORIGINAL five segments close the batch, in order.
                extra = self.fake_clip("save_extra.mp4")
                r = c.post("/ingest/event", json={
                    "path": extra, "worker_id": "w", "site_id": "save"})
                self.assertEqual(r.status_code, 200, r.text)
                batch = r.json()["batch"]
                self.assertIsNotNone(batch)
                self.assertEqual(batch["n"], 5)
                # the ORIGINAL five clips close the batch, in ingest order
                # (stub segment_id = seg-<file sha[:12]>-0000).
                expected_ids = [f"seg-{stubs.file_sha256(p)[:12]}-0000"
                                for p in paths]
                self.assertEqual(batch["segment_ids"], expected_ids)
                batches = c.get("/batches").json()["batches"]
                self.assertEqual(len(batches), 1)
                receipts = store_mod.Store(self.cfg.db_path
                                           ).get_receipts_for_batch(batch["batch_id"])
                self.assertEqual(len(receipts), 5)
                rv = c.post(f"/receipts/{receipts[0]['receipt_id']}/reverify")
                self.assertEqual(rv.status_code, 200)
                self.assertTrue(rv.json()["match"])

    def test_watcher_backoff_and_ingest_errors_exposed(self) -> None:
        from jadal import service as service_mod
        bad = self.cfg.bucket_dir / "poison.mp4"
        bad.write_bytes(b"poison-bytes")
        real = pipeline.process_clip

        def broken(path, worker_id, site_id, config):
            if str(path).endswith("poison.mp4"):
                raise RuntimeError("cannot decode")
            return real(path, worker_id, site_id, config)

        app = create_app(self.cfg, batch_max_age_s=60.0, watch_bucket=True,
                         watch_poll_s=0.05)
        with mock.patch.object(service_mod.pipeline, "process_clip",
                               side_effect=broken):
            with TestClient(app) as c:
                deadline = time.time() + 5.0
                errors: dict = {}
                while time.time() < deadline:
                    errors = c.get("/ingest/errors").json()["errors"]
                    if errors:
                        break
                    time.sleep(0.05)
                self.assertTrue(errors, "poison clip must surface in /ingest/errors")
                self.assertTrue(any("poison.mp4" in k for k in errors))
                svc = app.state.service
                gate = svc._watch_backoff.get("poison.mp4")
                self.assertIsNotNone(gate)
                self.assertGreaterEqual(gate["attempts"], 1)
                first_attempts = gate["attempts"]
                # backoff gate: an immediate tick must NOT retry yet
                time.sleep(0.2)
                gate2 = svc._watch_backoff.get("poison.mp4")
                self.assertIsNotNone(gate2)
                self.assertLessEqual(gate2["attempts"], first_attempts + 1)
                self.assertEqual(c.get("/batches").json()["batches"], [])


class TestStatsFromRealData(IntegrationCase):
    def _client(self, **kw):
        store_mod.init_db(self.cfg.db_path)
        return TestClient(create_app(self.cfg, batch_max_age_s=30.0,
                                     watch_bucket=False, **kw))

    def test_risk_coverage_empty_without_measured_data(self) -> None:
        with self._client() as c:
            r = c.get("/stats/risk_coverage")
            self.assertEqual(r.status_code, 200)
            # no store rows and no eval/results/ in this worktree: honest empty
            if not (ROOT / "eval" / "results" / "calibration.json").is_file():
                self.assertEqual(r.json(), [])

    def test_risk_coverage_from_eval_file(self) -> None:
        import tempfile
        fixture = {"risk_coverage": [
            {"threshold": 0.9, "coverage": 0.8, "error": 0.05},
            {"threshold": 0.5, "coverage": 1.0, "error": 0.2},
            {"threshold": "bad", "coverage": 1.0, "error": 0.2},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps(fixture))
            points = pipeline.risk_coverage_from_eval(path)
            self.assertEqual(len(points), 2)  # malformed row dropped
            self.assertEqual(points[0]["threshold"], 0.9)
            self.assertEqual(pipeline.risk_coverage_from_eval(Path(tmp) / "missing"), [])

    def test_sop_gaps_counts_real_stored_reasons(self) -> None:
        with self._client() as c:
            bad = self.fake_clip("gap_blur.mp4", signals={"blur_var": 2.0})
            oks = [self.fake_clip(f"gap_ok_{i}.mp4") for i in range(4)]
            for p in [bad, *oks]:
                r = c.post("/ingest/event", json={
                    "path": p, "worker_id": "w", "site_id": "gaps"})
                self.assertEqual(r.status_code, 200, r.text)
            gaps = c.get("/stats/sop_gaps").json()
            by_code = {g["reason_code"]: g for g in gaps}
            self.assertIn("BLUR", by_code)
            self.assertEqual(by_code["BLUR"]["count"], 1)
            for g in gaps:
                self.assertIn("count", g)
                self.assertIn("share", g)

    def test_reverify_calls_ensure_determinism(self) -> None:
        """reverify must seed exactly like run_batch (review Important #2)."""
        segs = pipeline.process_clip(self.fake_clip("detcall.mp4"), "w", "s",
                                     self.cfg)
        _, receipts = pipeline.run_batch(segs, self.cfg)
        with mock.patch.object(pipeline, "ensure_determinism") as det:
            result = pipeline.reverify_receipt(receipts[0], dict(segs[0]),
                                               self.cfg)
        self.assertTrue(result["match"])
        self.assertGreaterEqual(det.call_count, 1)

    def test_reverify_reports_version_comparison(self) -> None:
        with self._client() as c:
            paths = [self.fake_clip(f"rv_{i}.mp4") for i in range(5)]
            last = None
            for p in paths:
                last = c.post("/ingest/event", json={
                    "path": p, "worker_id": "w", "site_id": "rv"}).json()
            rid = last["batch"]["batch_id"]
            receipts = store_mod.Store(self.cfg.db_path).get_receipts_for_batch(rid)
            body = c.post(f"/receipts/{receipts[0]['receipt_id']}/reverify").json()
            self.assertTrue(body["match"])
            self.assertTrue(body["version_match"])
            self.assertEqual(body["version_diff"], [])


if __name__ == "__main__":
    unittest.main()
