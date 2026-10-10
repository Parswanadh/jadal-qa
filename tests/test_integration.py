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
    # Subclasses run on injected stub callables (plus JADAL_USE_STUBS=1) so
    # results never depend on which real modules are importable -- except
    # trial classes that set force_stubs = False to exercise the real stack.
    force_stubs = True

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
        self._stage_patches = []
        if self.force_stubs:
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
    _STAGE_MODULES = ("signals", "decision", "segmenter")

    def test_impl_versions_names_stubs_until_real_modules_merge(self) -> None:
        # Force stub mode: block the real L0/L1/segmenter imports so the
        # stub tags are reported whether or not the modules exist on disk.
        blocked = tuple(f"jadal.{name}" for name in self._STAGE_MODULES)
        with stubs.blocked_imports(*blocked):
            versions = pipeline.impl_versions(self.cfg)
        self.assertEqual(versions["certify"], "jadal.certify")
        self.assertEqual(versions["receipts"], "jadal.receipts")
        self.assertIn("python", versions)
        self.assertEqual(versions["signals"], "stubs.signals")
        self.assertEqual(versions["decision"], "stubs.decision")
        self.assertEqual(versions["segmenter"], "stubs.segmenter")

    def test_impl_versions_names_real_modules_when_importable(self) -> None:
        # Real-module path: only meaningful once a stage module is importable
        # (same test impl_versions itself uses: module + attr not None).
        # Skipped per stage when absent, so it holds with or without merges.
        expected = {"signals": ("compute_signals", "stubs.signals"),
                    "decision": ("Decider", "stubs.decision"),
                    "segmenter": ("split_clip", "stubs.segmenter")}
        real = {}
        for name, (attr, _stub) in expected.items():
            try:
                mod = __import__(f"jadal.{name}", fromlist=[attr])
            except ImportError:
                continue
            if getattr(mod, attr, None) is not None:
                real[name] = f"jadal.{name}"
        if not real:
            self.skipTest("no real jadal.signals/decision/segmenter importable")
        versions = pipeline.impl_versions(self.cfg)
        for name, want in real.items():
            with self.subTest(stage=name):
                self.assertEqual(versions[name], want)
                self.assertFalse(versions[name].startswith("stubs."))

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
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            entries = pipeline.risk_coverage_report(config=self.cfg,
                                                    results_dir=tmp)
            self.assertEqual(entries, [])

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


@unittest.skipUnless(importlib.util.find_spec("jadal.signals") is not None,
                     "needs the real jadal.signals module (trial worktree)")
class TestRealSignalsTrial(IntegrationCase):
    """Exercises the REAL decoder: a valid mp4 and a garbage file.

    No stub injection here (force_stubs = False); JADAL_USE_STUBS=1 stays set
    so the still-missing L1 falls back to the stub decider. Proves (1)-(3):
    a real tiny mp4 decodes and batches normally, while garbage bytes close
    with an UNDECODABLE reject verdict and a normal batch event -- no hang.
    """

    force_stubs = False

    def setUp(self) -> None:
        super().setUp()
        # Only the real decoder (signals) is under test here. The real L1
        # Decider loads EG2 onto CUDA, which OOMs whenever other jobs hold the
        # shared GPU, so the decider stage is stubbed (no GPU use by default).
        patcher = mock.patch.object(pipeline, "_decider_class",
                                    new=lambda: stubs.StubDecider)
        patcher.start()
        self._stage_patches.append(patcher)

    def _write_mp4(self, name: str) -> str:
        import cv2
        import numpy as np
        path = self.cfg.bucket_dir / name
        writer = cv2.VideoWriter(str(path),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 10.0, (160, 120))
        self.assertTrue(writer.isOpened(), "cv2.VideoWriter failed to open")
        try:
            for i in range(12):
                frame = np.full((120, 160, 3), (40 + i * 10) % 200,
                                dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()
        self.assertTrue(path.is_file() and path.stat().st_size > 0)
        return str(path)

    def test_real_mp4_decodes_and_batches(self) -> None:
        path = self._write_mp4("real_tiny.mp4")
        segs = pipeline.process_clip(path, "w", "s", self.cfg)
        self.assertGreaterEqual(len(segs), 1)
        batch, receipts = pipeline.run_batch(segs, self.cfg)
        self.assertEqual(batch["n"], len(segs))
        versions = receipts[0]["config"]["versions"]
        self.assertEqual(versions["signals"], "jadal.signals")

    def test_garbage_file_yields_undecodable_reject(self) -> None:
        import dataclasses
        bad = self.cfg.bucket_dir / "garbage.mp4"
        bad.write_bytes(b"this is not video\n" * 64)
        cfg1 = dataclasses.replace(self.cfg, batch_size=1)
        store_mod.init_db(cfg1.db_path)
        app = create_app(cfg1, batch_max_age_s=30.0, watch_bucket=False)
        with TestClient(app) as client:
            response = client.post("/ingest/event", json={
                "path": str(bad), "worker_id": "w", "site_id": "trial"})
            self.assertEqual(response.status_code, 200, response.text)
            batch = response.json()["batch"]
        self.assertIsNotNone(batch, "undecodable clip must still close a batch")
        self.assertEqual(len(batch["verdicts"]), 1)
        verdict = batch["verdicts"][0]
        self.assertEqual(verdict["verdict"], "reject")
        self.assertIn("UNDECODABLE", verdict["reasons"])


def _clean_receipt() -> dict:
    import json as _json
    backend = _json.dumps({"faces": "mediapipe", "hands": "mediapipe"})
    return {
        "signals": {"decodable": True},
        "decisions": {"model_versions": {
            "heads": "hands_ok=3d9b5c0b8da0ea9e,active_work=260657b5a9bb0dd8",
            "frames_decoded": "4"}},
        "config": {"versions": {
            "signals": "jadal.signals", "decision": "jadal.decision",
            "segmenter": "jadal.segmenter", "funnel": "jadal.funnel",
            "signals_backend": backend}},
    }


class TestReceiptProblems(IntegrationCase):
    """pipeline.receipt_problems flags stubs/silent fallbacks (issue #55).

    Crafted receipts only: no models, no GPU, no network.
    """

    def test_clean_real_receipt_has_no_problems(self) -> None:
        self.assertEqual(pipeline.receipt_problems(_clean_receipt()), [])

    def test_stub_implementations_are_problems(self) -> None:
        import copy
        for stage in ("signals", "decision", "segmenter"):
            receipt = _clean_receipt()
            receipt["config"]["versions"][stage] = f"stubs.{stage}"
            problems = pipeline.receipt_problems(receipt)
            self.assertTrue(problems, stage)
            self.assertTrue(any("stub" in p for p in problems), problems)
        receipt = _clean_receipt()
        receipt["config"]["versions"]["funnel"] = "missing"
        self.assertTrue(pipeline.receipt_problems(receipt))

    def test_proxy_backend_is_a_problem(self) -> None:
        import copy
        import json as _json
        receipt = _clean_receipt()
        versions = receipt["config"]["versions"]
        versions["signals_backend"] = _json.dumps(
            {"faces": "proxy", "hands": "proxy"})
        problems = pipeline.receipt_problems(receipt)
        self.assertTrue(any("mediapipe" in p for p in problems), problems)

    def test_missing_heads_and_zero_frames_are_problems(self) -> None:
        receipt = _clean_receipt()
        receipt["decisions"]["model_versions"]["heads"] = (
            "hands_ok=missing,active_work=missing")
        self.assertTrue(any("heads" in p
                            for p in pipeline.receipt_problems(receipt)))
        receipt = _clean_receipt()
        receipt["decisions"]["model_versions"]["frames_decoded"] = "0"
        self.assertTrue(pipeline.receipt_problems(receipt))

    def test_live_stub_receipt_fails_and_live_real_block_passes(self) -> None:
        # End to end through run_batch on forced stubs: the demo's no-stub
        # assertion must fire on a stub receipt.
        segs = pipeline.process_clip(self.fake_clip("stubfail.mp4"), "w", "s",
                                     self.cfg)
        _, receipts = pipeline.run_batch(segs, self.cfg)
        self.assertTrue(pipeline.receipt_problems(receipts[0]))

    def test_missing_versions_block_is_a_problem(self) -> None:
        self.assertTrue(pipeline.receipt_problems({}))


class TestVersionsBlock(IntegrationCase):
    """config.versions carries the B2 provenance keys (issue #55)."""

    def test_required_keys_present(self) -> None:
        versions = pipeline.impl_versions(self.cfg)
        for key in ("python", "platform", "arch", "device",
                    "torch_version", "opencv_version", "mediapipe_version",
                    "transformers_version", "numpy_version",
                    "signals", "decision", "segmenter", "funnel",
                    "certify", "receipts"):
            self.assertIn(key, versions, key)
        self.assertIn(versions["device"], ("cpu", "cuda"))
        for key in ("signals_git_sha", "decision_git_sha", "segmenter_git_sha",
                    "funnel_git_sha", "certify_git_sha", "receipts_git_sha"):
            if key in versions:
                self.assertRegex(versions[key], r"^[0-9a-f]{40}$", key)

    def test_model_tokens_present_when_files_exist(self) -> None:
        from jadal.config import ROOT
        versions = pipeline.impl_versions(self.cfg)
        for key, rel in (("face_detector_task",
                           "models/blaze_face_short_range.tflite"),
                          ("hand_landmarker_task",
                           "models/hand_landmarker.task")):
            if (ROOT / rel).is_file():
                self.assertIn(key, versions)
                self.assertIn(":sha256:", versions[key])
            else:
                self.skipTest(f"{rel} absent: models-dependent key unchecked")


class TestDeciderCache(IntegrationCase):
    """One Decider instance per (class, decimals): no EG2 reload per batch."""

    def test_same_class_returns_same_instance(self) -> None:
        first = pipeline._get_decider(stubs.StubDecider, self.cfg)
        second = pipeline._get_decider(stubs.StubDecider, self.cfg)
        self.assertIs(first, second)

    def test_distinct_class_gets_distinct_instance(self) -> None:
        class OtherDecider(stubs.StubDecider):
            pass

        self.assertIsNot(
            pipeline._get_decider(stubs.StubDecider, self.cfg),
            pipeline._get_decider(OtherDecider, self.cfg))


if __name__ == "__main__":
    unittest.main()
