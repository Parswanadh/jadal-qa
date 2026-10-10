"""Acceptance tests for the service module (GitHub issue #3).

Contract under test: docs/SPEC.md. Covers the FastAPI service (jadal/service.py),
the batch pipeline (jadal/pipeline.py), SQLite persistence (jadal/store.py) and the
deterministic stubs (jadal/stubs.py) that stand in for the signals/decision/certify/
receipts modules until those branches merge.

The service is exercised through fastapi's TestClient: ingest 5 fake segments ->
one batch with 5 verdicts + 5 receipts stored; re-verify returns match=true; the
batch rule applies k; GET /batches and /receipts work.

Fake clips are text files whose content is `JADALFAKE1\\n<json>`; the stub signals/
decisions read that JSON so tests can force exact values (deterministic, no video
decoding needed).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jadal.config import Config  # noqa: E402
from jadal import pipeline  # noqa: E402
from jadal.pipeline import thresholds_snapshot  # noqa: E402
from jadal.stubs import stub_receipt_hash  # noqa: E402
from jadal import store as store_mod  # noqa: E402
from jadal.service import create_app  # noqa: E402

FAKE_MAGIC = "JADALFAKE1\n"

PASSING = {
    "signals": {
        "decodable": True, "blur_var": 120.0, "brightness_mean": 90.0,
        "under_exposed_frac": 0.0, "over_exposed_frac": 0.02, "motion_mean": 2.0,
        "hands_visible_frac": 0.8, "hand_count_mode": 2, "faces_found": 0,
        "faces_blurred": 0, "dup_of": None,
    },
    "decisions": {
        "hands_ok": {"p_yes": 0.97},
        "active_work": {"p_yes": 0.93},
        "step": {"choice": "wash", "probs": {"wash": 0.93}, "answer_conf": 0.93},
    },
}


def dict_merge(base: dict, override: dict | None) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = dict_merge(out[k], v)
        else:
            out[k] = v
    return out


class ServiceAcceptanceCase(unittest.TestCase):
    """Shared fixtures: temp runtime dir, fake clip writer, service client."""

    def setUp(self) -> None:
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
        self.clips = tmp / "clips"
        self.clips.mkdir(parents=True)
        store_mod.init_db(self.cfg.db_path)
        self.app = create_app(self.cfg, batch_max_age_s=0.05)
        self._store = store_mod.Store(self.cfg.db_path)

    def tearDown(self) -> None:
        os.environ.pop("JADAL_WATCH_BUCKET", None)
        self._tmp.cleanup()

    # -- helpers -----------------------------------------------------------
    def fake_clip(self, name: str, payload: dict | None = None) -> str:
        body = dict_merge(PASSING, payload)
        body["clip"] = name  # keeps file bytes (and so ids) distinct per clip
        path = self.clips / name
        path.write_text(FAKE_MAGIC + json.dumps(body), encoding="utf-8")
        return str(path)

    def client(self) -> TestClient:
        return TestClient(self.app)

    def ingest(self, client: TestClient, *paths: str, site: str = "site-1", worker: str = "w-1"):
        last = None
        for p in paths:
            r = client.post("/ingest/event", json={
                "path": p, "worker_id": worker, "site_id": site,
            })
            self.assertEqual(r.status_code, 200, r.text)
            last = r.json()
        return last

    def ingest_n(self, client: TestClient, n: int, site: str = "site-1", payloads=None):
        paths = []
        for i in range(n):
            payload = (payloads or {}).get(i)
            paths.append(self.fake_clip(f"clip_{site}_{i}.mp4", payload))
        return self.ingest(client, *paths, site=site)


class TestStubsAndStore(unittest.TestCase):
    """Unit-level contract checks for the stubs and the SQLite store."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(db_path=Path(self._tmp.name) / "jadal.db")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_stub_signals_are_deterministic(self) -> None:
        from jadal.stubs import stub_compute_signals
        tmp = Path(self._tmp.name)
        p = tmp / "a.mp4"
        p.write_bytes(b"same-bytes")
        s1 = stub_compute_signals(str(p), self.cfg)
        s2 = stub_compute_signals(str(p), self.cfg)
        self.assertEqual(s1, s2)
        for key in ["decodable", "blur_var", "brightness_mean", "hands_visible_frac",
                    "phash", "dup_of", "faces_found"]:
            self.assertIn(key, s1)
        # floats rounded to 4 decimals before they enter a receipt
        self.assertEqual(s1["blur_var"], round(s1["blur_var"], 4))

    def test_stub_fake_payload_controls_signals(self) -> None:
        from jadal.stubs import stub_compute_signals
        p = Path(self._tmp.name) / "f.mp4"
        p.write_text(FAKE_MAGIC + json.dumps({"signals": {"blur_var": 3.0}}), encoding="utf-8")
        s = stub_compute_signals(str(p), self.cfg)
        self.assertEqual(s["blur_var"], 3.0)

    def test_stub_receipts_hash_ignores_id_and_time(self) -> None:
        receipt_a = {
            "receipt_id": "aaa", "created_utc": "2020-01-01T00:00:00Z",
            "segment_sha256": "ff", "receipt_sha256": "whatever",
            "signals": {"decodable": True}, "verdict": {"verdict": "accept"},
        }
        receipt_b = dict(receipt_a)
        receipt_b["receipt_id"] = "bbb"
        receipt_b["created_utc"] = "2030-01-01T00:00:00Z"
        self.assertEqual(stub_receipt_hash(receipt_a), stub_receipt_hash(receipt_b))

    def test_stub_batch_rule(self) -> None:
        from jadal.stubs import stub_batch_rule
        acc = {"verdict": "accept"}

        def v(x): return {"verdict": x}
        self.assertEqual(stub_batch_rule([acc, acc, v("reject")], 1), "accept")
        self.assertEqual(stub_batch_rule([acc, v("reject"), v("reject")], 1), "reject")
        self.assertEqual(stub_batch_rule([v("reject")] * 3, 3), "accept")
        self.assertEqual(stub_batch_rule([v("reject")] * 4, 3), "reject")

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


class TestPipeline(unittest.TestCase):
    """Pipeline behaviour: verdict reasons, batch rule wiring, determinism."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.cfg = Config(
            batch_size=5, batch_reject_k=1,
            bucket_dir=tmp / "bucket", segments_dir=tmp / "segments",
            blurred_dir=tmp / "blurred", db_path=tmp / "jadal.db",
        )
        self.clips = tmp / "clips"
        self.clips.mkdir(parents=True)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def fake_clip(self, name: str, payload: dict | None = None) -> str:
        body = dict_merge(PASSING, payload)
        body["clip"] = name  # keeps file bytes (and so ids) distinct per clip
        p = self.clips / name
        p.write_text(FAKE_MAGIC + json.dumps(body), encoding="utf-8")
        return str(p)

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
                segs = self.flat_segments(self.fake_clip(f"bad_{code}.mp4", payload))
                batch, _ = pipeline.run_batch(segs, self.cfg)
                self.assertEqual(batch["verdicts"][0]["verdict"], "reject", code)
                self.assertIn(code, batch["verdicts"][0]["reasons"], code)

    def test_low_confidence_is_review(self) -> None:
        segs = self.flat_segments(self.fake_clip("low.mp4", {
            "decisions": {"step": {"choice": "wash", "answer_conf": 0.2}},
        }))
        batch, _ = pipeline.run_batch(segs, self.cfg)
        self.assertEqual(batch["verdicts"][0]["verdict"], "review")
        self.assertIn("LOW_CONFIDENCE", batch["verdicts"][0]["reasons"])

    def test_duplicate_by_phash_within_batch(self) -> None:
        p1 = self.fake_clip("dup_a.mp4", {"signals": {"phash": "samephash1"}})
        p2 = self.fake_clip("dup_b.mp4", {"signals": {"phash": "samephash1"}})
        segs = self.flat_segments(p1, p2)
        batch, _ = pipeline.run_batch(segs, self.cfg)
        by_id = {v["segment_id"]: v for v in batch["verdicts"]}
        first_id, second_id = [s["segment_id"] for s in segs]
        self.assertEqual(by_id[first_id]["reasons"], [])
        self.assertEqual(by_id[second_id]["verdict"], "reject")
        self.assertIn("DUPLICATE", by_id[second_id]["reasons"])

    def test_batch_rule_two_rejects_beats_k1(self) -> None:
        paths = [
            self.fake_clip("k_a.mp4", {"signals": {"blur_var": 2.0}}),
            self.fake_clip("k_b.mp4", {"signals": {"motion_mean": 40.0}}),
        ]
        paths += [self.fake_clip(f"k_{i}.mp4") for i in range(3)]
        segs = self.flat_segments(*paths)
        batch, _ = pipeline.run_batch(segs, self.cfg)
        rejected = [v for v in batch["verdicts"] if v["verdict"] == "reject"]
        self.assertEqual(len(rejected), 2)
        self.assertEqual(batch["k"], 1)
        self.assertEqual(batch["batch_decision"], "reject")

    def test_one_reject_pass_with_k1(self) -> None:
        paths = [self.fake_clip("one_bad.mp4", {"signals": {"blur_var": 1.0}})]
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
        solo_batch, solo_receipts = pipeline.run_batch([target], self.cfg)
        batch, receipts = pipeline.run_batch(segs, self.cfg)
        self.assertEqual(len(solo_receipts), 1)
        solo_hash = solo_receipts[0]["receipt_sha256"]
        batch_hash = [r["receipt_sha256"] for r in receipts
                      if r["verdict"]["segment_id"] == target["segment_id"]][0]
        self.assertEqual(solo_hash, batch_hash)
        # running the same batch twice gives the same hashes
        _, receipts2 = pipeline.run_batch(segs, self.cfg)
        self.assertEqual([r["receipt_sha256"] for r in receipts],
                         [r["receipt_sha256"] for r in receipts2])

    def test_receipt_hash_matches_spec_definition(self) -> None:
        segs = self.flat_segments(self.fake_clip("spec.mp4"))
        _, receipts = pipeline.run_batch(segs, self.cfg)
        receipt = dict(receipts[0])
        expected = receipt["receipt_sha256"]
        # drop the three excluded fields, canonical-json the rest, hash it
        payload = {k: v for k, v in receipt.items()
                   if k not in ("receipt_sha256", "receipt_id", "created_utc")}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        import hashlib
        self.assertEqual(hashlib.sha256(blob).hexdigest(), expected)
        self.assertEqual(receipt["receipt_id"], expected[:16])

    def test_thresholds_snapshot_covers_l0_and_l2(self) -> None:
        snap = thresholds_snapshot(self.cfg)
        for key in ["blur_var_min", "brightness_min", "over_exposed_frac_max",
                    "motion_max", "hands_visible_frac_min", "answer_threshold",
                    "target_error", "delta", "min_accepted"]:
            self.assertIn(key, snap)


class TestServiceAcceptance(ServiceAcceptanceCase):
    """Issue #3 acceptance criteria, end to end through the HTTP API."""

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
            # 5 receipts stored
            r = c.get(f"/batches/{batch['batch_id']}")
            self.assertEqual(r.status_code, 200)
            stored = r.json()
            self.assertEqual(stored["n"], 5)
            receipts = self._store.get_receipts_for_batch(batch["batch_id"])
            self.assertEqual(len(receipts), 5)

    def test_batch_rule_applies_k(self) -> None:
        with self.client() as c:
            last = self.ingest_n(c, 5, payloads={
                1: {"signals": {"blur_var": 2.0}},
                3: {"signals": {"motion_mean": 44.0}},
            })
            batch = last["batch"]
            self.assertEqual(batch["k"], 1)
            rejected = [v for v in batch["verdicts"] if v["verdict"] == "reject"]
            self.assertEqual(len(rejected), 2)
            self.assertEqual(batch["batch_decision"], "reject")

    def test_reverify_returns_match_true(self) -> None:
        with self.client() as c:
            last = self.ingest_n(c, 5)
            batch = last["batch"]
            receipts = self._store.get_receipts_for_batch(batch["batch_id"])
            self.assertEqual(len(receipts), 5)
            for rec in receipts:
                r = c.post(f"/receipts/{rec['receipt_id']}/reverify")
                self.assertEqual(r.status_code, 200, r.text)
                body = r.json()
                self.assertTrue(body["match"],
                                f"{rec['receipt_id']}: {body['expected']} != {body['actual']}")
                self.assertEqual(body["diff"], [])
                self.assertEqual(body["expected"], body["actual"])
                self.assertEqual(body["actual"], rec["receipt_sha256"])

    def test_reverify_mismatch_when_clip_changed(self) -> None:
        with self.client() as c:
            last = self.ingest_n(c, 5)
            receipts = self._store.get_receipts_for_batch(last["batch"]["batch_id"])
            rid = receipts[0]["receipt_id"]
            # the source clip changed on disk after the receipt was written
            rec = self._store.get_receipt(rid)
            seg = self._store.get_segment(rec["verdict"]["segment_id"])
            Path(seg["path"]).write_text("tampered bytes", encoding="utf-8")
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
        with self.client() as c:
            self.ingest_n(c, 2, site="slow")
            r = c.get("/batches")
            self.assertEqual(r.json()["batches"], [], "batch must not fire before N")
            deadline = time.time() + 2.0
            batches = []
            while time.time() < deadline:
                batches = c.get("/batches").json()["batches"]
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

    def test_ingest_rejects_missing_clip(self) -> None:
        with self.client() as c:
            r = c.post("/ingest/event", json={
                "path": str(self.clips / "nope.mp4"), "worker_id": "w", "site_id": "s"})
            self.assertEqual(r.status_code, 400)

    def test_unknown_ids_are_404(self) -> None:
        with self.client() as c:
            self.assertEqual(c.get("/batches/missing").status_code, 404)
            self.assertEqual(c.get("/receipts/missing").status_code, 404)
            self.assertEqual(c.post("/receipts/missing/reverify").status_code, 404)

    def test_pico_status_answers(self) -> None:
        with self.client() as c:
            r = c.get("/pico/status")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json(), {"connected": False, "reason": "jadal.pico_link not installed"})

    def test_voice_endpoint_serves_and_404s(self) -> None:
        wav = self.cfg.voicebank_dir / "en-IN" / "BLUR.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"RIFFfake")
        with self.client() as c:
            r = c.get("/voice/BLUR/en-IN")
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.content.startswith(b"RIFF"))
            self.assertEqual(c.get("/voice/BLUR/fr-FR").status_code, 404)

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


class TestDockSim(unittest.TestCase):
    def test_dock_sim_module_imports_cleanly(self) -> None:
        path = ROOT / "sim" / "dock_sim.py"
        spec = importlib.util.spec_from_file_location("dock_sim_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertTrue(callable(mod.main))


if __name__ == "__main__":
    unittest.main()
