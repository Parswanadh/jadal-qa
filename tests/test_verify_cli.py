"""Tests for scripts/verify_receipt.py (offline hash check, MAC, re-run)."""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent


def _load_cli():
    path = ROOT / "scripts" / "verify_receipt.py"
    spec = importlib.util.spec_from_file_location("verify_receipt_cli", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CLI = _load_cli()


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    """Run the CLI in-process; returns (exit_code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = CLI.main(argv)
        except SystemExit as exc:
            code = int(exc.code) if isinstance(exc.code, int) else 2
    return code, out.getvalue(), err.getvalue()


def _set_path(obj: dict, dotted: str, value) -> None:
    node = obj
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


class _EnvCase(unittest.TestCase):
    def setUp(self) -> None:
        self._old_stubs = os.environ.get("JADAL_USE_STUBS")
        self._old_key = os.environ.get("JADAL_RECEIPT_KEY")
        os.environ.pop("JADAL_RECEIPT_KEY", None)
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        if self._old_stubs is None:
            os.environ.pop("JADAL_USE_STUBS", None)
        else:
            os.environ["JADAL_USE_STUBS"] = self._old_stubs
        if self._old_key is None:
            os.environ.pop("JADAL_RECEIPT_KEY", None)
        else:
            os.environ["JADAL_RECEIPT_KEY"] = self._old_key
        self._tmp.cleanup()

    def write_receipt(self, receipt: dict, name: str = "r.json") -> str:
        path = self.tmp / name
        path.write_text(json.dumps(receipt), encoding="utf-8")
        return str(path)


def _direct_fixture() -> dict:
    """Fixture built through jadal.receipts.make_receipt (offline tests)."""
    from jadal import receipts as receipts_mod
    from jadal.config import Config
    segment = {"sha256": "aa" * 32}
    signals = {
        "decodable": True, "blur_var": 120.0, "brightness_mean": 90.0,
        "under_exposed_frac": 0.0, "over_exposed_frac": 0.02,
        "motion_mean": 2.0, "hands_visible_frac": 0.8, "hand_count_mode": 2,
        "faces_found": 0, "faces_blurred": 0, "phash": "0123456789abcdef",
        "dup_of": None,
    }
    decisions = {
        "hands_ok": {"p_yes": 0.97},
        "active_work": {"p_yes": 0.93},
        "step": {"choice": "wash", "probs": {"wash": 0.93},
                 "answer_conf": 0.93},
        "model_versions": {"stub": "0.1.0"},
    }
    verdict = {"segment_id": "seg-001", "verdict": "accept", "reasons": [],
               "answer_conf": 0.93, "threshold": 0.9}
    return receipts_mod.make_receipt(
        segment, signals, decisions, verdict, Config(),
        created_utc="2026-10-10T00:00:00+00:00")


class OfflineCheckTest(_EnvCase):
    def test_valid_receipt_passes(self) -> None:
        path = self.write_receipt(_direct_fixture())
        code, out, _ = run_cli([path])
        self.assertEqual(code, 0, out)
        self.assertIn("hash: OK", out)
        self.assertIn("RESULT: PASS", out)

    def test_valid_receipt_json_mode(self) -> None:
        path = self.write_receipt(_direct_fixture())
        code, out, _ = run_cli([path, "--json"])
        self.assertEqual(code, 0, out)
        report = json.loads(out)
        self.assertTrue(report["ok"])
        self.assertTrue(report["hash"]["ok"])
        self.assertEqual(report["hash"]["diff"], [])

    def test_tampered_fields_fail_offline(self) -> None:
        # Offline the original bytes are unavailable, so the envelope path
        # is named (and the user is told to --rerun for exact field paths).
        for field, bad in [
            ("signals.blur_var", 1.0),
            ("signals.hands_visible_frac", 0.0),
            ("decisions.step.answer_conf", 0.11),
            ("verdict.verdict", "reject"),
            ("config.pipeline_version", "9.9.9"),
            ("segment_sha256", "bb" * 32),
        ]:
            with self.subTest(field=field):
                receipt = _direct_fixture()
                _set_path(receipt, field, bad)
                code, out, _ = run_cli([self.write_receipt(receipt)])
                self.assertEqual(code, 1, out)
                self.assertIn("receipt_sha256", out)
                self.assertIn("RESULT: FAIL", out)

    def test_tampered_hash_only_names_the_hash(self) -> None:
        receipt = _direct_fixture()
        receipt["receipt_sha256"] = "f" * 64
        code, out, _ = run_cli([self.write_receipt(receipt)])
        self.assertEqual(code, 1, out)
        self.assertIn("receipt_sha256", out)

    def test_wrong_id_fails_and_is_named(self) -> None:
        receipt = _direct_fixture()
        receipt["receipt_id"] = "0" * 16
        code, out, _ = run_cli([self.write_receipt(receipt)])
        self.assertEqual(code, 1, out)
        self.assertIn("receipt_id", out)

    def test_unreadable_receipt_is_exit_2(self) -> None:
        code, _, err = run_cli([str(self.tmp / "missing.json")])
        self.assertEqual(code, 2)
        bad = self.tmp / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        code, _, err = run_cli([str(bad)])
        self.assertEqual(code, 2, err)
        lst = self.tmp / "list.json"
        lst.write_text("[1, 2]", encoding="utf-8")
        code, _, _ = run_cli([str(lst)])
        self.assertEqual(code, 2)

    def test_rerun_without_clip_is_exit_2(self) -> None:
        path = self.write_receipt(_direct_fixture())
        code, _, _ = run_cli([path, "--rerun"])
        self.assertEqual(code, 2)

    def test_help_states_what_a_hash_check_proves(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit):
                CLI.main(["--help"])
        text = out.getvalue()
        self.assertIn("not edited", text)
        self.assertIn("does NOT prove the verdict was right", text)


class MacCliTest(_EnvCase):
    def test_mac_good_passes(self) -> None:
        from jadal import receipt_mac as mac_mod
        os.environ["JADAL_RECEIPT_KEY"] = "unit-test-key"
        receipt = _direct_fixture()
        tag = mac_mod.mac_for(receipt["receipt_sha256"])
        code, out, _ = run_cli([self.write_receipt(receipt), "--mac", tag])
        self.assertEqual(code, 0, out)
        self.assertIn("mac: OK", out)

    def test_mac_file_sidecar_passes(self) -> None:
        from jadal import receipt_mac as mac_mod
        os.environ["JADAL_RECEIPT_KEY"] = "unit-test-key"
        receipt = _direct_fixture()
        sidecar = self.tmp / "r.mac"
        sidecar.write_text(mac_mod.mac_for(receipt["receipt_sha256"]) + "\n")
        code, out, _ = run_cli(
            [self.write_receipt(receipt), "--mac-file", str(sidecar)])
        self.assertEqual(code, 0, out)
        self.assertIn("mac: OK", out)

    def test_mac_bad_fails(self) -> None:
        os.environ["JADAL_RECEIPT_KEY"] = "unit-test-key"
        receipt = _direct_fixture()
        code, out, _ = run_cli(
            [self.write_receipt(receipt), "--mac", "00" * 32])
        self.assertEqual(code, 1, out)
        self.assertIn("mac: MISMATCH", out)

    def test_mac_requested_without_key_is_exit_2(self) -> None:
        receipt = _direct_fixture()
        code, _, err = run_cli(
            [self.write_receipt(receipt), "--mac", "ab" * 32])
        self.assertEqual(code, 2)
        self.assertIn("JADAL_RECEIPT_KEY", err)

    def test_missing_mac_file_is_exit_2(self) -> None:
        os.environ["JADAL_RECEIPT_KEY"] = "unit-test-key"
        receipt = _direct_fixture()
        code, _, _ = run_cli(
            [self.write_receipt(receipt), "--mac-file", str(self.tmp / "no.mac")])
        self.assertEqual(code, 2)


class RerunTest(_EnvCase):
    def setUp(self) -> None:
        super().setUp()
        os.environ["JADAL_USE_STUBS"] = "1"
        from jadal import pipeline as pipeline_mod
        from jadal.config import Config
        clip = self.tmp / "clip.mp4"
        clip.write_bytes(b"rerun-clip-bytes" * 256)
        self.clip = str(clip)
        cfg = Config(
            bucket_dir=self.tmp / "bucket",
            segments_dir=self.tmp / "segments",
            blurred_dir=self.tmp / "blurred",
            db_path=self.tmp / "jadal.db",
            web_dir=self.tmp / "web",
            voicebank_dir=self.tmp / "voicebank",
        )
        cfg.bucket_dir.mkdir(parents=True, exist_ok=True)
        segs = pipeline_mod.process_clip(self.clip, "w", "s", cfg)
        _, receipts = pipeline_mod.run_batch(segs, cfg)
        self.fixture = receipts[0]

    def test_rerun_matching(self) -> None:
        path = self.write_receipt(copy.deepcopy(self.fixture))
        code, out, _ = run_cli([path, "--clip", self.clip, "--rerun"])
        self.assertEqual(code, 0, out)
        self.assertIn("rerun: MATCH", out)
        self.assertIn("RESULT: PASS", out)

    def test_rerun_names_each_tampered_field(self) -> None:
        cases = [
            ("signals.blur_var", 1.0, "signals.blur_var"),
            ("signals.hands_visible_frac", 0.01,
             "signals.hands_visible_frac"),
            ("decisions.step.answer_conf", 0.11,
             "decisions.step.answer_conf"),
            ("decisions.hands_ok.p_yes", 0.11,
             "decisions.hands_ok.p_yes"),
            ("segment_sha256", "bb" * 32, "segment_sha256"),
        ]
        orig_verdict = self.fixture["verdict"]["verdict"]
        flipped = "accept" if orig_verdict != "accept" else "reject"
        cases.append(("verdict.verdict", flipped, "verdict.verdict"))
        for field, bad, named in cases:
            with self.subTest(field=field):
                receipt = copy.deepcopy(self.fixture)
                _set_path(receipt, field, bad)
                code, out, _ = run_cli(
                    [self.write_receipt(receipt), "--clip", self.clip,
                     "--rerun"])
                self.assertEqual(code, 1, out)
                self.assertIn(named, out)
                self.assertIn("MISMATCH", out)

    def test_rerun_still_fails_when_only_replay_verbatim_config_edited(self) -> None:
        # The pipeline replays the stored config block verbatim, so an
        # edited config.* field cannot be told apart from a stale hash by
        # re-run alone — but the offline hash check still fails (exit 1).
        receipt = copy.deepcopy(self.fixture)
        receipt["config"]["pipeline_version"] = "9.9.9"
        code, out, _ = run_cli(
            [self.write_receipt(receipt), "--clip", self.clip, "--rerun"])
        self.assertEqual(code, 1, out)
        self.assertIn("receipt_sha256", out)

    def test_clip_sha_mismatch_without_rerun(self) -> None:
        other = self.tmp / "other.mp4"
        other.write_bytes(b"different-bytes" * 256)
        path = self.write_receipt(copy.deepcopy(self.fixture))
        code, out, _ = run_cli([path, "--clip", str(other)])
        self.assertEqual(code, 1, out)
        self.assertIn("clip: MISMATCH", out)

    def test_clip_match_without_rerun(self) -> None:
        path = self.write_receipt(copy.deepcopy(self.fixture))
        code, out, _ = run_cli([path, "--clip", self.clip])
        self.assertEqual(code, 0, out)
        self.assertIn("clip: OK", out)

    def test_cannot_rerun_is_exit_3_not_faked(self) -> None:
        from jadal import pipeline as pipeline_mod
        path = self.write_receipt(copy.deepcopy(self.fixture))
        with mock.patch.object(
                pipeline_mod, "reverify_receipt",
                side_effect=RuntimeError(
                    "no L0 signals implementation: jadal.signals."
                    "compute_signals is not importable and JADAL_USE_STUBS "
                    "is not '1'")):
            code, _, err = run_cli([path, "--clip", self.clip, "--rerun"])
        self.assertEqual(code, 3, err)
        self.assertIn("cannot re-run", err)


if __name__ == "__main__":
    unittest.main()
