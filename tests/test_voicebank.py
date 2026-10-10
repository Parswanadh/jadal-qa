"""Unit tests for voicebank/gen_voicebank.py and voicebank/messages.json.

Run from the repo root:

    PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests -v

No network, no API key, no sarvamai package needed: synthesis is exercised
with an injected fake TTS client, and all output files land in a scratch
directory under the repo's gitignored data_runtime/ (never /tmp).
"""
from __future__ import annotations

import base64
import builtins
import contextlib
import io
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jadal.config import LANGS, REASON_CODES  # noqa: E402  (read-only contract, no network)
from voicebank import gen_voicebank as gv  # noqa: E402

MESSAGES_PATH = REPO_ROOT / "voicebank" / "messages.json"

FULL_CATALOG_CHARS = 3570
FULL_CATALOG_COST_INR = 10.71


def _no_network(*args, **kwargs):
    raise AssertionError(
        "network access forbidden in voicebank tests "
        "(socket connection attempted)"
    )


def setUpModule() -> None:
    """Fail loudly if any voicebank test attempts a real socket connection."""
    for target in ("socket.socket.connect", "socket.create_connection"):
        patcher = mock.patch(target, side_effect=_no_network)
        patcher.start()
        setUpModule._patchers.append(patcher)  # type: ignore[attr-defined]


setUpModule._patchers = []  # type: ignore[attr-defined]


def tearDownModule() -> None:
    for patcher in setUpModule._patchers:  # type: ignore[attr-defined]
        patcher.stop()
    setUpModule._patchers.clear()  # type: ignore[attr-defined]


@contextlib.contextmanager
def simulate_missing_sdk():
    """Make `import sarvamai` fail, then restore the interpreter state.

    Works whether or not the sarvamai SDK is installed: sys.modules is
    seeded with None (the legacy "module is known to be absent" signal)
    and builtins.__import__ is patched to raise ImportError for sarvamai
    and any sarvamai.* submodule. Only sarvamai keys are touched: on exit
    the keys added during the context are deleted and the saved entries
    restored, so every other module object keeps its identity (no
    sys.modules.clear()).
    """
    saved = {
        key: value
        for key, value in sys.modules.items()
        if key == "sarvamai" or key.startswith("sarvamai.")
    }
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sarvamai" or name.startswith("sarvamai."):
            raise ImportError(
                "No module named 'sarvamai' (simulated missing SDK)"
            )
        return real_import(name, globals, locals, fromlist, level)

    sys.modules["sarvamai"] = None  # type: ignore[assignment]
    for key in [k for k in sys.modules if k.startswith("sarvamai.")]:
        del sys.modules[key]
    with mock.patch("builtins.__import__", side_effect=fake_import):
        sys.modules["sarvamai"] = None  # type: ignore[assignment]
        try:
            yield
        finally:
            for key in [
                k for k in sys.modules
                if k == "sarvamai" or k.startswith("sarvamai.")
            ]:
                del sys.modules[key]
            sys.modules.update(saved)


class ScratchTestCase(unittest.TestCase):
    """Scratch under the repo's gitignored data_runtime/; no /tmp fallback."""

    def setUp(self) -> None:
        base = REPO_ROOT / "data_runtime" / "test_scratch"
        base.mkdir(parents=True, exist_ok=True)
        self._scratch = tempfile.TemporaryDirectory(dir=str(base))
        self.addCleanup(self._scratch.cleanup)
        self.tmp = Path(self._scratch.name)


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.audios = [base64.b64encode(payload).decode("ascii")]


class FakeTTS:
    """Stands in for client.text_to_speech; records every convert call."""

    def __init__(self, payload: bytes = b"RIFF" + b"x" * 100) -> None:
        self.calls: list[dict] = []
        self.payload = payload

    def convert(self, **kwargs):
        self.calls.append(dict(kwargs))
        return FakeResponse(self.payload)


class FakeClient:
    def __init__(self, tts: FakeTTS) -> None:
        self.text_to_speech = tts


class CatalogCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = gv.load_catalog(MESSAGES_PATH)

    def test_language_and_code_lists_match_config(self) -> None:
        self.assertEqual(self.catalog["languages"], LANGS)
        self.assertEqual(self.catalog["reason_codes"], REASON_CODES)

    def test_every_reason_code_x_language_has_text(self) -> None:
        for code in REASON_CODES:
            entries = self.catalog["messages"].get(code)
            self.assertIsNotNone(entries, f"{code} missing from catalog")
            for lang in LANGS:
                text = (entries.get(lang) or {}).get("text", "")
                self.assertTrue(text.strip(), f"{code}/{lang} has no text")

    def test_batch_headers_cover_accept_and_reject(self) -> None:
        for decision in ("accept", "reject"):
            entries = self.catalog["batch_headers"].get(decision)
            self.assertIsNotNone(entries, f"batch header {decision} missing")
            for lang in LANGS:
                text = (entries.get(lang) or {}).get("text", "")
                self.assertTrue(text.strip(), f"batch header {decision}/{lang} has no text")

    def test_non_english_entries_are_machine_drafts(self) -> None:
        blocks = [self.catalog["messages"][c] for c in REASON_CODES]
        blocks += [self.catalog["batch_headers"][d] for d in ("accept", "reject")]
        for entries in blocks:
            for lang, entry in entries.items():
                self.assertEqual(entry["status"], "machine_draft", f"{lang} status")
                if lang == "en-IN":
                    self.assertFalse(entry["needs_native_review"], f"{lang} review flag")
                else:
                    self.assertTrue(entry["needs_native_review"], f"{lang} review flag")

    def test_every_text_fits_one_request(self) -> None:
        limit = self.catalog["tts"]["chars_per_request_max"]
        self.assertEqual(limit, 2500)
        for code in REASON_CODES:
            for lang, entry in self.catalog["messages"][code].items():
                self.assertLessEqual(len(entry["text"]), limit, f"{code}/{lang}")
        for decision in ("accept", "reject"):
            for lang, entry in self.catalog["batch_headers"][decision].items():
                self.assertLessEqual(len(entry["text"]), limit, f"{decision}/{lang}")


class CostMathTests(unittest.TestCase):
    def test_rate_matches_catalog(self) -> None:
        self.assertEqual(gv.PRICE_INR_PER_10K_CHARS, 30.0)
        self.assertEqual(self.catalog_rate(), 30.0)

    @staticmethod
    def catalog_rate() -> float:
        return gv.load_catalog(MESSAGES_PATH)["tts"]["price_inr_per_10k_chars"]

    def test_estimate_cost_inr(self) -> None:
        self.assertEqual(gv.estimate_cost_inr(0), 0.0)
        self.assertAlmostEqual(gv.estimate_cost_inr(10_000), 30.0)
        self.assertAlmostEqual(gv.estimate_cost_inr(FULL_CATALOG_CHARS), FULL_CATALOG_COST_INR, places=2)
        with self.assertRaises(ValueError):
            gv.estimate_cost_inr(-1)

    def test_full_catalog_chars_and_cost(self) -> None:
        catalog = gv.load_catalog(MESSAGES_PATH)
        jobs = gv.build_jobs(catalog, out_dir=Path("nowhere"))
        chars = sum(job.chars for job in jobs)
        self.assertEqual(len(jobs), 60)
        self.assertEqual(chars, FULL_CATALOG_CHARS)
        self.assertEqual(round(gv.estimate_cost_inr(chars), 2), FULL_CATALOG_COST_INR)

    def test_format_plan_reports_chars_and_cost(self) -> None:
        catalog = gv.load_catalog(MESSAGES_PATH)
        jobs = gv.build_jobs(catalog, out_dir=Path("nowhere"), langs=["hi-IN"], codes=["BLUR"])
        plan = gv.format_plan(jobs, [])
        self.assertIn("60 chars", plan)
        self.assertIn("Rs 0.18", plan)
        self.assertIn("BLUR.wav", plan)
        self.assertIn("skipped (already on disk): 0 files", plan)


class ValidationTests(unittest.TestCase):
    def test_real_catalog_validates(self) -> None:
        gv.validate_catalog(gv.load_catalog(MESSAGES_PATH))

    def test_missing_text_raises(self) -> None:
        catalog = gv.load_catalog(MESSAGES_PATH)
        catalog["messages"]["BLUR"]["hi-IN"]["text"] = "   "
        with self.assertRaises(gv.CatalogError):
            gv.validate_catalog(catalog)

    def test_overlong_text_raises(self) -> None:
        catalog = gv.load_catalog(MESSAGES_PATH)
        catalog["messages"]["BLUR"]["hi-IN"]["text"] = "x" * 2501
        with self.assertRaises(gv.CatalogError):
            gv.validate_catalog(catalog)

    def test_unknown_code_raises(self) -> None:
        catalog = gv.load_catalog(MESSAGES_PATH)
        with self.assertRaises(gv.CatalogError):
            gv.build_jobs(catalog, out_dir=Path("nowhere"), codes=["NOPE"])

    def test_resolve_speaker(self) -> None:
        self.assertEqual(gv.resolve_speaker("simran"), "simran")
        self.assertEqual(gv.resolve_speaker("SIMRAN"), "SIMRAN")
        with self.assertRaises(gv.CatalogError):
            gv.resolve_speaker("simran_en_customer")

    def test_default_model_and_speaker_are_explicit(self) -> None:
        self.assertEqual(gv.DEFAULT_MODEL, "bulbul:v3")
        self.assertIn(gv.DEFAULT_SPEAKER.lower(), gv.V3_SPEAKERS_LOWER)


class SynthesisTests(unittest.TestCase):
    def test_synthesise_sends_language_code_only(self) -> None:
        tts = FakeTTS()
        data = gv.synthesise("hello test", "hi-IN", client=FakeClient(tts))
        self.assertEqual(data, tts.payload)
        self.assertEqual(len(tts.calls), 1)
        call = tts.calls[0]
        self.assertEqual(call["text"], "hello test")
        self.assertEqual(call["language_code"], "hi-IN")
        self.assertEqual(call["model"], "bulbul:v3")
        self.assertEqual(call["speaker"], "simran")
        self.assertNotIn("target_language_code", call)

    def test_synthesise_type_error_is_not_swallowed(self) -> None:
        class BadTTS:
            def convert(self, **kwargs):
                raise TypeError("unexpected keyword argument 'language_code'")

        class BadClient:
            text_to_speech = BadTTS()

        with self.assertRaises(TypeError):
            gv.synthesise("hello", "hi-IN", client=BadClient())

    def test_synthesise_enforces_max_chars(self) -> None:
        with self.assertRaises(ValueError):
            gv.synthesise("x" * 2501, "hi-IN", client=FakeClient(FakeTTS()))

    def test_audio_bytes_accepts_object_and_dict(self) -> None:
        payload = b"RIFF...."
        encoded = base64.b64encode(payload).decode("ascii")
        self.assertEqual(gv.audio_bytes(FakeResponse(payload)), payload)
        self.assertEqual(gv.audio_bytes({"audios": [encoded]}), payload)
        with self.assertRaises(RuntimeError):
            gv.audio_bytes({"audios": []})

    def test_no_sarvamai_imported(self) -> None:
        # Laziness must hold in a fresh interpreter: importing
        # voicebank.gen_voicebank must never pull in the SDK, regardless of
        # what this test process has already imported.
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import voicebank.gen_voicebank; "
                "assert 'sarvamai' not in sys.modules, "
                "sorted(m for m in sys.modules if 'sarvam' in m.lower())",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr[-2000:])


class SimulateMissingSdkTests(unittest.TestCase):
    def test_leaves_rest_of_sys_modules_untouched(self) -> None:
        import os as os_mod

        before = dict(sys.modules)
        probe = object()
        with simulate_missing_sdk():
            with self.assertRaises(ImportError):
                import sarvamai  # noqa: F401
            # A non-sarvamai entry added inside the context must survive:
            # the finally block restores by key diff, never sys.modules.clear().
            sys.modules["test_voicebank_probe_module"] = probe
        self.assertIs(sys.modules.get("test_voicebank_probe_module"), probe)
        del sys.modules["test_voicebank_probe_module"]
        # Every pre-existing non-sarvamai entry keeps its object identity.
        for key, value in before.items():
            if key == "sarvamai" or key.startswith("sarvamai."):
                continue
            self.assertIs(sys.modules.get(key), value, f"replaced: {key}")
        self.assertIs(sys.modules["os"], os_mod)
        # Saved sarvamai entries are restored as-is.
        for key, value in before.items():
            if key == "sarvamai" or key.startswith("sarvamai."):
                self.assertIs(sys.modules.get(key), value, f"not restored: {key}")


class AtomicWriteTests(ScratchTestCase):
    def test_write_is_atomic_and_leaves_no_temp_files(self) -> None:
        target = self.tmp / "hi-IN" / "BLUR.wav"
        gv.write_wav_atomic(target, b"RIFF....v1....")
        self.assertEqual(target.read_bytes(), b"RIFF....v1....")
        self.assertEqual(list(self.tmp.rglob("*.tmp")), [])
        gv.write_wav_atomic(target, b"RIFF....v2....")
        self.assertEqual(target.read_bytes(), b"RIFF....v2....")
        self.assertEqual(list(self.tmp.rglob("*.tmp")), [])

    def test_failed_write_cleans_up_temp_file(self) -> None:
        target = self.tmp / "BLUR.wav"
        with mock.patch("os.replace", side_effect=OSError("disk gone")):
            with self.assertRaises(OSError):
                gv.write_wav_atomic(target, b"data")
        self.assertFalse(target.exists())
        self.assertEqual(list(self.tmp.rglob("*.tmp")), [])


class SplitJobsTests(ScratchTestCase):
    def _job(self, name: str, text: str = "hello") -> gv.Job:
        return gv.Job(lang="hi-IN", code=name, kind="reason", text=text,
                      out_path=self.tmp / f"{name}.wav")

    def test_missing_file_is_todo(self) -> None:
        todo, skipped = gv.split_jobs([self._job("BLUR")])
        self.assertEqual(len(todo), 1)
        self.assertEqual(skipped, [])

    def test_complete_file_is_skipped(self) -> None:
        job = self._job("BLUR")
        job.out_path.write_bytes(b"x" * 100)
        todo, skipped = gv.split_jobs([job])
        self.assertEqual(todo, [])
        self.assertEqual([j.code for j in skipped], ["BLUR"])

    def test_empty_file_is_regenerated(self) -> None:
        job = self._job("BLUR")
        job.out_path.write_bytes(b"")
        todo, skipped = gv.split_jobs([job])
        self.assertEqual([j.code for j in todo], ["BLUR"])
        self.assertEqual(skipped, [])

    def test_header_only_file_is_regenerated(self) -> None:
        job = self._job("BLUR")
        job.out_path.write_bytes(b"x" * gv.MIN_WAV_BYTES)
        todo, _ = gv.split_jobs([job])
        self.assertEqual([j.code for j in todo], ["BLUR"])

    def test_force_regenerates_complete_files(self) -> None:
        job = self._job("BLUR")
        job.out_path.write_bytes(b"x" * 100)
        todo, skipped = gv.split_jobs([job], force=True)
        self.assertEqual([j.code for j in todo], ["BLUR"])
        self.assertEqual(skipped, [])


class RunEndToEndTests(ScratchTestCase):
    ARGS = ["--lang", "hi-IN", "--code", "BLUR", "--code", "BATCH_ACCEPT"]

    def _run(self, client, extra=()):
        out = io.StringIO()
        with redirect_stdout(out):
            code = gv.run(
                ["--messages", str(MESSAGES_PATH), "--out-dir", str(self.tmp),
                 *self.ARGS, *extra],
                client=client,
            )
        return code, out.getvalue()

    def test_run_writes_wavs_then_skips(self) -> None:
        tts = FakeTTS()
        code, _ = self._run(FakeClient(tts))
        self.assertEqual(code, 0)
        self.assertEqual(len(tts.calls), 2)
        self.assertEqual(
            {c["language_code"] for c in tts.calls}, {"hi-IN"}
        )
        blur = self.tmp / "hi-IN" / "BLUR.wav"
        batch = self.tmp / "hi-IN" / "BATCH_ACCEPT.wav"
        self.assertEqual(blur.read_bytes(), tts.payload)
        self.assertEqual(batch.read_bytes(), tts.payload)
        self.assertEqual(list(self.tmp.rglob("*.tmp")), [])

        code, text = self._run(FakeClient(tts))
        self.assertEqual(code, 0)
        self.assertIn("nothing to do", text)

    def test_run_regenerates_truncated_file(self) -> None:
        tts = FakeTTS()
        self._run(FakeClient(tts))
        blur = self.tmp / "hi-IN" / "BLUR.wav"
        blur.write_bytes(b"")
        tts2 = FakeTTS(payload=b"RIFF" + b"y" * 100)
        code, _ = self._run(FakeClient(tts2))
        self.assertEqual(code, 0)
        self.assertEqual(len(tts2.calls), 1)
        self.assertEqual(tts2.calls[0]["text"], gv.load_catalog(MESSAGES_PATH)["messages"]["BLUR"]["hi-IN"]["text"])
        self.assertEqual(blur.read_bytes(), b"RIFF" + b"y" * 100)

    def test_run_without_client_and_without_sdk_returns_2(self) -> None:
        err = io.StringIO()
        with mock.patch.dict(os.environ, {"SARVAM_API_KEY": "dummy-not-a-real-key"}):
            with simulate_missing_sdk():
                with redirect_stdout(io.StringIO()), redirect_stderr(err):
                    code = gv.main(
                        ["--messages", str(MESSAGES_PATH), "--out-dir", str(self.tmp),
                         *self.ARGS]
                    )
        self.assertEqual(code, 2)
        self.assertIn("sarvamai", err.getvalue())

    def test_dry_run_writes_nothing(self) -> None:
        code, text = self._run(FakeClient(FakeTTS()), extra=("--dry-run",))
        self.assertEqual(code, 0)
        self.assertIn("dry run: no API call made", text)
        self.assertEqual(list(self.tmp.rglob("*.wav")), [])


if __name__ == "__main__":
    unittest.main()
