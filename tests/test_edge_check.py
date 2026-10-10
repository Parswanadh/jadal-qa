"""Hermetic unit tests for the edge tier scripts (B9).

Everything here runs WITHOUT jadal.signals, WITHOUT the IndEgo clips, and
WITHOUT spawning subprocesses: the forbidden-import detector, the
requirements parser check, and the result-file schema validators are all
pure functions tested with fake inputs. Fast (<20 s added) and hang-free.
"""

import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


edge_check = _load("edge_check_mod", "scripts/edge_check.py")
edge_bench = _load("edge_bench_mod", "eval/edge_bench.py")


class ForbiddenDetectorTest(unittest.TestCase):
    def test_clean_edge_modules_pass(self):
        mods = ["jadal.signals", "jadal.funnel", "numpy", "cv2", "scipy",
                "PIL.Image", "imagehash", "os", "sys"]
        self.assertEqual(edge_check.check_forbidden(mods), [])

    def test_dict_input_uses_keys(self):
        self.assertEqual(edge_check.check_forbidden({"torch": None}), ["torch"])

    def test_each_forbidden_exact(self):
        for mod in ["torch", "torchvision", "sentence_transformers",
                    "transformers", "sklearn", "pandas", "matplotlib",
                    "fastapi", "uvicorn", "onnxruntime", "tensorflow"]:
            with self.subTest(mod=mod):
                self.assertEqual(edge_check.check_forbidden([mod, "cv2"]), [mod])

    def test_submodule_counts(self):
        self.assertEqual(edge_check.check_forbidden(["torch.cuda"]), ["torch"])
        self.assertEqual(edge_check.check_forbidden(["uvicorn.main"]),
                         ["uvicorn"])

    def test_cuda_prefixes(self):
        for mod in ["nvidia.cublas.lib", "cupy", "triton.language", "cuda"]:
            with self.subTest(mod=mod):
                self.assertEqual(edge_check.check_forbidden([mod]), [mod.split(".")[0]])

    def test_similar_names_not_flagged(self):
        self.assertEqual(edge_check.check_forbidden(
            ["mytorchio", "fastapix", "pandascore"]), [])

    def test_sorted_unique(self):
        self.assertEqual(edge_check.check_forbidden(["uvicorn", "torch", "torch"]),
                         ["torch", "uvicorn"])


class RequirementsTest(unittest.TestCase):
    REQ = ROOT / "requirements-edge.txt"
    NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+(\[[A-Za-z0-9_,.\-]+\])?$")

    def _entries(self):
        lines = self.REQ.read_text().splitlines()
        return [ln.strip() for ln in lines
                if ln.strip() and not ln.strip().startswith("#")]

    def test_file_exists_and_minimal(self):
        self.assertTrue(self.REQ.is_file())
        self.assertLessEqual(len(self._entries()), 10)

    def test_bare_names_no_pins(self):
        for entry in self._entries():
            with self.subTest(entry=entry):
                self.assertRegex(entry, self.NAME_RE)
                self.assertNotRegex(entry, r"[<>=!~]")

    def test_no_torch_family(self):
        text = self.REQ.read_text().lower()
        for pkg in ["torch", "sentence-transformers", "transformers",
                    "scikit-learn", "sklearn", "pandas", "matplotlib",
                    "fastapi", "uvicorn"]:
            self.assertNotIn(pkg, [ln.strip().lower() for ln in text.splitlines()
                                   if ln.strip() and not ln.strip().startswith("#")])

    def test_mediapipe_optional_comment(self):
        text = self.REQ.read_text()
        self.assertIn("mediapipe", text)
        self.assertIn("torch is NOT needed", text)


class CheckSchemaTest(unittest.TestCase):
    def _valid(self):
        return {
            "device_label": "x86-laptop-proxy",
            "disclaimer": edge_check.DISCLAIMER,
            "platform": "Linux-x", "python": "3.12", "git_sha": "abc123",
            "forbidden_found": [], "heavy_present": {"cv2": True},
            "import_time_s": 0.5,
            "edge_modules_imported": ["jadal.funnel"],
            "edge_modules_missing": [],
        }

    def test_valid_passes(self):
        self.assertEqual(edge_check.validate_result(self._valid()), [])

    def test_missing_key_and_disclaimer(self):
        bad = self._valid()
        del bad["git_sha"]
        bad["disclaimer"] = "runs on a Pi!"
        errors = edge_check.validate_result(bad)
        self.assertTrue(any("git_sha" in e for e in errors))
        self.assertTrue(any("disclaimer" in e for e in errors))

    def test_disclaimer_carries_proxy_label(self):
        self.assertIn("NOT measured on a Raspberry Pi", edge_check.DISCLAIMER)

    def test_non_dict_rejected(self):
        self.assertTrue(edge_check.validate_result([]))

    def test_runtime_forbidden_field(self):
        good = self._valid()
        good["runtime_forbidden_found"] = []
        self.assertEqual(edge_check.validate_result(good), [])
        bad = self._valid()
        bad["runtime_forbidden_found"] = "torch"
        self.assertTrue(any("runtime_forbidden" in e
                            for e in edge_check.validate_result(bad)))

    def test_runtime_probe_modules_detect_lazy_torch(self):
        # Important 1: a lazy `import torch` at call time must fail the gate
        # even when the import-time module list is clean.
        self.assertEqual(edge_check.check_forbidden(["jadal.signals", "cv2"]), [])
        self.assertEqual(
            edge_check.check_forbidden(
                ["jadal.signals", "cv2", "torch", "torch._C"]),
            ["torch"])


    def test_process_child_emits_modules(self):
        # Important 1 wiring: the call-time probe child must export
        # sys.modules so main() can forbid-gate lazy imports.
        self.assertIn('"modules"', edge_check._PROCESS_CHILD)


class OverwriteRuleTest(unittest.TestCase):
    def test_measured_result_protected(self):
        old = {"process_probe": {"peak_rss_MB": 1.0}}
        self.assertFalse(edge_check.should_overwrite(old, False, False))

    def test_force_allows(self):
        old = {"process_probe": {"peak_rss_MB": 1.0}}
        self.assertTrue(edge_check.should_overwrite(old, False, True))

    def test_new_measurements_overwrite_stub(self):
        self.assertTrue(edge_check.should_overwrite({"process_probe": None},
                                                    True, False))
        self.assertTrue(edge_check.should_overwrite(None, False, False))
        self.assertTrue(edge_check.should_overwrite("garbage", False, False))


class BenchSchemaTest(unittest.TestCase):
    def _valid(self):
        return {
            "device_label": "x86-laptop-proxy",
            "disclaimer": edge_bench.DISCLAIMER,
            "placeholder_thresholds_note": "x",
            "platform": "Linux-x", "machine": "x86_64", "cpu_model": "Y",
            "python": "3.12", "opencv": "5.0", "numpy": "2.5",
            "git_sha": "abc123", "seg_seconds": 10.0, "n_segments": 30,
            "repeats": 3, "core_configs": ["0", "0-3"], "results": {},
            "determinism": {"identical_across_repeats": True,
                            "identical_across_cores": True},
        }

    def test_valid_passes(self):
        self.assertEqual(edge_bench.validate_bench_result(self._valid()), [])

    def test_disclaimer_required(self):
        bad = self._valid()
        bad["disclaimer"] = "fast on a Pi"
        self.assertTrue(any("disclaimer" in e
                            for e in edge_bench.validate_bench_result(bad)))

    def test_determinism_keys_required(self):
        bad = self._valid()
        del bad["determinism"]["identical_across_cores"]
        self.assertTrue(edge_bench.validate_bench_result(bad))

    def test_hash_stable_and_order_insensitive(self):
        a = {"route": {"verdict": "review", "reasons": ["BLUR"]}, "n": 1}
        b = {"n": 1, "route": {"reasons": ["BLUR"], "verdict": "review"}}
        self.assertEqual(edge_bench.hash_route_output(a),
                         edge_bench.hash_route_output(b))
        c = dict(b)
        c["route"] = dict(b["route"], verdict="accept")
        self.assertNotEqual(edge_bench.hash_route_output(a),
                            edge_bench.hash_route_output(c))


class BenchHelpersTest(unittest.TestCase):
    def test_parse_core_list(self):
        self.assertEqual(edge_bench.parse_core_list("0"), [0])
        self.assertEqual(edge_bench.parse_core_list("0-3"), [0, 1, 2, 3])
        self.assertEqual(edge_bench.parse_core_list("0,2"), [0, 2])
        self.assertEqual(edge_bench.parse_core_list("0-1,3"), [0, 1, 3])

    def test_parse_core_list_garbage(self):
        for bad in ["", "x", "3-1", "0-"]:
            with self.subTest(bad=bad):
                self.assertRaises(ValueError,
                                  edge_bench.parse_core_list, bad)

    def test_seg_id_keys_path_and_length(self):
        # Important 2: same stem from different dirs, or different
        # --seg-seconds, must never share a cache file.
        a = edge_bench.make_seg_id("/d1/clip_480.mp4", 60.0, 10.0)
        b = edge_bench.make_seg_id("/d2/clip_480.mp4", 60.0, 10.0)
        c = edge_bench.make_seg_id("/d1/clip_480.mp4", 60.0, 5.0)
        d = edge_bench.make_seg_id("/d1/clip_480.mp4", 60.0, 10.0)
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(a, d)


if __name__ == "__main__":
    unittest.main()
