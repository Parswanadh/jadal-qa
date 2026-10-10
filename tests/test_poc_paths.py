"""Default data/clip path resolution for the PoC scripts (scripts/poc_paths.py).

Uses tmp dirs for every layout. The subprocess tests only exercise the
explicit-failure paths, so they do not depend on the real checkout layout.
"""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import poc_paths as P  # noqa: E402


def mk(*parts):
    os.makedirs(os.path.join(*parts), exist_ok=True)
    return os.path.join(*parts)


class ResolutionTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        base = self._td.name
        # <base>/proj/wt is the repo root; siblings live under <base>/proj.
        self.root = mk(base, "proj", "wt")
        self.proj = os.path.join(base, "proj")

    def tearDown(self):
        self._td.cleanup()

    def sibling_data(self):
        return mk(self.proj, "poc_data")

    def inner_data(self):
        return mk(self.root, "poc_data")

    def clips(self, *, outer=False):
        if outer:  # <root>/../../awing/...
            return mk(self.proj, "..", "awing", "data", "ego10k_scale", "clips")
        return mk(self.proj, "awing", "data", "ego10k_scale", "clips")

    # --- data dir -------------------------------------------------------
    def test_env_var_wins_over_candidates(self):
        self.sibling_data()
        env_dir = mk(self.proj, "elsewhere")
        got = P.resolve_poc_data(self.root, env={P.DATA_ENV: env_dir})
        self.assertEqual(got, os.path.normpath(env_dir))

    def test_sibling_preferred_over_inner(self):
        sib = self.sibling_data()
        self.inner_data()
        self.assertEqual(P.resolve_poc_data(self.root, env={}), os.path.normpath(sib))

    def test_inner_used_when_sibling_missing(self):
        inner = self.inner_data()
        self.assertEqual(P.resolve_poc_data(self.root, env={}), os.path.normpath(inner))

    def test_candidate_must_exist_as_directory(self):
        # A plain file named poc_data is not a data dir.
        open(os.path.join(self.proj, "poc_data"), "w").close()
        with self.assertRaises(P.PathError):
            P.resolve_poc_data(self.root, env={})

    def test_missing_everything_fails_loudly_naming_env_var(self):
        with self.assertRaises(P.PathError) as cm:
            P.resolve_poc_data(self.root, env={})
        self.assertIn(P.DATA_ENV, str(cm.exception))

    def test_missing_env_dir_fails_and_does_not_fall_back(self):
        # Even though a candidate exists, an explicit but wrong env var is an error.
        self.sibling_data()
        with self.assertRaises(P.PathError) as cm:
            P.resolve_poc_data(self.root, env={P.DATA_ENV: os.path.join(self.proj, "nope")})
        self.assertIn(P.DATA_ENV, str(cm.exception))

    def test_empty_env_value_is_ignored(self):
        sib = self.sibling_data()
        self.assertEqual(P.resolve_poc_data(self.root, env={P.DATA_ENV: ""}),
                         os.path.normpath(sib))

    # --- clip root ------------------------------------------------------
    def test_clip_env_var_wins(self):
        self.clips(outer=True)
        env_dir = mk(self.proj, "clips_here")
        self.assertEqual(P.resolve_clip_root(self.root, env={P.CLIP_ENV: env_dir}),
                         os.path.normpath(env_dir))

    def test_clip_outer_awing_preferred(self):
        self.clips(outer=True)
        self.clips(outer=False)
        got = P.resolve_clip_root(self.root, env={})
        self.assertEqual(got, os.path.normpath(
            os.path.join(self.proj, "..", "awing", "data", "ego10k_scale", "clips")))

    def test_clip_second_candidate_used_when_first_missing(self):
        inner = self.clips(outer=False)
        self.assertEqual(P.resolve_clip_root(self.root, env={}), os.path.normpath(inner))

    def test_clip_missing_fails_loudly_naming_env_var(self):
        with self.assertRaises(P.PathError) as cm:
            P.resolve_clip_root(self.root, env={})
        self.assertIn(P.CLIP_ENV, str(cm.exception))

    def test_clip_missing_env_dir_fails(self):
        self.clips(outer=True)
        with self.assertRaises(P.PathError) as cm:
            P.resolve_clip_root(self.root, env={P.CLIP_ENV: os.path.join(self.proj, "x")})
        self.assertIn(P.CLIP_ENV, str(cm.exception))


class NoSilentFixtureTest(unittest.TestCase):
    """The server must never serve tests/fixtures/poc unless --demo-fixture is passed."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "poc_server_nofix", os.path.join(ROOT, "scripts", "poc_server.py"))
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_using_demo_false_without_flag_even_if_data_missing(self):
        with tempfile.TemporaryDirectory() as td:
            self.mod.DEMO_MODE = False
            self.mod.DATA_DIR = td  # no summary.json here
            try:
                self.assertFalse(self.mod.using_demo())
            finally:
                self.mod.DATA_DIR = None

    def test_server_exits_nonzero_when_data_unresolvable(self):
        # Env var names a missing dir: the server must refuse to start (no port bound).
        env = dict(os.environ, JADAL_POC_DATA=os.path.join(ROOT, "no-such-dir-xyz"))
        env.pop("JADAL_PY", None)
        q = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "poc_server.py"), "--port", "8847"],
            capture_output=True, text=True, timeout=30, env=env)
        self.assertEqual(q.returncode, 2, q.stderr)
        self.assertIn("JADAL_POC_DATA", q.stderr)
        self.assertNotIn("poc dashboard on", q.stdout)

    def test_server_explicit_missing_data_dir_fails(self):
        q = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "poc_server.py"), "--port", "8847",
             "--data-dir", os.path.join(ROOT, "no-such-dir-xyz")],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(q.returncode, 2, q.stderr)
        self.assertIn("does not exist", q.stderr)


if __name__ == "__main__":
    unittest.main()
