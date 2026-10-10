"""Tests for tools/pod/union_registry.py (no network, < 30 s)."""
import importlib.util
import json
import os
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "union_registry", os.path.join(REPO, "tools", "pod", "union_registry.py"))
ur = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ur)


def write(path, entries):
    with open(path, "w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


def read(path):
    with open(path) as fh:
        return fh.read()


class UnionRegistryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def p(self, name):
        return os.path.join(self.tmp.name, name)

    def test_later_line_and_later_source_win(self):
        a = self.p("a.jsonl")
        b = self.p("b.jsonl")
        write(a, [{"id": "X", "title": "new", "status": "running"},
                  {"id": "X", "status": "done", "verdict": "adopt"},
                  {"id": "Y", "title": "only-in-a"}])
        write(b, [{"id": "X", "verdict": "drop"}])
        merged = ur.merge([a, b])
        self.assertEqual(merged["X"]["title"], "new")      # partial lines merge
        self.assertEqual(merged["X"]["status"], "done")
        self.assertEqual(merged["X"]["verdict"], "drop")   # later source wins
        self.assertEqual(merged["Y"]["title"], "only-in-a")

    def test_seed_is_append_only_and_idempotent(self):
        live = self.p("registry.jsonl")
        snap = self.p("snapshot.jsonl")
        write(live, [{"id": "X", "status": "running"}])
        write(snap, [{"id": "X", "title": "should-not-be-appended"},
                     {"id": "Y", "title": "seeded", "status": "running"},
                     {"id": "Y", "status": "done", "verdict": "adopt"}])
        before = read(live)
        added = ur.seed(live, [snap])
        self.assertEqual(added, ["Y", "Y"])  # every line of the new id, in order
        self.assertTrue(read(live).startswith(before))  # nothing rewritten
        self.assertEqual(ur.seed(live, [snap]), [])     # re-run is a no-op
        merged = ur.merge([live])
        self.assertEqual(merged["X"]["status"], "running")
        self.assertEqual(merged["Y"]["status"], "done")  # latest seeded line still wins
        self.assertEqual(merged["Y"]["title"], "seeded")

    def test_render_delegates_to_exp_py(self):
        a = self.p("a.jsonl")
        write(a, [{"id": "E9.01", "track": "E9", "title": "render me", "status": "done",
                   "verdict": "adopt", "data": "synthetic", "regime": "n/a"}])
        out = self.p("EXPERIMENTS.md")
        ur.render(ur.merge([a]), out, exp_py=os.path.join(REPO, "tools", "exp.py"))
        text = read(out)
        self.assertIn("E9.01", text)
        self.assertIn("render me", text)
        self.assertIn("Experiment log", text)

    def test_main_seed_then_render(self):
        live = self.p("registry.jsonl")
        snap = self.p("snapshot.jsonl")
        out = self.p("EXPERIMENTS.md")
        merged = self.p("merged.jsonl")
        write(snap, [{"id": "E1", "track": "E1", "title": "t", "status": "done", "verdict": "maybe"}])
        rc = ur.main(["--source", snap, "--source", live, "--seed", "--seed-target", live,
                      "--out", out, "--merged", merged])
        self.assertEqual(rc, 0)
        self.assertEqual(len(read(live).strip().splitlines()), 1)
        self.assertIn("E1", read(out))
        self.assertEqual(len(read(merged).strip().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
