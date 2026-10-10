"""PoC dashboard server tests — stdlib only (unittest + urllib), no GPU/network.

Covers: API shape (summary/windows/stream), path-traversal rejection,
pagination, partial-run flag, reverify id validation + 'not available yet'.
Runs against the demo fixture on an ephemeral port. Must finish < 30 s.
"""
import json
import os
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PORT = 8849
BASE = f"http://127.0.0.1:{PORT}"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


def get_json(path):
    status, _, blob = get(path)
    return status, json.loads(blob)


def post(path):
    req = urllib.request.Request(BASE + path, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class PocServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Regenerate the fixture (idempotent) so thumbs exist even though
        # generated JPEGs are git-ignored.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "make_poc_fixture", os.path.join(HERE, "make_poc_fixture.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.build(os.path.join(HERE, "fixtures", "poc"), n=60, partial=True)
        cls.proc = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "scripts", "poc_server.py"),
             "--demo-fixture", "--port", str(PORT)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                get_json("/api/summary")
                break
            except OSError:
                time.sleep(0.2)
        else:
            cls.proc.terminate()
            raise RuntimeError("server did not start")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=10)

    def test_summary_shape(self):
        status, s = get_json("/api/summary")
        self.assertEqual(status, 200)
        for key in ("n_windows", "hours", "by_verdict", "by_reason",
                    "by_factory", "throughput", "partial"):
            self.assertIn(key, s, key)
        self.assertIn("partial", s)
        self.assertIsInstance(s["partial"], bool)
        self.assertIn("gpu", s["throughput"])
        self.assertIn("live", s)
        self.assertIn("by_stage", s["live"])

    def test_windows_shape_and_pagination(self):
        status, d = get_json("/api/windows?per_page=5&page=1")
        self.assertEqual(status, 200)
        self.assertEqual(d["per_page"], 5)
        self.assertGreaterEqual(d["total"], len(d["windows"]))
        w = d["windows"][0]
        for key in ("window_id", "factory", "worker", "start_s", "end_s",
                    "verdict", "stage", "reasons", "score", "features",
                    "receipt_sha256", "thumbs", "label_source"):
            self.assertIn(key, w, key)
        _, d2 = get_json("/api/windows?per_page=5&page=2")
        ids1 = [x["window_id"] for x in d["windows"]]
        ids2 = [x["window_id"] for x in d2["windows"]]
        self.assertFalse(set(ids1) & set(ids2), "pages must not overlap")
        # per_page cap
        _, d3 = get_json("/api/windows?per_page=9999")
        self.assertLessEqual(d3["per_page"], 100)

    def test_windows_filters(self):
        _, d = get_json("/api/windows?verdict=reject&per_page=50")
        self.assertGreater(d["total"], 0)
        self.assertTrue(all(w["verdict"] == "reject" for w in d["windows"]))
        _, d = get_json("/api/windows?q=f061&per_page=50")
        self.assertTrue(all("f061" in w["window_id"] for w in d["windows"]))
        self.assertIn("facets", d)

    def test_single_window_and_404(self):
        _, d = get_json("/api/windows?per_page=1")
        wid = d["windows"][0]["window_id"]
        status, w = get_json(f"/api/windows/{wid}")
        self.assertEqual(status, 200)
        self.assertEqual(w["window_id"], wid)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            get("/api/windows/does_not_exist_zzz")
        self.assertEqual(cm.exception.code, 404)

    def test_path_traversal_rejected(self):
        for bad in ("/thumbs/%2e%2e/poc_server.py", "/thumbs/%2Fetc%2Fpasswd",
                    "/thumbs/ok/../../x.jpg"):
            try:
                status, _, _ = get(bad)
            except urllib.error.HTTPError as e:
                status = e.code
            self.assertIn(status, (400, 404), bad)

    def test_thumb_served(self):
        _, d = get_json("/api/windows?per_page=1")
        thumb = d["windows"][0]["thumbs"][0]
        self.assertTrue(thumb.startswith("thumbs/"))
        status, ctype, blob = get("/" + thumb)
        self.assertEqual(status, 200)
        self.assertIn("image/jpeg", ctype)
        self.assertTrue(blob[:2] == b"\xff\xd8")

    def test_stream_shape(self):
        status, d = get_json("/api/stream")
        self.assertEqual(status, 200)
        self.assertTrue(d["available"])
        e = d["events"][0]
        for key in ("t", "window_id", "chunk", "running_score", "decision"):
            self.assertIn(key, e, key)

    def test_reverify_validation(self):
        status, body = post("/api/reverify/bad!id/here")
        self.assertEqual(status, 400)
        status, body = post("/api/reverify/ok_window-1.2")
        # Pre-integration the run worker had not provided scripts/poc_run.py
        # (-> clear 503). Post-integration poc_run.py exists, so the server
        # attempts verification and reports ok|mismatch with HTTP 200.
        # Accept either, but validate the shape of each.
        if status == 503:
            self.assertEqual(body["status"], "unavailable")
            self.assertIn("not available yet", body["message"])
        else:
            self.assertEqual(status, 200)
            self.assertIn(body["status"], ("ok", "mismatch"))
            self.assertIn("returncode", body)

    def test_index_served_offline(self):
        status, ctype, blob = get("/")
        self.assertEqual(status, 200)
        html = blob.decode()
        self.assertIn("honesty", html.lower())
        self.assertNotIn("src=\"http", html)
        self.assertNotIn("href=\"http", html)
        self.assertNotIn("srcset=", html)

    def test_banner_states_out_of_domain_finding(self):
        _, _, blob = get("/")
        html = blob.decode()
        start = html.index('class="honesty"')
        banner = html[start:html.index("</div>", start)]
        self.assertIn("0.529", banner)
        self.assertIn("false reject", banner)
        self.assertIn('href="#/about"', banner)
        self.assertIn('id="claims"', html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
