"""Tests for the Jadal QA dashboard (web/).

Runs with the project python:
    /home/parshu/envs/jadal/bin/python -m unittest discover -s web -t . -v

NOT DONE:
- No browser assertions here; visual checks (1280 / 390 / dark screenshots
  and a no-horizontal-scroll probe) live in docs/screens/take_shots.mjs and
  are run manually because they need Playwright.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

WEB = Path(__file__).resolve().parent
MOCK = WEB / "mock"

REASON_CODES = {
    "UNDECODABLE", "HANDS_NOT_VISIBLE", "BLUR", "TOO_DARK", "OVEREXPOSED",
    "SHAKY", "DUPLICATE", "WRONG_STEP", "NO_ACTIVE_WORK", "LOW_CONFIDENCE",
}
VERDICTS = {"accept", "reject", "review"}
LANGS = ["hi-IN", "kn-IN", "ta-IN", "te-IN", "en-IN"]


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def load(name: str):
    return json.loads((MOCK / name).read_text(encoding="utf-8"))


class StaticAssets(unittest.TestCase):
    def test_files_exist(self):
        for name in ["index.html", "app.js", "style.css",
                     "mock/batches.json", "mock/receipts.json",
                     "mock/risk_coverage.json", "mock/sop_gaps.json", "mock/pico_status.json"]:
            p = WEB / name
            self.assertTrue(p.is_file(), f"missing {name}")
            self.assertGreater(p.stat().st_size, 0, f"empty {name}")

    def test_no_external_resources(self):
        """No CDN, no remote assets: src/href/@import/url() must be local."""
        html = (WEB / "index.html").read_text(encoding="utf-8")
        css = (WEB / "style.css").read_text(encoding="utf-8")
        js = (WEB / "app.js").read_text(encoding="utf-8")
        for attr in re.findall(r'(?:src|href)\s*=\s*"([^"]+)"', html):
            self.assertFalse(attr.startswith(("http://", "https://", "//")),
                             f"external asset in index.html: {attr}")
        for blob, where in [(css, "CSS"), (js, "JS")]:
            for m in re.findall(r'(?:@import\s+|url\(\s*)["\']?([^"\')]+)', blob):
                self.assertFalse(m.startswith(("http://", "https://", "//", "data:")) or "cdn" in m,
                                 f"{where} pulls a remote asset: {m}")

    def test_html_wiring(self):
        html = (WEB / "index.html").read_text(encoding="utf-8")
        self.assertIn('name="viewport"', html)
        self.assertIn('href="style.css"', html)
        self.assertIn('src="app.js"', html)
        for view in ["view-batches", "view-segment", "view-insights"]:
            self.assertIn(f'id="{view}"', html)
        self.assertIn('id="pico-status"', html)

    def test_css_has_dark_mode(self):
        css = (WEB / "style.css").read_text(encoding="utf-8")
        self.assertIn("@media (prefers-color-scheme: dark)", css)
        # fixed min-widths (outside @media breakpoints) wider than a 390px
        # phone would force horizontal scroll
        no_media = "\n".join(l for l in css.splitlines() if not l.strip().startswith("@media"))
        for m in re.finditer(r"min-width:\s*([\d.]+)(px|rem)", no_media):
            val, unit = float(m.group(1)), m.group(2)
            px = val * 16 if unit == "rem" else val
            self.assertLessEqual(px, 0.1, f"fixed min-width {m.group(0)} risks sideways scroll")

    def test_js_touches_every_endpoint(self):
        js = (WEB / "app.js").read_text(encoding="utf-8")
        for ep in ["/ws", "/batches", "/receipts/", "/reverify",
                   "/voice/", "/stats/risk_coverage", "/stats/sop_gaps", "/pico/status"]:
            self.assertIn(ep, js, f"app.js never calls {ep}")
        for lang in LANGS:
            self.assertIn(lang, js, f"language {lang} missing from app.js")
        self.assertIn("mock=1", js + (WEB / "index.html").read_text(encoding="utf-8"),
                      "?mock=1 switch not advertised")

    def test_js_syntax_with_node(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node not on PATH")
        r = subprocess.run([node, "--check", str(WEB / "app.js")],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, f"node --check failed: {r.stderr}")


class MockContract(unittest.TestCase):
    def test_batch_shapes_and_rule(self):
        batches = load("batches.json")
        self.assertIsInstance(batches, list)
        self.assertGreaterEqual(len(batches), 1)
        for b in batches:
            self.assertIn(b["batch_decision"], {"accept", "reject"})
            self.assertEqual(b["rule"], "reject if rejected_count > k")
            rejected = sum(1 for v in b["verdicts"] if v["verdict"] == "reject")
            expect = "reject" if rejected > b["k"] else "accept"
            self.assertEqual(b["batch_decision"], expect,
                             f"{b['batch_id']}: rule gives {expect}")
            self.assertEqual(b["n"], len(b["segment_ids"]))
            for v in b["verdicts"]:
                self.assertIn(v["verdict"], VERDICTS)
                self.assertTrue(set(v["reasons"]) <= REASON_CODES,
                                f"unknown reason code in {v}")
                self.assertIn(v["segment_id"], b["segment_ids"])
            seg_ids = [s["segment_id"] for s in b.get("segments", [])]
            self.assertEqual(sorted(seg_ids), sorted(b["segment_ids"]),
                             f"{b['batch_id']}: segments[] does not cover segment_ids[]")
            for s in b.get("segments", []):
                self.assertIn("receipt_id", s)
                self.assertIn("worker_id", s)
                self.assertIn("latency_ms", s)

    def test_receipt_hashes_follow_spec(self):
        receipts = {k: v for k, v in load("receipts.json").items() if not k.startswith("_")}
        batches = load("batches.json")
        linked = {s["receipt_id"] for b in batches for s in b.get("segments", [])}
        self.assertTrue(linked <= set(receipts), "segments link to unknown receipts")
        for rid, r in receipts.items():
            body = {k: v for k, v in r.items() if k not in ("receipt_sha256", "receipt_id", "created_utc")}
            h = hashlib.sha256(canonical(body)).hexdigest()
            self.assertEqual(r["receipt_sha256"], h, f"bad receipt_sha256 for {rid}")
            self.assertEqual(r["receipt_id"], h[:16], f"bad receipt_id for {rid}")
            self.assertEqual(rid, h[:16])
            self.assertTrue(set(r["verdict"]) >= {"segment_id", "verdict", "reasons", "answer_conf", "threshold"},
                            f"verdict keys off in {rid}")

    def test_floats_rounded_to_4dp(self):
        receipts = load("receipts.json")
        def check(x, path=""):
            if isinstance(x, float):
                self.assertEqual(round(x, 4), x, f"unrounded float at {path}: {x}")
            elif isinstance(x, dict):
                for k, v in x.items():
                    check(v, f"{path}.{k}")
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    check(v, f"{path}[{i}]")
        for rid, r in receipts.items():
            if rid.startswith("_"):
                continue
            check(r.get("signals"), f"{rid}.signals")
            check(r.get("decisions"), f"{rid}.decisions")

    def test_risk_and_gaps_shapes(self):
        risk = load("risk_coverage.json")
        self.assertGreaterEqual(len(risk["points"]), 2)
        for p in risk["points"]:
            self.assertEqual(set(p), {"threshold", "coverage", "error"})
            self.assertTrue(0.0 <= p["coverage"] <= 1.0)
        for key in ["error_bound", "n", "errors", "target_error", "delta"]:
            self.assertIn(key, risk)
        gaps = load("sop_gaps.json")
        self.assertGreaterEqual(len(gaps["gaps"]), 1)
        total = sum(g["count"] for g in gaps["gaps"])
        self.assertGreater(total, 0)
        for g in gaps["gaps"]:
            self.assertIn(g["reason_code"], REASON_CODES)

    def test_pico_status_shape(self):
        pico = load("pico_status.json")
        self.assertIn(pico.get("led"), {"G", "A", "R", "OFF"})
        self.assertIn("connected", pico)
        self.assertIn("last_heartbeat_utc", pico)


if __name__ == "__main__":
    unittest.main(verbosity=2)
