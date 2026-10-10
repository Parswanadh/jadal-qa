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

    def test_colour_rule_and_no_overflow_mask(self):
        """Green/amber/red only on verdict chips + Pico LED; no body
        overflow-x mask (real overflow must be fixed, not hidden)."""
        css = (WEB / "style.css").read_text(encoding="utf-8")
        self.assertNotRegex(css, r"overflow-x:\s*(hidden|clip|scroll)",
                            "page-level overflow must be fixed, not masked")
        self.assertIn("overflow-wrap: anywhere", css,
                      "long hashes must wrap at phone width")
        for var in ["--accept", "--review", "--reject"]:
            uses = [l.strip() for l in css.splitlines()
                    if var in l and not re.match(r"\s*--[\w-]+:", l)]
            for line in uses:
                self.assertRegex(
                    line, r"verdict|status-led",
                    f"verdict colour used outside verdicts/Pico LED: {line}")

    def test_js_touches_every_endpoint(self):
        js = (WEB / "app.js").read_text(encoding="utf-8")
        for ep in ["/ws", "/batches", "/receipts/", "/reverify",
                   "/voice/", "/stats/risk_coverage", "/stats/sop_gaps", "/pico/status"]:
            self.assertIn(ep, js, f"app.js never calls {ep}")
        # exact live field names (upper_bound is certify's name, not error_bound)
        for field in ["receipt_ids", "upper_bound", "last_hb_led", "last_hb_age_s",
                      "blurred_sha256", "resolveReceiptId", "rvSeq"]:
            self.assertIn(field, js, f"app.js never reads {field}")
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
    """web/mock/*.json mirrors the ACTUAL service (origin/feat/service):
    batches.json == GET /batches, risk_coverage.json == GET /stats/risk_coverage
    (bare list), sop_gaps.json == GET /stats/sop_gaps (bare list),
    pico_status.json == GET /pico/status (PicoLink.status shape)."""

    BATCH_KEYS = {"batch_id", "site_id", "segment_ids", "verdicts",
                  "batch_decision", "rule", "k", "n", "created_utc"}

    def batches(self):
        doc = load("batches.json")
        self.assertEqual(set(doc), {"batches"}, "GET /batches returns {batches: [...]}")
        return doc["batches"]

    def test_batch_shapes_and_rule(self):
        batches = self.batches()
        self.assertIsInstance(batches, list)
        self.assertGreaterEqual(len(batches), 1)
        for b in batches:
            self.assertEqual(set(b), self.BATCH_KEYS,
                             f"{b.get('batch_id')}: keys drifted from the store shape")
            self.assertIn(b["batch_decision"], {"accept", "reject"})
            self.assertEqual(b["rule"], "reject if rejected_count > k")
            rejected = sum(1 for v in b["verdicts"] if v["verdict"] == "reject")
            expect = "reject" if rejected > b["k"] else "accept"
            self.assertEqual(b["batch_decision"], expect,
                             f"{b['batch_id']}: rule gives {expect}")
            self.assertEqual(b["n"], len(b["segment_ids"]))
            self.assertRegex(b["batch_id"], r"^b[0-9a-f]{16}$")
            for v in b["verdicts"]:
                self.assertIn(v["verdict"], VERDICTS)
                self.assertTrue(set(v["reasons"]) <= REASON_CODES,
                                f"unknown reason code in {v}")
                self.assertIn(v["segment_id"], b["segment_ids"])

    def test_every_segment_resolves_to_a_receipt(self):
        """The UI links segment -> receipt via verdict.segment_id, so every
        segment_id must appear in exactly one receipt verdict."""
        batches = self.batches()
        receipts = {k: v for k, v in load("receipts.json").items() if not k.startswith("_")}
        by_seg: dict[str, list[str]] = {}
        for rid, r in receipts.items():
            by_seg.setdefault(r["verdict"]["segment_id"], []).append(rid)
        for b in batches:
            # live orders GET /batches/{id}.receipt_ids by receipt_id
            rids = sorted(rid for rid, r in receipts.items()
                          if r["verdict"]["segment_id"] in b["segment_ids"])
            self.assertEqual(len(rids), len(b["segment_ids"]),
                             f"{b['batch_id']}: segments without receipts")
            for sid in b["segment_ids"]:
                self.assertEqual(len(by_seg.get(sid, [])), 1,
                                 f"segment {sid} resolves to {by_seg.get(sid)}")

    def test_receipt_hashes_follow_spec(self):
        receipts = {k: v for k, v in load("receipts.json").items() if not k.startswith("_")}
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
        self.assertIsInstance(risk, list, "GET /stats/risk_coverage returns a bare list")
        self.assertGreaterEqual(len(risk), 2)
        for p in risk:
            self.assertEqual(set(p), {"threshold", "coverage", "error"})
            self.assertTrue(0.0 <= p["coverage"] <= 1.0)
        cov = [p["coverage"] for p in risk]
        self.assertEqual(cov, sorted(cov), "certify orders strictest-first (coverage ascending)")
        gaps = load("sop_gaps.json")
        self.assertIsInstance(gaps, list, "GET /stats/sop_gaps returns a bare list")
        self.assertGreaterEqual(len(gaps), 1)
        total = sum(g["count"] for g in gaps)
        self.assertGreater(total, 0)
        for g in gaps:
            self.assertIn(g["reason_code"], REASON_CODES)
            self.assertIn("count", g)

    def test_pico_status_shape(self):
        pico = load("pico_status.json")
        self.assertEqual(set(pico), {"connected", "port", "last_hb_age_s",
                                     "last_hb_ticks_ms", "last_hb_led"},
                         "must match PicoLink.status() keys")
        self.assertIn(pico["last_hb_led"], {"G", "A", "R", "OFF"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
