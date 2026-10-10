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
        for name in ["index.html", "api.js", "app.js", "style.css",
                     "mock/batches.json", "mock/receipts.json",
                     "mock/risk_coverage.json", "mock/sop_gaps.json", "mock/pico_status.json"]:
            p = WEB / name
            self.assertTrue(p.is_file(), f"missing {name}")
            self.assertGreater(p.stat().st_size, 0, f"empty {name}")

    def test_no_external_resources(self):
        """No CDN, no remote assets: src/href/@import/url() must be local."""
        html = (WEB / "index.html").read_text(encoding="utf-8")
        css = (WEB / "style.css").read_text(encoding="utf-8")
        js = ((WEB / "api.js").read_text(encoding="utf-8")
              + (WEB / "app.js").read_text(encoding="utf-8"))
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
        self.assertIn('src="api.js"', html)
        self.assertIn('src="app.js"', html)
        for view in ["view-batches", "view-segment", "view-insights",
                     "view-queue", "view-videos"]:
            self.assertIn(f'id="{view}"', html)
        for tab in ["tab-batches", "tab-segment", "tab-insights",
                    "tab-queue", "tab-videos"]:
            self.assertIn(f'id="{tab}"', html)
        # v2: system drawer + global filter bar of real <select>s + search
        self.assertIn('id="sys-drawer"', html)
        self.assertIn('id="pico-led"', html)
        self.assertIn('id="filterbar"', html)
        for sel in ["f-site", "f-worker", "f-batch", "f-verdict", "f-reason", "f-lang", "f-q",
                    "q-reason", "v-q", "v-site", "v-verdict", "v-reason",
                    "videos-form", "queue-body", "videos-body"]:
            self.assertIn(f'id="{sel}"', html)
        for tag in ["<select", "<details"]:
            self.assertIn(tag, html)

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
        api = (WEB / "api.js").read_text(encoding="utf-8")
        app = (WEB / "app.js").read_text(encoding="utf-8")
        # all transport lives in the adapter; app.js only goes through it
        for ep in ["/ws", "/batches", "/receipts/", "/reverify",
                   "/voice/", "/stats/risk_coverage", "/stats/sop_gaps", "/pico/status",
                   "/hub/search", "/hub/thumb"]:
            self.assertIn(ep, api, f"api.js never calls {ep}")
        # exact live field names (upper_bound is certify's name, not error_bound)
        for field in ["receipt_ids", "upper_bound", "last_hb_led", "last_hb_age_s",
                      "blurred_sha256", "findReceiptForSegment", "parseHash", "buildHash",
                      "filterOptions", "applyFilters", "normReceipt", "createClient",
                      "isBound", "boundCaption", "funnelCounts", "queueRows", "normHub", "hubSearch"]:
            self.assertIn(field, api, f"api.js never defines {field}")
        self.assertIn("JadalApi", app, "app.js must go through the web/api.js adapter")
        self.assertIn("createClient", app)
        # app.js reaches every operation through the client (never its own fetch):
        for call in ["client.batches(", "client.batchDetail(", "client.receipt(",
                     "client.reverify(", "client.risk(", "client.gaps(",
                     "client.pico(", "client.voiceUrl(", "client.hubSearch(",
                     "client.hubThumbUrl(", "client.hubVideoUrl("]:
            self.assertIn(call, app, f"app.js never calls {call}")
        for helper in ["A.normBatch", "A.normReceipt", "A.normRisk", "A.normGaps",
                       "A.normPico", "A.applyFilters", "A.filterOptions",
                       "A.parseHash", "A.buildHash", "A.shortHash",
                       "A.findReceiptForSegment", "A.wsUrl", "A.isBound",
                       "A.boundCaption",
                       "A.funnelCounts", "A.queueRows", "A.queueReasons"]:
            self.assertIn(helper, app, f"app.js never uses adapter {helper}")
        # ... and builds no transport URLs of its own
        self.assertNotIn("fetch(API", app, "app.js bypasses the adapter transport")
        self.assertNotIn("fetch('mock", app, "app.js fetches mock files directly")
        self.assertNotIn("/hub/search", app, "hub URLs must come from the adapter")
        # v2 components: expandable rows, receipt tree, timeline, voice select
        for token in ["segdetails", "receipt-tree", "timeline", "voice-lang", "sys-body", "f-head",
                      "funnel", "queue-body", "videos-body", "tl-compact", "more-body",
                      "video-grid", "q-reason", "videos-form"]:
            self.assertIn(token, app, f"app.js never builds {token}")
        for lang in LANGS:
            self.assertIn(lang, api, f"language {lang} missing from api.js")
        self.assertIn("mock=1", app + (WEB / "index.html").read_text(encoding="utf-8"),
                      "?mock=1 switch not advertised")

    def test_js_syntax_with_node(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node not on PATH")
        for name in ["api.js", "app.js"]:
            r = subprocess.run([node, "--check", str(WEB / name)],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, f"node --check {name} failed: {r.stderr}")


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


NODE_ADAPTER_CASES = r"""
const A = require(API_PATH);
const batches = A.normBatchList(require(BATCHES));
const receipts = require(RECEIPTS);
const assert = require('node:assert');

// normalizers
assert.strictEqual(batches.length, 3);
assert.ok(batches.every(b => b.batch_id && b.site_id && Array.isArray(b.rows)));
const total = batches.reduce((n, b) => n + b.rows.length, 0);
assert.strictEqual(total, 20);

// filter options come from data
const opts = A.filterOptions(batches);
assert.deepStrictEqual(opts.sites, ['site_ludhiana_07', 'site_surat_01', 'site_tirupur_02']);
assert.deepStrictEqual(opts.verdicts, ['accept', 'reject', 'review']);
assert.ok(opts.reasons.includes('BLUR') && opts.reasons.includes('WRONG_STEP'));
assert.deepStrictEqual(opts.langs, ['hi-IN', 'kn-IN', 'ta-IN', 'te-IN', 'en-IN']);
assert.strictEqual(opts.batches.length, 3);

// verdict filter keeps only matching rows
let f = A.applyFilters(batches, { verdict: 'reject' });
assert.ok(f.length >= 1);
assert.ok(f.every(x => x.rows.length > 0 && x.rows.every(r => r.verdict === 'reject')));
assert.ok(f.length < batches.length || f[0].rows.length < batches[0].rows.length);

// site + reason filters compose
f = A.applyFilters(batches, { site: 'site_tirupur_02', reason: 'LOW_CONFIDENCE' });
assert.ok(f.length >= 1);
assert.ok(f.every(x => x.batch.site_id === 'site_tirupur_02'));
assert.ok(f.every(x => x.rows.every(r => r.reasons.includes('LOW_CONFIDENCE'))));

// search matches segment ids case-insensitively
const someSeg = batches[0].rows[0].segment_id;
f = A.applyFilters(batches, { q: someSeg.slice(-4).toUpperCase() });
assert.ok(f.some(x => x.rows.some(r => r.segment_id === someSeg)));
// empty filters return everything
f = A.applyFilters(batches, {});
assert.strictEqual(f.reduce((n, x) => n + x.rows.length, 0), total);
// impossible filter returns nothing
assert.deepStrictEqual(A.applyFilters(batches, { site: 'no-such-site' }), []);

// hash router round-trips
const h = A.buildHash('batches', null, { site: 'a', verdict: 'reject', q: 'x y' }, null);
const p = A.parseHash(h);
assert.strictEqual(p.route, 'batches');
assert.strictEqual(p.filters.site, 'a');
assert.strictEqual(p.filters.verdict, 'reject');
assert.strictEqual(p.filters.q, 'x y');
const bad = A.parseHash('#/batches?verdict=bogus&reason=BOGUS');
assert.strictEqual(bad.filters.verdict, '');
assert.strictEqual(bad.filters.reason, '');
assert.strictEqual(A.parseHash('#/garbage').route, 'batches');
const seg = A.parseHash('#/segment/abc123?lang=ta-IN');
assert.strictEqual(seg.route, 'segment');
assert.strictEqual(seg.id, 'abc123');
assert.strictEqual(seg.lang, 'ta-IN');

// receipt normalizer + hash helpers
const rid = Object.keys(receipts).filter(k => k[0] !== '_')[0];
const r = A.normReceipt(receipts[rid]);
assert.strictEqual(r.receipt_id, rid);
assert.ok(r.step_names.length > 0 && r.step_names.includes(r.step_choice));
assert.strictEqual(typeof r.pipeline_version, 'string');
assert.strictEqual(A.shortHash('abcdef1234567890'), 'abcdef123456…');
assert.strictEqual(A.shortHash('—'), '—');

// segment -> receipt resolution is pure over fetched bodies
const detail = { receipt_ids: [rid] };
assert.strictEqual(A.findReceiptForSegment(detail, { [rid]: receipts[rid] }, r.segment_id), rid);
assert.strictEqual(A.findReceiptForSegment(detail, { [rid]: receipts[rid] }, 'nope'), null);
console.log('ADAPTER-OK total=' + total);
"""


class AdapterLogic(unittest.TestCase):
    """web/api.js pure functions (normalizers, filters, hash router,
    receipt resolution) exercised in node against the real fixtures."""

    def test_adapter_in_node(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node not on PATH")
        script = (NODE_ADAPTER_CASES
                  .replace("API_PATH", repr(str(WEB / "api.js")))
                  .replace("BATCHES", repr(str(MOCK / "batches.json")))
                  .replace("RECEIPTS", repr(str(MOCK / "receipts.json"))))
        r = subprocess.run([node, "-e", script],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0,
                         f"adapter node cases failed:\n{r.stdout}\n{r.stderr}")
        self.assertIn("ADAPTER-OK", r.stdout)


class FixRound1(unittest.TestCase):
    """PR #21 fix round 1: no stray re-verify GET, hash-proof routing,
    absent-vs-null threshold, bound caption, adapter-only reads."""

    def test_no_stray_unused_get(self):
        api = (WEB / "api.js").read_text(encoding="utf-8")
        self.assertNotIn("__unused__", api,
                         "live re-verify must not fire a stray GET")

    def test_route_param_validation(self):
        app = (WEB / "app.js").read_text(encoding="utf-8")
        # a hash like head=a"b must never be interpolated into a selector
        self.assertNotIn("querySelector('option[value=\"'", app,
                         "route param interpolated into querySelector (throws)")
        self.assertIn("sel.options", app,
                      "head option must be matched via dataset/options lookup")
        node = shutil.which("node")
        if node is None:
            self.skipTest("node not on PATH")
        script = (
            "const A = require(" + repr(str(WEB / "api.js")) + ");\n"
            "const assert = require('node:assert');\n"
            "const p = A.parseHash('#/insights?head=a\"b');\n"
            "assert.strictEqual(p.route, 'insights');\n"
            "assert.strictEqual(p.head, 'a\"b');\n"
            "const q = A.parseHash('#/segment/abc%22def?lang=xx&verdict=bogus');\n"
            "assert.strictEqual(q.id, 'abc\"def');\n"
            "assert.strictEqual(q.lang, 'hi-IN');\n"
            "assert.strictEqual(q.filters.verdict, '');\n"
            "console.log('ROUTE-OK');\n"
        )
        r = subprocess.run([node, "-e", script],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0,
                         f"route-param node cases failed:\n{r.stdout}\n{r.stderr}")
        self.assertIn("ROUTE-OK", r.stdout)

    def test_threshold_absent_vs_null(self):
        app = (WEB / "app.js").read_text(encoding="utf-8")
        api = (WEB / "api.js").read_text(encoding="utf-8")
        self.assertIn("no certified threshold reported", app,
                      "absent chosen_threshold needs a neutral 'not reported'")
        self.assertIn("bound not reported", api,
                      "absent error bound needs a 'bound not reported' caption (shared helper)")
        node = shutil.which("node")
        if node is None:
            self.skipTest("node not on PATH")
        script = (
            "const A = require(" + repr(str(WEB / "api.js")) + ");\n"
            "const assert = require('node:assert');\n"
            "// bare list (live risk shape): metadata absent -> undefined\n"
            "let d = A.normRisk([{threshold: 0.5, coverage: 0.8, error: 0.05}]);\n"
            "assert.strictEqual(d.chosen_threshold, undefined);\n"
            "assert.strictEqual(d.error_bound, undefined);\n"
            "// object without the keys: still absent -> undefined\n"
            "d = A.normRisk({points: [{threshold: 0.5, coverage: 0.8, error: 0.05}]});\n"
            "assert.strictEqual(d.chosen_threshold, undefined);\n"
            "// explicit null survives (means 'not certified', not 'unreported')\n"
            "d = A.normRisk({points: [], chosen_threshold: null, upper_bound: 0.07});\n"
            "assert.strictEqual(d.chosen_threshold, null);\n"
            "assert.strictEqual(d.error_bound, 0.07);\n"
            "// a real number survives too\n"
            "d = A.normRisk({points: [], chosen_threshold: 0.62});\n"
            "assert.strictEqual(d.chosen_threshold, 0.62);\n"
            "console.log('THRESHOLD-OK');\n"
        )
        r = subprocess.run([node, "-e", script],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0,
                         f"threshold node cases failed:\n{r.stdout}\n{r.stderr}")
        self.assertIn("THRESHOLD-OK", r.stdout)

    def test_named_reads_go_through_adapter(self):
        app = (WEB / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("raw.signals", app, "signals must come from normReceipt")
        self.assertNotIn("raw.decisions", app, "decisions must come from normReceipt")
        self.assertIn("r.hands_p_yes", app)
        self.assertIn("r.active_work_p_yes", app)
        self.assertIn("A.first(state.segBatch", app)

    def test_copy_feedback_is_inline(self):
        app = (WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn("copy-toast", app, "copy feedback must render next to the button")
        css = (WEB / "style.css").read_text(encoding="utf-8")
        self.assertIn(".copy-toast", css)


class Dashboard3(unittest.TestCase):
    """dashboard3 round: null-vs-absent bound tri-state, chart tokens,
    funnel/queue/video-hub adapter coverage, new routes."""

    def test_bound_tristate_strings(self):
        app = (WEB / "app.js").read_text(encoding="utf-8")
        api = (WEB / "api.js").read_text(encoding="utf-8")
        self.assertIn("A.boundCaption", app, "bound checks must use the adapter gate")
        self.assertIn("bound not reported", api)
        self.assertIn("not certified (no threshold meets the target)", api)
        # no bare isFinite(null)-style bound test may remain (isFinite(null) is true)
        for line in app.splitlines():
            if "error_bound" in line and "isFinite" in line and "isBound" not in line:
                self.fail(f"bare isFinite on error_bound (null coerces to 0): {line.strip()}")

    def test_bound_tristate_in_node(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node not on PATH")
        script = (
            "const A = require(" + repr(str(WEB / "api.js")) + ");\n"
            "const assert = require('node:assert');\n"
            "assert.strictEqual(A.isBound(undefined), false);\n"
            "assert.strictEqual(A.isBound(null), false);\n"
            "assert.strictEqual(A.isBound(0.07), true);\n"
            "assert.strictEqual(A.isBound(0), true);\n"
            "assert.strictEqual(A.isBound(NaN), false);\n"
            "assert.strictEqual(A.isBound('0.07'), false);\n"
            "let d = A.normRisk({points: [{threshold: 0.5, coverage: 0.8, error: 0.05}], upper_bound: null});\n"
            "assert.strictEqual(d.error_bound, null);\n"
            "assert.strictEqual(A.isBound(d.error_bound), false);\n"
            "d = A.normRisk({points: [{threshold: 0.5, coverage: 0.8, error: 0.05}]});\n"
            "assert.strictEqual(d.error_bound, undefined);\n"
            "assert.strictEqual(A.isBound(d.error_bound), false);\n"
            "d = A.normRisk({points: [], upper_bound: 0.07});\n"
            "assert.strictEqual(A.isBound(d.error_bound), true);\n"
            "console.log('BOUND-OK');\n"
        )
        r = subprocess.run([node, "-e", script],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0,
                         f"bound node cases failed:\n{r.stdout}\n{r.stderr}")
        self.assertIn("BOUND-OK", r.stdout)

    def test_funnel_and_queue_in_node(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node not on PATH")
        script = (
            "const A = require(" + repr(str(WEB / "api.js")) + ");\n"
            "const assert = require('node:assert');\n"
            "const c = A.funnelCounts([{verdict: 'accept'}, {verdict: 'reject'},"
            " {verdict: 'review'}, {verdict: 'review'}, {verdict: 'weird'}]);\n"
            "assert.deepStrictEqual(c, {accept: 1, reject: 1, review: 2, other: 1, total: 5});\n"
            "const batches = [{batch_id: 'b1', site_id: 's', created_utc: '2026-10-10T02:00:00Z', rows: [\n"
            " {segment_id: 's2', site_id: 's', verdict: 'review', reasons: ['BLUR'], queue_priority: null},\n"
            " {segment_id: 's1', site_id: 's', verdict: 'review', reasons: [], queue_priority: 5},\n"
            " {segment_id: 's0', site_id: 's', verdict: 'accept', reasons: [], queue_priority: 9}]} ,\n"
            " {batch_id: 'b2', site_id: 's', created_utc: '2026-10-10T03:00:00Z', rows: [\n"
            " {segment_id: 's3', site_id: 's', verdict: 'review', reasons: [], queue_priority: null}]}];\n"
            "const q = A.queueRows(batches);\n"
            "assert.deepStrictEqual(q.map(r => r.segment_id), ['s1', 's3', 's2']);\n"
            "assert.ok(q.every(r => r.verdict === 'review'));\n"
            "assert.deepStrictEqual(A.queueReasons(q), ['BLUR']);\n"
            "assert.strictEqual(A.parseHash('#/queue').route, 'queue');\n"
            "assert.strictEqual(A.parseHash('#/videos?q=x').route, 'videos');\n"
            "console.log('QUEUE-OK');\n"
        )
        r = subprocess.run([node, "-e", script],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0,
                         f"queue node cases failed:\n{r.stdout}\n{r.stderr}")
        self.assertIn("QUEUE-OK", r.stdout)

    def test_hub_adapter_shape(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node not on PATH")
        script = (
            "const A = require(" + repr(str(WEB / "api.js")) + ");\n"
            "const assert = require('node:assert');\n"
            "const h = A.normHub({items: [{segment_id: 's1', verdict: 'review', reasons: ['BLUR']}], total: 1});\n"
            "assert.strictEqual(h.items.length, 1);\n"
            "assert.strictEqual(h.items[0].thumb_url, null);\n"
            "assert.strictEqual(h.total, 1);\n"
            "console.log('HUB-OK');\n"
        )
        r = subprocess.run([node, "-e", script],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0,
                         f"hub node cases failed:\n{r.stdout}\n{r.stderr}")
        self.assertIn("HUB-OK", r.stdout)

    def test_hub_unavailable_copy(self):
        app = (WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn("Video hub not available in this deployment", app)

    def test_hash_round_trips_queue_and_hub(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node not on PATH")
        script = (
            "const A = require(" + repr(str(WEB / "api.js")) + ");\n"
            "const assert = require('node:assert');\n"
            "let p = A.parseHash('#/queue?qr=BLUR');\n"
            "assert.strictEqual(p.route, 'queue');\n"
            "assert.strictEqual(p.queueReason, 'BLUR');\n"
            "assert.strictEqual(A.buildHash('queue', null, null, {qr: 'BLUR'}), '#/queue?qr=BLUR');\n"
            "p = A.parseHash('#/videos?hq=seg+12&hsite=site_a&hverdict=review&hreason=BLUR');\n"
            "assert.strictEqual(p.route, 'videos');\n"
            "assert.deepStrictEqual(p.hub, {q: 'seg 12', site: 'site_a', verdict: 'review', reason: 'BLUR'});\n"
            "const h = A.buildHash('videos', null, null, {hq: 'seg 12', hsite: 'site_a', hverdict: 'review', hreason: 'BLUR'});\n"
            "assert.deepStrictEqual(A.parseHash(h).hub, p.hub);\n"
            "assert.strictEqual(A.buildHash('queue', null, null, null), '#/queue');\n"
            "assert.strictEqual(A.parseHash('#/queue').queueReason, '');\n"
            "assert.strictEqual(A.parseHash('#/batches?verdict=accept').queueReason, '');\n"
            "console.log('HASH-OK');\n"
        )
        r = subprocess.run([node, "-e", script],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0,
                         f"hash node cases failed:\n{r.stdout}\n{r.stderr}")
        self.assertIn("HASH-OK", r.stdout)

    def test_bound_caption_is_shared(self):
        app = (WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn("A.boundCaption", app,
                      "stats row, caption and aria-label must share one helper")
        node = shutil.which("node")
        if node is None:
            self.skipTest("node not on PATH")
        script = (
            "const A = require(" + repr(str(WEB / "api.js")) + ");\n"
            "const assert = require('node:assert');\n"
            "assert.strictEqual(A.boundCaption(0.07), null);\n"
            "assert.strictEqual(A.boundCaption(undefined), 'bound not reported');\n"
            "assert.strictEqual(A.boundCaption(null), 'not certified (no threshold meets the target)');\n"
            "assert.strictEqual(A.boundCaption('x'), 'bound not reported');\n"
            "console.log('CAPTION-OK');\n"
        )
        r = subprocess.run([node, "-e", script],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0,
                         f"caption node cases failed:\n{r.stdout}\n{r.stderr}")
        self.assertIn("CAPTION-OK", r.stdout)

    def test_funnel_counts_other(self):
        app = (WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn("is-other", app, "funnel must render an 'other' segment")
        self.assertIn("other-count", app)

    def test_no_legacy_layout_traps_in_css(self):
        css = (WEB / "style.css").read_text(encoding="utf-8")
        for dead in [".tl-scroll", ".tl-step", "pre.pre", "130vw", "150vw"]:
            self.assertNotIn(dead, css, f"dead layout trap remains: {dead}")
        self.assertNotRegex(css, r"overflow-x:\s*auto",
                            "no horizontal-scroll containers may remain")
        self.assertNotRegex(css, r"min-width:\s*[^;]*vw",
                            "viewport-width min-widths risk sideways scroll")


if __name__ == "__main__":
    unittest.main(verbosity=2)
