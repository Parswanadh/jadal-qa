"""PoC dashboard server: tiny stdlib HTTP server on 127.0.0.1:8840 (read-only).

Serves web/poc/ (no build step, no CDN) and a JSON API over
poc_data/{summary.json,windows.jsonl,stream_events.jsonl,thumbs/}.

  python3 scripts/poc_server.py [--port 8840] [--demo-fixture] [--data-dir DIR]

--demo-fixture serves tests/fixtures/poc/ instead of the live poc_data dir.
Re-reads data files on every request (the run worker appends while we serve).
"""
import argparse
import collections
import json
import mimetypes
import os
import re
import subprocess
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repo / worktree root
WEB_DIR = os.path.join(ROOT, "web", "poc")
REAL_DATA = "/home/parshu/projects/jadal-qa-wt/poc_data"
FIXTURE_DATA = os.path.join(ROOT, "tests", "fixtures", "poc")

WINDOW_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
THUMB_RE = re.compile(r"^[A-Za-z0-9_.\-]+\.jpg$")
MAX_PER_PAGE = 100
DEFAULT_PER_PAGE = 24

DATA_DIR = REAL_DATA
DEMO_MODE = False


def data_path(name):
    return os.path.join(DATA_DIR, name)


def using_demo():
    return DEMO_MODE or not os.path.exists(data_path("summary.json"))


def read_summary():
    path = data_path("summary.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def iter_windows():
    """Yield window dicts in file order. Files are small (~4k lines max); a
    single streaming pass keeps memory flat and works while appended to."""
    path = data_path("windows.jsonl")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def tail_windows(limit):
    """Efficient tail read: last `limit` windows without parsing the head."""
    path = data_path("windows.jsonl")
    if not os.path.exists(path):
        return []
    dq = collections.deque(maxlen=limit)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                dq.append(line)
    out = []
    for line in dq:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def query_windows(params):
    verdict = params.get("verdict", "")
    reason = params.get("reason", "")
    factory = params.get("factory", "")
    worker = params.get("worker", "")
    stage = params.get("stage", "")
    q = params.get("q", "").strip()
    sort = params.get("sort", "time")
    order = params.get("order", "desc")
    try:
        page = max(1, int(params.get("page", "1")))
    except ValueError:
        page = 1
    try:
        per_page = min(MAX_PER_PAGE, max(1, int(params.get("per_page", str(DEFAULT_PER_PAGE)))))
    except ValueError:
        per_page = DEFAULT_PER_PAGE

    unfiltered = not (verdict or reason or factory or worker or stage or q)
    items = None
    if unfiltered and sort == "time" and order == "desc":
        # latest-first default view: tail-read only what the page needs
        items = tail_windows(page * per_page)
        items.reverse()
        total = count_windows()
        page_items = items[(page - 1) * per_page:page * per_page]
        return {"total": total, "page": page, "per_page": per_page,
                "windows": page_items, "facets": collect_facets()}

    def keep(w):
        if verdict and w.get("verdict") != verdict:
            return False
        if factory and w.get("factory") != factory:
            return False
        if worker and w.get("worker") != worker:
            return False
        if stage and w.get("stage") != stage:
            return False
        if reason and reason not in [r.get("code") for r in w.get("reasons", [])]:
            return False
        if q and q.lower() not in str(w.get("window_id", "")).lower():
            return False
        return True

    items = [w for w in iter_windows() if keep(w)]
    if sort == "score":
        items.sort(key=lambda w: (w.get("score") is None, w.get("score") or 0),
                   reverse=(order == "desc"))
    else:
        items.sort(key=lambda w: (w.get("start_s") or 0, w.get("window_id", "")),
                   reverse=(order == "desc"))
    total = len(items)
    page_items = items[(page - 1) * per_page:page * per_page]
    return {"total": total, "page": page, "per_page": per_page,
            "windows": page_items, "facets": collect_facets()}


def collect_facets():
    """Distinct filter values in one streaming pass (cheap; reused by UI)."""
    factories, workers, reasons = set(), set(), set()
    for w in iter_windows():
        if w.get("factory"):
            factories.add(w["factory"])
        if w.get("worker"):
            workers.add(w["worker"])
        for r in w.get("reasons", []):
            if r.get("code"):
                reasons.add(r["code"])
    return {"factories": sorted(factories), "workers": sorted(workers),
            "reasons": sorted(reasons)}


def count_windows():
    n = 0
    for _ in iter_windows():
        n += 1
    return n


def find_window(window_id):
    for w in iter_windows():
        if w.get("window_id") == window_id:
            return w
    return None


def read_stream(limit=2000):
    path = data_path("stream_events.jsonl")
    if not os.path.exists(path):
        return None
    dq = collections.deque(maxlen=limit)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                dq.append(line)
    out = []
    for line in dq:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "PocDash/1.0"

    def log_message(self, fmt, *args):  # quieter logs
        pass

    # -- helpers ---------------------------------------------------------
    def send_json(self, obj, status=200):
        blob = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def send_file(self, path, content_type=None):
        try:
            with open(path, "rb") as fh:
                blob = fh.read()
        except OSError:
            self.send_error(404, "not found")
            return
        ctype = content_type or mimetypes.guess_type(path)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        if path.endswith((".html", ".js", ".css")):
            self.send_header("Cache-Control", "no-store")
        else:
            self.send_header("Cache-Control", "max-age=60")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    # -- routes ----------------------------------------------------------
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        params = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}

        if route in ("/", "/index.html"):
            return self.send_file(os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8")
        if route == "/api/summary":
            s = read_summary()
            if s is None:
                return self.send_json({"error": "no data yet (run worker has not written summary.json)"},
                                      status=503)
            s = dict(s)
            s["source"] = "demo-fixture" if using_demo() else "poc_data"
            s["has_stream"] = os.path.exists(data_path("stream_events.jsonl"))
            # Live stage split (time-to-verdict): computed from windows.jsonl
            # in one streaming pass so the chart stays fresh as the run appends.
            by_stage = {"S0": 0, "S1": 0, "S2": 0}
            s0_reject = 0
            n_reject = 0
            for w in iter_windows():
                st = w.get("stage")
                if st in by_stage:
                    by_stage[st] += 1
                if w.get("verdict") == "reject":
                    n_reject += 1
                    if st == "S0":
                        s0_reject += 1
            s["live"] = {"by_stage": by_stage,
                         "early_reject_share": (s0_reject / n_reject) if n_reject else 0.0}
            return self.send_json(s)
        if route == "/api/windows":
            return self.send_json(query_windows(params))
        if route.startswith("/api/windows/"):
            wid = urllib.parse.unquote(route[len("/api/windows/"):])
            if not WINDOW_ID_RE.match(wid):
                return self.send_json({"error": "bad window id"}, status=400)
            w = find_window(wid)
            if w is None:
                return self.send_json({"error": "not found"}, status=404)
            return self.send_json(w)
        if route == "/api/stream":
            ev = read_stream()
            if ev is None:
                return self.send_json({"events": [], "available": False})
            return self.send_json({"events": ev, "available": True})
        if route.startswith("/thumbs/"):
            name = urllib.parse.unquote(route[len("/thumbs/"):])
            if "/" in name or not THUMB_RE.match(name):
                return self.send_json({"error": "bad thumbnail name"}, status=400)
            base = os.path.realpath(data_path("thumbs"))
            target = os.path.realpath(os.path.join(base, name))
            if os.path.commonpath([base, target]) != base or not os.path.isfile(target):
                return self.send_json({"error": "not found"}, status=404)
            return self.send_file(target, "image/jpeg")
        # static under web/poc/
        if route.startswith("/") and ".." not in route:
            cand = os.path.realpath(os.path.join(WEB_DIR, route.lstrip("/")))
            if cand.startswith(os.path.realpath(WEB_DIR)) and os.path.isfile(cand):
                return self.send_file(cand)
        return self.send_error(404, "not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        if route.startswith("/api/reverify/"):
            wid = urllib.parse.unquote(route[len("/api/reverify/"):])
            if not WINDOW_ID_RE.match(wid):
                return self.send_json({"error": "bad window id (allowed: A-Za-z0-9_.-)"}, status=400)
            script = os.path.join(ROOT, "scripts", "poc_run.py")
            if not os.path.isfile(script):
                return self.send_json({"status": "unavailable",
                                       "message": "Re-verify is not available yet: the run worker has not provided scripts/poc_run.py."},
                                      status=503)
            try:
                proc = subprocess.run(
                    ["python3", script, "--verify", wid],
                    capture_output=True, text=True, timeout=60, cwd=ROOT)
            except subprocess.TimeoutExpired:
                return self.send_json({"status": "timeout",
                                       "message": "poc_run.py --verify timed out after 60 s."},
                                      status=504)
            return self.send_json({"status": "ok" if proc.returncode == 0 else "mismatch",
                                   "returncode": proc.returncode,
                                   "stdout": proc.stdout[-4000:],
                                   "stderr": proc.stderr[-4000:]})
        return self.send_error(404, "not found")


def main():
    global DATA_DIR, DEMO_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8840)
    ap.add_argument("--demo-fixture", action="store_true")
    ap.add_argument("--data-dir", default=REAL_DATA)
    args = ap.parse_args()
    DEMO_MODE = args.demo_fixture
    DATA_DIR = FIXTURE_DATA if args.demo_fixture else args.data_dir
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"poc dashboard on http://127.0.0.1:{args.port}/ data={DATA_DIR}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
