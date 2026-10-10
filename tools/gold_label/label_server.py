"""Standalone gold-label page (stdlib only; issue #26, Phase 3).

Serves 127.0.0.1:8811. Per segment it shows a contact sheet of 6 evenly
spaced frames (JPEGs generated on demand with OpenCV, cached under
data_runtime/gold/), big GOOD / BAD buttons + keys (g / b / s=skip /
u=undo), a reason dropdown (existing reason codes) shown when BAD is
chosen, a free-text note, and a progress counter with ETA. It NEVER shows
any model verdict (avoids bias). Every click is saved atomically
(temp file + os.replace) to eval/gold/labels.json as append-only history
plus a latest-per-segment map, so the page is resume-safe.

Usage:
    PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python \\
        tools/gold_label/label_server.py [--port 8811]
    then open http://127.0.0.1:8811/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent.parent
SEGMENTS_PATH = ROOT / "eval" / "gold" / "segments.json"
LABELS_PATH = ROOT / "eval" / "gold" / "labels.json"
FRAMES_DIR = ROOT / "data_runtime" / "gold"

REASON_CODES = [
    "UNDECODABLE", "HANDS_NOT_VISIBLE", "BLUR", "TOO_DARK", "OVEREXPOSED",
    "SHAKY", "DUPLICATE", "WRONG_STEP", "NO_ACTIVE_WORK", "LOW_CONFIDENCE",
]
N_FRAMES = 6
FRAME_WIDTH = 360


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    """Resume-safe label store: append-only history + latest map."""

    def __init__(self, segments_path: Path, labels_path: Path):
        with open(segments_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        self.segments: list[dict] = manifest["segments"]
        self.by_id = {s["segment_id"]: s for s in self.segments}
        self.labels_path = labels_path
        self.history: list[dict] = []
        self.latest: dict[str, dict] = {}
        self.lock = threading.Lock()
        if labels_path.is_file():
            with open(labels_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.history = payload.get("history", [])
            self._rebuild_latest()

    def _rebuild_latest(self) -> None:
        self.latest = {}
        for entry in self.history:
            self.latest[entry["segment_id"]] = entry

    def _save_locked(self) -> None:
        payload = {"history": self.history, "latest": self.latest}
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self.labels_path.parent), prefix="labels_", suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1)
                fh.write("\n")
            os.replace(tmp_path, self.labels_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def add(self, segment_id: str, verdict: str, reason: str | None,
            note: str) -> dict:
        with self.lock:
            if segment_id not in self.by_id:
                raise KeyError(f"unknown segment {segment_id!r}")
            if verdict not in ("good", "bad"):
                raise ValueError("verdict must be good|bad")
            if verdict == "bad" and reason not in REASON_CODES:
                raise ValueError("bad verdict needs a valid reason code")
            entry = {"ts": utcnow(), "segment_id": segment_id,
                     "verdict": verdict,
                     "reason": reason if verdict == "bad" else None,
                     "note": note or ""}
            self.history.append(entry)
            self.latest[segment_id] = entry
            self._save_locked()
            return entry

    def undo(self) -> dict | None:
        with self.lock:
            if not self.history:
                return None
            undone = self.history.pop()
            self._rebuild_latest()
            self._save_locked()
            return undone

    def state(self) -> dict:
        with self.lock:
            return {"n": len(self.segments), "labeled": len(self.latest),
                    "segments": self.segments, "latest": self.latest}


def frame_path(frames_dir: Path, segment_id: str, k: int) -> Path:
    return frames_dir / f"{segment_id}_f{k}.jpg"


def render_frame(segment: dict, k: int, frames_dir: Path) -> Path:
    """Return the cached JPEG for frame k (generate on demand with OpenCV)."""
    out = frame_path(frames_dir, segment["segment_id"], k)
    if out.is_file():
        return out
    import cv2
    start, end = float(segment["start_s"]), float(segment["end_s"])
    t = start + (k + 0.5) * (end - start) / N_FRAMES
    cap = cv2.VideoCapture(segment["path"])
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"cannot read t={t}s from {segment['path']}")
    finally:
        cap.release()
    height, width = frame.shape[:2]
    small = cv2.resize(frame, (FRAME_WIDTH, FRAME_WIDTH * height // width))
    frames_dir.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", small,
                           [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(frames_dir), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            fh.write(bytes(buf))
        os.replace(tmp_path, out)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return out


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Jadal gold labels (no model verdicts shown)</title>
<style>
body{font-family:sans-serif;max-width:1100px;margin:0 auto;padding:16px;}
#grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:12px 0;}
#grid img{width:100%;display:block;background:#111;}
button{font-size:22px;padding:14px 34px;margin:4px;cursor:pointer;}
#good{background:#cfe8cf;} #bad{background:#f3c9c9;}
#meta{color:#444;margin:8px 0;} #prog{font-size:18px;margin:8px 0;}
select,input[type=text]{font-size:16px;padding:6px;margin:4px;}
#reasonRow{display:none;}
</style>
</head>
<body>
<h2>Gold label <span id="segid"></span> <small>(unbiased: no model output shown)</small></h2>
<div id="prog"></div>
<div id="meta"></div>
<div>
<button id="good">GOOD (g)</button><button id="bad">BAD (b)</button>
<button id="skip">skip (s)</button><button id="undo">undo (u)</button>
<button id="prev">&lt; prev</button><button id="next">next &gt;</button>
<button id="first">first unlabeled</button>
</div>
<div id="grid"></div>
<div id="reasonRow">Reason (required for BAD):
<select id="reason"><option value="">-- pick --</option></select>
</div>
<div>Note: <input type="text" id="note" size="60" placeholder="optional free text"></div>
<div id="msg"></div>
<script>
const REASONS = __REASONS__;
let S = null, idx = 0, t0 = Date.now();
const reasonSel = document.getElementById('reason');
REASONS.forEach(c => { const o = document.createElement('option'); o.value = c; o.textContent = c; reasonSel.appendChild(o); });
async function load() {
  const r = await fetch('/api/state'); S = await r.json();
  idx = firstUnlabeled(); render();
}
function firstUnlabeled() {
  for (let i = 0; i < S.segments.length; i++)
    if (!S.latest[S.segments[i].segment_id]) return i;
  return S.segments.length - 1;
}
function render() {
  const seg = S.segments[idx];
  document.getElementById('segid').textContent = seg.segment_id + '  ' + seg.clip + '  ' + seg.start_s + 's-' + seg.end_s + 's';
  const lab = S.latest[seg.segment_id];
  document.getElementById('meta').textContent = lab ? ('current label: ' + lab.verdict.toUpperCase() + (lab.reason ? ' / ' + lab.reason : '')) : 'current label: —';
  const grid = document.getElementById('grid'); grid.innerHTML = '';
  for (let k = 0; k < 6; k++) {
    const img = document.createElement('img');
    img.src = '/frame?seg=' + seg.segment_id + '&k=' + k;
    grid.appendChild(img);
  }
  const n = Object.keys(S.latest).length;
  const el = (Date.now() - t0) / 1000, rate = n / Math.max(el, 1);
  const eta = rate > 0 ? Math.round((S.n - n) / rate / 60) : '?';
  document.getElementById('prog').textContent = 'labeled ' + n + '/' + S.n + '  (~' + eta + ' min left)';
  document.getElementById('reasonRow').style.display = 'block';
  document.getElementById('note').value = lab ? lab.note : '';
  if (lab && lab.reason) reasonSel.value = lab.reason; else reasonSel.value = '';
}
async function send(verdict) {
  const seg = S.segments[idx];
  let reason = reasonSel.value || null;
  if (verdict === 'bad' && !reason) { msg('BAD needs a reason code'); return; }
  const note = document.getElementById('note').value;
  const r = await fetch('/api/label', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({segment_id: seg.segment_id, verdict, reason, note})});
  if (!r.ok) { msg('save failed: ' + await r.text()); return; }
  const e = await r.json(); S.latest[e.segment_id] = e; msg('saved ' + e.verdict);
  step(1);
}
function step(d) { idx = Math.min(S.segments.length - 1, Math.max(0, idx + d)); render(); window.scrollTo(0, 0); }
async function undo() {
  const r = await fetch('/api/undo', {method: 'POST'});
  const u = await r.json();
  if (u && u.undone) { await load(); msg('undid ' + u.undone.segment_id); } else msg('nothing to undo');
}
function msg(t) { document.getElementById('msg').textContent = t; }
document.getElementById('good').onclick = () => send('good');
document.getElementById('bad').onclick = () => send('bad');
document.getElementById('skip').onclick = () => step(1);
document.getElementById('undo').onclick = undo;
document.getElementById('prev').onclick = () => step(-1);
document.getElementById('next').onclick = () => step(1);
document.getElementById('first').onclick = () => { idx = firstUnlabeled(); render(); };
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'g') send('good'); else if (e.key === 'b') send('bad');
  else if (e.key === 's') step(1); else if (e.key === 'u') undo();
});
load();
</script>
</body>
</html>
""".replace("__REASONS__", json.dumps(REASON_CODES))


class Handler(BaseHTTPRequestHandler):
    store: Store
    frames_dir: Path = FRAMES_DIR

    def log_message(self, *args):  # keep test output clean
        pass

    def _json(self, obj: object, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/state":
            self._json(self.store.state())
        elif parsed.path == "/frame":
            query = parse_qs(parsed.query)
            seg_id = (query.get("seg", [None]))[0]
            try:
                k = int((query.get("k", ["0"]))[0])
            except ValueError:
                k = -1
            segment = self.store.by_id.get(seg_id)
            if segment is None or not 0 <= k < N_FRAMES:
                self.send_error(404, "unknown segment/frame")
                return
            try:
                path = render_frame(segment, k, self.frames_dir)
            except Exception as exc:  # noqa: BLE001
                self.send_error(500, f"frame render failed: {exc}")
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/undo":
            self._json({"undone": self.store.undo()})
            return
        if parsed.path != "/api/label":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            entry = self.store.add(
                payload.get("segment_id", ""), payload.get("verdict", ""),
                payload.get("reason"), payload.get("note", ""))
        except (KeyError, ValueError) as exc:
            self._json({"error": str(exc)}, status=400)
            return
        self._json(entry)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8811)
    parser.add_argument("--segments", type=Path, default=SEGMENTS_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    args = parser.parse_args(argv)
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    Handler.store = Store(args.segments, args.labels)
    Handler.frames_dir = args.frames_dir
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"gold labels: http://127.0.0.1:{args.port}/ "
          f"({len(Handler.store.segments)} segments, "
          f"{len(Handler.store.latest)} already labeled)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
