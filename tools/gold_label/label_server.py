"""Gold-label page with video player (stdlib only).

Serves 127.0.0.1:8811. Per segment it shows an on-demand 10 s video cut
(<video autoplay loop muted playsinline controls> + speed selector), a
dataset task-context line (IndEgo annotated key steps overlapping the
window — ground truth, NOT a model verdict), a GOOD/BAD definition panel,
the 6-frame contact sheet inside a collapsed <details>, big GOOD / BAD
buttons + keys (g / b / s=skip / u=undo, arrows, space, 1-8 = BAD reason),
and a progress bar with ETA from the real median seconds per label.
It NEVER shows any model verdict (avoids bias). Every click is saved
atomically (temp file + os.replace) as append-only history plus a
latest-per-segment map, so the page is resume-safe. Existing entries in
--labels are always preserved and reused, never deleted or rewritten.

Usage:
    PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python \\
        tools/gold_label/label_server.py \\
        --labels /home/parshu/projects/jadal-qa-wt/funnel/eval/gold/labels.json \\
        [--manifest eval/gold/segments.json] [--port 8811]
    then open http://127.0.0.1:8811/
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent.parent
SEGMENTS_PATH = ROOT / "eval" / "gold" / "segments.json"
LABELS_PATH = ROOT / "eval" / "gold" / "labels.json"
FRAMES_DIR = ROOT / "data_runtime" / "gold"
VIDEO_DIR = ROOT / "data_runtime" / "gold" / "video"

INDEGO_ROOT = Path("/home/parshu/projects/awing/data/indego")
ANNOTATION_DIRS = [
    INDEGO_ROOT / "1_Assembly" / "annotated_keysteps_assembly_disassembly",
    INDEGO_ROOT / "3_Logistics_Organization"
    / "annotated_keysteps_logistics_organisation",
]

# (code, one-line example). Order here defines the 1-8 shortcut numbers.
REASONS: list[tuple[str, str]] = [
    ("HANDS_NOT_VISIBLE",
     "task needs hands but they are out of view most of the segment"),
    ("BLUR", "motion blur / out of focus, detail unreadable"),
    ("TOO_DARK", "too dark to see the work area"),
    ("OVEREXPOSED", "washed out / blown highlights hide the work"),
    ("SHAKY", "camera shaking so hard the scene is unusable"),
    ("NO_ACTIVE_WORK", "walking / idle / nothing task-relevant happening"),
    ("UNDECODABLE", "blocked camera / lens covered / stream corrupt"),
    ("DUPLICATE", "frozen / repeated frames, no real motion"),
]
REASON_CODES = [code for code, _ in REASONS]

# Dataset task ground truth per manifest clip id (from CATALOG.md).
CLIP_TASKS = {
    "ecu_full": ("ECU assembly", True),
    "ecu_late": ("ECU assembly (late steps)", True),
    "desk": ("mechanical desk assembly (final steps)", True),
    "pallet2": ("logistics \u2014 pallet transport", False),
    "pallet": ("logistics \u2014 pallet transport", False),
    "organise": ("logistics \u2014 organise / clean-up", True),
    "logistics4": ("logistics (unlabelled folder clip)", False),
}

N_FRAMES = 6
FRAME_WIDTH = 360
VIDEO_CRF = "28"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Label store (unchanged semantics: append-only history + latest, atomic save)
# --------------------------------------------------------------------------

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
            # Preserve every existing entry; tolerate either shape.
            self.history = payload.get("history", [])
            self._rebuild_latest()

    def _rebuild_latest(self) -> None:
        self.latest = {}
        for entry in self.history:
            self.latest[entry["segment_id"]] = entry

    def _save_locked(self) -> None:
        payload = {"history": self.history, "latest": self.latest}
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
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

    def median_seconds_per_label(self) -> float | None:
        """Median gap between consecutive label timestamps (capped at 10 min)."""
        with self.lock:
            stamps = [e.get("ts") for e in self.history]
        gaps: list[float] = []
        prev: datetime | None = None
        for ts in stamps:
            try:
                cur = datetime.fromisoformat(ts)
            except (TypeError, ValueError):
                continue
            if prev is not None:
                gap = (cur - prev).total_seconds()
                if 0 < gap <= 600:
                    gaps.append(gap)
            prev = cur
        if not gaps:
            return None
        gaps.sort()
        mid = len(gaps) // 2
        if len(gaps) % 2:
            return gaps[mid]
        return (gaps[mid - 1] + gaps[mid]) / 2.0

    def state(self) -> dict:
        with self.lock:
            counts = {"good": 0, "bad": 0}
            for entry in self.latest.values():
                if entry.get("verdict") in counts:
                    counts[entry["verdict"]] += 1
            return {"n": len(self.segments), "labeled": len(self.latest),
                    "n_good": counts["good"], "n_bad": counts["bad"],
                    "segments": self.segments, "latest": self.latest}


# --------------------------------------------------------------------------
# Task context: VIA annotation overlap lookup
# --------------------------------------------------------------------------

def parse_via_steps(via: dict) -> list[tuple[float, float, str]]:
    """Extract (start_s, end_s, step_label) from a VIA project dict."""
    attrs = via.get("attribute", {})
    options: dict[str, str] = {}
    for attr in attrs.values():
        if isinstance(attr, dict) and isinstance(attr.get("options"), dict):
            options = {str(k): str(v)
                        for k, v in attr["options"].items()}
            break
    out: list[tuple[float, float, str]] = []
    for meta in via.get("metadata", {}).values():
        try:
            start, end = float(meta["z"][0]), float(meta["z"][1])
            opt_id = str(meta["av"]["1"])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        out.append((start, end, options.get(opt_id, opt_id)))
    out.sort()
    return out


def _annotation_candidate(clip_path: str) -> list[Path]:
    """Candidate VIA JSON files for a clip (assembly + logistics dirs)."""
    stem = Path(clip_path).stem  # e.g. User_15_410_0411_2_480
    short = stem[:-4] if stem.endswith("_480") else stem
    cands = []
    for directory in ANNOTATION_DIRS:
        cands.append(directory / f"AS_{short}.json")
    return cands


def load_step_index(clip_paths: list[str]) -> dict[str, list[tuple]]:
    """Map clip path -> sorted [(start, end, step)] (missing file -> [])."""
    index: dict[str, list[tuple]] = {}
    for clip_path in clip_paths:
        steps: list[tuple] = []
        for cand in _annotation_candidate(clip_path):
            if cand.is_file():
                try:
                    with open(cand, encoding="utf-8") as fh:
                        steps = parse_via_steps(json.load(fh))
                except (json.JSONDecodeError, OSError):
                    steps = []
                break
        index[clip_path] = steps
    return index


def steps_in_window(index: dict[str, list[tuple]], clip_path: str,
                    start: float, end: float) -> list[str]:
    """Step labels whose [s, e) overlaps [start, end); order-preserved, deduped."""
    found: list[str] = []
    for s, e, label in index.get(clip_path, []):
        if s < end and e > start and label not in found:
            found.append(label)
    return found


def task_line(segment: dict, index: dict[str, list[tuple]]) -> str:
    """Human-readable dataset-ground-truth line for a segment."""
    clip_id = segment.get("clip", "")
    task, has_steps = CLIP_TASKS.get(clip_id, ("unknown", False))
    if task == "unknown":
        # Fall back to folder name from the clip path.
        m = re.search(r"data/indego/([^/]+)/", segment.get("path", ""))
        task = m.group(1).replace("_", " ") if m else "unknown"
        has_steps = bool(index.get(segment.get("path", "")))
    if has_steps:
        steps = steps_in_window(index, segment.get("path", ""),
                                float(segment["start_s"]),
                                float(segment["end_s"]))
        if steps:
            return f"Task: {task} \u00b7 annotated step(s) in this window: " \
                + ", ".join(steps)
        return f"Task: {task} \u00b7 between annotated steps"
    return f"Task: {task}"


# --------------------------------------------------------------------------
# Frames + video cutting (ffmpeg)
# --------------------------------------------------------------------------

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


def video_path(video_dir: Path, segment_id: str) -> Path:
    return video_dir / f"{segment_id}.mp4"


_video_locks: dict[str, threading.Lock] = {}
_video_locks_guard = threading.Lock()


def ensure_video(segment: dict, video_dir: Path) -> Path:
    """Cut (and cache) the 10 s clip with ffmpeg; returns the mp4 path."""
    out = video_path(video_dir, segment["segment_id"])
    if out.is_file() and out.stat().st_size > 0:
        return out
    with _video_locks_guard:
        lock = _video_locks.setdefault(segment["segment_id"],
                                       threading.Lock())
    with lock:
        if out.is_file() and out.stat().st_size > 0:
            return out
        video_dir.mkdir(parents=True, exist_ok=True)
        dur = float(segment["end_s"]) - float(segment["start_s"])
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(video_dir),
                                            prefix="cut_", suffix=".tmp.mp4")
        os.close(tmp_fd)
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y",
                 "-ss", str(float(segment["start_s"])),
                 "-t", str(dur),
                 "-i", segment["path"],
                 "-an", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", VIDEO_CRF, "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", tmp_path],
                capture_output=True, text=True, timeout=120)
            if proc.returncode != 0 or not Path(tmp_path).is_file():
                raise RuntimeError(
                    f"ffmpeg failed: {proc.stderr.strip()[-500:]}")
            os.replace(tmp_path, out)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    return out


PREFETCH = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")


def prefetch_next(store: Store, video_dir: Path, seg_id: str,
                  count: int = 3) -> None:
    """Cut the next `count` segments in a background thread (cache warm)."""
    try:
        ids = [s["segment_id"] for s in store.segments]
        pos = ids.index(seg_id)
    except ValueError:
        return

    def _work() -> None:
        for nxt in ids[pos + 1:pos + 1 + count]:
            try:
                ensure_video(store.by_id[nxt], video_dir)
            except Exception:  # noqa: BLE001 - prefetch must never break serving
                pass

    PREFETCH.submit(_work)


# --------------------------------------------------------------------------
# HTTP Range helper
# --------------------------------------------------------------------------

def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Parse a single `bytes=start-end` Range header.

    Returns (start, end_inclusive) or None if absent/invalid.
    Suffix ranges (`bytes=-N`) and open ranges (`bytes=N-`) supported.
    """
    if not header or not header.startswith("bytes="):
        return None
    spec = header[len("bytes="):].strip()
    if "," in spec:  # multipart not supported
        return None
    m = re.fullmatch(r"(\d*)-(\d*)", spec)
    if not m or (not m.group(1) and not m.group(2)):
        return None
    if not m.group(1):  # suffix: last N bytes
        n = int(m.group(2))
        if n <= 0:
            return None
        start = max(0, size - n)
        return (start, size - 1)
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else size - 1
    if start >= size:
        return None
    return (start, min(end, size - 1))


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Jadal gold labels (no model verdicts shown)</title>
<style>
body{font-family:sans-serif;max-width:720px;margin:0 auto;padding:16px 16px 64px;
  background:#fafaf8;color:#222;line-height:1.45;}
video{width:480px;max-width:100%;display:block;background:#111;border-radius:8px;}
button{font-size:22px;padding:14px 30px;margin:6px 4px;cursor:pointer;border-radius:8px;border:1px solid #999;}
#good{background:#cfe8cf;} #bad{background:#f3c9c9;}
#progWrap{background:#e4e4e4;border-radius:8px;height:22px;margin:10px 0;overflow:hidden;}
#progBar{background:#5a9e5a;height:100%;width:0%;}
#prog{font-size:17px;margin:6px 0;} #counts{font-size:16px;color:#333;margin:4px 0;}
#taskline{font-size:17px;background:#eef3ff;border:1px solid #bcd;padding:8px 10px;border-radius:8px;margin:8px 0;}
details{margin:10px 0;border:1px solid #ccc;border-radius:8px;padding:8px 10px;background:#fff;}
summary{font-size:17px;cursor:pointer;font-weight:bold;}
table{border-collapse:collapse;margin-top:8px;font-size:15px;}
td,th{border:1px solid #ccc;padding:4px 8px;text-align:left;}
#grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:12px 0;}
#grid img{width:100%;display:block;background:#111;}
select{font-size:17px;padding:8px;margin:4px;}
input[type=text]{font-size:16px;padding:8px;margin:4px;width:90%;}
#reasonRow{margin:8px 0;}
#msg{min-height:1.5em;color:#064;}
kbd{background:#eee;border:1px solid #bbb;border-radius:4px;padding:0 5px;font-size:14px;}
</style>
</head>
<body>
<h2>Gold label <span id="segid"></span></h2>
<div id="progWrap"><div id="progBar"></div></div>
<div id="prog"></div>
<div id="counts"></div>
<div id="taskline"></div>
<details id="defn">
<summary>What counts as GOOD / BAD? (read me first)</summary>
<p>Imagine you are a QA reviewer at a data company. <b>GOOD</b> = usable
training data of someone doing the task: camera works (sharp enough, not too
dark or washed out, not blocked), the work area is in view, and the activity
is task-relevant. <b>BAD</b> = not usable (choose a reason). <b>SKIP</b> =
you honestly cannot tell. Do not guess; do not be nice; label what you would
really reject. <i>No model output is shown on this page \u2014 judge the footage only.</i></p>
<table>
<tr><th>#</th><th>Reason</th><th>Example</th></tr>
__REASON_ROWS__
</table>
</details>
<div>
<video id="vid" autoplay loop muted playsinline controls preload="auto"></video>
<div>Speed: <select id="speed">
<option value="1">1x</option><option value="1.5">1.5x</option><option value="2">2x</option><option value="3">3x</option><option value="4">4x</option>
</select></div>
</div>
<div>
<button id="good">GOOD (<kbd>g</kbd>)</button><button id="bad">BAD (<kbd>b</kbd>)</button>
<button id="skip">skip (<kbd>s</kbd>)</button><button id="undo">undo (<kbd>u</kbd>)</button>
<br><button id="prev">&lt; prev</button><button id="next">next &gt;</button>
<button id="first">first unlabeled</button>
</div>
<details id="sheet">
<summary>Contact sheet (6 still frames)</summary>
<div id="grid"></div>
</details>
<div id="reasonRow">Reason (required for BAD, or press <kbd>1</kbd>\u2013<kbd>8</kbd>):
<select id="reason"><option value="">-- pick --</option></select>
</div>
<div>Note: <input type="text" id="note" placeholder="optional free text"></div>
<div id="msg"></div>
<div><small>Keys: <kbd>g</kbd> good, <kbd>b</kbd> bad, <kbd>s</kbd> skip,
<kbd>u</kbd> undo, <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> prev/next,
<kbd>space</kbd> play/pause, <kbd>1</kbd>\u2013<kbd>8</kbd> pick BAD reason &amp; save.</small></div>
<script>
const REASONS = __REASONS__;
let S = null, idx = 0, t0 = Date.now(), nSkip = 0;
const reasonSel = document.getElementById('reason');
REASONS.forEach((c, i) => { const o = document.createElement('option'); o.value = c; o.textContent = (i+1) + ': ' + c; reasonSel.appendChild(o); });
const vid = document.getElementById('vid');
const speedSel = document.getElementById('speed');
speedSel.value = localStorage.getItem('gold_speed') || '1';
function applySpeed() { const r = parseFloat(speedSel.value) || 1; vid.defaultPlaybackRate = r; vid.playbackRate = r; }
speedSel.onchange = () => { localStorage.setItem('gold_speed', speedSel.value); applySpeed(); };
['loadstart', 'loadedmetadata', 'canplay', 'playing'].forEach(ev => vid.addEventListener(ev, applySpeed));
const defn = document.getElementById('defn');
if (localStorage.getItem('gold_defn_seen')) { defn.open = false; } else { defn.open = true; }
defn.addEventListener('toggle', () => { if (!defn.open) localStorage.setItem('gold_defn_seen', '1'); });
async function load() {
  const r = await fetch('/api/state'); S = await r.json();
  idx = firstUnlabeled(); render();
}
function firstUnlabeled() {
  for (let i = 0; i < S.segments.length; i++)
    if (!S.latest[S.segments[i].segment_id]) return i;
  return S.segments.length - 1;
}
function nextUnlabeled(from) {
  for (let i = from + 1; i < S.segments.length; i++)
    if (!S.latest[S.segments[i].segment_id]) return i;
  for (let i = 0; i <= from; i++)
    if (!S.latest[S.segments[i].segment_id]) return i;
  return Math.min(from + 1, S.segments.length - 1);
}
function fmtETA(sec) {
  if (sec == null) return '~? min left';
  if (sec < 90) return '~1 min left';
  if (sec < 3600) return '~' + Math.round(sec / 60) + ' min left';
  return '~' + (sec / 3600).toFixed(1) + ' h left';
}
function render() {
  const seg = S.segments[idx];
  document.getElementById('segid').textContent =
    seg.segment_id + '  (segment ' + (idx + 1) + ' of ' + S.n + ')';
  const lab = S.latest[seg.segment_id];
  document.getElementById('taskline').textContent = seg.task_line +
    '  \u00b7  ' + seg.clip + ' ' + seg.start_s + 's\u2013' + seg.end_s + 's';
  applySpeed();
  vid.src = '/video?seg=' + seg.segment_id;
  vid.play().catch(() => {});
  const grid = document.getElementById('grid'); grid.innerHTML = '';
  if (document.getElementById('sheet').open) fillSheet(seg);
  const n = Object.keys(S.latest).length;
  document.getElementById('progBar').style.width = (100 * n / S.n) + '%';
  document.getElementById('prog').textContent =
    'labeled ' + n + '/' + S.n + '  (' + fmtETA(S.eta_s) + ', ~' + S.median_s + 's/label)';
  document.getElementById('counts').textContent =
    'good: ' + S.n_good + '   bad: ' + S.n_bad + '   skipped (this session): ' + nSkip;
  document.getElementById('note').value = lab ? lab.note : '';
  if (lab && lab.reason) reasonSel.value = lab.reason; else reasonSel.value = '';
}
function fillSheet(seg) {
  const grid = document.getElementById('grid'); grid.innerHTML = '';
  for (let k = 0; k < 6; k++) {
    const img = document.createElement('img');
    img.src = '/frame?seg=' + seg.segment_id + '&k=' + k;
    img.loading = 'lazy';
    grid.appendChild(img);
  }
}
document.getElementById('sheet').addEventListener('toggle', e => {
  if (e.target.open) fillSheet(S.segments[idx]);
});
async function send(verdict) {
  const seg = S.segments[idx];
  let reason = reasonSel.value || null;
  if (verdict === 'bad' && !reason) { msg('BAD needs a reason code (pick 1-8)'); return; }
  const note = document.getElementById('note').value;
  const r = await fetch('/api/label', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({segment_id: seg.segment_id, verdict, reason, note})});
  if (!r.ok) { msg('save failed: ' + await r.text()); return; }
  const e = await r.json(); S.latest[e.segment_id] = e;
  S.n_good += (e.verdict === 'good' ? 1 : 0); S.n_bad += (e.verdict === 'bad' ? 1 : 0);
  S.labeled = Object.keys(S.latest).length;
  msg('saved ' + e.verdict.toUpperCase());
  idx = nextUnlabeled(idx); await refreshStats(); render(); window.scrollTo(0, 0);
}
async function refreshStats() {
  const r = await fetch('/api/state?light=1'); const L = await r.json();
  S.n_good = L.n_good; S.n_bad = L.n_bad; S.labeled = L.labeled;
  S.eta_s = L.eta_s; S.median_s = L.median_s;
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
document.getElementById('skip').onclick = () => { nSkip++; step(1); };
document.getElementById('undo').onclick = undo;
document.getElementById('prev').onclick = () => step(-1);
document.getElementById('next').onclick = () => step(1);
document.getElementById('first').onclick = () => { idx = firstUnlabeled(); render(); };
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === ' ') { e.preventDefault(); if (vid.paused) vid.play(); else vid.pause(); }
  else if (e.key === 'g') send('good');
  else if (e.key === 'b') send('bad');
  else if (e.key === 's') { nSkip++; step(1); }
  else if (e.key === 'u') undo();
  else if (e.key === 'ArrowLeft') step(-1);
  else if (e.key === 'ArrowRight') step(1);
  else if (e.key >= '1' && e.key <= '8') {
    reasonSel.value = REASONS[parseInt(e.key) - 1];
    send('bad');
  }
});
load();
</script>
</body>
</html>
"""

_REASON_ROWS = "\n".join(
    f"<tr><td>{i + 1}</td><td>{code}</td><td>{ex}</td></tr>"
    for i, (code, ex) in enumerate(REASONS))
_RENDERED_PAGE = (PAGE.replace("__REASONS__", json.dumps(REASON_CODES))
                      .replace("__REASON_ROWS__", _REASON_ROWS))


class Handler(BaseHTTPRequestHandler):
    store: Store
    frames_dir: Path = FRAMES_DIR
    video_dir: Path = VIDEO_DIR
    step_index: dict = {}

    def log_message(self, *args):  # keep test output clean
        pass

    def _json(self, obj: object, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path, content_type: str) -> None:
        size = path.stat().st_size
        if self.command == "HEAD":
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return
        rng = parse_range(self.headers.get("Range"), size)
        if self.headers.get("Range") and rng is None:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        start, end = rng if rng else (0, size - 1)
        length = end - start + 1
        if rng:
            self.send_response(206)
            self.send_header("Content-Range",
                             f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _state_payload(self, light: bool) -> dict:
        st = self.store.state()
        median = self.store.median_seconds_per_label()
        unlabeled = st["n"] - st["labeled"]
        if median is None:
            eta = None
            median_display = "?"
        else:
            eta = median * unlabeled
            median_display = f"{median:.0f}"
        payload = {"n": st["n"], "labeled": st["labeled"],
                   "n_good": st["n_good"], "n_bad": st["n_bad"],
                   "eta_s": round(eta) if eta is not None else None,
                   "median_s": median_display,
                   "latest": st["latest"]}
        if not light:
            payload["segments"] = [
                {**seg, "task_line": task_line(seg, self.step_index)}
                for seg in st["segments"]
            ]
        return payload

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = _RENDERED_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/state":
            query = parse_qs(parsed.query)
            self._json(self._state_payload(light=query.get("light") == ["1"]))
        elif parsed.path == "/api/prefetch":
            query = parse_qs(parsed.query)
            seg_id = (query.get("seg", [None]))[0]
            if seg_id in self.store.by_id:
                prefetch_next(self.store, self.video_dir, seg_id)
                self._json({"ok": True})
            else:
                self._json({"error": "unknown segment"}, status=404)
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
            self._serve_file(path, "image/jpeg")
        elif parsed.path == "/video":
            query = parse_qs(parsed.query)
            seg_id = (query.get("seg", [None]))[0]
            segment = self.store.by_id.get(seg_id)
            if segment is None:
                self.send_error(404, "unknown segment")
                return
            try:
                path = ensure_video(segment, self.video_dir)
            except Exception as exc:  # noqa: BLE001
                self.send_error(500, f"video cut failed: {exc}")
                return
            prefetch_next(self.store, self.video_dir, seg_id)
            self._serve_file(path, "video/mp4")
        else:
            self.send_error(404)

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/video":
            query = parse_qs(parsed.query)
            segment = self.store.by_id.get((query.get("seg", [None]))[0])
            if segment is None:
                self.send_error(404, "unknown segment")
                return
            try:
                path = ensure_video(segment, self.video_dir)
            except Exception as exc:  # noqa: BLE001
                self.send_error(500, f"video cut failed: {exc}")
                return
            self._serve_file(path, "video/mp4")
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
    parser.add_argument("--manifest", type=Path, default=None,
                        help="alias for --segments (segment manifest JSON)")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH,
                        help="labels JSON (existing entries are preserved)")
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    args = parser.parse_args(argv)
    manifest = args.manifest or args.segments
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    Handler.store = Store(manifest, args.labels)
    Handler.frames_dir = args.frames_dir
    Handler.video_dir = args.video_dir
    Handler.step_index = load_step_index(
        sorted({s["path"] for s in Handler.store.segments}))
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
