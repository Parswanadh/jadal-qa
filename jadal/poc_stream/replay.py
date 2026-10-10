"""Live stream-replay core for the PoC demo (issue #64).

Replays one real Egocentric-10K clip as if a worker were recording: the clip
is delivered to the funnel in 2-second chunks, each chunk is decoded on its
own (``ffmpeg -ss/-t``, small working resolution, O(window) memory), S0
early-reject rules are evaluated on the running window, and after every
chunk one event is appended (and flushed) to ``stream_events.jsonl``::

    {"t", "window_id", "chunk", "running_score", "decision", "reason"}

``decision`` is ``continue`` | ``early_reject`` | ``accept``.

Reuse strategy (read-only, best-effort): S0 rule thresholds are loaded by
path from the gpu-pipeline worker's ``eval/gpu/cascade.py`` (``S0_RULES``);
the streaming per-pixel path tries to load ``SpectralStream`` by path from
the spectral-features worker's ``jadal/spectral/stream.py``. Both imports are
wrapped in try/except: the other workers' trees may not be checked out, and
their module imports may assume their own ``jadal`` package first on
``sys.path``. On any failure the replay falls back to a small local
implementation of the same rules/features operating on the same decoded
frames, and records which engine was used in every event (``engine`` field).
Nothing outside OWNED PATHS is written.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

import cv2
import numpy as np

# Fallback S0 rules: identical (rule, feature, op, threshold) shape to
# eval/gpu/cascade.py::S0_RULES on origin/feat/gpu-pipeline.
S0_RULES_FALLBACK = [
    ("black", "p95", "<", 0.03),
    ("dark", "mean", "<", 0.05),
    ("lens_blocked", "spatial_var", "<", 1e-5),
    ("overexposed", "p5", ">", 0.70),
    ("frozen", "temp_e", "<", 5e-4),
]

DECISION_TO_LED = {"accept": "G", "early_reject": "R", "continue": "A"}


def load_by_path(modname, path):
    """Import a module read-only from an absolute path; raise on failure."""
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_s0_rules():
    """Best-effort read-only reuse of the gpu-pipeline S0 rules.

    Returns (rules, source_tag). Never raises: falls back to the local copy
    of the same constants.
    """
    cands = [
        "/home/parshu/projects/jadal-qa-wt/gpu-pipeline/eval/gpu/cascade.py",
        "/home/parshu/projects/jadal-qa-wt/poc-demo/eval/gpu/cascade.py",
    ]
    for path in cands:
        if not os.path.exists(path):
            continue
        try:
            mod = load_by_path("poc_s0_cascade", path)
            rules = list(mod.S0_RULES)
            if rules and all(len(r) == 4 for r in rules):
                return rules, f"cascade.py:{path}"
        except Exception:
            continue
    return list(S0_RULES_FALLBACK), "local-fallback"


def make_spectral_stream():
    """Best-effort read-only reuse of the spectral-features SpectralStream.

    Returns (stream_or_None, source_tag). The spectral module does
    ``from jadal.spectral.features import ...`` which resolves against this
    worktree's own ``jadal`` package unless the spectral-features tree is
    first on ``sys.path``; so the import is attempted with that tree
    temporarily prepended, then ``sys.path`` is restored.
    """
    tree = "/home/parshu/projects/jadal-qa-wt/spectral-features"
    spath = os.path.join(tree, "jadal", "spectral", "stream.py")
    if not os.path.exists(spath):
        return None, "local-fallback:stream.py not checked out"
    saved = list(sys.path)
    try:
        sys.path.insert(0, tree)
        for m in ("jadal.spectral.stream", "jadal.spectral.features"):
            sys.modules.pop(m, None)
        sys.modules.pop("jadal", None)
        mod = load_by_path("poc_spectral_stream", spath)
        stream = mod.SpectralStream(tier="T1")
        # Smoke-test on a dummy chunk so a broken import fails here, not
        # mid-demo.
        dummy = [np.zeros((90, 160, 3), np.uint8)]
        feats = stream.feed(dummy)
        if not isinstance(feats, dict) or not feats:
            raise ValueError("SpectralStream.feed returned no features")
        return stream, f"SpectralStream:{spath}"
    except Exception as exc:
        return None, f"local-fallback ({type(exc).__name__}: {exc})"
    finally:
        sys.path[:] = saved


# --------------------------------------------------------------------------- #
# Decode: one chunk at a time, small frames, bounded memory                    #
# --------------------------------------------------------------------------- #

def decode_chunk(clip, start_s, chunk_s, fps=5, width=160, use_cuda=False):
    """Decode [start_s, start_s+chunk_s) to a list of BGR uint8 frames.

    Returns (frames, decode_backend). Small working resolution (default
    160x90 @ 5 fps => 10 frames/chunk, ~0.4 MB) keeps RAM far under the
    1.5 GB budget. NVDEC is tried only when requested AND available;
    otherwise CPU, recorded in the return value.
    """
    height = max(2, int(round(width * 9 / 16)) // 2 * 2)
    size = f"{width}x{height}"
    vf = f"scale={width}:-1,fps={fps}"
    backend = "cpu"
    pre = []
    if use_cuda:
        # hevc_cuvid probe: fail soft to CPU.
        try:
            probe = subprocess.run(
                ["ffmpeg", "-hide_banner", "-hwaccel", "cuda",
                 "-c:v", "hevc_cuvid", "-i", clip, "-t", "0.1",
                 "-f", "null", "-"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=30)
            if probe.returncode == 0:
                pre = ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
                backend = "cuda"
        except Exception:
            pre = []
    cmd = (["ffmpeg", "-hide_banner", "-loglevel", "error"]
           + pre + ["-ss", f"{start_s:.3f}", "-t", f"{chunk_s:.3f}",
                    "-i", clip, "-vf", vf, "-f", "rawvideo",
                    "-pix_fmt", "bgr24", "-vsync", "0", "pipe:1"])
    proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed t={start_s}: "
                           f"{proc.stderr.decode()[:300]}")
    raw = proc.stdout
    npx = width * height * 3
    # Height may differ by a pixel from rounding; reshape what we can.
    nframes = len(raw) // npx
    frames = [np.frombuffer(raw[i * npx:(i + 1) * npx], np.uint8)
              .reshape(height, width, 3).copy()
              for i in range(nframes)]
    return frames, backend


# --------------------------------------------------------------------------- #
# Degradations (applied in-memory to decoded frames, from --degrade-at)        #
# --------------------------------------------------------------------------- #

def apply_degrade(frames, kind, freeze_frame=None):
    """Apply a known synthetic degradation to decoded chunk frames."""
    if kind in (None, "none"):
        return frames, freeze_frame
    if kind == "blackout":
        return [np.zeros_like(f) for f in frames], freeze_frame
    if kind == "freeze":
        if freeze_frame is None and frames:
            freeze_frame = frames[0].copy()
        base = freeze_frame if freeze_frame is not None else frames[0]
        return [base.copy() for base in [base] for _ in frames], freeze_frame
    if kind == "blur":
        return [cv2.GaussianBlur(f, (31, 31), 0) for f in frames], freeze_frame
    raise ValueError(f"unknown degradation {kind!r}")


# --------------------------------------------------------------------------- #
# Features + S0 + running score (local path, same rule shape as cascade)       #
# --------------------------------------------------------------------------- #

def window_stats(gray_frames):
    """Luma stats over the running window: mean, p5, p95, spatial_var, temp_e."""
    if not gray_frames:
        return {"mean": 0.0, "p5": 0.0, "p95": 0.0,
                "spatial_var": 0.0, "temp_e": 0.0}
    arr = np.stack(gray_frames).astype(np.float64) / 255.0
    flat = arr.ravel()
    # Quantiles on a strided sample bound memory/time on long windows.
    sample = flat[::max(1, flat.size // 200000)]
    spatial_var = float(np.mean([float(g.var()) for g in
                                 (arr * 255.0)])) / (255.0 ** 2)
    if len(gray_frames) > 1:
        diffs = np.abs(np.diff(arr, axis=0))
        temp_e = float(diffs.mean())
    else:
        temp_e = 1.0  # single frame: do not claim frozen
    return {"mean": float(arr.mean()),
            "p5": float(np.quantile(sample, 0.05)),
            "p95": float(np.quantile(sample, 0.95)),
            "spatial_var": spatial_var,
            "temp_e": temp_e}


def s0_fire(stats, rules):
    """First firing S0 rule -> evidence dict; else None."""
    for rule, key, op, thr in rules:
        if key not in stats:
            continue
        val = float(stats[key])
        if (val < thr) if op == "<" else (val > thr):
            return {"rule": rule, "feature": key, "value": round(val, 6),
                    "op": op, "threshold": thr}
    return None


def running_score(stats, gray_frames):
    """Descriptive running usability score in [0,1] (NOT accuracy).

    Brightness prefers mid-luma, motion prefers temporal energy above the
    codec-noise floor, sharpness prefers Laplacian variance. Fixed a priori
    weights; the accept threshold is a demo operating point, not certified.
    """
    b = 1.0 - min(1.0, abs(stats["mean"] - 0.35) / 0.35)
    m = min(1.0, stats["temp_e"] / 0.02)
    if gray_frames:
        lap = cv2.Laplacian(gray_frames[-1], cv2.CV_64F).var()
        s = min(1.0, float(lap) / 60.0)
    else:
        s = 0.0
    return round(max(0.0, min(1.0, 0.4 * b + 0.3 * m + 0.3 * s)), 4)


def window_id_for(clip, start_s):
    base = os.path.splitext(os.path.basename(clip))[0]
    m = re.search(r"(factory_?(\d+)).*?(worker_?(\d+))", clip)
    if m:
        return (f"f{m.group(2).lstrip('0') or '0'}_w"
                f"{m.group(4).lstrip('0') or '0'}_t{int(start_s):04d}")
    return f"{base}_t{int(start_s):04d}"


# --------------------------------------------------------------------------- #
# Pico hook                                                                    #
# --------------------------------------------------------------------------- #

def send_pico(decision, lock_path="tools/pico.lock", serial_obj=None):
    """Send the running verdict to the Pico LED under flock.

    accept->green, continue(uncertain)->amber, early_reject->red. Works
    headless: with no hardware it logs 'pico: not connected' and returns
    False. ``serial_obj`` injects a fake serial (tests/demo without hw).
    """
    color = DECISION_TO_LED.get(decision, "OFF")
    try:
        from jadal.pico_link import PicoLink
        os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
        with open(lock_path, "a+"):
            pass
        with open(lock_path, "r") as lockf:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            try:
                if serial_obj is not None:
                    link = PicoLink(serial=serial_obj)
                else:
                    link = PicoLink()
                link.connect()
                try:
                    link.set_led(color)
                finally:
                    link.disconnect()
            finally:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
        print(f"pico: LED {color} for {decision}", flush=True)
        return True
    except Exception as exc:
        print(f"pico: not connected ({type(exc).__name__})", flush=True)
        return False


# --------------------------------------------------------------------------- #
# Replay driver                                                                #
# --------------------------------------------------------------------------- #

def replay(clip, out_path, chunk_s=2.0, window_s=10.0, speed=20.0,
           max_chunks=None, degrade="none", degrade_at=None,
           accept_thr=0.5, use_pico=False, fps=5, width=160,
           use_cuda=False, lock_path="tools/pico.lock",
           serial_obj=None):
    """Run the replay; return a summary dict. Appends+flushes per chunk."""
    rules, s0_source = load_s0_rules()
    stream, engine = make_spectral_stream()
    wid = window_id_for(clip, 0.0)
    per_window = max(1, int(round(window_s / chunk_s)))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    freeze_frame = None
    gray_window = []
    degraded_seen_at = None
    rejected_at = None
    decisions = []
    lat_ms = []
    t0 = time.perf_counter()
    n = 0
    while True:
        if max_chunks is not None and n >= max_chunks:
            break
        t = n * chunk_s
        c0 = time.perf_counter()
        try:
            frames, backend = decode_chunk(clip, t, chunk_s, fps=fps,
                                           width=width, use_cuda=use_cuda)
        except Exception as exc:
            if n == 0:
                raise
            print(f"replay: end of clip at t={t:.0f}s ({exc})", flush=True)
            break
        if not frames:
            break
        degraded = (degrade not in (None, "none") and degrade_at is not None
                    and t >= degrade_at)
        if degraded and degraded_seen_at is None:
            degraded_seen_at = t
        frames, freeze_frame = apply_degrade(frames, degrade if degraded
                                             else "none", freeze_frame)
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        gray_window.extend(gray)
        gray_window = gray_window[-(per_window * fps + 8):]
        stats = window_stats(gray_window)
        score = running_score(stats, gray_window)
        # S0 fires on the running window OR on the current chunk alone:
        # a sudden blackout/freeze must reject on the chunk it arrives in,
        # not after the window dilutes it. The chunk-level check is the
        # same rules, same thresholds, narrower scope (recorded in reason).
        ev = s0_fire(stats, rules)
        scope = "running"
        if ev is None:
            ev = s0_fire(window_stats(gray), rules)
            scope = "chunk"
        chunk_in_window = (n % per_window) + 1
        window_done = chunk_in_window == per_window
        if ev is not None:
            decision, reason = "early_reject", {"s0": ev, "scope": scope}
            if rejected_at is None:
                rejected_at = t
        elif window_done and score >= accept_thr:
            decision, reason = "accept", {"score_ge": accept_thr}
        else:
            decision, reason = "continue", {}
        # Optional spectral-stream running features alongside (best effort).
        spec_feats = None
        if stream is not None:
            try:
                spec_feats = {k: round(float(v), 4)
                              for k, v in stream.feed(frames).items()
                              if isinstance(v, (int, float))}
            except Exception:
                spec_feats = None
        ms = (time.perf_counter() - c0) * 1000.0
        lat_ms.append(ms)
        try:
            import resource
            ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            ram_mb = -1.0
        event = {"t": round(t, 1), "window_id": wid, "chunk": n + 1,
                 "running_score": score, "decision": decision,
                 "reason": reason, "latency_ms": round(ms, 1),
                 "ram_mb": round(ram_mb, 1), "decode": backend,
                 "engine": engine, "s0_source": s0_source}
        if spec_feats is not None:
            event["spectral"] = {k: spec_feats[k]
                                 for k in ("brightness", "spec_hfratio",
                                           "motion_energy", "temp_frozen_frac",
                                           "under_frac", "over_frac")
                                 if k in spec_feats}
        with open(out_path, "a") as f:
            f.write(json.dumps(event) + "\n")
            f.flush()
        decisions.append(decision)
        print(f"chunk {n + 1} t={t:5.0f}s score={score:.2f} {decision} "
              f"{ms:5.0f}ms", flush=True)
        if use_pico:
            send_pico(decision, lock_path=lock_path, serial_obj=serial_obj)
        # Pacing: 20x default => 0.1 s per 2-s chunk; sleep only the remainder.
        pace = chunk_s / max(speed, 1e-9) - (time.perf_counter() - c0)
        if pace > 0:
            time.sleep(pace)
        n += 1
    wall = time.perf_counter() - t0
    ttr = (None if degraded_seen_at is None or rejected_at is None
           else round(rejected_at - degraded_seen_at, 1))
    return {"chunks": n, "decisions": decisions,
            "time_to_reject_s": ttr,
            "degrade": degrade, "degrade_at": degrade_at,
            "mean_latency_ms": round(float(np.mean(lat_ms)), 1) if lat_ms else -1,
            "wall_s": round(wall, 1), "events": out_path,
            "engine": engine, "s0_source": s0_source}
