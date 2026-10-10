"""Window decode helpers (CPU ffmpeg). No GPU, no lock needed."""
import subprocess

import numpy as np


def clip_duration(path):
    """Duration in seconds via ffprobe (cached per path)."""
    if path not in clip_duration._cache:
        p = subprocess.run(["ffprobe", "-hide_banner", "-v", "error",
                            "-show_entries", "format=duration",
                            "-of", "default=noprint_wrappers=1:nokey=1", path],
                           capture_output=True, text=True, timeout=30)
        clip_duration._cache[path] = float(p.stdout.strip())
    return clip_duration._cache[path]


clip_duration._cache = {}


def safe_offset(path, i, dur_s, step=37):
    dur = clip_duration(path)
    span = max(dur - dur_s - 1, 1)
    return (i * step) % span


def decode_window(path, start_s, dur_s, w=160, h=90, pix_fmt="rgb24",
                  timeout_s=120):
    """Decode [start_s, start_s+dur_s) scaled to w,h. Returns (arr, fps).

    arr: (T,H,W,3) float64 in [0,1] for rgb24, (T,H,W) for gray.
    """
    vf = f"scale={w}:{h}"
    if pix_fmt == "rgb24":
        vf += ",format=rgb24"
        ch = 3
    else:
        vf += ",format=gray"
        ch = 1
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
           "-ss", str(start_s), "-t", str(dur_s), "-i", path,
           "-vf", vf, "-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.DEVNULL, timeout=timeout_s)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg rc={p.returncode}: {p.stderr[-500:].decode('utf-8', 'replace')}")
    px = w * h * ch
    n = len(p.stdout) // px
    if n == 0:
        raise RuntimeError(f"no frames decoded from {path}@{start_s}")
    arr = np.frombuffer(p.stdout[:n * px], dtype=np.uint8).astype(np.float64) / 255.0
    if ch == 3:
        arr = arr.reshape(n, h, w, 3)
    else:
        arr = arr.reshape(n, h, w)
    return arr, n / dur_s
