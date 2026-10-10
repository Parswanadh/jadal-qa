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


def nv12_to_rgb(y, uv):
    """y (H,W), uv (H/2,W/2,2) uint8 limited-range BT.601 -> (H,W,3) float64."""
    u = uv[..., 0].repeat(2, axis=0).repeat(2, axis=1).astype(np.float64)
    v = uv[..., 1].repeat(2, axis=0).repeat(2, axis=1).astype(np.float64)
    c = y.astype(np.float64) - 16.0
    d, e = u - 128.0, v - 128.0
    r = 1.164 * c + 1.596 * e
    g = 1.164 * c - 0.391 * d - 0.813 * e
    b = 1.164 * c + 2.018 * d
    rgb = np.stack([r, g, b], axis=-1).clip(0, 255) / 255.0
    return rgb


NVDEC_CHAIN = (  # proven by nvdec_probe.json chain B (rc=0, exact bytes)
    ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"],
    "scale_cuda={w}:{h}:format=nv12,hwdownload,format=nv12",
)


def decode_window_nvdec(path, start_s, dur_s, w=160, h=90, timeout_s=120):
    """NVDEC decode [start_s, start_s+dur_s) at w,h. Returns (rgb, fps)."""
    acc, filt = NVDEC_CHAIN
    cmd = (["ffmpeg", "-hide_banner", "-v", "error", "-nostdin"] + acc +
           ["-ss", str(start_s), "-t", str(dur_s), "-i", path,
            "-vf", filt.format(w=w, h=h),
            "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"])
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.DEVNULL, timeout=timeout_s)
    if p.returncode != 0:
        raise RuntimeError(f"nvdec rc={p.returncode}: {p.stderr[-300:].decode('utf-8', 'replace')}")
    fs = w * h * 3 // 2
    n = len(p.stdout) // fs
    if n == 0:
        raise RuntimeError(f"nvdec: no frames from {path}@{start_s}")
    out = np.empty((n, h, w, 3))
    for i in range(n):
        f = np.frombuffer(p.stdout[i * fs:(i + 1) * fs], dtype=np.uint8)
        out[i] = nv12_to_rgb(f[:w * h].reshape(h, w),
                             f[w * h:].reshape(h // 2, w // 2, 2))
    return out, n / dur_s
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
