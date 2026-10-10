"""Deterministic, cheap, hashable per-window regime descriptors (E4.01).

No GPU. Reads K sampled frames per window with cv2, computes scalar
descriptors, returns a canonical dict + sha256 record hash.
"""
import hashlib
import json
import subprocess

import cv2
import numpy as np

N_FRAMES_DEFAULT = 8
IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def ffprobe_meta(path):
    """Container-level descriptors via ffprobe. Returns dict."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", path]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    info = json.loads(out.stdout)
    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise ValueError(f"no video stream: {path}")
    v = streams[0]
    w, h = int(v["width"]), int(v["height"])
    fps_s = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1"
    num, den = fps_s.split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    fmt = info.get("format", {})
    size_b = int(float(fmt.get("size", 0)))
    dur_s = float(fmt.get("duration", 0))
    codec = v.get("codec_name", "?")
    pix_fmt = v.get("pix_fmt", "?")
    br = float(fmt.get("bit_rate", 0)) / 1000.0  # kbps
    meta = {
        "width": w, "height": h, "fps": round(fps, 3),
        "codec": codec, "pix_fmt": pix_fmt,
        "bitrate_kbps": round(br, 1),
        "bytes_per_frame": round(size_b / max(dur_s * fps, 1e-9), 1) if dur_s and fps else 0.0,
        "duration_s": round(dur_s, 2),
    }
    return meta


def sample_frames(path, start_s, end_s, k=N_FRAMES_DEFAULT):
    """K evenly spaced BGR frames across [start_s, end_s). Returns list or None entries."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [None] * k
    ts = [start_s + (end_s - start_s) * (i + 0.5) / k for i in range(k)]
    frames = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, fr = cap.read()
        frames.append(fr if ok and fr is not None else None)
    cap.release()
    return frames


def frame_stats(bgr):
    """Scalar stats for one BGR frame."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    luma = gray / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lapvar = float(lap.var())
    lmean = float(luma.mean())
    # Immerkaer noise sigma on luma
    m = cv2.filter2D(gray, -1, IMMERKAER)
    sigma = float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(m)) / 6.0 / 255.0)
    # edge density: sobel magnitude mean
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = float(np.mean(np.sqrt(sx * sx + sy * sy)) / 255.0)
    p = np.percentile(luma, [5, 50, 95])
    dark = float(np.mean(luma < 16 / 255.0))
    bright = float(np.mean(luma > 235 / 255.0))
    # blockiness: 8px-boundary discontinuity ratio (horizontal + vertical)
    gx = np.abs(np.diff(gray, axis=1))
    bnd = gx[:, 7::8].mean() if gx.shape[1] > 8 else 0.0
    mask = np.ones(gx.shape[1], bool)
    mask[7::8] = False
    inn = gx[:, mask].mean() if mask.any() else 1e-9
    gy = np.abs(np.diff(gray, axis=0))
    bnd_v = gy[7::8, :].mean() if gy.shape[0] > 8 else 0.0
    maskv = np.ones(gy.shape[0], bool)
    maskv[7::8] = False
    inn_v = gy[maskv, :].mean() if maskv.any() else 1e-9
    blocky = float(0.5 * (bnd / (inn + 1e-9) + bnd_v / (inn_v + 1e-9)))
    # chroma spread (scene-colour proxy): std of U,V at small size
    small = cv2.resize(bgr, (80, 45))
    yuv = cv2.cvtColor(small, cv2.COLOR_BGR2YUV).astype(np.float32)
    chroma = float(yuv[:, :, 1].std() + yuv[:, :, 2].std())
    return {
        "sharp_lapvar": lapvar, "luma_mean": lmean,
        "noise_sigma": sigma, "edge_density": edge,
        "luma_p05": float(p[0]), "luma_p50": float(p[1]), "luma_p95": float(p[2]),
        "dark_frac": dark, "bright_frac": bright,
        "blockiness": blocky, "chroma_spread": chroma,
    }


def motion_stats(frames):
    """Ego-motion proxies between consecutive sampled frames (160px gray)."""
    grays = []
    for fr in frames:
        if fr is None:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        sc = 160.0 / g.shape[1]
        g = cv2.resize(g, (160, max(8, int(g.shape[0] * sc))))
        grays.append(g.astype(np.float32))
    shifts, diffs = [], []
    for a, b in zip(grays[:-1], grays[1:]):
        try:
            (dx, dy), _ = cv2.phaseCorrelate(a, b)
        except cv2.error:
            dx, dy = 0.0, 0.0
        shifts.append(float(np.hypot(dx, dy)))
        diffs.append(float(np.mean(np.abs(b - a)) / 255.0))
    if not shifts:
        return {"shift_mean": 0.0, "shift_max": 0.0, "static_frac": 1.0,
                "temp_energy": 0.0, "dup_frac": 1.0}
    shifts = np.array(shifts)
    diffs = np.array(diffs)
    return {
        "shift_mean": float(shifts.mean()), "shift_max": float(shifts.max()),
        "static_frac": float(np.mean(shifts < 1.0)),
        "temp_energy": float(diffs.mean()),
        "dup_frac": float(np.mean(diffs < 0.003)),
    }


def describe_window(path, start_s, end_s, k=N_FRAMES_DEFAULT):
    """Full per-window descriptor record (deterministic given file bytes)."""
    meta = ffprobe_meta(path)
    frames = sample_frames(path, start_s, end_s, k)
    good = [f for f in frames if f is not None]
    if not good:
        raise ValueError(f"no frames decoded: {path} [{start_s},{end_s}]")
    per = [frame_stats(f) for f in good]
    agg = {}
    for key in per[0]:
        vals = [p[key] for p in per]
        agg[key] = round(float(np.mean(vals)), 6)
    agg["sharp_norm"] = round(agg["sharp_lapvar"] / (agg["luma_mean"] + 1e-3), 4)
    agg["clip_frac"] = round(agg["dark_frac"] + agg["bright_frac"], 6)
    mo = motion_stats(frames)
    desc = {**meta, **agg, **{kk: round(v, 6) for kk, v in mo.items()},
            "n_frames_used": len(good), "n_frames_req": k}
    rec = json.dumps(desc, sort_keys=True)
    desc["_sha256"] = hashlib.sha256(rec.encode()).hexdigest()[:16]
    return desc
