"""Nonce-LED FRESHNESS check: prove a clip was recorded live in view of our Pico beacon.

Protocol (firmware unchanged): host issues a random 32-bit nonce, the Pico
flashes it with ``FLASH <8 hex>`` (100 ms/bit, MSB first, onboard LED), and the
webcam records the LED. The verifier decodes the 32 bits from the clip and
correlates against the expected pattern. A replayed clip (recorded with an
earlier nonce) FAILS because it shows the wrong pattern.

Claim supported: "a party that did not know the nonce during its validity
window could not have produced a clip that shows it". NOT supported: forgery
by an attacker who watches the nonce live and renders it.

NOT DONE:
  * No defence against a live-render attacker (watches the nonce, synthesises
    the LED); deployment must keep the validity window short.
  * ROI finder assumes the beacon LED is the dominant temporal-variance blob;
    other blinking lights in frame can confuse it (returns inconclusive when
    the signal is weak, but a strong distractor is out of scope).
  * Auto-exposure hunting on cheap webcams smears the bit edges; we lock
    exposure/gain when the driver allows, but some drivers ignore it.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

__all__ = [
    "BIT_MS",
    "BITS_TOTAL",
    "DEFAULT_FPS",
    "DEFAULT_Z_THRESH",
    "MIN_BITS_FRESH",
    "NULL_PATTERNS",
    "Nonce",
    "issue_nonce",
    "nonce_hash",
    "bits_of",
    "synthesize_frames",
    "write_video",
    "read_video",
    "find_led_roi",
    "roi_series",
    "verify_clip",
    "record_live",
]

BIT_MS = 100  # firmware bit period; MSB first
BITS_TOTAL = 32
DEFAULT_FPS = 30.0
DEFAULT_Z_THRESH = 4.0  # justified in docs/reports/freshness.md (synthetic + live trials)
MIN_BITS_FRESH = 30  # >= 30/32 bits must match for 'fresh'
NULL_PATTERNS = 2000  # random patterns for the z null distribution
_ROI_SIZE = 41  # square ROI edge (px) centred on peak variance
_VAR_FLOOR = 100.0  # min peak temporal variance (8-bit gray units^2) to claim LED found
_MIN_PEAK_TO_PEAK = 10.0  # min ROI series swing; below => inconclusive (too weak)


@dataclass
class Nonce:
    value: int
    hex: str
    issued_at: float
    validity_s: float = 30.0

    @property
    def expired(self, now: Optional[float] = None) -> bool:
        return (time.time() if now is None else now) - self.issued_at > self.validity_s


def nonce_hash(nonce: Union[Nonce, str, int]) -> str:
    """SHA-256 of the lowercase 8-hex nonce. Never log the nonce itself."""
    if isinstance(nonce, Nonce):
        h = nonce.hex
    elif isinstance(nonce, int):
        h = f"{nonce:08x}"
    else:
        h = str(nonce).strip().lower()
        if h.startswith("0x"):
            h = h[2:]
    return hashlib.sha256(h.encode("ascii")).hexdigest()


def issue_nonce(link=None, rng=None, validity_s: float = 30.0) -> Nonce:
    """Issue a fresh 32-bit nonce. ``rng`` is injectable for tests.

    ``rng`` may be ``random.Random``/``random.Random().getrandbits`` style
    (has ``getrandbits``) or anything with ``randint``; default is
    ``secrets``. ``link`` is accepted for API symmetry (unused; flashing
    happens in :func:`record_live`).
    """
    _ = link
    if rng is None:
        value = secrets.randbits(32)
    elif hasattr(rng, "getrandbits"):
        value = rng.getrandbits(32)
    elif hasattr(rng, "randint"):
        value = int(rng.randint(0, 2**32 - 1))
    else:  # callable returning int
        value = int(rng(2**32))
    value &= 0xFFFFFFFF
    return Nonce(value=value, hex=f"{value:08x}", issued_at=time.time(),
                 validity_s=validity_s)


def bits_of(nonce: Union[Nonce, str, int]) -> List[int]:
    """32 bits, MSB first."""
    if isinstance(nonce, Nonce):
        v = nonce.value
    elif isinstance(nonce, int):
        v = nonce & 0xFFFFFFFF
    else:
        s = str(nonce).strip().lower()
        if s.startswith("0x"):
            s = s[2:]
        v = int(s, 16) & 0xFFFFFFFF
    return [(v >> (31 - i)) & 1 for i in range(32)]


# --------------------------------------------------------------------------- #
# Synthetic generator (hardware-free)                                         #
# --------------------------------------------------------------------------- #
def synthesize_frames(
    nonce: Union[Nonce, str, int],
    fps: float = DEFAULT_FPS,
    pre_s: float = 1.0,
    post_s: float = 1.0,
    frame_size: Tuple[int, int] = (640, 480),
    roi: Tuple[int, int, int, int] = (300, 220, 40, 40),
    on: float = 220.0,
    off: float = 30.0,
    bg: float = 20.0,
    noise_sigma: float = 5.0,
    seed: int = 0,
) -> Tuple[np.ndarray, List[float], int]:
    """Render a synthetic LED clip showing ``nonce``. Deterministic in ``seed``.

    Returns ``(frames, timestamps, start_sample)`` where frames is uint8
    ``(N, H, W, 3)``, timestamps are seconds from clip start, and
    ``start_sample`` is the frame index where the 32-bit pattern starts.
    ``noise_sigma=0`` with ``on/off`` distinct gives a near-perfect clip;
    ``on == off`` (or equal levels) gives a no-LED clip for inconclusive tests.
    """
    rng = np.random.RandomState(seed)
    w, h = frame_size
    bit_s = BIT_MS / 1000.0
    pattern_s = BITS_TOTAL * bit_s
    total_s = pre_s + pattern_s + post_s
    n = int(round(total_s * fps))
    start_sample = int(round(pre_s * fps))
    bit_frames = max(1, int(round(bit_s * fps)))
    bits = bits_of(nonce)
    x, y, rw, rh = roi
    base = np.full((h, w), bg, dtype=np.float64)
    frames = np.empty((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        frame = base.copy()
        rel = i - start_sample
        if 0 <= rel < BITS_TOTAL * bit_frames:
            bit = bits[rel // bit_frames]
            frame[y : y + rh, x : x + rw] = on if bit else off
        else:
            frame[y : y + rh, x : x + rw] = off
        if noise_sigma > 0:
            frame = frame + rng.normal(0.0, noise_sigma, size=(h, w))
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames[i] = np.stack([frame, frame, frame], axis=-1)
    timestamps = [i / fps for i in range(n)]
    return frames, timestamps, start_sample


def write_video(frames: np.ndarray, path: Union[str, Path],
                fps: float = DEFAULT_FPS) -> Path:
    """Write ``(N,H,W,3)`` uint8 frames to mp4. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1], frames.shape[2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {path}")
    for i in range(frames.shape[0]):
        vw.write(frames[i])
    vw.release()
    return path


def read_video(path: Union[str, Path]) -> Tuple[np.ndarray, float]:
    """Read a video file into ``(frames, fps)``."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    frames: List[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return np.stack(frames, axis=0), float(fps)


def _to_gray(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 4:
        g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
        return g.astype(np.float64)
    return np.asarray(frames, dtype=np.float64)


def find_led_roi(frames: np.ndarray,
                 roi_size: int = _ROI_SIZE) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    """Locate the LED as the peak of the temporal-variance map.

    Returns ``(roi, peak_var)``; ``roi`` is None when the peak variance is
    below the detection floor (LED not found).
    """
    gray = _to_gray(frames)
    if gray.shape[0] < 8:
        return None, 0.0
    varmap = np.var(gray, axis=0)
    k = max(3, roi_size // 4)
    blur = cv2.blur(varmap.astype(np.float32), (k, k))
    _, peak, _, peak_loc = cv2.minMaxLoc(blur)
    if float(peak) < _VAR_FLOOR:
        return None, float(peak)
    cx, cy = peak_loc
    half = roi_size // 2
    h, w = varmap.shape
    x0 = min(max(0, cx - half), max(0, w - roi_size))
    y0 = min(max(0, cy - half), max(0, h - roi_size))
    rw = min(roi_size, w - x0)
    rh = min(roi_size, h - y0)
    return (x0, y0, rw, rh), float(peak)


def roi_series(frames: np.ndarray,
               roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Mean grayscale inside ``roi`` per frame."""
    gray = _to_gray(frames)
    x, y, w, h = roi
    return gray[:, y : y + h, x : x + w].mean(axis=(1, 2))


def _otsu_threshold(vals: np.ndarray) -> float:
    """1-D Otsu threshold over 32 window means."""
    s = np.sort(vals)
    best_var, best_thr = -1.0, float((s[0] + s[-1]) / 2.0)
    for k in range(1, len(s)):
        w0, w1 = k / len(s), 1 - k / len(s)
        mu0, mu1 = s[:k].mean(), s[k:].mean()
        bv = w0 * w1 * (mu0 - mu1) ** 2
        if bv > best_var:
            best_var = bv
            best_thr = float((s[k - 1] + s[k]) / 2.0)
    return best_thr


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sa, sb = a.std(), b.std()
    if sa == 0.0 or sb == 0.0:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _slice_means(series: np.ndarray, start: int, bit_frames: int) -> np.ndarray:
    return np.array([series[start + i * bit_frames: start + (i + 1) * bit_frames].mean()
                     for i in range(BITS_TOTAL)])


def verify_clip(
    path_or_frames: Union[str, Path, np.ndarray],
    nonce: Union[Nonce, str, int],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Verify a clip shows ``nonce``. Deterministic: same input => same dict.

    Returns ``{nonce_hash, status, lag_ms, z, bits_matched, bits_total}`` with
    status ``'fresh'`` | ``'stale'`` | ``'inconclusive'``.
    """
    cfg = dict(cfg or {})
    fps: float = float(cfg.get("fps", DEFAULT_FPS))
    z_thresh: float = float(cfg.get("z_thresh", DEFAULT_Z_THRESH))
    min_bits: int = int(cfg.get("min_bits", MIN_BITS_FRESH))
    null_n: int = int(cfg.get("null_n", NULL_PATTERNS))
    null_seed: int = int(cfg.get("null_seed", 1234))
    roi_size: int = int(cfg.get("roi_size", _ROI_SIZE))
    timestamps: Optional[Sequence[float]] = cfg.get("timestamps")

    if isinstance(path_or_frames, np.ndarray):
        frames = path_or_frames
        if timestamps is not None and len(timestamps) >= 2:
            dt = float(np.median(np.diff(np.asarray(timestamps, dtype=float))))
            if dt > 0:
                fps = 1.0 / dt
    else:
        frames, file_fps = read_video(path_or_frames)
        if "fps" not in cfg and file_fps > 0:
            fps = float(file_fps)

    n_frames = frames.shape[0]
    bit_frames = max(1, int(round(BIT_MS / 1000.0 * fps)))
    need = BITS_TOTAL * bit_frames
    result_base = {"nonce_hash": nonce_hash(nonce), "bits_total": BITS_TOTAL}
    if n_frames < need + 1:
        return {**result_base, "status": "inconclusive", "lag_ms": None,
                "z": None, "bits_matched": 0}

    roi, _peak = find_led_roi(frames, roi_size=roi_size)
    if roi is None:
        return {**result_base, "status": "inconclusive", "lag_ms": None,
                "z": None, "bits_matched": 0}
    series = roi_series(frames, roi)
    if float(series.max() - series.min()) < _MIN_PEAK_TO_PEAK or float(series.std()) == 0.0:
        return {**result_base, "status": "inconclusive", "lag_ms": None,
                "z": None, "bits_matched": 0}

    expected = np.array(bits_of(nonce), dtype=np.float64)
    exp_bip = 2 * expected - 1.0
    # Degenerate nonce (all 0s / all 1s) carries no timing info: cannot be fresh.
    if float(exp_bip.std()) == 0.0:
        return {**result_base, "status": "inconclusive", "lag_ms": None,
                "z": None, "bits_matched": 0}

    # Search pattern start over the whole clip; pick max correlation.
    best_corr, best_start = -2.0, 0
    n_starts = n_frames - need + 1
    step = max(1, bit_frames // 3)
    for start in range(0, n_starts, step):
        means = _slice_means(series, start, bit_frames)
        c = _pearson(means, exp_bip)
        if c > best_corr:
            best_corr, best_start = c, start
    # Refine around the coarse best at single-sample resolution.
    lo = max(0, best_start - step)
    hi = min(n_starts, best_start + step + 1)
    for start in range(lo, hi):
        means = _slice_means(series, start, bit_frames)
        c = _pearson(means, exp_bip)
        if c > best_corr:
            best_corr, best_start = c, start
    means = _slice_means(series, best_start, bit_frames)

    thr = _otsu_threshold(means)
    decoded = (means > thr).astype(int)
    # Otsu is polarity-free; keep the polarity that matches better.
    m1 = int((decoded == expected.astype(int)).sum())
    m0 = int(((1 - decoded) == expected.astype(int)).sum())
    if m0 > m1:
        decoded = 1 - decoded
        m1 = m0
    bits_matched = m1

    # Null: same slice vs >=2000 random patterns (seeded, deterministic).
    rng = np.random.RandomState(null_seed)
    rand_bits = rng.randint(0, 2, size=(null_n, BITS_TOTAL)).astype(np.float64)
    rand_bip = 2 * rand_bits - 1.0
    m_mean, m_std = float(means.mean()), float(means.std())
    if m_std == 0.0:
        return {**result_base, "status": "inconclusive",
                "lag_ms": round(best_start / fps * 1000.0, 4),
                "z": None, "bits_matched": 0}
    zn = (means - m_mean) / m_std
    null_corrs = (rand_bip * zn).mean(axis=1)
    mu, sd = float(null_corrs.mean()), float(null_corrs.std())
    z = 0.0 if sd == 0.0 else float((best_corr - mu) / sd)

    lag_ms = round(best_start / fps * 1000.0, 4)
    z_r = round(z, 4)
    status = "fresh" if (bits_matched >= min_bits and z >= z_thresh) else "stale"
    return {"nonce_hash": nonce_hash(nonce), "status": status,
            "lag_ms": lag_ms, "z": z_r, "bits_matched": bits_matched,
            "bits_total": BITS_TOTAL}


# --------------------------------------------------------------------------- #
# Live capture (hardware)                                                     #
# --------------------------------------------------------------------------- #
def _open_camera(cam_index: int = 0, fps: float = DEFAULT_FPS,
                 width: int = 640, height: int = 480):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if not cap.isOpened():  # fall back to default backend
        cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {cam_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # Lock exposure / auto-gain when the driver allows (best effort).
    for prop, val in ((cv2.CAP_PROP_AUTO_EXPOSURE, 1),
                      (cv2.CAP_PROP_GAIN, 0),
                      (cv2.CAP_PROP_AUTOFOCUS, 0)):
        try:
            cap.set(prop, val)
        except Exception:
            pass
    return cap


def record_live(nonce: Union[Nonce, str, int], link, cam_index: int = 0,
                seconds: float = 6.0, out_dir: Union[str, Path, None] = None,
                fps: float = DEFAULT_FPS) -> Dict[str, Any]:
    """Capture frames while flashing ``nonce`` on the Pico. Returns paths+times.

    Captures ``seconds`` of video (must cover 3.2 s pattern + margins),
    sends ``FLASH`` ~0.5 s after capture starts, writes mp4 + timestamps json
    under ``data_runtime/freshness/``. Returns ``{path, ts_path, t_flash,
    frame_times, fps, nonce_hash}``.
    """
    from jadal.config import RUNTIME  # local import: keeps import side-effect free

    hexs = nonce.hex if isinstance(nonce, Nonce) else (
        f"{int(nonce):08x}" if isinstance(nonce, int)
        else str(nonce).strip().lower().removeprefix("0x"))
    if link is None:
        raise ValueError("record_live needs a PicoLink (link=None)")
    out = Path(out_dir) if out_dir else (RUNTIME / "freshness")
    out.mkdir(parents=True, exist_ok=True)

    cap = _open_camera(cam_index, fps=fps)
    frames: List[np.ndarray] = []
    frame_times: List[float] = []
    # Warm-up: let auto-gain settle, discard a few frames.
    for _ in range(5):
        cap.read()
    t_start = time.monotonic()
    t_flash: Optional[float] = None
    flashed = False
    while True:
        ok, fr = cap.read()
        now = time.monotonic()
        if not ok:
            continue
        frames.append(fr)
        frame_times.append(now)
        if not flashed and now - t_start >= 0.5:
            link.flash(hexs)
            t_flash = time.monotonic()
            flashed = True
        if now - t_start >= seconds:
            break
    cap.release()
    if t_flash is None:  # very short captures: flash at end
        link.flash(hexs)
        t_flash = time.monotonic()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    vpath = out / f"live_{stamp}_{hexs}.mp4"
    arr = np.stack(frames, axis=0)
    write_video(arr, vpath, fps=fps)
    import json

    tspath = vpath.with_suffix(".timestamps.json")
    tspath.write_text(json.dumps({"t_start": t_start, "t_flash": t_flash,
                                  "frame_times": frame_times, "fps": fps}))
    # Save one preview frame for aiming the camera.
    cv2.imwrite(str(out / "frame.jpg"), frames[len(frames) // 2])
    return {"path": str(vpath), "ts_path": str(tspath), "t_flash": t_flash,
            "frame_times": frame_times, "fps": fps, "nonce_hash": nonce_hash(hexs)}
