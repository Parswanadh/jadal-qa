"""Nonce-LED FRESHNESS check: prove a clip was recorded live in view of our Pico beacon.

Protocol (firmware unchanged): host issues a random 32-bit nonce, the Pico
flashes it with ``FLASH <8 hex>`` (100 ms/bit, MSB first, onboard LED), and the
webcam records the LED. The verifier resamples the clip to the bit grid using
real per-frame timestamps, decodes the 32 bits, and correlates against the
expected pattern. A replayed clip (recorded with an earlier nonce) FAILS
because it shows the wrong pattern.

Decision rule: ``fresh`` requires an exact 32/32-bit match (primary) AND
``z >= Z_THRESH`` (supporting). ``z`` compares the best lag-search correlation
against a null of >= 2000 seeded random wrong nonces run through EXACTLY the
same procedure (same lag search, same Pearson normalisation, same polarity
handling), so the threshold means what it says.

On ``nonce_hash``: it is a SHA-256 COMMITMENT (binding), NOT a hiding
commitment — the preimage is only 32 bits and is trivially enumerable
(~2.1 M hashes/s single-core, ~33 min worst case). Secrecy of the nonce value
matters only inside its validity window (default 30 s); after that the value
is worthless to an attacker because no fresh clip can be back-dated into the
window. The value itself NEVER appears in filenames, logs, sidecars, or
results — only ``nonce_hash[:16]`` is used for naming.

Claim supported: "a party that did not know the nonce during its validity
window could not have produced a clip that shows it". NOT supported: forgery
by an attacker who watches the nonce live and renders it.

NOT DONE:
  * No defence against a live-render attacker (watches the nonce, synthesises
    the LED); deployment must keep the validity window short.
  * ROI finder assumes the beacon LED is the dominant temporal-variance blob;
    other blinking lights in frame can confuse it (returns inconclusive when
    the signal is weak, but a strong distractor is out of scope).
  * Auto-exposure hunting on cheap webcams smears the bit edges; we request
    manual exposure/gain when the driver allows, but some drivers ignore it
    (``CAP_PROP_AUTO_EXPOSURE=1`` means "manual" on most V4L2 backends and
    "auto" on some others — verify on the device).
  * Privacy: captures record whatever the camera sees (office, bystanders).
    Keep the beacon framing tight and delete raw clips after the trials.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

__all__ = [
    "BIT_S",
    "BITS_TOTAL",
    "BITS_REQUIRED",
    "DEFAULT_FPS",
    "DEFAULT_Z_THRESH",
    "NULL_PATTERNS",
    "LAG_STEP_S",
    "Nonce",
    "issue_nonce",
    "nonce_hash",
    "bits_of",
    "synthesize_frames",
    "write_video",
    "read_video",
    "clip_basename",
    "save_recording",
    "find_led_roi",
    "roi_series",
    "means_matrix",
    "null_maxcorr",
    "decode_bits",
    "verify_clip",
    "record_live",
]

BIT_S = 0.1  # firmware bit period, seconds; MSB first
BITS_TOTAL = 32
BITS_REQUIRED = 32  # exact-match primary decision: 32/32 bits
DEFAULT_FPS = 30.0  # last-resort fallback only; real timestamps always preferred
# Calibrated 2026-10-10 (fix round 1): pooled null of 40000 seeded wrong
# nonces (4 SNR levels x 2 clips x 5000, full lag search each — the same
# statistic as observed); pooled p99 = 2.56, per-level p99 2.53-2.68;
# threshold 3.0 sits above the pooled p99. Full-rule (exact 32/32 + z>=3.0)
# false accepts: 0/10000 per level (Clopper-Pearson upper 0.023% at
# delta=0.1), so FA <= 1% on wrong nonces. True-nonce z 6.6-7.0, bits 32/32
# at every level. See eval/results/freshness_synthetic.json. NOT justified by
# live trials (N=0); re-check on hardware before the go/no-go gate.
DEFAULT_Z_THRESH = 3.0
NULL_PATTERNS = 2000  # default null size for single verify calls
LAG_STEP_S = 0.01  # lag-search grid step (10 ms)
_ROI_SIZE = 41  # square ROI edge (px) centred on peak variance
_ROI_MAX_FRAMES = 60  # subsample cap for the variance map
_VAR_FLOOR = 100.0  # min peak temporal variance (8-bit gray units^2) to claim LED found
_MIN_PEAK_TO_PEAK = 10.0  # min ROI series swing; below => inconclusive (too weak)


@dataclass
class Nonce:
    value: int
    hex: str
    issued_at: float  # wall-clock (time.time), comparable to clip capture times
    validity_s: float = 30.0

    def is_expired(self, now: Optional[float] = None) -> bool:
        return (time.time() if now is None else now) - self.issued_at > self.validity_s

    @property
    def expired(self) -> bool:
        return self.is_expired()


def nonce_hash(nonce: Union[Nonce, str, int]) -> str:
    """SHA-256 commitment to the lowercase 8-hex nonce.

    Binding, not hiding: the 32-bit preimage is enumerable, so this hash
    protects integrity, not secrecy. Secrecy lives only in the validity
    window (see :class:`Nonce`); never log the value itself.
    """
    if isinstance(nonce, Nonce):
        h = nonce.hex
    elif isinstance(nonce, int):
        h = f"{nonce & 0xFFFFFFFF:08x}"
    else:
        h = str(nonce).strip().lower()
        if h.startswith("0x"):
            h = h[2:]
    return hashlib.sha256(h.encode("ascii")).hexdigest()


def issue_nonce(link=None, rng=None, validity_s: float = 30.0) -> Nonce:
    """Issue a fresh 32-bit nonce. ``rng`` is injectable for tests.

    ``rng`` may expose ``getrandbits`` or ``randint``; default is ``secrets``.
    ``link`` is accepted for API symmetry (unused; flashing happens in
    :func:`record_live`). The returned value must be treated as secret only
    until ``issued_at + validity_s``; afterwards it is worthless.
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


def _bipolar(bits: np.ndarray) -> np.ndarray:
    return 2.0 * np.asarray(bits, dtype=np.float64) - 1.0


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
    ``(N, H, W, 3)`` and timestamps are true per-frame capture times in seconds
    from clip start. ``fps`` models the camera rate (15/25/30 all supported by
    the timestamp-based decoder). ``on == off`` gives a no-LED clip.
    """
    rng = np.random.RandomState(seed)
    w, h = frame_size
    pattern_s = BITS_TOTAL * BIT_S
    total_s = pre_s + pattern_s + post_s
    n = int(round(total_s * fps))
    start_sample = int(round(pre_s * fps))
    bit_frames = pre_s  # not used for rendering grid; kept explicit below
    _ = bit_frames
    bits = bits_of(nonce)
    x, y, rw, rh = roi
    base = np.full((h, w), bg, dtype=np.float64)
    frames = np.empty((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        frame = base.copy()
        t = i / fps - pre_s  # seconds since pattern start
        if 0.0 <= t < pattern_s:
            bit = bits[min(int(t / BIT_S), BITS_TOTAL - 1)]
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


def read_video(path: Union[str, Path]) -> Tuple[np.ndarray, float, List[float]]:
    """Read a video file into ``(frames, container_fps, container_ts)``.

    ``container_ts`` are per-frame presentation times in seconds from
    ``CAP_PROP_POS_MSEC`` (0.0-based). Falls back to a nominal grid when the
    container reports nothing usable.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    frames: List[np.ndarray] = []
    ts: List[float] = []
    while True:
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
        ts.append(max(0.0, float(pos_ms)) / 1000.0)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    if ts[-1] <= ts[0]:  # container gave no timing; nominal grid
        ts = [i / fps for i in range(len(frames))]
    return np.stack(frames, axis=0), float(fps), ts


def clip_basename(stamp: str, nonce_hash_hex: str) -> str:
    """Clip filename from the hash prefix ONLY — never the nonce value."""
    return f"live_{stamp}_{nonce_hash_hex[:16]}.mp4"


def save_recording(frames: np.ndarray, meta: Dict[str, Any],
                   out_dir: Union[str, Path], nonce_hash_hex: str,
                   stamp: Optional[str] = None) -> Dict[str, str]:
    """Write mp4 + ``.timestamps.json`` sidecar. ``meta`` must not contain the
    nonce value (only its hash); returns ``{path, ts_path}``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    vpath = out / clip_basename(stamp, nonce_hash_hex)
    fps = float(meta.get("fps", DEFAULT_FPS))
    write_video(frames, vpath, fps=fps)
    tspath = vpath.with_suffix(".timestamps.json")
    tspath.write_text(json.dumps(meta, indent=2))
    return {"path": str(vpath), "ts_path": str(tspath)}


def _to_gray(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 4:
        g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
        return g.astype(np.float64)
    return np.asarray(frames, dtype=np.float64)


def find_led_roi(frames: np.ndarray,
                 roi_size: int = _ROI_SIZE,
                 max_frames: int = _ROI_MAX_FRAMES
                 ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
    """Locate the LED as the peak of the temporal-variance map.

    Returns ``(roi, peak_var)``; ``roi`` is None when the peak variance is
    below the detection floor (LED not found). At most ``max_frames`` evenly
    spaced frames are used for the map (finding needs change, not density).
    """
    n = frames.shape[0]
    if n < 8:
        return None, 0.0
    idx = np.linspace(0, n - 1, num=min(n, max(8, max_frames))).astype(int)
    gray = _to_gray(frames[idx])
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


def means_matrix(series: np.ndarray, ts: Sequence[float],
                 lag_step_s: float = LAG_STEP_S) -> Tuple[np.ndarray, np.ndarray]:
    """Window-mean rows for every candidate pattern start ``t0``.

    Row ``s`` holds the 32 means of samples in
    ``[t0_s + i*BIT_S, t0_s + (i+1)*BIT_S)``. Starts whose windows lack enough
    span are dropped. Returns ``(M, starts)`` with ``M`` shape ``(S, 32)``.
    """
    t = np.asarray(ts, dtype=np.float64)
    need = BITS_TOTAL * BIT_S
    if t.size < 2 or t[-1] - t[0] < need:
        return np.empty((0, BITS_TOTAL)), np.empty((0,))
    n_steps = int(np.floor((t[-1] - t[0] - need) / lag_step_s)) + 1
    starts = t[0] + np.arange(max(n_steps, 1)) * lag_step_s
    edges = starts[:, None] + np.arange(BITS_TOTAL + 1)[None, :] * BIT_S
    idx = np.searchsorted(t, edges)  # (S, 33)
    M = np.empty((starts.size, BITS_TOTAL))
    ok = np.ones(starts.size, dtype=bool)
    for s in range(starts.size):
        for i in range(BITS_TOTAL):
            seg = series[idx[s, i] : idx[s, i + 1]]
            if seg.size == 0:
                ok[s] = False
                break
            M[s, i] = seg.mean()
    return M[ok], starts[ok]


def _standardize_rows(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    zero = (sd == 0.0).ravel()
    Z = np.zeros_like(X)
    nz = ~zero
    Z[nz] = (X[nz] - mu[nz]) / sd[nz]
    return Z, zero


def maxcorr_over_lags(M: np.ndarray, pattern_bipolar: np.ndarray) -> Tuple[float, int]:
    """Max Pearson correlation of the pattern against rows of ``M``.

    Same normalisation as the null (:func:`null_maxcorr`); degenerate patterns
    (std 0) score 0. Returns ``(best_corr, best_row)``.
    """
    Z, zero = _standardize_rows(M)
    p = np.asarray(pattern_bipolar, dtype=np.float64).ravel()
    if p.std() == 0.0:
        return 0.0, 0
    zp = (p - p.mean()) / p.std()
    corrs = (Z * zp).mean(axis=1)
    b = int(np.argmax(corrs))
    return float(corrs[b]), b


def null_maxcorr(M: np.ndarray, null_n: int, seed: int,
                 return_argmax: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Max-over-lags correlation for ``null_n`` seeded random patterns.

    EXACTLY the same statistic as :func:`maxcorr_over_lags` (same search, same
    Pearson normalisation per pattern). Degenerate draws (all-equal bits,
    probability 2/2^32) contribute 0. Returns array of ``null_n`` maxima, or
    ``(maxima, argmax_rows)`` with ``return_argmax=True`` (for full-rule
    false-accept evaluation: decode at each pattern's own best lag).
    """
    rng = np.random.RandomState(seed)
    P = rng.randint(0, 2, size=(null_n, BITS_TOTAL)).astype(np.float64)
    Pb = 2.0 * P - 1.0
    Z, _ = _standardize_rows(M)
    Pz, zero = _standardize_rows(Pb)
    Pz[zero] = 0.0
    corrs = (Z @ Pz.T) / BITS_TOTAL  # (S, null_n)
    maxima = corrs.max(axis=0)
    if return_argmax:
        return maxima, corrs.argmax(axis=0)
    return maxima


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


def decode_bits(means: np.ndarray, expected: np.ndarray) -> Tuple[np.ndarray, int]:
    """Otsu-decode 32 window means; polarity follows the better match.

    Returns ``(decoded, bits_matched)``.
    """
    thr = _otsu_threshold(np.asarray(means, dtype=np.float64))
    dec = (np.asarray(means) > thr).astype(int)
    exp = np.asarray(expected).astype(int)
    m1 = int((dec == exp).sum())
    m0 = int(((1 - dec) == exp).sum())
    if m0 > m1:
        dec = 1 - dec
        m1 = m0
    return dec, m1


def _measured_fps(ts: Sequence[float], fallback: float) -> float:
    t = np.asarray(ts, dtype=np.float64)
    if t.size >= 2 and t[-1] > t[0]:
        return float((t.size - 1) / (t[-1] - t[0]))
    return float(fallback)


def verify_clip(
    path_or_frames: Union[str, Path, np.ndarray],
    nonce: Union[Nonce, str, int],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Verify a clip shows ``nonce``. Deterministic: same input => same dict.

    Timestamp sources (first wins): ``cfg["timestamps"]`` > sibling
    ``.timestamps.json`` sidecar (``frame_times``) > container presentation
    times > nominal ``cfg["fps"]`` grid. The measured rate is reported as
    ``fps_measured`` (with ``fps_source``); the bit grid is always wall-clock
    (100 ms), so off-rate cameras decode correctly.

    Validity: when ``nonce`` is a :class:`Nonce` and the clip wall-clock start
    is known (``cfg["clip_start_wall"]``, sidecar ``capture_start_wall``, or
    file mtime), a clip wholly outside ``[issued_at, issued_at+validity_s]``
    is ``stale`` with ``reason="expired"`` without decoding.

    Returns ``{nonce_hash, status, lag_ms, z, bits_matched, bits_total,
    reason, fps_measured, fps_source}``; ``reason`` is one of
    ``ok`` | ``mismatch`` | ``expired`` | ``no-signal``.
    """
    cfg = dict(cfg or {})
    z_thresh: float = float(cfg.get("z_thresh", DEFAULT_Z_THRESH))
    null_n: int = int(cfg.get("null_n", NULL_PATTERNS))
    null_seed: int = int(cfg.get("null_seed", 1234))
    roi_size: int = int(cfg.get("roi_size", _ROI_SIZE))
    lag_step: float = float(cfg.get("lag_step_s", LAG_STEP_S))
    nominal_fps: float = float(cfg.get("fps", DEFAULT_FPS))
    base = {"nonce_hash": nonce_hash(nonce), "bits_total": BITS_TOTAL}

    frames: np.ndarray
    ts: Optional[List[float]] = None
    fps_source = "nominal"
    clip_wall: Optional[float] = cfg.get("clip_start_wall")
    sidecar: Dict[str, Any] = {}
    if isinstance(path_or_frames, np.ndarray):
        frames = path_or_frames
        if cfg.get("timestamps") is not None:
            ts = [float(v) for v in cfg["timestamps"]]
            fps_source = "explicit"
    else:
        frames, container_fps, container_ts = read_video(path_or_frames)
        if container_fps > 0:
            nominal_fps = float(container_fps)
        sib = Path(path_or_frames).with_suffix(".timestamps.json")
        if sib.exists():
            try:
                sidecar = json.loads(sib.read_text())
            except Exception:
                sidecar = {}
        if cfg.get("timestamps") is not None:
            ts = [float(v) for v in cfg["timestamps"]]
            fps_source = "explicit"
        elif isinstance(sidecar.get("frame_times"), list) and len(sidecar["frame_times"]) == frames.shape[0]:
            ts = [float(v) - float(sidecar["frame_times"][0]) for v in sidecar["frame_times"]]
            fps_source = "sidecar"
        else:
            ts = container_ts
            fps_source = "container"
        if clip_wall is None:
            if sidecar.get("capture_start_wall") is not None:
                clip_wall = float(sidecar["capture_start_wall"])
            else:
                try:
                    clip_wall = float(Path(path_or_frames).stat().st_mtime)
                except OSError:
                    clip_wall = None
    if ts is None:  # in-memory frames, no timestamps: nominal grid
        ts = [i / nominal_fps for i in range(frames.shape[0])]
    fps_measured = round(_measured_fps(ts, nominal_fps), 4)

    def out(**kw: Any) -> Dict[str, Any]:
        d = {**base, "fps_measured": fps_measured, "fps_source": fps_source}
        d.update(kw)
        return d

    # Validity window (wall-clock vs wall-clock only).
    if isinstance(nonce, Nonce) and clip_wall is not None:
        dur = ts[-1] - ts[0] if len(ts) >= 2 else 0.0
        if clip_wall + dur < nonce.issued_at or clip_wall > nonce.issued_at + nonce.validity_s:
            return out(status="stale", lag_ms=None, z=None, bits_matched=0,
                       reason="expired")

    roi, _peak = find_led_roi(frames, roi_size=roi_size)
    if roi is None:
        return out(status="inconclusive", lag_ms=None, z=None, bits_matched=0,
                   reason="no-signal")
    series = roi_series(frames, roi)
    if float(series.max() - series.min()) < _MIN_PEAK_TO_PEAK or float(series.std()) == 0.0:
        return out(status="inconclusive", lag_ms=None, z=None, bits_matched=0,
                   reason="no-signal")

    expected = np.array(bits_of(nonce), dtype=np.float64)
    if float(_bipolar(expected).std()) == 0.0:  # degenerate nonce: no timing info
        return out(status="inconclusive", lag_ms=None, z=None, bits_matched=0,
                   reason="no-signal")

    M, starts = means_matrix(series, ts, lag_step_s=lag_step)
    if M.shape[0] == 0:
        return out(status="inconclusive", lag_ms=None, z=None, bits_matched=0,
                   reason="no-signal")
    exp_bip = _bipolar(expected)
    best_corr, best_row = maxcorr_over_lags(M, exp_bip)
    means = M[best_row]
    _, bits_matched = decode_bits(means, expected)

    null = null_maxcorr(M, null_n, null_seed)
    mu, sd = float(null.mean()), float(null.std())
    z = 0.0 if sd == 0.0 else float((best_corr - mu) / sd)

    lag_ms = round(float((starts[best_row] - ts[0]) * 1000.0), 4)
    z_r = round(z, 4)
    fresh = bits_matched == BITS_REQUIRED and z >= z_thresh
    return out(status="fresh" if fresh else "stale", lag_ms=lag_ms, z=z_r,
               bits_matched=bits_matched,
               reason="ok" if fresh else "mismatch")


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
    # Best effort: request manual exposure + zero gain so bit edges don't swim.
    # NOTE: CAP_PROP_AUTO_EXPOSURE=1 means "manual" on most V4L2 backends but
    # "auto" on some others — verify the effective rate on the device (we
    # record per-frame times, so any residual drift is measured, not assumed).
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
                fps: float = DEFAULT_FPS, fail_max: int = 30) -> Dict[str, Any]:
    """Capture frames while flashing ``nonce`` on the Pico. Returns paths+times.

    Captures ``seconds`` of video (must cover the 3.2 s pattern + margins),
    sends ``FLASH`` ~0.5 s after capture starts, and saves mp4 + sidecar under
    ``data_runtime/freshness/`` named by ``nonce_hash[:16]`` (the value never
    touches disk). The sidecar keeps per-frame monotonic times plus the
    wall-clock capture start, so :func:`verify_clip` resamples on real timing.
    Returns ``{path, ts_path, t_flash, frame_times, fps, fps_source,
    capture_start_wall, nonce_hash}`` — no nonce value anywhere.
    """
    from jadal.config import RUNTIME  # local import: keeps import side-effect free

    h16 = nonce_hash(nonce)[:16]
    hexs: Optional[str] = None
    if isinstance(nonce, Nonce):
        hexs = nonce.hex
    elif isinstance(nonce, int):
        hexs = f"{nonce & 0xFFFFFFFF:08x}"
    else:
        s = str(nonce).strip().lower().removeprefix("0x")
        hexs = s if len(s) == 8 else None
    if link is None:
        raise ValueError("record_live needs a PicoLink (link=None)")
    out = Path(out_dir) if out_dir else (RUNTIME / "freshness")
    out.mkdir(parents=True, exist_ok=True)

    cap = _open_camera(cam_index, fps=fps)
    frames: List[np.ndarray] = []
    frame_times: List[float] = []
    for _ in range(5):  # warm-up: let auto-gain settle
        cap.read()
    wall_start = time.time()
    t_start = time.monotonic()
    t_flash: Optional[float] = None
    flashed = False
    fails = 0
    while True:
        ok, fr = cap.read()
        now = time.monotonic()
        if not ok:
            fails += 1
            if fails >= fail_max:
                cap.release()
                raise RuntimeError(f"camera {cam_index} gave {fails} empty reads; aborting")
            continue
        fails = 0
        frames.append(fr)
        frame_times.append(now)
        if not flashed and now - t_start >= 0.5:
            if hexs is None:
                cap.release()
                raise ValueError("record_live needs the nonce value to flash it")
            link.flash(hexs)
            t_flash = time.monotonic()
            flashed = True
        if now - t_start >= seconds:
            break
    cap.release()
    if t_flash is None:
        if hexs is None:
            raise ValueError("record_live needs the nonce value to flash it")
        link.flash(hexs)
        t_flash = time.monotonic()

    rel = [t - frame_times[0] for t in frame_times]
    fps_meas = _measured_fps(rel, fps)
    meta = {"capture_start_wall": wall_start, "t_start_mono": t_start,
            "t_flash_mono": t_flash, "frame_times": frame_times,
            "fps": round(fps_meas, 4), "fps_source": "capture",
            "nonce_hash": nonce_hash(nonce)}
    saved = save_recording(np.stack(frames, axis=0), meta, out, h16)
    # Preview frame for aiming the camera (records surroundings — see privacy note).
    cv2.imwrite(str(out / "frame.jpg"), frames[len(frames) // 2])
    return {"path": saved["path"], "ts_path": saved["ts_path"],
            "t_flash": t_flash, "frame_times": frame_times,
            "fps": round(fps_meas, 4), "fps_source": "capture",
            "capture_start_wall": wall_start, "nonce_hash": nonce_hash(nonce)}
