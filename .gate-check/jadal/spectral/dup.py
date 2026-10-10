"""Duplicate-detection signature (arm B, spectral): multi-scale DCT hash + temporal-spectral fingerprint.

Pure numpy/cv2, CPU-only, deterministic. No torch, no network.
No horizontal-flip invariance: hashes use signed DCT coefficients, so a
mirrored copy scores lower (it is NOT a duplicate per the contract).
"""

import os

import cv2
import numpy as np

_HASH_BITS = 64
_TEMP_SAMPLES = 32
_TEMP_HALF = 16


def _dct_hash(mean_img: np.ndarray) -> str:
    """64-bit hash from top-left 8x8 DCT coeffs, AC-median thresholded.

    mean_img: float64 2-D array (NxN mean gray frame).
    Median is taken over the 63 AC coeffs (DC excluded); all 64 coeffs in
    row-major order are thresholded against it (> median -> 1).
    Returns 16-char zero-padded hex string.
    """
    d = cv2.dct(np.float32(mean_img))
    low = np.float64(d[:8, :8])
    ac = low.copy()
    # Median over AC coefficients only (exclude DC at [0,0]).
    med = float(np.median(np.delete(ac.reshape(-1), 0)))
    bits = 0
    for v in low.reshape(-1):
        bits = (bits << 1) | (1 if float(v) > med else 0)
    return format(bits, "016x")


def _spec_norm(series: np.ndarray) -> np.ndarray:
    """Resample series to 32 samples, z-normalize, rFFT magnitude -> 16-vector summing to 1.

    Excludes the DC bin (magnitudes[1:17]). Zero-variance or zero-energy
    inputs map to the zero vector (the cosine guard in similarity handles it).
    """
    s = np.asarray(series, dtype=np.float64).reshape(-1)
    n = s.shape[0]
    if n == 0:
        return np.zeros(_TEMP_HALF, dtype=np.float64)
    if n == 1:
        r = np.full(_TEMP_SAMPLES, float(s[0]), dtype=np.float64)
    else:
        old = np.linspace(0.0, 1.0, n, dtype=np.float64)
        new = np.linspace(0.0, 1.0, _TEMP_SAMPLES, dtype=np.float64)
        r = np.interp(new, old, s).astype(np.float64)
    mu = float(np.mean(r))
    sd = float(np.std(r))
    if sd == 0.0 or not np.isfinite(sd):
        z = np.zeros_like(r)
    else:
        z = (r - mu) / sd
    mag = np.abs(np.fft.rfft(z))
    v = np.float64(mag[1 : _TEMP_HALF + 1])
    if v.shape[0] < _TEMP_HALF:
        v = np.pad(v, (0, _TEMP_HALF - v.shape[0]))
    total = float(np.sum(v))
    if total == 0.0 or not np.isfinite(total):
        return np.zeros(_TEMP_HALF, dtype=np.float64)
    return v / total


def signature(video_path: str, start_s: float = 0.0, end_s: float | None = None) -> dict:
    """Compute the duplicate signature over [start_s, end_s) using ALL decoded frames.

    Returns {'dct64': hex16, 'dct32': hex16, 'temp': [32 floats],
             'nframes': int, 'fps': float}.
    Raises FileNotFoundError if the path does not exist, ValueError if the
    video cannot be opened/decoded or the range yields no frames.
    """
    if not isinstance(video_path, str) or not os.path.exists(video_path):
        raise FileNotFoundError(f"video not found: {video_path}")
    if end_s is not None and end_s <= start_s:
        raise ValueError(f"empty time range: [{start_s}, {end_s})")
    if start_s < 0:
        raise ValueError(f"negative start_s: {start_s}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video (unsupported/corrupt): {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 0.0
        if fps > 0:
            start_frame = int(start_s * fps)
            end_frame = int(end_s * fps) if end_s is not None else None
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            start_frame, end_frame = 0, None
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)

        acc64 = np.zeros((64, 64), dtype=np.float64)
        acc32 = np.zeros((32, 32), dtype=np.float64)
        means: list[float] = []
        diffs: list[float] = []
        prev: np.ndarray | None = None
        idx = 0
        nframes = 0
        while True:
            if end_frame is not None and (start_frame + idx) >= end_frame:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if fps <= 0 and end_s is not None:
                pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                if pos_ms >= end_s * 1000.0:
                    break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g64 = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            g32 = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
            acc64 += np.float64(g64)
            acc32 += np.float64(g32)
            means.append(float(np.mean(np.float64(gray))))
            if prev is None:
                diffs.append(0.0)
            else:
                diffs.append(float(np.mean(np.abs(np.float64(gray) - prev))))
            prev = np.float64(gray)
            nframes += 1
            idx += 1
    finally:
        cap.release()

    if nframes == 0:
        raise ValueError(f"no frames decoded in range [{start_s}, {end_s}) of {video_path}")

    mean64 = acc64 / float(nframes)
    mean32 = acc32 / float(nframes)
    temp = np.concatenate([_spec_norm(np.array(means)), _spec_norm(np.array(diffs[1:] if nframes > 1 else [0.0]))])
    return {
        "dct64": _dct_hash(mean64),
        "dct32": _dct_hash(mean32),
        "temp": [float(v) for v in temp],
        "nframes": int(nframes),
        "fps": float(fps),
    }


def _hamming(hex_a: str, hex_b: str) -> int:
    return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()


def similarity(a: dict, b: dict) -> float:
    """Duplicate similarity in [0,1].

    0.25*(1-ham64/64) + 0.25*(1-ham32/64) + 0.5*cosine(temp).
    Zero-norm temp vectors guard to cosine 0.5 (neutral).
    """
    t1 = 1.0 - _hamming(a["dct64"], b["dct64"]) / _HASH_BITS
    t2 = 1.0 - _hamming(a["dct32"], b["dct32"]) / _HASH_BITS
    ta = np.asarray(a["temp"], dtype=np.float64).reshape(-1)
    tb = np.asarray(b["temp"], dtype=np.float64).reshape(-1)
    na = float(np.sqrt(np.sum(ta * ta)))
    nb = float(np.sqrt(np.sum(tb * tb)))
    if na == 0.0 or nb == 0.0 or not (np.isfinite(na) and np.isfinite(nb)):
        cos = 0.5
    else:
        cos = float(np.sum(ta * tb) / (na * nb))
        cos = min(1.0, max(0.0, cos))
    sim = 0.25 * t1 + 0.25 * t2 + 0.5 * cos
    return min(1.0, max(0.0, float(sim)))
