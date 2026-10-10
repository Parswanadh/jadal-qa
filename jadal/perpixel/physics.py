"""Physics/signal features per research3/physics_features.md top specs.

Implemented: 1 (Poisson-Gaussian noise + SNR), 2 (H.264 bytes/frame via
ffprobe, decode-free), 3 (frame diff vs noise floor), 5 (noise-normalised
Tenengrad sharpness), 6 (residual optical flow after median ego-motion
removal), 4 (dHash + loop check), exposure-clip + histogram entropy (#6-doc).
SKIPPED per instructions: heat-equation scale-space (#11), CCT (#13),
MSCN (#10), max-entropy (#14).

Deviations from the doc, documented: pixels come from the 160x90 stream
decode (fixed downscale, so thresholds still transfer across source
resolutions); Y is 0-255 DN; Farneback runs at 160x90 (params fixed).
All math float64, deterministic, fixed params. Window aggregates = median
over frames (robust). <20 usable blocks -> NaN (never guessed).

Units: a unitless, b DN^2, SNR dB, bytes in bytes, MAD DN, Tenengrad DN^2,
flow px/frame, dHash bits, clip fraction [0,1], entropy bits.
"""

from __future__ import annotations

import json
import subprocess

import cv2
import numpy as np

PHYSICS_FEATURES = [
    "pg_a", "pg_b", "snr_db",          # spec 1
    "frm_bytes_med", "frm_frozen_frac",  # spec 2
    "i_size", "p_size_med",
    "mad_mean", "mad_frozen_frac",      # spec 3
    "ten_raw", "ten_norm",              # spec 5
    "flow_act", "flow_med",             # spec 6
    "dhash_loop_min",                   # spec 4
    "clip_frac", "hist_ent",            # exposure + entropy
]

FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                 poly_n=5, poly_sigma=1.2, flags=0)


def _blocks(Y: np.ndarray):
    """16x16 blocks with mean in [25.5,229.5] and low gradient energy."""
    H, W = Y.shape
    gy, gx = np.gradient(Y)
    ge = gx ** 2 + gy ** 2
    out = []
    for y in range(0, H - 15, 16):
        for x in range(0, W - 15, 16):
            b = Y[y:y + 16, x:x + 16]
            mu = float(b.mean())
            if 25.5 <= mu <= 229.5:
                out.append((mu, float(ge[y:y + 16, x:x + 16].mean()), b))
    return out


def noise_model(Y: np.ndarray):
    """Spec 1 on one frame (kept for API compat; under-powered at 160x90)."""
    a, b, s = noise_model_stack(Y[None], np.array([0]))
    return a, b, s


def noise_model_stack(Ys: np.ndarray, samp: np.ndarray):
    """Spec 1, window-pooled: 16x16 blocks from all sampled frames.

    At 160x90 one frame yields ~50 candidate blocks; the 30% flat cut then
    leaves <20 and the fit always fails. Pooling sampled frames (~12x540
    candidates) restores the spec's intent (same block size, cut, minimum).
    <20 pooled blocks -> (NaN, NaN, NaN), never guessed.
    """
    mus, vs = [], []
    for i in samp:
        for mu, g, b in _blocks(Ys[i]):
            r = b - cv2.blur(b, (3, 3))
            mus.append(mu)
            vs.append((float(r.var()) * 1.5, g))
    if len(mus) < 20:
        return float("nan"), float("nan"), float("nan")
    gs = np.array([g for _, g in vs])
    cut = np.percentile(gs, 30)
    m2 = [m for m, (v, g) in zip(mus, vs) if g <= cut]
    v2 = [v for v, g in vs if g <= cut]
    if len(m2) < 20:
        return float("nan"), float("nan"), float("nan")
    A = np.column_stack([np.asarray(m2), np.ones(len(m2))])
    (a, b), *_ = np.linalg.lstsq(A, np.asarray(v2), rcond=None)
    a, b = float(a), float(max(b, 1e-9))
    snr = float(np.median([m / np.sqrt(a * m + b) for m in m2]))
    return a, b, float(10 * np.log10(max(snr, 1e-12)))


def frame_bytes(path: str, start_s: float, end_s: float, fps_hint: float = 10.0):
    """Spec 2: decode-free per-frame packet sizes in [start,end).

    Uses -read_intervals (input -ss returns no packets on this build).
    Fail-LOUD (fix round 1/I1): missing file, ffprobe error, or zero
    in-window packets all raise — never silent NaN/zero features.
    (jadal/spectral compressed_stats swallows the same failures into zeros,
    but that file is read-only / another worker's.)
    """
    import os
    if not isinstance(path, str) or not os.path.exists(path):
        raise FileNotFoundError(f"video not found: {path}")
    n_exp = max(1, int(round((end_s - start_s) * fps_hint)) + 60)
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_packets", "-select_streams", "v:0",
         "-read_intervals", f"{start_s}%+{n_exp}",
         "-show_entries", "packet=size,pts_time,flags", "-of", "json", path],
        capture_output=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed rc={p.returncode}: "
                           f"{p.stderr[:200]!r}")
    try:
        pkts = json.loads(p.stdout or b"{}").get("packets", [])
    except (ValueError, AttributeError) as e:
        raise RuntimeError(f"ffprobe output unparseable: {e}") from e
    sizes, keys = [], []
    for q in pkts:
        try:
            t = float(q.get("pts_time", -1))
        except (TypeError, ValueError):
            continue
        if not (start_s <= t < end_s):
            continue
        try:
            s = int(q["size"])
        except (KeyError, TypeError, ValueError):
            continue
        sizes.append(s)
        keys.append("K" in str(q.get("flags", "")))
    sizes = np.asarray(sizes, dtype=np.float64)
    if len(sizes) < 3:
        raise RuntimeError(
            f"ffprobe returned only {len(sizes)} in-window packets for "
            f"[{start_s},{end_s}) of {path} (need >= 3)")
    med = float(np.median(sizes))
    nonkey = sizes[np.array(keys) == False]  # noqa: E712
    frozen = float((nonkey < 0.05 * med).mean()) if len(nonkey) else 0.0
    ik = sizes[np.array(keys)]
    return {"frm_bytes_med": med, "frm_frozen_frac": frozen,
            "i_size": float(ik.mean()) if len(ik) else 0.0,
            "p_size_med": float(np.median(nonkey)) if len(nonkey) else 0.0}


def dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return int(np.packbits((small[:, 1:] > small[:, :-1]).ravel()).tobytes().hex(), 16)


def _ham(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def extract_physics(frames90: np.ndarray, fps: float, path: str,
                    start_s: float, end_s: float) -> dict:
    """frames90 (T,H,W,3) float64 [0,1] + source window -> PHYSICS_FEATURES."""
    T = frames90.shape[0]
    Y = ((0.299 * frames90[..., 0] + 0.587 * frames90[..., 1]
          + 0.114 * frames90[..., 2]) * 255.0)
    clip = Y <= 2
    clip |= Y >= 253
    samp = np.linspace(0, T - 1, min(T, 12)).astype(int)
    # spec 1, window-pooled (per-frame fits are under-powered at 160x90)
    a, b, snr = noise_model_stack(Y, samp)
    TEN, TENN, MAD, FRZ = [], [], [], []
    from jadal.perpixel import theory as _T
    sig_floor = float(np.sqrt(b)) if np.isfinite(b) else float(
        np.median([_T.immerkaer_sigma(Y[i]) for i in samp]))
    if not np.isfinite(sig_floor) or sig_floor <= 0:
        sig_floor = 1.0
    for i in samp:
        y = Y[i]
        # normalisation uses the spec-1 floor when valid, else Immerkaer.
        sig2 = sig_floor ** 2 + 1e-12
        h, w = y.shape
        roi = y[h // 4:3 * h // 4, w // 4:3 * w // 4]
        gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1)
        t = float((gx ** 2 + gy ** 2).mean())
        TEN.append(t)
        TENN.append(t / sig2)
        if i > 0:
            d = np.abs(y - Y[i - 1])
            m = float(d[~clip[i] & ~clip[i - 1]].mean()
                      if (~clip[i] & ~clip[i - 1]).any() else d.mean())
            MAD.append(m)
            FRZ.append(1.0 if m < 0.5 * sig_floor else 0.0)
    # spec 6: residual flow after median (ego-motion) removal, lower-60% ROI
    H, W = Y.shape[1:]
    acts, meds = [], []
    step = max(1, T // 25)  # <=25 pairs per window (cost cap)
    for i in range(0, T - 1, step):
        p0 = Y[i].astype(np.float32)
        if float(p0.std()) < 1e-6:
            continue
        F = cv2.calcOpticalFlowFarneback(p0, Y[i + 1].astype(np.float32),
                                         None, **FARNEBACK)
        g = np.median(F.reshape(-1, 2), axis=0)
        R = np.hypot(*(F - g).transpose(2, 0, 1))
        roi = R[int(0.4 * H):, :]
        acts.append(float((roi > 1.0).mean()))
        meds.append(float(np.median(roi)))
    fb = frame_bytes(path, start_s, end_s)
    # spec 4: loop check — min Hamming to frames >= 2 s earlier
    hs = [dhash(Y[i]) for i in range(T)]
    gap = max(1, int(round(2 * fps)))
    loop = min((_ham(hs[i], hs[j]) for i in range(gap, T)
                for j in range(0, i - gap + 1)), default=64)
    ent = []
    for i in samp:
        h, _ = np.histogram(Y[i], bins=256, range=(0, 255))
        q = h / (h.sum() + 1e-12)
        q = q[q > 0]
        ent.append(float(-(q * np.log2(q)).sum()))
    out = {
        "pg_a": a, "pg_b": b, "snr_db": snr,
        "mad_mean": float(np.median(MAD)) if MAD else 0.0,
        "mad_frozen_frac": float(np.mean(FRZ)) if FRZ else 0.0,
        "ten_raw": float(np.median(TEN)), "ten_norm": float(np.median(TENN)),
        "flow_act": float(np.median(acts)) if acts else 0.0,
        "flow_med": float(np.median(meds)) if meds else 0.0,
        "dhash_loop_min": float(loop),
        "clip_frac": float(clip.mean()),
        "hist_ent": float(np.median(ent)),
    }
    out.update(fb)
    return {k: float(v) for k, v in out.items()}
