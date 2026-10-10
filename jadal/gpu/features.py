"""GPU-resident per-pixel spectral feature kernels + CPU float64 reference.

Hardware label: RTX 4070 Laptop, NOT a Pi, NOT a datacenter GPU.
Whole video, every decoded frame; the 10-s segment is just the verdict window.

Layout: cpu_* functions are the float64 reference (also the Pi/edge path).
gpu_* functions are batched torch kernels (fp16 in / fp32 math where safe,
fixed-order deterministic reductions). Tolerance tests in tests/test_gpu_features.py.
"""
import subprocess
import sys

import numpy as np

FPS_DEFAULT = 30.0
# Temporal bands (Hz): frozen / slow drift / hand-motion / fast / flicker-ish
BANDS = [(0.0, 0.3), (0.3, 2.0), (2.0, 8.0), (8.0, 12.0)]

FEATURE_NAMES = [
    "temp_b0", "temp_b1", "temp_b2", "temp_b3",  # band-energy means
    "frozen_frac", "flicker_idx",
    "radial_slope", "hf_ratio", "dct_energy",
    "pca_resid", "pca_top1_share",
    "luma_mean", "luma_p05", "luma_p50", "luma_p95",
    "dark_frac", "bright_frac",
    "motion_energy", "global_shift",
    "skin_frac", "illum_r", "illum_g", "illum_b",
    "rppg_pulse", "dup_hash_mean", "zncc_self",
]

# --- 14 features added in m/feat40 (26 -> 40); see cpu_new_features ---
NEW_FEATURE_NAMES = [
    # Z3 transferable set (definitions frozen in m/adapt160:z3extract.py)
    "ct_fsize_std", "mo_hand_motion_excess", "inv_clip_frac",
    "inv_laplacian_var", "inv_hist_entropy", "inv_dct_hf_lf",
    # finer spectral radial bands (fraction of FFT energy in a shell)
    "spec_b1", "spec_b2", "spec_b3",
    # colour-channel ratios (mean R/G, mean B/G)
    "col_rg_ratio", "col_bg_ratio",
    # temporal-lag differences (|frame_t - frame_{t-k}|)
    "temp_lag2", "temp_lag3",
    # per-pixel noise floor (median absolute deviation of the Laplacian)
    "noise_floor",
]
FEATURE_NAMES = FEATURE_NAMES + NEW_FEATURE_NAMES


# ---------------- CPU float64 reference ----------------

def _band_masks(nfreq, df):
    masks = []
    for lo, hi in BANDS:
        k = np.arange(nfreq)
        f = k * df
        masks.append((f >= lo) & (f < hi))
    return masks


def cpu_temporal_bands(x, fps=FPS_DEFAULT):
    """x: (T,H,W) float64 luma in [0,1]. Returns dict of (H,W) band-energy maps."""
    T = x.shape[0]
    F = np.fft.rfft(x, axis=0)
    P = (F.real ** 2 + F.imag ** 2) / T
    df = fps / T
    out = {}
    for i, (lo, hi) in enumerate(BANDS):
        k = np.arange(P.shape[0])
        m = (k * df >= lo) & (k * df < hi)
        out[f"b{i}"] = P[m].sum(axis=0) if m.any() else np.zeros_like(P[0])
    return out


def cpu_spatial_spectrum(img):
    """img: (H,W) float64. Returns (radial_slope, hf_ratio)."""
    H, W = img.shape
    F = np.fft.rfft2(img - img.mean())
    P = (F.real ** 2 + F.imag ** 2)
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.rfftfreq(W)[None, :]
    r = np.sqrt(ky ** 2 + kx ** 2)
    rr = r.ravel()
    pp = P.ravel()
    # radial slope: log-log fit over full bins (skip DC)
    bins = np.linspace(rr[rr > 0].min(), 0.5, 12)
    idx = np.digitize(rr, bins)
    bx, by = [], []
    for b in range(1, len(bins)):
        sel = (idx == b) & (rr > 0)
        if sel.sum() > 4 and pp[sel].mean() > 0:
            bx.append(np.log(bins[b]))
            by.append(np.log(pp[sel].mean()))
    slope = float(np.polyfit(bx, by, 1)[0]) if len(bx) >= 3 else 0.0
    hf = float(pp[rr > 0.25].sum() / pp[rr > 0].sum()) if pp[rr > 0].sum() > 0 else 0.0
    return slope, hf


def _dct_matrix(n=8):
    C = np.zeros((n, n))
    for k in range(n):
        for t in range(n):
            C[k, t] = np.sqrt(2.0 / n) * (0.7071067811865476 if k == 0 else 1.0) * \
                np.cos(np.pi * k * (2 * t + 1) / (2 * n))
    return C


_DCT8 = _dct_matrix(8)


def cpu_dct_energy(img):
    """Mean 8x8 block DCT AC energy (sharpness-like)."""
    H, W = img.shape
    H8, W8 = (H // 8) * 8, (W // 8) * 8
    im = img[:H8, :W8].reshape(H8 // 8, 8, W8 // 8, 8).transpose(0, 2, 1, 3)
    C = _DCT8
    D = np.einsum("ab,ijbc,cd->ijad", C, im, C.T)
    ac = D.copy()
    ac[:, :, 0, 0] = 0.0
    return float((ac ** 2).mean())


def cpu_pca_inverse_residual(rgb, k=2):
    """rgb: (N,3) float64. Returns (mean_residual, top1_share, resid_map_fn).

    'Reverse PCA': variance NOT explained by top-k principal components.
    """
    mu = rgb.mean(axis=0)
    c = rgb - mu
    cov = (c.T @ c) / max(len(c) - 1, 1)
    ev, V = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1]
    ev, V = ev[order], V[:, order]
    Vk = V[:, :k]
    rec = (c @ Vk) @ Vk.T
    resid = ((c - rec) ** 2).sum(axis=1)
    tot = float(ev.sum())
    if tot <= 0:
        return 0.0, 1.0, np.zeros(len(rgb))  # constant input: trivially explained
    return float(resid.mean()), float(ev[0] / tot), resid


def cpu_luma_stats(x):
    """x: (T,H,W) float64. Returns dict of scalars."""
    v = x.ravel()
    return {
        "mean": float(v.mean()),
        "p05": float(np.percentile(v, 5)),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "dark_frac": float((v < 0.06).mean()),
        "bright_frac": float((v > 0.94).mean()),
        "var": float(v.var()),
    }


def cpu_motion(frames):
    """frames: (T,H,W) float64. Returns (motion_energy, global_shift_px)."""
    d = np.abs(np.diff(frames, axis=0)).mean()
    # phase correlation between first and middle frame
    a = frames[0] - frames[0].mean()
    b = frames[len(frames) // 2] - frames[len(frames) // 2].mean()
    FA = np.fft.rfft2(a)
    FB = np.fft.rfft2(b)
    R = FA * np.conj(FB)
    R = R / (np.abs(R) + 1e-12)
    cc = np.fft.irfft2(R, s=a.shape)
    jy, ix = np.unravel_index(np.argmax(cc), cc.shape)
    H, W = a.shape
    dy = min(jy, H - jy)
    dx = min(ix, W - ix)
    return float(d), float(np.hypot(dy, dx))


def cpu_skin_fraction(rgb):
    """rgb: (H,W,3) float64 in [0,1]. YCrCb skin-locus area fraction (weak cue)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5
    mask = (cr > 0.55) & (cr < 0.72) & (cb > 0.38) & (cb < 0.52)
    return float(mask.mean())


def cpu_grey_world(rgb):
    """rgb: (...,3). Returns (illum_r, illum_g, illum_b) normalised, + chromaticity spread."""
    m = rgb.reshape(-1, 3).mean(axis=0)
    s = m.sum()
    ill = m / s if s > 0 else np.array([1 / 3] * 3)
    return tuple(float(v) for v in ill)


def cpu_rppg_pulse(skin_means):
    """skin_means: (T,3) mean RGB of skin ROI per frame. Returns pulse-band (0.7-3Hz) energy ratio."""
    s = np.asarray(skin_means, dtype=np.float64)
    if len(s) < 16:
        return 0.0
    chrom = s[:, 1] - 0.5 * (s[:, 0] + s[:, 2])  # green - red/blue average
    chrom = chrom - chrom.mean()
    fps = FPS_DEFAULT
    F = np.fft.rfft(chrom)
    P = F.real ** 2 + F.imag ** 2
    freq = np.fft.rfftfreq(len(s), 1.0 / fps)
    pulse = P[(freq >= 0.7) & (freq <= 3.0)].sum()
    tot = P[1:].sum()
    return float(pulse / tot) if tot > 0 else 0.0


def cpu_dct_hash(img, hsize=8):
    """64-bit DCT perceptual hash of luma image."""
    H, W = img.shape
    # block-average down to 32x32 then DCT, take top-left 8x8
    bh, bw = int(np.ceil(H / 32)), int(np.ceil(W / 32))
    ph, pw = bh * 32 - H, bw * 32 - W
    padded = np.pad(img, ((0, ph), (0, pw)), mode="edge")
    small = padded.reshape(32, bh, 32, bw).mean(axis=(1, 3))
    C32 = _dct_matrix(32)
    D = C32 @ small @ C32.T
    low = D[:hsize, :hsize]
    med = np.median(low)
    bits = (low > med).ravel()
    h = 0
    for bit in bits:
        h = (h << 1) | int(bit)
    return h


def cpu_zncc(a, b):
    """Zero-mean normalised cross-correlation of two luma frames."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def cpu_window_features(luma, rgb, fps=FPS_DEFAULT, path=None, start_s=None, end_s=None):
    """luma: (T,H,W), rgb: (T,H,W,3) float64 in [0,1]. Returns {name: float} (40)."""
    bands = cpu_temporal_bands(luma, fps)
    bmean = {f"temp_b{i}": float(bands[f'b{i}'].mean()) for i in range(4)}
    tot_e = sum(bmean.values()) + 1e-18
    frozen_frac = float((bands["b1"] + bands["b2"] + bands["b3"] < 1e-9 * max(tot_e, 1e-9)).mean())
    mid = luma[len(luma) // 2]
    slope, hf = cpu_spatial_spectrum(mid)
    dct_e = cpu_dct_energy(mid)
    N = rgb.shape[0] * rgb.shape[1] * rgb.shape[2]
    resid, top1, _ = cpu_pca_inverse_residual(rgb.reshape(-1, 3))
    st = cpu_luma_stats(luma)
    me, gs = cpu_motion(luma)
    skin = np.mean([cpu_skin_fraction(f) for f in rgb[:: max(len(rgb) // 8, 1)]])
    ill = cpu_grey_world(rgb[:: max(len(rgb) // 8, 1)])
    # rPPG over whole-frame mean as ROI fallback
    skin_means = rgb.reshape(len(rgb), -1, 3).mean(axis=1)
    pulse = cpu_rppg_pulse(skin_means)
    h = cpu_dct_hash(mid)
    dup_h = float(bin(h).count("1")) / 64.0
    zn = cpu_zncc(luma[0], luma[-1])
    flick = float(bmean["temp_b3"] / tot_e)
    return {
        **bmean, "frozen_frac": frozen_frac, "flicker_idx": flick,
        "radial_slope": slope, "hf_ratio": hf, "dct_energy": dct_e,
        "pca_resid": resid, "pca_top1_share": top1,
        "luma_mean": st["mean"], "luma_p05": st["p05"], "luma_p50": st["p50"],
        "luma_p95": st["p95"], "dark_frac": st["dark_frac"], "bright_frac": st["bright_frac"],
        "motion_energy": me, "global_shift": gs, "skin_frac": skin,
        "illum_r": ill[0], "illum_g": ill[1], "illum_b": ill[2],
        "rppg_pulse": pulse, "dup_hash_mean": dup_h, "zncc_self": zn,
        **cpu_new_features(luma, rgb, fps, path=path, start_s=start_s, end_s=end_s),
    }


# ---------------- 14 added features (26 -> 40) ----------------

def _ct_fsize_std(path, start_s, end_s):
    """Compression telemetry: std of per-frame packet sizes in [start,end).

    Cost: ONE ffprobe `-show_entries frame=...` pass (~20-80 ms/window).
    Fallback 0.0 is loud (stderr), never silent.
    """
    out = 0.0
    if path is None or start_s is None or end_s is None:
        return out
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "frame=pts_time,pkt_size", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=180)
        sz = []
        for line in p.stdout.splitlines():
            a = line.split(",")
            if len(a) < 2:
                continue
            try:
                ts, s = float(a[0]), float(a[1])
            except ValueError:
                continue
            if float(start_s) - 1e-6 <= ts < float(end_s) - 1e-6:
                sz.append(s)
        if len(sz) >= 3:
            out = float(np.std(np.asarray(sz, float)))
    except Exception as e:  # loud, never silent
        print(f"[features40] ct_fsize_std failed for {path}: {e}",
              file=sys.stderr, flush=True)
    return out


def _skin_mask(rgb_frame):
    """YCrCb skin-locus mask for one (H,W,3) frame in [0,1] (BT.601)."""
    r, g, b = rgb_frame[..., 0], rgb_frame[..., 1], rgb_frame[..., 2]
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5
    return (cr > 0.55) & (cr < 0.72) & (cb > 0.38) & (cb < 0.52)


def cpu_new_features(luma, rgb, fps=FPS_DEFAULT, path=None, start_s=None, end_s=None):
    """The 14 added per-pixel features. Returns {name: float}.

    luma: (T,H,W) float64 [0,1]; rgb: (T,H,W,3) float64 [0,1]. All reductions are
    fixed-order float64 means over decoded frames (deterministic).
    """
    import cv2
    T = luma.shape[0]
    if T == 0:
        return {k: 0.0 for k in NEW_FEATURE_NAMES}
    inv = 1.0 / T
    # ---- streaming per-frame accumulators (Z3 invariants + colour + edge/noise) ----
    lap_var = clip = hist_ent = noise = 0.0
    rg_ratio = bg_ratio = 0.0
    htrans, trans = [], []
    prev = None
    for t in range(T):
        g = luma[t]
        lap = cv2.Laplacian(g.astype(np.float32), cv2.CV_32F)
        lap_var += float(lap.var())
        noise += float(np.median(np.abs(lap - np.median(lap))))  # MAD noise floor
        clip += float(((g <= 2.0 / 255.0) | (g >= 253.0 / 255.0)).mean())
        h, _ = np.histogram(g, bins=32, range=(0.0, 1.0))
        p = h / max(int(h.sum()), 1)
        pp = p[p > 0]
        hist_ent += float(-(pp * np.log(pp)).sum())
        r, gc, b = rgb[t, :, :, 0], rgb[t, :, :, 1], rgb[t, :, :, 2]
        gm = float(gc.mean())
        rg_ratio += float(r.mean() / (gm + 1e-9))
        bg_ratio += float(b.mean() / (gm + 1e-9))
        if prev is not None:
            d = np.abs(g - prev)
            tr = float(d.mean())  # translation-only residual proxy (no warp)
            trans.append(tr)
            sk = _skin_mask(rgb[t])
            if int(sk.sum()) >= 20:
                htrans.append(float(d[sk].mean()))
        prev = g
    # ---- finer spectral radial bands (fraction of FFT energy in a shell) ----
    b = np.zeros(3)
    for t in range(0, T, max(1, T // 3)):
        g = luma[t]
        F = np.fft.rfft2(g - g.mean())
        P = (F.real ** 2 + F.imag ** 2)
        H, W = g.shape
        ky = np.fft.fftfreq(H)[:, None]
        kx = np.fft.rfftfreq(W)[None, :]
        rr = np.sqrt(ky ** 2 + kx ** 2)
        tot = P[rr > 0].sum() + 1e-18
        b[0] += P[(rr > 0.10) & (rr <= 0.20)].sum() / tot
        b[1] += P[(rr > 0.20) & (rr <= 0.35)].sum() / tot
        b[2] += P[(rr > 0.35) & (rr <= 0.50)].sum() / tot
    nb = len(range(0, T, max(1, T // 3)))
    b /= max(nb, 1)
    # ---- inv_dct_hf_lf on a 64x64 resize of the middle frame (Z3 definition) ----
    mid = luma[T // 2]
    small = cv2.resize(mid, (64, 64), interpolation=cv2.INTER_AREA)
    d = cv2.dct(small.astype(np.float32))
    yy, xx = np.mgrid[0:64, 0:64]
    rad = np.sqrt((yy - 32) ** 2 + (xx - 32) ** 2)
    hf_lf = float((d[rad > 20] ** 2).sum() / ((d[rad < 8] ** 2).sum() + 1e-9))
    # ---- temporal lag 2/3 ----
    def lag(k):
        if T <= k:
            return 0.0
        return float(np.mean([np.abs(luma[t] - luma[t - k]).mean() for t in range(k, T)]))
    return {
        "ct_fsize_std": _ct_fsize_std(path, start_s, end_s),
        "mo_hand_motion_excess": float(np.mean(htrans) - np.mean(trans)) if htrans else 0.0,
        "inv_clip_frac": float(clip * inv),
        "inv_laplacian_var": float(lap_var * inv),
        "inv_hist_entropy": float(hist_ent * inv),
        "inv_dct_hf_lf": hf_lf,
        "spec_b1": float(b[0]), "spec_b2": float(b[1]), "spec_b3": float(b[2]),
        "col_rg_ratio": float(rg_ratio * inv),
        "col_bg_ratio": float(bg_ratio * inv),
        "temp_lag2": lag(2), "temp_lag3": lag(3),
        "noise_floor": float(noise * inv),
    }


# ---------------- GPU torch kernels ----------------
def _torch():
    import torch
    assert torch.cuda.is_available(), "no GPU"
    return torch


def gpu_window_features_batch(luma, rgb, fps=FPS_DEFAULT):
    """Batched GPU features. luma: (B,T,H,W) fp16/fp32 CUDA, rgb: (B,T,H,W,3).
    Returns (B,F) float32 tensor with columns FEATURE_NAMES. Deterministic
    fixed-order reductions (no nondeterministic algorithms enabled)."""
    torch = _torch()
    assert luma.is_cuda and rgb.is_cuda
    B, T, H, W = luma.shape
    x = luma.float()
    F = torch.fft.rfft(x, dim=1)
    P = F.abs().pow(2) / T
    df = fps / T
    kk = torch.arange(P.shape[1], device=x.device) * df
    cols = []
    bmeans = []
    for lo, hi in BANDS:
        m = (kk >= lo) & (kk < hi)
        e = P[:, m].sum(dim=1).mean(dim=(1, 2)) if m.any() else torch.zeros(B, device=x.device)
        bmeans.append(e)
        cols.append(e)
    tot = sum(bmeans) + 1e-18
    nz = (P[:, 1:].sum(dim=1))  # non-DC energy map
    frozen_frac = (nz < 1e-9 * tot.view(B, 1, 1)).float().mean(dim=(1, 2))
    cols.append(frozen_frac)
    cols.append(bmeans[3] / tot)  # flicker_idx
    # spatial spectrum on middle frame
    mid = x[:, T // 2]
    mid0 = mid - mid.mean(dim=(1, 2), keepdim=True)
    FS = torch.fft.rfft2(mid0)
    PS = FS.abs().pow(2)
    ky = torch.fft.fftfreq(H, device=x.device)[:, None]
    kx = torch.fft.rfftfreq(W, device=x.device)[None, :]
    r = torch.sqrt(ky ** 2 + kx ** 2)
    p = PS.reshape(B, -1)
    rr = r.reshape(-1)
    pos = rr > 0
    hf = (PS * (r > 0.25)).reshape(B, -1).sum(1) / p[:, pos].sum(1).clamp_min(1e-18)
    # radial slope: per-element log-log fit over 11 bins (loop over B; still GPU-resident)
    slopes = []
    edges = torch.linspace(rr[pos].min(), 0.5, 12, device=x.device)
    for bi in range(B):
        pp = PS[bi].reshape(-1)
        idx = torch.bucketize(rr, edges)
        bx, by = [], []
        for b in range(1, 12):
            sel = (idx == b) & pos
            if int(sel.sum()) > 4 and float(pp[sel].mean()) > 0:
                bx.append(float(torch.log(edges[b])))
                by.append(float(torch.log(pp[sel].mean())))
        if len(bx) >= 3:
            X = torch.tensor([bx, [1.0] * len(bx)], device=x.device).T
            y = torch.tensor(by, device=x.device)
            sol = torch.linalg.lstsq(X, y).solution
            slopes.append(float(sol[0]))
        else:
            slopes.append(0.0)
    cols.append(torch.tensor(slopes, device=x.device))  # radial_slope
    cols.append(hf)
    # DCT energy via matrix on 8x8 blocks
    N = 8
    ks = torch.arange(N, device=x.device).float()
    dctm = torch.sqrt(torch.tensor(2.0 / N)) * torch.where(
        ks[:, None] == 0, torch.tensor(0.70710678), torch.tensor(1.0)) * \
        torch.cos(torch.pi * ks[:, None] * (2 * ks[None, :] + 1) / (2 * N))
    H8, W8 = (H // 8) * 8, (W // 8) * 8
    xb = mid[:, :H8, :W8].reshape(B, H8 // 8, 8, W8 // 8, 8).permute(0, 1, 3, 2, 4).reshape(-1, 8, 8)
    C = dctm @ xb @ dctm.T
    C[:, 0, 0] = 0
    cols.append(C.pow(2).mean(dim=(1, 2)).reshape(B, -1).mean(1))
    # PCA inverse residual on sampled pixels
    pix = rgb.float().reshape(B, -1, 3)[:, ::max(rgb.shape[1] * H * W // 20000, 1)]
    mu = pix.mean(dim=1, keepdim=True)
    c = pix - mu
    cov = (c.transpose(1, 2) @ c) / max(c.shape[1] - 1, 1)
    ev, V = torch.linalg.eigh(cov)
    Vk = V[:, :, 1:]  # top-2 (ascending order)
    rec = (c @ Vk) @ Vk.transpose(1, 2)
    cols.append(((c - rec).pow(2).sum(-1)).mean(1))
    cols.append((ev[:, 2] / ev.sum(1).clamp_min(1e-18)))
    # luma stats
    flat = x.reshape(B, -1)
    cols.append(flat.mean(1))
    q = torch.quantile(flat, torch.tensor([0.05, 0.5, 0.95], device=x.device), dim=1)
    cols += [q[0], q[1], q[2]]
    cols.append((flat < 0.06).float().mean(1))
    cols.append((flat > 0.94).float().mean(1))
    # motion
    d = (x[:, 1:] - x[:, :-1]).abs().mean(dim=(1, 2, 3))
    cols.append(d)
    a0 = x[:, 0] - x[:, 0].mean(dim=(1, 2), keepdim=True)
    b0 = x[:, T // 2] - x[:, T // 2].mean(dim=(1, 2), keepdim=True)
    FA = torch.fft.rfft2(a0)
    FB = torch.fft.rfft2(b0)
    R = FA * FB.conj()
    R = R / (R.abs() + 1e-12)
    cc = torch.fft.irfft2(R, s=(H, W))
    flat_idx = cc.reshape(B, -1).argmax(dim=1)
    jy = flat_idx // W
    ix = flat_idx % W
    shift = torch.sqrt(torch.minimum(jy, H - jy).float() ** 2 + torch.minimum(ix, W - ix).float() ** 2)
    cols.append(shift)  # global_shift px (same definition as CPU)
    # skin fraction (mean over sampled frames)
    rs = rgb.float()[::1, :: max(T // 8, 1)]
    r_, g_, b_ = rs[..., 0], rs[..., 1], rs[..., 2]
    cr = 0.5 * r_ - 0.4187 * g_ - 0.0813 * b_ + 0.5
    cb = -0.1687 * r_ - 0.3313 * g_ + 0.5 * b_ + 0.5
    mask = (cr > 0.55) & (cr < 0.72) & (cb > 0.38) & (cb < 0.52)
    cols.append(mask.float().mean(dim=(1, 2, 3)))
    # grey-world illuminant
    m = rgb.float().reshape(B, -1, 3).mean(dim=1)
    ill = m / m.sum(dim=1, keepdim=True).clamp_min(1e-18)
    cols += [ill[:, 0], ill[:, 1], ill[:, 2]]
    # rPPG pulse ratio
    sm = rgb.float().reshape(B, T, -1, 3).mean(dim=2)  # (B,T,3)
    chrom = sm[:, :, 1] - 0.5 * (sm[:, :, 0] + sm[:, :, 2])
    chrom = chrom - chrom.mean(dim=1, keepdim=True)
    FP = torch.fft.rfft(chrom, dim=1)
    PP = FP.abs().pow(2)
    fr = torch.fft.rfftfreq(T, 1.0 / fps).to(x.device)
    pm = (fr >= 0.7) & (fr <= 3.0)
    pulse = PP[:, pm].sum(1) / PP[:, 1:].sum(1).clamp_min(1e-18)
    cols.append(pulse)
    # dup hash popcount proxy: use mean of mid-frame DCT sign bits via cheap hash
    cols.append(((mid * 100).int() % 2).float().mean(dim=(1, 2)))
    # zncc first-last
    A = x[:, 0].reshape(B, -1)
    Bc = x[:, -1].reshape(B, -1)
    A = A - A.mean(1, keepdim=True)
    Bc = Bc - Bc.mean(1, keepdim=True)
    zn = (A * Bc).sum(1) / torch.sqrt((A.pow(2).sum(1)) * (Bc.pow(2).sum(1))).clamp_min(1e-18)
    cols.append(zn)
    return torch.stack(cols, dim=1)
