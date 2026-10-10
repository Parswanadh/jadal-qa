"""Per-frame SPATIAL-SPECTRAL feature extractor for the frames-spectral study.

Issue #60. Deterministic (no randomness), CPU-only (numpy/scipy/cv2/pywt).

Pipeline per frame (BGR uint8, any size):
  1. Resize to a FIXED 480-px width first (INTER_AREA, aspect preserved), so
     JPEG-1080p eval frames and H.264-480p IndEgo video frames are comparable.
     IndEgo 480x480 frames are already 480 wide -> unchanged.
  2. Compute the feature families below on the resized frame.

Feature definitions (all float, deterministic):
  Radial power spectrum (2-D FFT of luma gray):
    spec_slope   : slope of polyfit(log1p(f), log1p(P(f)/Ptot)) over radial
                   bins (f = bin/(nbins), DC excluded). Natural images ~ -2.
    spec_total   : log1p(total radial power).
    spec_lf/mf/hf: share of power in f<=0.15 / 0.15<f<=0.5 / f>0.5.
    spec_hfratio : hf share (dup of spec_hf, kept explicit per task text).
  DCT (global 2-D DCT of gray):
    dct_dc       : |DC| coefficient.
    dct_hfenergy : log1p(mean |DCT| with top-left 8x8 zeroed).
    dct_hfratio  : hf-energy / (total |DCT| + eps).
  Wavelet (db1, 3 levels via wavedec2; db2 level-1 detail share):
    wav_*        : energy share of each detail subband (H/V/D per level 1-3)
                   + final approx, normalised by total energy (10 values:
                   wav_db1_L1H/L1V/L1D, L2H/L2V/L2D, L3H/L3V/L3D, A3, and
                   wav_db2_HH = db2 level-1 HH share).
  Local variance map (5x5 box mean of gray and gray^2):
    lvar_mean/std/p5/p95 : stats of the local-variance map.
  Edges:
    edge_density : fraction of Canny edge pixels (50/150 thresholds).
    sobel_energy : log1p(mean(gx^2+gy^2)), Sobel 3x3.
  Sharpness:
    lap_var      : log1p(variance of Laplacian).
    tenengrad    : log1p(mean Sobel-gradient magnitude^2) (= same energy
                   without log? kept as mean(gmag^2) raw-scale log1p; documents
                   focus measure).
  Luma/chroma (YCrCb BT.601):
    y_mean/y_std, cr_mean/cr_std, cb_mean/cb_std.
  Exposure:
    under_frac   : fraction luma < 0.05.  over_frac: fraction luma > 0.95.
    sat_mean/sat_std : HSV saturation mean/std.
    sat_low_frac : fraction saturation < 0.08 (washed-out proxy).
    val_mean     : HSV value mean.
  Skin mask (YCrCb rule 133<=Cr<=173, 77<=Cb<=127, 8-bit scale):
    skin_frac    : mask area fraction (hand proxy).
    skin_ncomp   : number of connected components (capped count, raw int).
    skin_largest : largest-component area fraction of frame.
    skin_cx/cy   : centroid of largest component, normalised [0,1]
                   (0.5/0.5 when empty).
  Colour opponent:
    opp_rg_mean/std : mean/std of R-G.
    opp_yb_mean/std : mean/std of 0.5*(R+G)-B.
    colorfulness : Hasler-Susstrunk style sqrt(var(rg)+var(yb))+0.3*mean term.
  JPEG blockiness:
    jpeg_block   : mean |adjacent-pixel diff| on 8-px grid boundaries minus
                   mean diff off-grid (luma). JPEG frames show a positive
                   ridge; clean/resized video ~ 0.
  Noise estimators (QUALITY group, computed on RAW resized pixels — cleaning
  would erase exactly what they measure):
    noise_immerkaer : Immerkaer fast noise sigma (Laplacian-mask based):
                   sigma = sqrt(pi/2) * mean(|Laplacian(gray)|) / 6.
    noise_mad_hh1   : MAD of the db1 level-1 diagonal (HH1) detail coeffs
                   scaled to sigma (MAD/0.6745). Robust to edges.
    NOTE on QUALITY vs ACTIVITY split (amendment "clean the noise first"):
    QUALITY features (blur/sharpness/noise/blockiness/exposure: lap_var,
    tenengrad, sobel_energy, edge_density, lvar_*, noise_*, jpeg_block,
    under/over_frac, y_*) are computed on RAW pixels here. ACTIVITY/TEMPORAL
    features (motion rhythm, hands, duplicates) are computed on CLEANED video
    at the window stage (see window_heads.py CLEAN_STEPS: area downscale,
    luma normalisation, global-motion compensation, detrending,
    Hann-Welch spectra, median/MAD robust stats, median-of-3).

Output order: FEATURES list. extract_frame_bgr() -> np.float64 vector.
"""

from __future__ import annotations

import cv2
import numpy as np
import pywt

TARGET_WIDTH = 480

FEATURES: list[str] = [
    "spec_slope", "spec_total", "spec_lf", "spec_mf", "spec_hf", "spec_hfratio",
    "dct_dc", "dct_hfenergy", "dct_hfratio",
    "wav_L1H", "wav_L1V", "wav_L1D",
    "wav_L2H", "wav_L2V", "wav_L2D",
    "wav_L3H", "wav_L3V", "wav_L3D",
    "wav_A3", "wav_db2_HH",
    "lvar_mean", "lvar_std", "lvar_p5", "lvar_p95",
    "edge_density", "sobel_energy",
    "lap_var", "tenengrad",
    "y_mean", "y_std", "cr_mean", "cr_std", "cb_mean", "cb_std",
    "under_frac", "over_frac",
    "sat_mean", "sat_std", "sat_low_frac", "val_mean",
    "skin_frac", "skin_ncomp", "skin_largest", "skin_cx", "skin_cy",
    "opp_rg_mean", "opp_rg_std", "opp_yb_mean", "opp_yb_std", "colorfulness",
    "jpeg_block",
    "noise_immerkaer", "noise_mad_hh1",
]

N_FEATURES = len(FEATURES)

FAMILIES: dict[str, list[str]] = {
    "fft": ["spec_slope", "spec_total", "spec_lf", "spec_mf", "spec_hf",
            "spec_hfratio"],
    "dct": ["dct_dc", "dct_hfenergy", "dct_hfratio"],
    "wavelet": ["wav_L1H", "wav_L1V", "wav_L1D", "wav_L2H", "wav_L2V",
                "wav_L2D", "wav_L3H", "wav_L3V", "wav_L3D", "wav_A3",
                "wav_db2_HH"],
    "texture": ["lvar_mean", "lvar_std", "lvar_p5", "lvar_p95",
                "edge_density", "sobel_energy", "lap_var", "tenengrad"],
    "luma_exposure": ["y_mean", "y_std", "under_frac", "over_frac",
                      "sat_mean", "sat_std", "sat_low_frac", "val_mean"],
    "chroma": ["cr_mean", "cr_std", "cb_mean", "cb_std"],
    "skin": ["skin_frac", "skin_ncomp", "skin_largest", "skin_cx", "skin_cy"],
    "opponent": ["opp_rg_mean", "opp_rg_std", "opp_yb_mean", "opp_yb_std",
                 "colorfulness"],
    "jpeg": ["jpeg_block"],
    "quality_noise": ["noise_immerkaer", "noise_mad_hh1", "jpeg_block",
                      "lap_var", "tenengrad", "sobel_energy", "edge_density",
                      "under_frac", "over_frac", "y_mean", "y_std"],
}

# QUALITY = measured on RAW pixels (cleaning would erase the signal).
# ACTIVITY = cleaned at window stage before temporal pooling.
QUALITY_FEATURES = FAMILIES["quality_noise"]
ACTIVITY_FAMILIES = ["skin", "opponent", "chroma", "fft", "dct", "wavelet",
                     "texture", "luma_exposure"]


def resize_to_480w(bgr: np.ndarray) -> np.ndarray:
    """Resize BGR image to fixed 480-px width, aspect preserved (INTER_AREA)."""
    h, w = bgr.shape[:2]
    if w == TARGET_WIDTH:
        return bgr
    scale = TARGET_WIDTH / float(w)
    nh = max(1, int(round(h * scale)))
    return cv2.resize(bgr, (TARGET_WIDTH, nh), interpolation=cv2.INTER_AREA)


def extract_frame_bgr(bgr: np.ndarray) -> np.ndarray:
    """Full extractor: resize to 480-px width, then all families. float64."""
    img = resize_to_480w(bgr)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    out: dict[str, float] = {}

    # --- radial power spectrum (2-D FFT) ---
    F = np.fft.rfft2(gray)
    power = F.real ** 2 + F.conj().imag ** 2
    h, w = power.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt(((xx - w // 2) / (w / 2)) ** 2 + ((yy - h // 2) / (h / 2)) ** 2)
    nb = min(h, w) // 2
    rbin = np.clip((rr * nb).astype(np.int64), 0, nb)
    pw = power.ravel().astype(np.float64)
    sums = np.bincount(rbin.ravel(), weights=pw, minlength=nb + 1)
    cnts = np.bincount(rbin.ravel(), minlength=nb + 1)
    prof = (sums / np.maximum(cnts, 1))[1:]
    freqs = np.arange(1, len(prof) + 1, dtype=np.float64) / max(len(prof), 1)
    tot = float(prof.sum()) + 1e-12
    out["spec_slope"] = float(np.polyfit(np.log1p(freqs), np.log1p(prof / tot), 1)[0])
    out["spec_total"] = float(np.log1p(tot))
    out["spec_lf"] = float(prof[freqs <= 0.15].sum() / tot)
    out["spec_mf"] = float(prof[(freqs > 0.15) & (freqs <= 0.5)].sum() / tot)
    out["spec_hf"] = float(prof[freqs > 0.5].sum() / tot)
    out["spec_hfratio"] = out["spec_hf"]

    # --- DCT ---
    D = cv2.dct(gray.astype(np.float32)).astype(np.float64)
    h8, w8 = min(8, D.shape[0]), min(8, D.shape[1])
    hf = D.copy()
    hf[:h8, :w8] = 0.0
    out["dct_dc"] = float(abs(D[0, 0]))
    out["dct_hfenergy"] = float(np.log1p(np.mean(np.abs(hf))))
    out["dct_hfratio"] = float(np.mean(np.abs(hf)) / (np.mean(np.abs(D)) + 1e-12))

    # --- wavelet db1 3 levels + db2 L1 HH share ---
    coeffs = pywt.wavedec2(gray, "db1", level=3)
    A3, details = coeffs[0], coeffs[1:]
    names = [("L3H", details[0][0]), ("L3V", details[0][1]), ("L3D", details[0][2]),
             ("L2H", details[1][0]), ("L2V", details[1][1]), ("L2D", details[1][2]),
             ("L1H", details[2][0]), ("L1V", details[2][1]), ("L1D", details[2][2])]
    en = {k: float(np.sum(v ** 2)) for k, v in names}
    en["A3"] = float(np.sum(A3 ** 2))
    etot = sum(en.values()) + 1e-12
    for k, v in en.items():
        out["wav_" + k] = float(v / etot)
    cA2, (cH2, cV2, cD2) = pywt.dwt2(gray, "db2")
    e2 = float(np.sum(cA2 ** 2) + np.sum(cH2 ** 2) + np.sum(cV2 ** 2)
               + np.sum(cD2 ** 2)) + 1e-12
    out["wav_db2_HH"] = float(np.sum(cD2 ** 2) / e2)

    # --- local variance map ---
    k = np.ones((5, 5), dtype=np.float64) / 25.0
    mu = cv2.filter2D(gray, -1, k)
    mu2 = cv2.filter2D(gray * gray, -1, k)
    lvar = np.maximum(mu2 - mu * mu, 0.0)
    out["lvar_mean"] = float(lvar.mean())
    out["lvar_std"] = float(lvar.std())
    out["lvar_p5"] = float(np.percentile(lvar, 5))
    out["lvar_p95"] = float(np.percentile(lvar, 95))

    # --- edges / sharpness ---
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gmag2 = gx * gx + gy * gy
    out["sobel_energy"] = float(np.log1p(np.mean(gmag2)))
    gmag = np.sqrt(gmag2)
    thr = float(gmag.mean() + gmag.std())
    strong = gmag2[gmag > thr]
    out["tenengrad"] = float(np.log1p(float(strong.mean()) if strong.size else 0.0))
    out["lap_var"] = float(np.log1p(float(cv2.Laplacian(gray, cv2.CV_64F).var())))
    edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
    out["edge_density"] = float(np.mean(edges > 0))

    # --- luma/chroma ---
    cr = 0.5 * (r - gray) / ((1 - 0.299) + 1e-12) + 0.5
    cb = 0.5 * (b - gray) / ((1 - 0.114) + 1e-12) + 0.5
    out["y_mean"] = float(gray.mean())
    out["y_std"] = float(gray.std())
    out["cr_mean"] = float(cr.mean())
    out["cr_std"] = float(cr.std())
    out["cb_mean"] = float(cb.mean())
    out["cb_std"] = float(cb.std())
    out["under_frac"] = float(np.mean(gray < 0.05))
    out["over_frac"] = float(np.mean(gray > 0.95))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
    sat = hsv[:, :, 1] / 255.0
    val = hsv[:, :, 2] / 255.0
    out["sat_mean"] = float(sat.mean())
    out["sat_std"] = float(sat.std())
    out["sat_low_frac"] = float(np.mean(sat < 0.08))
    out["val_mean"] = float(val.mean())

    # --- skin mask ---
    Cr8 = (cr * 255.0).astype(np.float64)
    Cb8 = (cb * 255.0).astype(np.float64)
    skin = ((Cr8 >= 133) & (Cr8 <= 173) & (Cb8 >= 77) & (Cb8 <= 127)).astype(np.uint8)
    out["skin_frac"] = float(skin.mean())
    ncomp, lab, stats, cent = cv2.connectedComponentsWithStats(skin, 8)
    out["skin_ncomp"] = float(ncomp - 1)
    H, W = skin.shape
    if ncomp > 1:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        big = int(1 + int(np.argmax(areas)))
        out["skin_largest"] = float(areas.max() / (H * W))
        out["skin_cx"] = float(cent[big][0] / W)
        out["skin_cy"] = float(cent[big][1] / H)
    else:
        out["skin_largest"] = 0.0
        out["skin_cx"] = 0.5
        out["skin_cy"] = 0.5

    # --- colour opponent ---
    rg = r - g
    yb = 0.5 * (r + g) - b
    out["opp_rg_mean"] = float(rg.mean())
    out["opp_rg_std"] = float(rg.std())
    out["opp_yb_mean"] = float(yb.mean())
    out["opp_yb_std"] = float(yb.std())
    out["colorfulness"] = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                                + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    # --- JPEG blockiness ---
    gd = gray.astype(np.float64)
    dh = np.abs(np.diff(gd, axis=1))
    dw = np.abs(np.diff(gd, axis=0))
    bnd_h = dh[:, 7::8]
    off_h = np.delete(dh, slice(7, None, 8), axis=1)
    bnd_v = dw[7::8, :]
    off_v = np.delete(dw, slice(7, None, 8), axis=0)
    out["jpeg_block"] = float(
        0.5 * (bnd_h.mean() - off_h.mean()) + 0.5 * (bnd_v.mean() - off_v.mean()))

    # --- noise estimators (RAW pixels) ---
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    out["noise_immerkaer"] = float(
        np.sqrt(np.pi / 2.0) * np.mean(np.abs(lap)) / 6.0)
    # MAD of db1 L1 diagonal detail -> sigma (recompute cheap via dwt2 on gray
    # downscaled x2 for speed; use full gray if tiny)
    small = gray if min(gray.shape) < 64 else cv2.resize(
        gray, (gray.shape[1] // 2, gray.shape[0] // 2),
        interpolation=cv2.INTER_AREA)
    _, (_, _, cD1) = pywt.dwt2(small, "db1")
    out["noise_mad_hh1"] = float(
        np.median(np.abs(cD1 - np.median(cD1))) / 0.6745)

    return np.asarray([out[k] for k in FEATURES], dtype=np.float64)
