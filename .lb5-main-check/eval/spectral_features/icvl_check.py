"""Arm C bounded hyperspectral feasibility check (ICVL surrogate).

DATA STATUS (verified 2026-10-10): the local HF cache at
/home/parshu/.cache/huggingface/hub/datasets--danaroth--icvl contains ONLY the
dataset README blob (~16 KB). No image cubes (.mat/.raw), no icvl.py, no
dataset_infos.json are present locally. No large downloads were performed.

Therefore everything below runs on SYNTHETIC spectra whose shapes are chosen to
mimic qualitatively known natural-reflectance structure (vegetation red edge,
soil ramp, skin-like visible rise + 970nm water dip, blue-peaked fabric/sky),
observed through 3 Gaussian RGB response curves (centres 460/540/620 nm,
sigma 40 nm) over 519 bands spanning 400-1000 nm linear (per the ICVL README:
519 bands, 400-1000 nm). This measures what a 3-broadband camera *mathematically*
must lose; it is NOT evidence about real ICVL cubes and NOT evidence on Awign
footage (no real hyperspectral sensor was used).

Analyses:
  (a) ridge regression RGB -> full spectrum (train/test split), mean R^2 per
      band and in 3 regimes (visible 400-700, red-edge 700-800, NIR 800-1000).
  (b) separability: k-means (k=4, random_state=0) on full-spectrum vs on RGB,
      adjusted mutual information between the two clusterings (= information
      lost by RGB), plus each clustering's AMI vs ground-truth class.
  (c) skin-proxy check: SKIPPED with reason (no validated skin reflectance
      template; synthetic "skin-like" class is illustrative only).

Stdlib + numpy/sklearn only. CPU, deterministic (seeded), <10 min.
"""

import os
import glob

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_mutual_info_score, r2_score
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(0)

ICVL_CACHE = "/home/parshu/.cache/huggingface/hub/datasets--danaroth--icvl"
N_BANDS = 519
WL_MIN, WL_MAX = 400.0, 1000.0  # nm, per ICVL README
RGB_CENTERS = (460.0, 540.0, 620.0)  # nm
RGB_SIGMA = 40.0  # nm
N_PIXELS = 6000
K = 4


def probe_local_icvl():
    """Inspect what ICVL data is actually on disk. Returns (mode, report)."""
    info = {"cache": ICVL_CACHE, "blobs": [], "mat_files": [], "total_bytes": 0}
    blobs = sorted(glob.glob(os.path.join(ICVL_CACHE, "blobs", "*")))
    for b in blobs:
        try:
            s = os.path.getsize(b)
        except OSError:
            s = -1
        info["blobs"].append((os.path.basename(b)[:12], s))
        if s > 0:
            info["total_bytes"] += s
    info["mat_files"] = sorted(
        glob.glob(os.path.join(ICVL_CACHE, "**", "*.mat"), recursive=True)
    )
    # Any blob big enough to be an image cube? README blob is ~16 KB;
    # a single ICVL cube (1392x1300x31 float) would be >200 MB.
    big = [s for _, s in info["blobs"] if s > 1_000_000]
    mode = "real" if (info["mat_files"] or big) else "synthetic"
    return mode, info


def wavelengths(n=N_BANDS):
    return np.linspace(WL_MIN, WL_MAX, n)


def rgb_responses(wl):
    """(3, n_bands) Gaussian response curves, each normalised to sum 1."""
    wl = np.asarray(wl)
    R = np.array(
        [np.exp(-0.5 * ((wl - c) / RGB_SIGMA) ** 2) for c in RGB_CENTERS]
    )
    return R / R.sum(axis=1, keepdims=True)


def synth_templates(wl):
    """(4, n_bands) illustrative class-mean reflectance shapes in [0,1]."""
    t = np.zeros((4, len(wl)))
    # 0: vegetation - low visible w/ green bump, sharp red edge, NIR plateau
    t[0] = 0.05 + 0.06 * np.exp(-0.5 * ((wl - 550) / 25) ** 2)
    t[0] += 0.55 / (1 + np.exp(-(wl - 720) / 12))
    # 1: soil/concrete - gentle ramp
    t[1] = 0.15 + 0.35 * (wl - 400) / 600
    # 2: skin-like (ILLUSTRATIVE, not a validated template) - visible rise
    #    with haemoglobin wiggles + NIR water dip near 970 nm
    t[2] = 0.25 + 0.35 * (wl - 400) / 600
    t[2] += 0.04 * np.exp(-0.5 * ((wl - 540) / 12) ** 2)
    t[2] += 0.04 * np.exp(-0.5 * ((wl - 575) / 12) ** 2)
    t[2] -= 0.18 * np.exp(-0.5 * ((wl - 970) / 18) ** 2)
    # 3: blue fabric/sky - blue peak, declining to NIR
    t[3] = 0.55 * np.exp(-0.5 * ((wl - 450) / 60) ** 2) + 0.12
    return np.clip(t, 0.02, 0.95)


def synth_cube(n=N_PIXELS, noise=0.01):
    """Return (spectra (n, bands), labels (n,)) with mixing + illumination jitter."""
    wl = wavelengths()
    T = synth_templates(wl)
    labels = RNG.integers(0, K, size=n)
    X = T[labels].copy()
    # per-pixel mixing toward a random other class (natural pixels are mixed)
    mix = RNG.uniform(0, 0.25, size=n)
    other = RNG.integers(0, K, size=n)
    X = (1 - mix)[:, None] * X + mix[:, None] * T[other]
    # illumination scale jitter + sensor noise
    X *= RNG.uniform(0.7, 1.3, size=(n, 1))
    X += RNG.normal(0, noise, size=X.shape)
    return np.clip(X, 0.0, 1.5), labels


def regime_masks(wl):
    return {
        "visible 400-700": (wl >= 400) & (wl < 700),
        "red-edge 700-800": (wl >= 700) & (wl < 800),
        "NIR 800-1000": (wl >= 800),
    }


def main():
    mode, info = probe_local_icvl()
    print("== ICVL local-data probe ==")
    print(f"cache: {info['cache']}")
    print(f"blobs: {info['blobs']} (total {info['total_bytes']} bytes)")
    print(f".mat files on disk: {info['mat_files']}")
    print(f"DATA MODE: {mode.upper()}")
    if mode != "synthetic":
        print("WARNING: real cubes found; this script still runs the synthetic "
              "surrogate only (bounded effort).")
    print()

    wl = wavelengths()
    R = rgb_responses(wl)
    X, y = synth_cube()
    rgb = X @ R.T  # (n, 3) simulated camera

    # RGB response weight per regime: how much of each regime the camera sees
    print("== RGB response coverage per regime (fraction of curve mass) ==")
    for name, m in regime_masks(wl).items():
        frac = R[:, m].sum() / R.sum()
        print(f"  {name}: {frac:.3f}")
    print("  (NIR mass ~ 0: the camera barely measures it; "
          "NIR reconstruction must be prior-driven.)")
    print()

    # (a) ridge RGB -> spectrum
    n = len(X)
    idx = RNG.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    reg = Ridge(alpha=1.0).fit(rgb[tr], X[tr])
    pred = reg.predict(rgb[te])
    r2_per_band = np.array(
        [r2_score(X[te][:, j], pred[:, j]) for j in range(X.shape[1])]
    )
    print("== (a) ridge RGB->spectrum, test R^2 ==")
    print(f"  mean over all {N_BANDS} bands: {r2_per_band.mean():.3f}")
    r2_reg = {}
    for name, m in regime_masks(wl).items():
        r2_reg[name] = r2_per_band[m].mean()
        print(f"  {name}: {r2_reg[name]:.3f}")
    print()

    # (b) separability: k-means on full spectrum vs on RGB
    kf = KMeans(n_clusters=K, n_init=10, random_state=0).fit(
        StandardScaler().fit_transform(X)
    )
    kr = KMeans(n_clusters=K, n_init=10, random_state=0).fit(
        StandardScaler().fit_transform(rgb)
    )
    ami_full_rgb = adjusted_mutual_info_score(kf.labels_, kr.labels_)
    ami_full_true = adjusted_mutual_info_score(y, kf.labels_)
    ami_rgb_true = adjusted_mutual_info_score(y, kr.labels_)
    print("== (b) k-means separability (k=4, random_state=0) ==")
    print(f"  AMI(full-spectrum clusters, RGB clusters): {ami_full_rgb:.3f} "
          "(<1 = structure lost by RGB)")
    print(f"  AMI(full-spectrum clusters, true class):   {ami_full_true:.3f}")
    print(f"  AMI(RGB clusters, true class):            {ami_rgb_true:.3f}")
    print()

    # (c) skin-proxy: skipped
    print("== (c) skin-proxy check: SKIPPED ==")
    print("  Reason: no validated skin reflectance template available; the "
          "synthetic 'skin-like' class is illustrative only, so matching a "
          "cluster to 'literature skin shape' would be circular.")
    print()

    print("== HONEST VERDICT ==")
    print(f"  On SYNTHETIC data only (no ICVL cubes on disk; total cached "
          f"{info['total_bytes']} bytes = README text). RGB simulation used "
          "Gaussian responses at 460/540/620 nm, sigma 40 nm.")
    print(f"  RGB preserves most visible-range class structure on this synthetic "
          f"mix (RGB-cluster AMI vs truth {ami_rgb_true:.2f} ~= full-spectrum "
          f"{ami_full_true:.2f}), but the clusterings still disagree "
          f"(cross-space AMI {ami_full_rgb:.2f} < 1): RGB merges structure "
          "that the full spectrum resolves, and the gap would widen for "
          "materials differing mainly outside the visible range.")
    print(f"  RGB->spectral ridge reconstruction recovers visible bands "
          f"(R^2 {r2_reg['visible 400-700']:.2f}) but adds little usable "
          f"information in red-edge/NIR (R^2 "
          f"{r2_reg['red-edge 700-800']:.2f}/{r2_reg['NIR 800-1000']:.2f}): "
          "it interpolates from RGB priors, it does not measure NIR. A real "
          "hyperspectral sensor would be needed for any NIR claim.")
    print("  Nothing here is evidence on Awign footage: no real "
          "hyperspectral sensor was used and no Awign/worker video entered "
          "this check.")


if __name__ == "__main__":
    main()
