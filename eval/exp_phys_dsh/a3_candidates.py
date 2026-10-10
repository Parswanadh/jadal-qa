#!/usr/bin/env python3
"""DSH-PHYS A3: cheap-signal deterministic rule candidates + NEW closed-form
composites from the per-frame series, evaluated LOCO on IndEgo gold-150.

Protocol (mirrors eval/exp_phys/run_exp_phys.py, Q1 PR #141):
  * every candidate is a single-feature, fixed-integer Q=4 comparison ladder:
    direction (low-is-bad / high-is-bad) and the four thresholds are derived
    from the TRAINING clips only, at quantiles p01/p05/p10/p25 (mirrored to
    p99/p95/p90/p75 for high-is-bad), quantised to 4 decimals;
  * LOCO = leave-one-clip-out over the 7 IndEgo clips; a window's indicator is
    produced with its own clip excluded from threshold/direction fitting;
  * indicator = at least one threshold crossed (matches Q1 per-rule number);
    severity = integer count of crossed thresholds (0..4);
  * CI = clip bootstrap (B=2000, seed 0) over the LOCO indicators;
  * no learning, no RNG beyond the seeded bootstrap, no multiprocessing.

Reads (read-only):
  $JADAL_BUNDLE/caches/perpixel/gold.npz         X (150,121), y, clips, ids, names
  $JADAL_BUNDLE/caches/perpixel/series_gold.npz  S (150,100,8)
Writes:
  eval/results/exp_phys_dsh/A3_new_rules.json
  eval/results/exp_phys_dsh/A3_rule_table_candidates.json

Run: OMP_NUM_THREADS=2 python3 eval/exp_phys_dsh/a3_candidates.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
OUTDIR = os.path.join(ROOT, "eval", "results", "exp_phys_dsh")
Q = 4
QUANTILES = (0.01, 0.05, 0.10, 0.25)
BOOT = 2000
SEED = 0

# Q1 (PR #141) per-named-rule LOCO indicator AUROC, as published in the BRIEF.
Q1_RULE_AUROC = {
    "BLUR": 0.695, "CODING_NOISE": 0.695, "NO_ACTIVE_WORK": 0.695,
    "DUPLICATE": 0.675, "HANDS_NOT_VISIBLE": 0.610, "OVEREXPOSED": 0.565,
    "TOO_DARK": 0.550, "FROZEN_STATIC": 0.500,
}
# gold.npz columns implementing the Q1 rule table (BRIEF name map).
Q1_COLUMNS = {
    "full.luma_mean": "G_luma_mean", "full.dark_frac": "G_dark_frac",
    "full.bright_frac": "G_bright_frac", "full.hf_ratio": "G_hf_ratio",
    "full.dct_energy": "G_dct_energy", "full.temp_b0": "G_temp_b0",
    "full.frozen_frac": "G_frozen_frac", "full.motion_energy": "G_motion_energy",
    "full.skin_frac": "G_skin_frac",
}
Q1_RULE_FEATURE = {
    "TOO_DARK": ["full.luma_mean", "full.dark_frac"],
    "OVEREXPOSED": ["full.bright_frac"],
    "BLUR": ["full.hf_ratio", "full.dct_energy"],
    "FROZEN_STATIC": ["full.temp_b0", "full.frozen_frac"],
    "NO_ACTIVE_WORK": ["full.motion_energy"],
    "HANDS_NOT_VISIBLE": ["full.skin_frac"],
    "DUPLICATE": [], "CODING_NOISE": [],
}

# ---------------------------------------------------------------------------
# closed-form composites from S (150,100,8)
# channels: 0 luma_mean 1 lap_var 2 diff_e 3 skin_frac 4 shift_px
#           5 under_frac 6 over_frac 7 dct_hf
CH = {"luma_mean": 0, "lap_var": 1, "diff_e": 2, "skin_frac": 3, "shift_px": 4,
      "under_frac": 5, "over_frac": 6, "dct_hf": 7}
EPS = 1e-6


def _ac1(x: np.ndarray) -> float:
    """Biased lag-1 autocorrelation, closed form: sum(x1*x0)/sum(x^2) after demean."""
    x = np.asarray(x, float) - float(np.mean(x))
    d = float(np.dot(x, x))
    if d <= 0:
        return 0.0
    return float(np.dot(x[1:], x[:-1]) / d)


def _spec_flat(x: np.ndarray) -> float:
    """Spectral flatness (Wiener entropy) of the AC spectrum, bins 1..N/2."""
    x = np.asarray(x, float) - float(np.mean(x))
    P = np.abs(np.fft.rfft(x)) ** 2
    P = P[1:len(x) // 2 + 1] + 1e-12
    return float(np.exp(np.mean(np.log(P))) / np.mean(P))


def _excess(d: np.ndarray) -> float:
    """Noise-floor-relative motion excess: mean(max(d-F,0))/(F+eps), F=med+MAD."""
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    F = med + mad
    if F <= 0:
        return 0.0
    return float(np.mean(np.maximum(d - F, 0.0)) / (F + EPS))


def _onset_rate(d: np.ndarray) -> float:
    """Burst onsets per frame: P(d_t > F and d_{t-1} <= F), F = med+MAD."""
    med = float(np.median(d))
    F = med + float(np.median(np.abs(d - med)))
    above = d > F
    return float(np.mean(above[1:] & ~above[:-1]))


def new_composites(S: np.ndarray) -> dict[str, np.ndarray]:
    """The NEW deterministic composites (fixed constants only, no fitting)."""
    n = len(S)
    luma = np.zeros(n); lap = np.zeros(n); dif = np.zeros(n)
    clip_tot = np.zeros(n); sharp_fl = np.zeros(n); acx = np.zeros(n)
    flat = np.zeros(n); onset = np.zeros(n); exc = np.zeros(n)
    for i in range(n):
        a = np.asarray(S[i], dtype=float)
        luma[i] = a[:, CH["luma_mean"]].mean()
        d = a[:, CH["diff_e"]]
        lv = a[:, CH["lap_var"]]
        dif[i] = d.mean()
        clip_tot[i] = a[:, CH["under_frac"]].mean() + a[:, CH["over_frac"]].mean()
        sharp_fl[i] = (np.mean(np.abs(np.diff(lv))) / (lv.mean() + EPS))
        acx[i] = _ac1(d) - _ac1(a[:, CH["luma_mean"]])
        flat[i] = _spec_flat(d)
        onset[i] = _onset_rate(d)
        exc[i] = _excess(d)
    return {
        "P_motion_excess_nf": exc,                   # noise-floor-relative motion excess
        "P_ac1_contrast": acx,                       # lag-1 AC(motion) - AC(luma)
        "P_clip_total": clip_tot,                    # mean under_frac + mean over_frac
        "P_sharp_flicker": sharp_fl,                 # mean|d lap_var| / mean lap_var
        "P_motion_tonality": flat,                   # spectral flatness of diff_e
        "P_burst_onset_rate": onset,                 # burst onsets per frame
        # raw channel means used only as denominators/context (not new rules)
        "P_series_luma_mean": luma,
        "P_series_diff_e_mean": dif,
    }


DERIVATIONS = {
    "P_motion_excess_nf": "closed form: d=diff_e series, F=median(d)+MAD(d) (per-window noise floor), value=mean(max(d-F,0))/(F+1e-6); how far motion spikes rise ABOVE the window's own noise floor, not how often",
    "P_ac1_contrast": "closed form: ac1(x)=sum((x_t-m)(x_{t-1}-m))/sum((x_t-m)^2); value=ac1(diff_e)-ac1(luma_mean); motion that is bursty while luma is static gives a positive contrast (handheld jitter / strobing), smooth global motion drives both together",
    "P_clip_total": "closed form: mean_t under_frac(t) + mean_t over_frac(t) with under=luma<0.05, over=luma>0.95; total hard-clipped area, direction-free (fails either way)",
    "P_sharp_flicker": "closed form: mean_t |lap_var(t)-lap_var(t-1)| / (mean_t lap_var(t)+1e-6); relative frame-to-frame sharpness instability (focus hunting / rolling-shutter strobe) at zero extra decode cost",
    "P_motion_tonality": "closed form: Wiener entropy of the diff_e spectrum, exp(mean(log(P+1e-12)))/mean(P+1e-12) over rfft bins 1..T/2; ~0 = tonal/periodic motion (oscillation, strobe), ~1 = broadband noise motion (coding noise / shake)",
    "P_burst_onset_rate": "closed form: F=median(diff_e)+MAD(diff_e); value=mean_t[ d_t>F and d_{t-1}<=F ]; rate of activity onsets per frame (burst/gap statistic)",
    "P_series_luma_mean": "context denominator only: mean luma_mean over the 100-frame series",
    "P_series_diff_e_mean": "context denominator only: mean diff_e over the 100-frame series",
}

# ---------------------------------------------------------------------------
# candidate specification: pool candidates from gold.npz + new S composites
# (rule, source, prior direction guess, derivation sentence)
POOL = [
    # --- compression telemetry -> CODING_NOISE ---
    ("B_i_size", "CODING_NOISE", "high", "mean I-frame packet size (bytes) over the window; large I-frames at fixed CRF mean fine detail/noise, tiny/zero means no intra refresh or a blank keyframe"),
    ("B_p_size_mean", "CODING_NOISE", "high", "mean P-frame packet size (bytes); P bytes rise with residual motion and coding noise at fixed quality"),
    ("B_pb_size_ratio", "CODING_NOISE", "high", "mean P-frame bytes / mean I-frame bytes; high = inter frames nearly as expensive as intra = high residual/noise, low = static over-compressed scene"),
    ("B_frame_size_std", "CODING_NOISE", "high", "std of all packet sizes / mean, i.e. bitrate instability across the window; a flickering/duplicated/strobing source spends unevenly"),
    ("B_p_count_frac", "FROZEN_STATIC", "low", "fraction of packets that are non-key frames; low means the encoder keeps re-sending I-frames (scene cuts/hard changes) or almost no coded frames"),
    ("B_mv_per_frame", "NO_ACTIVE_WORK", "high", "mean number of exported motion vectors per decoded frame; motion-vector count is encoder flow activity, high = busy camera/hands"),
    # --- DCT / spectral band ratios -> BLUR (low) or CODING_NOISE (high) ---
    ("B_dct_hfenergy", "BLUR", "low", "mean |DCT| energy outside the top-left 8x8 block; drops when the frame is defocused/motion-blurred, rises with sensor/compression noise"),
    ("B_spec_hfratio", "BLUR", "low", "share of FFT power above the mid band; a flat/smooth image loses high frequencies"),
    ("B_spec_slope", "BLUR", "low", "log-log FFT power-spectrum slope; steeper (more negative) = energy concentrated at low frequencies = blurred"),
    ("B_wav_HH", "BLUR", "low", "Haar HH (diagonal) sub-band energy share; diagonal detail is the first thing blur removes"),
    ("B_wav_HL", "BLUR", "low", "Haar HL (vertical-edge) sub-band energy share; vertical edges vanish under horizontal motion blur"),
    ("B_wav_LH", "BLUR", "low", "Haar LH (horizontal-edge) sub-band energy share; horizontal edges vanish under vertical motion blur"),
    ("B_lap_var", "BLUR", "low", "variance of the Laplacian (raw sharpness units, window mean); the classic focus measure"),
    ("P_dct_hf_mean", "BLUR", "low", "per-frame mean high-frequency DCT energy averaged over the 100-frame series (same quantity as B_dct_hfenergy but series-averaged)"),
    ("P_dct_hf_std", "CODING_NOISE", "high", "std over frames of the per-frame HF DCT energy; unstable HF energy despite static content = coding noise/flicker"),
    # --- ego-motion / affine residual -> NO_ACTIVE_WORK / FROZEN_STATIC ---
    ("G_global_shift", "NO_ACTIVE_WORK", "high", "median global (phase-correlation) shift magnitude in px; large = camera/ego motion, near-zero = static viewpoint"),
    ("G_pca_resid", "NO_ACTIVE_WORK", "high", "residual of the dominant affine/PCA motion model; high = motion the global model cannot explain (independent object/hand motion)"),
    ("G_pca_top1_share", "FROZEN_STATIC", "low", "share of motion energy in the top PCA component; a single dominant translation = clean ego-motion"),
    ("G_radial_slope", "BLUR", "low", "slope of radially averaged power spectrum; flatter spectra (less negative) carry relatively more high-frequency energy"),
    ("P_shift_mean", "NO_ACTIVE_WORK", "high", "mean over frames of phase-correlation shift vs the first frame (px at 160x90); accumulated viewpoint displacement"),
    ("P_shift_max", "NO_ACTIVE_WORK", "high", "max over frames of the shift vs the first frame; a single jump (scene change / hard cut)"),
    ("P_motion_floor", "NO_ACTIVE_WORK", "high", "data-driven motion floor of diff_e (median+MAD of the window); the window's own noise/jitter level"),
    ("P_motion_frac_above", "NO_ACTIVE_WORK", "high", "fraction of frames with diff_e above the window's median+MAD floor; duty cycle of real motion"),
    ("P_still_run_frac", "FROZEN_STATIC", "high", "longest run of frames with diff_e below the floor, as a fraction of the window; high = long dead time"),
    # --- periodicity / duplicate ---
    ("G_dup_hash_mean", "DUPLICATE", "high", "mean dHash Hamming distance of consecutive frames (GPU window aggregate)"),
    ("G_zncc_self", "DUPLICATE", "high", "self zero-normalised cross-correlation of the window against its own frames; high = the window repeats its own content"),
    ("B_temp_period", "DUPLICATE", "high", "dominant temporal period (frames) of the luma signal; a fixed period indicates a loop/repeat rather than live work"),
    ("G_flicker_idx", "CODING_NOISE", "high", "GPU flicker index (illumination instability)"),
    ("P_luma_mean_ac_peak", "DUPLICATE", "high", "peak normalised autocorrelation of the luma_mean series (lag>=1); high = the window's brightness pattern repeats"),
    ("P_lap_var_ac_lag", "CODING_NOISE", "high", "lag (frames) of the lap_var autocorrelation peak"),
]

Q1_TABLE_ROWS = [
    ("G_hf_ratio", "BLUR"), ("G_dct_energy", "BLUR"), ("G_luma_mean", "TOO_DARK"),
    ("G_dark_frac", "TOO_DARK"), ("G_bright_frac", "OVEREXPOSED"),
    ("G_temp_b0", "FROZEN_STATIC"), ("G_frozen_frac", "FROZEN_STATIC"),
    ("G_motion_energy", "NO_ACTIVE_WORK"), ("G_skin_frac", "HANDS_NOT_VISIBLE"),
]


# ---------------------------------------------------------------------------


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC P(score_bad > score_good) with average ranks.

    Identical to sklearn.metrics.roc_auc_score(1 - y, s) (verified in main),
    but ~100x faster, which matters for the 200k bootstrap evaluations.
    """
    s = np.asarray(s, float)
    pos = (y == 0)                                    # y=0 is bad = positive class
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def derive_thresholds(v: np.ndarray, side: str) -> list[float]:
    """Q1 helper semantics, plus dedup (identical quantiles -> one rung)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 8 or len(np.unique(v)) < 2:
        return []
    qs = np.array(QUANTILES) if side == "low" else 1.0 - np.array(QUANTILES)
    ths = [round(float(t), Q) for t in np.quantile(v, qs)]
    out = []
    for t in ths:                      # keep severity order, drop exact duplicates
        if not out or t != out[-1]:
            out.append(t)
    return out


def crosses(value: float, ths: list[float], side: str) -> int:
    if not np.isfinite(value):
        return 0
    if side == "low":
        return int(sum(1 for t in ths if value < t))
    return int(sum(1 for t in ths if value > t))


def score_candidate(v: np.ndarray, y: np.ndarray, clips: np.ndarray, fixed_side=None):
    """LOCO direction+thresholds from the 6 training clips; fixed_side for the
    frozen all-clip table."""
    n = len(v)
    ind = np.zeros(n); sev = np.zeros(n); side_fold = {}
    for c in sorted(set(clips)):
        te = clips == c
        tr = ~te
        m = tr & np.isfinite(v)
        if m.sum() < 8 or len(np.unique(v[m])) < 2:
            continue
        if fixed_side is None:
            a_tr = auc(y[m], v[m])
            side = "high" if a_tr >= 0.5 else "low"
        else:
            side = fixed_side
        ths = derive_thresholds(v[m], side)
        if not ths:
            continue
        side_fold[str(c)] = side
        for i in np.where(te)[0]:
            k = crosses(v[i], ths, side)
            sev[i] = k
            ind[i] = 1.0 if k >= 1 else 0.0
    return ind, sev, side_fold


def clip_boot_ci(y, s, clips, n_boot=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    uc = sorted(set(clips))
    out = []
    for _ in range(n_boot):
        cs = rng.choice(uc, size=len(uc), replace=True)
        idx = np.concatenate([np.where(clips == c)[0] for c in cs])
        if len(set(y[idx])) < 2:
            continue
        out.append(auc(y[idx], s[idx]))
    if not out:
        return [None, None]
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def n_unique(v):
    return int(len(np.unique(np.asarray(v, float)[np.isfinite(v)])))


def main() -> int:
    t0 = time.perf_counter()
    g = np.load(os.path.join(BUNDLE, "caches", "perpixel", "gold.npz"), allow_pickle=True)
    sg = np.load(os.path.join(BUNDLE, "caches", "perpixel", "series_gold.npz"), allow_pickle=True)
    names = [str(x) for x in g["names"]]
    X = np.asarray(g["X"], float)
    y = np.asarray(g["y"], int)
    clips = np.asarray(g["clips"])
    ids = [str(i) for i in g["ids"]]
    assert all(str(a) == str(b) for a, b in zip(sg["ids"], g["ids"]))
    S = sg["S"]
    comp = new_composites(S)
    idx = {nm: i for i, nm in enumerate(names)}
    # AUROC implementation check (rank-based vs sklearn) on deterministic vectors
    auc_check = 0.0
    for k in range(5):
        s = np.sin(np.arange(len(y)) * (k + 1)) + k
        auc_check = max(auc_check, abs(auc(y, s) - roc_auc_score(1 - y, s)))
        auc_check = max(auc_check, abs(auc(y, (s > 0).astype(float)) - roc_auc_score(1 - y, (s > 0).astype(float))))

    # candidate value vectors
    values: dict[str, np.ndarray] = {}
    spec: dict[str, dict] = {}
    for feat, rule, prior, deriv in POOL:
        if feat not in idx:
            continue
        values[feat] = X[:, idx[feat]]
        spec[feat] = {"rule": rule, "prior_direction": prior, "derivation": deriv,
                      "source": "gold.npz:" + feat}
    for feat, v in comp.items():
        if feat in DERIVATIONS and not feat.startswith("P_series"):
            values[feat] = np.asarray(v, float)
            spec[feat] = {"rule": None, "prior_direction": None,
                          "derivation": DERIVATIONS[feat], "source": "series_gold.npz"}
    # B_temp_period stands in for the missing G_temp_period column
    for f in ("B_temp_period",):
        if f in spec:
            spec[f]["note"] = "G_temp_period is absent from gold.npz names; B_temp_period is the same quantity from the CPU series extractor"

    results = []
    for feat in list(spec):
        v = np.asarray(values[feat], float)
        ind, sev, side_fold = score_candidate(v, y, clips)
        # raw signed per-fold AUROC (threshold-free ceiling for the feature)
        z = np.zeros(len(v))
        for c in sorted(set(clips)):
            te = clips == c
            tr = ~te
            m = tr & np.isfinite(v)
            if m.sum() < 8 or len(np.unique(v[m])) < 2:
                continue
            side = "high" if auc(y[m], v[m]) >= 0.5 else "low"
            z[te] = v[te] if side == "high" else -v[te]
        raw_auroc = auc(y, z) if len(np.unique(z)) > 1 else None
        # frozen all-clip ladder (what the lead would merge)
        m_all = np.isfinite(v)
        side_all = None
        if m_all.sum() >= 8 and len(np.unique(v[m_all])) >= 2:
            side_all = "high" if auc(y[m_all], v[m_all]) >= 0.5 else "low"
        ths = derive_thresholds(v, side_all) if side_all else []
        # pooled fire counts with the frozen ladder
        pooled_fire = np.array([crosses(v[i], ths, side_all) for i in range(len(v))]) if ths else np.zeros(len(v))
        pooled_ext = np.array([crosses(v[i], ths[:1], side_all) for i in range(len(v))]) if ths else np.zeros(len(v))
        res = {
            "feature": feat,
            "rule": spec[feat]["rule"],
            "source": spec[feat]["source"],
            "derivation": spec[feat]["derivation"],
            "n_unique": n_unique(v),
            "loco_direction": (sorted(set(side_fold.values()))[0] if len(set(side_fold.values())) == 1
                               else "mixed:" + ",".join(sorted(set(side_fold.values())))),
            "frozen_direction": side_all,
            "frozen_op": ("<" if side_all == "low" else ">") if side_all else None,
            "frozen_thresholds": ths,
            "frozen_thresholds_q": [int(round(t * 10 ** Q)) for t in ths],
            "loco_indicator_auroc": round(auc(y, ind), 4),
            "loco_indicator_ci95": clip_boot_ci(y, ind, clips),
            "loco_severity_auroc": round(auc(y, sev), 4),
            "loco_severity_ci95": clip_boot_ci(y, sev, clips),
            "loco_raw_signed_auroc": round(raw_auroc, 4) if raw_auroc is not None else None,
            "fire_any_loco_good": int(((ind > 0) & (y == 1)).sum()),
            "fire_any_loco_bad": int(((ind > 0) & (y == 0)).sum()),
            "n_good": int((y == 1).sum()), "n_bad": int((y == 0).sum()),
            "fire_any_frozen_good": int(((pooled_fire > 0) & (y == 1)).sum()),
            "fire_any_frozen_bad": int(((pooled_fire > 0) & (y == 0)).sum()),
            "fire_extreme_frozen_good": int(((pooled_ext > 0) & (y == 1)).sum()),
            "fire_extreme_frozen_bad": int(((pooled_ext > 0) & (y == 0)).sum()),
            "fire_extreme_loco_good": None, "fire_extreme_loco_bad": None,
            "loco_fold_sides": side_fold,
        }
        # true extreme-rung fire count under LOCO (crosses rung 0 only)
        ex = np.zeros(len(v))
        for c in sorted(set(clips)):
            te = clips == c
            tr = ~te
            m = tr & np.isfinite(v)
            if m.sum() < 8 or len(np.unique(v[m])) < 2:
                continue
            side = "high" if auc(y[m], v[m]) >= 0.5 else "low"
            t0_ = derive_thresholds(v[m], side)
            if not t0_:
                continue
            for i in np.where(te)[0]:
                ex[i] = 1.0 if crosses(v[i], t0_[:1], side) >= 1 else 0.0
        res["fire_extreme_loco_good"] = int(((ex > 0) & (y == 1)).sum())
        res["fire_extreme_loco_bad"] = int(((ex > 0) & (y == 0)).sum())
        results.append(res)

    # Q1 reference rows through the same pipeline (pipeline sanity check)
    ref = []
    for feat, rule in Q1_TABLE_ROWS:
        if feat not in idx:
            continue
        v = X[:, idx[feat]]
        ind, sev, side_fold = score_candidate(v, y, clips)
        a_ind = round(auc(y, ind), 4)
        ref.append({"feature": feat, "q1_rule": rule, "q1_published_auroc": Q1_RULE_AUROC[rule],
                    "pipeline_indicator_auroc": a_ind,
                    "pipeline_indicator_ci95": clip_boot_ci(y, ind, clips),
                    "pipeline_severity_auroc": round(auc(y, sev), 4),
                    "note": "Q1 published value is the named-rule indicator (union of that rule's features, "
                            "per-fold table); this row is the single feature through the A3 pipeline"})

    # redundancy: |spearman| vs the Q1-implementing gold columns
    from scipy.stats import spearmanr
    q1_vals = {k: X[:, idx[v]] for k, v in Q1_COLUMNS.items() if v in idx}
    for r in results:
        v = np.asarray(values[r["feature"]], float)
        best = []
        for k, w in q1_vals.items():
            if not np.isfinite(v).all() or len(np.unique(v)) < 2:
                continue
            rho = float(spearmanr(v, w).statistic)
            best.append((abs(rho), rho, k))
        best.sort(reverse=True)
        r["max_abs_spearman_with_q1"] = round(best[0][0], 4) if best else None
        r["max_abs_spearman_partner"] = best[0][2] if best else None
        r["spearman_top3_q1"] = [{"feature": k, "rho": round(rho, 4)} for _, rho, k in best[:3]]

    # verdicts
    for r in results:
        a = r["loco_indicator_auroc"]; lo = r["loco_indicator_ci95"][0]
        sa = r["loco_severity_auroc"]; slo = r["loco_severity_ci95"][0]
        best_a = max(a, sa); best_lo = max(lo, slo)
        rule = r["rule"]
        q1a = Q1_RULE_AUROC.get(rule) if rule else None
        rho = r["max_abs_spearman_with_q1"]
        if r["n_unique"] <= 2:
            v_ = "DUD_degenerate"
            why = f"only {r['n_unique']} distinct values on gold-150 (constant: cannot threshold)"
        elif best_a <= 0.55:
            v_ = "DUD"
            why = f"best LOCO AUROC {best_a:.3f} <= 0.55 (near chance)"
        elif rule and q1a is not None and best_a <= q1a + 0.02 and (rho or 0) >= 0.85:
            v_ = "REDUNDANT"
            why = f"same rule {rule} already at {q1a:.3f}; rho={rho:.2f} with the Q1 feature"
        elif best_a >= 0.65 and best_lo and best_lo > 0.52:
            v_ = "ADDS"
            why = f"best LOCO AUROC {best_a:.3f} CI-low {best_lo:.3f}"
        elif best_a >= 0.60 and best_lo and best_lo > 0.50:
            v_ = "WEAK_ADD"
            why = f"best LOCO AUROC {best_a:.3f} CI-low {best_lo:.3f}"
        else:
            v_ = "DUD"
            why = f"best LOCO AUROC {best_a:.3f} CI-low {best_lo}"
        if rule is None and v_ in ("ADDS", "WEAK_ADD"):
            v_ = "ADDS_NEW_RULE" if v_ == "ADDS" else v_
        r["verdict"] = v_
        r["verdict_reason"] = why
        if rule:
            r["delta_vs_rule_table_point"] = round(best_a - q1a, 4) if q1a is not None else None

    results.sort(key=lambda r: -max(r["loco_indicator_auroc"], r["loco_severity_auroc"]))
    ranked = [r["feature"] for r in results]

    out = {
        "experiment": "A3",
        "title": "Cheap-signal deterministic rule candidates and NEW closed-form series composites (IndEgo gold-150, LOCO)",
        "determinism": ("no learned weights; per-fold direction from the training-clip AUROC sign; "
                        "thresholds = quantiles p01/p05/p10/p25 (mirrored) of TRAINING clips, quantised to "
                        "4 dp; integer Q=4 comparisons (thresholds_q = int(round(t*1e4))); indicator = any rung "
                        "crossed; severity = integer count of rungs crossed; RNG only in the seeded clip bootstrap"),
        "data": {"source": os.path.join(BUNDLE, "caches", "perpixel", "gold.npz"),
                 "series": os.path.join(BUNDLE, "caches", "perpixel", "series_gold.npz"),
                 "n": int(len(y)), "n_good": int((y == 1).sum()), "n_bad": int((y == 0).sum()),
                 "n_clips": int(len(set(clips))),
                 "clips": {str(c): int((clips == c).sum()) for c in sorted(set(clips))},
                 "label_source": "human (IndEgo gold, labels/manifest_gold.json)",
                 "window": "10 s, 100 frames decoded at 160x90 (P_nframes=100 for all 150)"},
        "protocol": {"in_domain": "leave-one-clip-out over the 7 IndEgo clips; direction+thresholds from the 6 training clips only",
                     "ci": f"clip bootstrap B={BOOT} seed={SEED}, percentile 95%",
                     "indicator": "score >= 1 rung crossed (comparable to Q1 per-rule indicator AUROC)",
                     "severity": "integer 0..4 count of rungs crossed",
                     "q1_reference": Q1_RULE_AUROC},
        "exact_command": "OMP_NUM_THREADS=2 python3 eval/exp_phys_dsh/a3_candidates.py",
        "device": "CPU only (no GPU; torch present but unused)",
        "wall_s": None,
        "auc_impl_max_abs_diff_vs_sklearn": auc_check,
        "ranked_features": ranked,
        "candidates": results,
        "q1_reference_pipeline_check": ref,
        "new_composites_closed_form": {k: DERIVATIONS[k] for k in DERIVATIONS},
        "limitations": [
            "n=150 with only 7 clips; clip bootstrap has 7 resample units, so CIs are wide and the indicator AUROC is a coarse statistic (binary).",
            "Direction is chosen inside each fold from training labels (that is LOCO-honest but means the reported direction can differ from a physical prior).",
            "Severity AUROC and indicator AUROC rank differently when a feature's tail is informative but its bulk is not.",
            "The frozen_thresholds_* fields are fitted on all 7 clips (what a deployment table would carry); they are NOT used for the LOCO columns.",
            "0.5-granularity: with a binary indicator the AUROC is determined by TPR/FPR at one operating point (equivalent to balanced accuracy), which is why Q1 per-rule values sit near 0.5.",
            "No OOD labels were read for this experiment; no hold-out 108 windows were touched.",
        ],
    }
    os.makedirs(OUTDIR, exist_ok=True)

    # ---- mergeable rule-table candidates -------------------------------
    shape_ok = {"ADDS", "ADDS_NEW_RULE", "WEAK_ADD"}
    rules_out, excluded = [], []
    for r in results:
        entry = {
            "rule": r["rule"] if r["rule"] else "NEW_RULE",
            "feature": r["feature"],
            "op": r["frozen_op"], "side": r["frozen_direction"],
            "strength": "weak" if r["loco_indicator_auroc"] < 0.75 else "quality",
            "thresholds": r["frozen_thresholds"],
            "thresholds_q": r["frozen_thresholds_q"],
            "quantiles": list(QUANTILES),
            "source_quantile": "p01/p05/p10/p25" if r["frozen_direction"] == "low" else "p99/p95/p90/p75",
            "derivation": r["derivation"],
            "loco_indicator_auroc": r["loco_indicator_auroc"],
            "loco_indicator_ci95": r["loco_indicator_ci95"],
            "loco_severity_auroc": r["loco_severity_auroc"],
            "verdict": r["verdict"],
            "robustness": None,   # filled by the sensitivity run
        }
        if r["verdict"] in shape_ok:
            rules_out.append(entry)
        else:
            excluded.append({"rule": entry["rule"], "feature": entry["feature"],
                             "verdict": r["verdict"],
                             "loco_indicator_auroc": r["loco_indicator_auroc"],
                             "loco_indicator_ci95": r["loco_indicator_ci95"],
                             "n_unique": r["n_unique"],
                             "max_abs_spearman_with_q1": r["max_abs_spearman_with_q1"],
                             "reason": r["verdict_reason"]})
    rt = {
        "experiment": "A3",
        "description": "Merge candidates for the Q1 rule table: NEW deterministic cheap-signal features (fixed integer Q=4 thresholds).",
        "derivation": ("thresholds = quantiles p01/p05/p10/p25 (mirrored to p99/p95/p90/p75 for high-is-bad) of the "
                       "feature on the 150 IndEgo gold windows, quantised to 4 dp; thresholds_q = int(round(t*1e4)); "
                       "score contribution = 1 per crossed rung, no weights. LOCO columns use per-fold (6-clip) tables."),
        "quantisation": Q,
        "n": int(len(y)), "label_source": "human (IndEgo gold)",
        "protocol": "leave-one-clip-out over 7 clips; clip-bootstrap 95% CI (B=2000, seed 0)",
        "exact_command": "OMP_NUM_THREADS=2 python3 eval/exp_phys_dsh/a3_candidates.py",
        "rules": rules_out,
        "excluded_candidates": excluded,
        "notes": [
            "Only candidates whose best LOCO AUROC clears 0.60 with CI-low > 0.50 are listed in rules[]; everything else is in excluded_candidates with the reason.",
            "Features whose robustness field is null were not exercised in the resolution/fps sensitivity run (pool/other-source features).",
        ],
    }
    out["wall_s"] = round(time.perf_counter() - t0, 2)
    with open(os.path.join(OUTDIR, "A3_new_rules.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=False)
        f.write("\n")
    with open(os.path.join(OUTDIR, "A3_rule_table_candidates.json"), "w") as f:
        json.dump(rt, f, indent=1, sort_keys=False)
        f.write("\n")

    print(f"wall {out['wall_s']}s  n_cand={len(results)}  mergeable={len(rules_out)}  excluded={len(excluded)}")
    print(f"{'feature':24s} {'rule':16s} {'dir':5s} {'ind':>6s} {'CI':>16s} {'sev':>6s} {'raw':>6s} {'verdict':14s}")
    for r in results:
        ci = r["loco_indicator_ci95"]
        print(f"{r['feature']:24s} {str(r['rule']):16s} {str(r['frozen_direction']):5s} "
              f"{r['loco_indicator_auroc']:6.3f} [{ci[0]:.3f},{ci[1]:.3f}] {r['loco_severity_auroc']:6.3f} "
              f"{str(r['loco_raw_signed_auroc']):>6s} {r['verdict']:14s} {r['verdict_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
