#!/usr/bin/env python3
"""Koopman / exact-DMD prototype for egocentric activity (direction 2).

Method (exact DMD, Schmid 2010; Tu et al. 2014): given snapshot sequence
X = [x1..xm], X1 = X[:,:-1], X2 = X[:,1:], X' = A X. Reduced operator via
truncated SVD X1 = U S V*, Atilde = U' X2 V S^-1, eigendecompose Atilde,
modes Phi = X2 V S^-1 W. Windowed/streaming variant: sliding 2-s exact-DMD
(Hemati et al. 2014 style update approximated by independent short windows).

INPUT CHOICE (important): sibling direction-1 found raw-luma rank features
ANTI-correlate with usable (ego-motion swamps hands). So we do NOT run DMD on
raw frames: input snapshots are FRAME DIFFERENCES d_t = f_{t+1} - f_t
(grayscale, 32x32, column vectors). This is a cheap motion-compensated input:
static background and constant ego-translation cancel to first order, leaving
transients (hands/tools). Documented; no optical-flow warping (CPU budget).

Deterministic: uniform frame sampling, fixed sizes, numpy SVD only.
CPU-only, single core.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("need opencv (cv2)")

SIZE = 32          # spatial downscale
NFRAMES = 60       # frames per 10-s window (6 fps effective)
DIFF = True        # frame-difference input (motion-compensated, cheap)
RANK = 15          # exact-DMD truncation rank (full window)
SUB_LEN = 12       # 2-s subwindow in frames (at 6fps)
SUB_STRIDE = 6     # 1-s stride
SUB_RANK = 8
FMIN, FMAX = 0.3, 8.0   # rhythmic band Hz
MAG_LO, MAG_HI = 0.80, 1.20  # near-unit-circle band


def decode_clip(path, size=SIZE, max_frames=20000):
    """Sequential single-pass decode of a clip to gray downscaled frames + fps.
    Sequential (no seeks): H.264 random seeks are ~100x slower."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    frames = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(g.astype(np.float32) / 255.0)
    cap.release()
    return np.stack(frames, axis=0), fps


def sample_window(clip_frames, fps, start_s, end_s, nframes=NFRAMES):
    ts = np.linspace(start_s + 0.05, end_s - 0.05, nframes)
    idx = np.clip((ts * fps).astype(int), 0, len(clip_frames) - 1)
    return clip_frames[idx]


def exact_dmd(X, dt, rank):
    """Exact DMD (Tu et al.). X: (n,m) snapshots. Returns dict of spectra."""
    n, m = X.shape
    X1, X2 = X[:, :-1], X[:, 1:]
    r = int(min(rank, min(X1.shape) - 1))
    r = max(r, 1)
    U, s, Vh = np.linalg.svd(X1, full_matrices=False)
    U, s, V = U[:, :r], s[:r], Vh[:r, :].T
    Atilde = (U.T @ X2 @ V) / s[None, :]  # (r,r)
    lam, W = np.linalg.eig(Atilde)
    Phi = (X2 @ V) / s[None, :] @ W  # exact modes (n,r)
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = np.log(lam + 0j) / dt
    freq = np.abs(omega.imag) / (2 * np.pi)
    growth = omega.real
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta = -growth / np.abs(omega)
    zeta = np.where(np.isfinite(zeta), zeta, 0.0)
    # amplitudes: least-squares fit of first snapshot
    b, *_ = np.linalg.lstsq(Phi, X1[:, 0], rcond=None)
    energy = np.abs(b) ** 2
    # one-step residual ||X2 - Phi diag(lam) ... || via projected operator
    X1p = U @ np.diag(s) @ V.T
    Aapprox = (X2 @ V / s[None, :]) @ U.T
    resid = float(np.linalg.norm(X2 - Aapprox @ X1) / (np.linalg.norm(X2) + 1e-12))
    return {"lambda": lam, "freq": np.asarray(freq.real), "growth": np.asarray(growth.real),
            "zeta": np.asarray(zeta.real), "energy": np.asarray(energy.real),
            "mag": np.abs(lam), "resid": resid, "rank": r,
            "spec_radius": float(np.max(np.abs(lam))) if len(lam) else 0.0}


def band_features(d, dt, rank):
    X = d.reshape(d.shape[0], -1).T  # (n, m)
    # guard degenerate (all-zero diffs e.g. dark/frozen)
    nrm = float(np.linalg.norm(X))
    if nrm < 1e-9 or X.shape[1] < 4:
        return {"rhythmic_energy": 0.0, "rhythmic_frac": 0.0, "n_rhythmic": 0,
                "dom_freq": 0.0, "dom_zeta": 0.0, "dom_mag": 0.0, "resid": 0.0,
                "spec_radius": 0.0, "static_frac": 1.0, "total_energy": 0.0,
                "mean_mag": 0.0}
    res = exact_dmd(X, dt, rank)
    e = res["energy"]
    tot = float(e.sum()) + 1e-12
    rhythmic = (res["freq"] >= FMIN) & (res["freq"] <= FMAX) & \
               (res["mag"] >= MAG_LO) & (res["mag"] <= MAG_HI)
    static = (res["freq"] < 0.15) & (np.abs(res["mag"] - 1.0) < 0.10)
    re_ = float(e[rhythmic].sum())
    k = int(np.argmax(e))
    return {"rhythmic_energy": re_, "log_rhythmic_energy": float(np.log10(re_ + 1e-12)),
            "rhythmic_frac": float(re_ / tot), "n_rhythmic": int(rhythmic.sum()),
            "dom_freq": float(res["freq"][k]), "dom_zeta": float(res["zeta"][k]),
            "dom_mag": float(res["mag"][k]), "resid": res["resid"],
            "spec_radius": res["spec_radius"], "static_frac": float(e[static].sum() / tot),
            "total_energy": float(np.log10(tot)),
            "mean_mag": float(np.average(res["mag"], weights=e + 1e-12))}


def extract(frames, fps_eff):
    dt = 1.0 / fps_eff
    if DIFF:
        D = np.diff(frames, axis=0)  # (T-1,H,W)
    else:
        D = frames
    full = band_features(D, dt, RANK)
    # windowed / streaming approx: sliding 2-s exact DMD
    T = D.shape[0]
    subs = []
    t0 = 0
    while t0 + SUB_LEN <= T:
        subs.append(band_features(D[t0:t0 + SUB_LEN], dt, SUB_RANK))
        t0 += SUB_STRIDE
    if not subs:
        subs = [full]
    re_vals = np.array([s["rhythmic_frac"] for s in subs])
    df_vals = np.array([s["dom_freq"] for s in subs])
    out = {f"full_{k}": v for k, v in full.items()}
    out.update({
        "win_mean_rhythmic_frac": float(re_vals.mean()),
        "win_std_rhythmic_frac": float(re_vals.std()),
        "win_frac_active": float(np.mean(re_vals > 0.25)),
        "win_std_domfreq": float(df_vals.std()),
        "win_mean_domfreq": float(df_vals.mean()),
        "n_sub": len(subs),
    })
    # primary score: mean windowed rhythmic fraction (streaming-capable scalar)
    out["score"] = out["win_mean_rhythmic_frac"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-windows", type=int, default=150)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.outdir, "results"), exist_ok=True)
    man = json.load(open(args.manifest))
    items = man["items"][:args.max_windows]
    fps_eff = NFRAMES / 10.0
    rows, t_dec, t_dmd = [], [], []
    # single-pass sequential decode per clip (no random seeks)
    clips_needed = sorted(set(it["path"] for it in items))
    cache = {}
    t_dec_total = 0.0
    for p in clips_needed:
        t0 = time.perf_counter()
        cache[p] = decode_clip(p)
        t_dec_total += time.perf_counter() - t0
    ms_decode_share = t_dec_total * 1000 / len(items)
    for it in items:
        clip_frames, clip_fps = cache[it["path"]]
        t0 = time.perf_counter()
        frames = sample_window(clip_frames, clip_fps, it["start_s"], it["end_s"])
        t1 = time.perf_counter()
        feats = extract(frames, fps_eff)
        t2 = time.perf_counter()
        feats.update({"id": it["id"], "clip": it["clip"], "label": it["label"],
                      "reason": it.get("reason", "?"),
                      "ms_decode": ms_decode_share, "ms_dmd": (t2 - t1) * 1000})
        rows.append(feats)
        t_dec.append(ms_decode_share); t_dmd.append((t2 - t1) * 1000)
    # ---- evaluation: leave-one-clip-out ----
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    clips = sorted(set(r["clip"] for r in rows))
    feat_keys = ["full_rhythmic_frac", "full_log_rhythmic_energy", "full_dom_freq",
                 "full_dom_zeta", "full_resid", "win_mean_rhythmic_frac",
                 "win_frac_active", "full_static_frac"]
    for r in rows:  # sanitize
        for k in feat_keys:
            v = r.get(k, 0.0)
            if not np.isfinite(v):
                r[k] = 0.0
    oof_score, oof_lr, y_all, clip_of = [], [], [], []
    per_clip = {}
    rng = np.random.RandomState(0)
    for c in clips:
        tr = [r for r in rows if r["clip"] != c]
        te = [r for r in rows if r["clip"] == c]
        ytr = np.array([r["label"] for r in tr]); yte = np.array([r["label"] for r in te])
        str_ = np.array([r["score"] for r in tr]); ste = np.array([r["score"] for r in te])
        # threshold via Youden J on train
        thrs = np.unique(str_)
        bestJ, bestT = -1, float(np.median(str_))
        for t in thrs:
            p = (str_ >= t).astype(int)
            sens = (p[ytr == 1] == 1).mean() if (ytr == 1).any() else 0
            spec = (p[ytr == 0] == 0).mean() if (ytr == 0).any() else 0
            J = sens + spec - 1
            if J > bestJ:
                bestJ, bestT = J, float(t)
        # logistic on train clips only
        Xtr = np.array([[r[k] for k in feat_keys] for r in tr])
        Xte = np.array([[r[k] for k in feat_keys] for r in te])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        lr = LogisticRegression(max_iter=2000)
        lr.fit((Xtr - mu) / sd, ytr)
        plr = lr.predict_proba((Xte - mu) / sd)[:, 1]
        auc_s = float(roc_auc_score(yte, ste)) if len(set(yte)) > 1 else float("nan")
        auc_l = float(roc_auc_score(yte, plr)) if len(set(yte)) > 1 else float("nan")
        per_clip[c] = {"n": len(te), "n_pos": int(yte.sum()), "thresh_train": bestT,
                       "auc_thresh_score": auc_s, "auc_logistic": auc_l}
        for r, s, p in zip(te, ste, plr):
            oof_score.append(float(s)); oof_lr.append(float(p))
            y_all.append(r["label"]); clip_of.append(r["clip"])
    y_all = np.array(y_all)
    pooled = {"auc_score": float(roc_auc_score(y_all, oof_score)),
              "auc_logistic": float(roc_auc_score(y_all, oof_lr))}
    # clip bootstrap CI (resample clips w/ replacement, 1000 reps)
    B = 1000
    aucs_s, aucs_l = [], []
    clip_list = np.array(clip_of)
    oof_s, oof_l = np.array(oof_score), np.array(oof_lr)
    for b in range(B):
        cs = rng.choice(clips, size=len(clips), replace=True)
        idx = np.concatenate([np.where(clip_list == c)[0] for c in cs])
        if len(set(y_all[idx])) < 2:
            continue
        aucs_s.append(roc_auc_score(y_all[idx], oof_s[idx]))
        aucs_l.append(roc_auc_score(y_all[idx], oof_l[idx]))
    pooled["auc_score_ci95"] = [float(np.percentile(aucs_s, 2.5)), float(np.percentile(aucs_s, 97.5))]
    pooled["auc_logistic_ci95"] = [float(np.percentile(aucs_l, 2.5)), float(np.percentile(aucs_l, 97.5))]
    # per-reason descriptives
    reasons = {}
    for r in rows:
        d = reasons.setdefault(r["reason"], {"n": 0, "pos": 0, "scores": []})
        d["n"] += 1; d["pos"] += r["label"]; d["scores"].append(r["score"])
    for k, d in reasons.items():
        s = np.array(d["scores"])
        d.update({"mean_score": float(s.mean()), "std_score": float(s.std()),
                  "frac_usable": float(np.mean([rr["label"] for rr in rows if rr["reason"] == k]))})
        del d["scores"]
    # determinism + hashability check: re-run first window
    it0 = items[0]
    cf0, cfps0 = cache[it0["path"]]
    f1 = sample_window(cf0, cfps0, it0["start_s"], it0["end_s"])
    r1, r2 = extract(f1, fps_eff)["score"], extract(f1, fps_eff)["score"]
    det = {"rerun_abs_diff": float(abs(r1 - r2)), "deterministic": bool(r1 == r2)}
    h = hashlib.sha256(json.dumps(
        {k: round(rows[0][k], 6) for k in feat_keys}, sort_keys=True).encode()).hexdigest()[:16]
    summary = {
        "n_windows": len(rows), "n_clips": len(clips), "pooled": pooled,
        "per_clip": per_clip, "per_reason": reasons,
        "cost_ms": {"decode_mean": float(np.mean(t_dec)), "decode_p50": float(np.median(t_dec)),
                    "dmd_mean": float(np.mean(t_dmd)), "dmd_p50": float(np.median(t_dmd)),
                    "total_mean_incl_decode": float(np.mean(np.add(t_dec, t_dmd)))},
        "determinism": det, "hash_example": h,
        "streaming": True, "streaming_note": "windowed DMD uses only past 2-s buffer; no full-window state needed",
        "hashable": True, "hash_how": "round(feature_vector,6) -> json(sort_keys) -> sha256",
        "input": f"frame-difference {SIZE}x{SIZE}, {NFRAMES} frames/10s, rank {RANK}, sub {SUB_LEN}/{SUB_STRIDE} rank {SUB_RANK}",
        "baseline_compare": {"task_baseline": 0.80, "direction1": 0.31},
    }
    json.dump(rows, open(os.path.join(args.outdir, "results", "per_window.json"), "w"), indent=1)
    json.dump(summary, open(os.path.join(args.outdir, "results", "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
