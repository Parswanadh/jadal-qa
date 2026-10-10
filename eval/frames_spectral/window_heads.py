"""Key test: frame heads -> every frame of the 150 human windows -> pooled
window score -> HUMAN usable label, leave-one-clip-out CV.

Cleaning pipeline (amendment A) applied to ACTIVITY/TEMPORAL series; QUALITY
features stay RAW. CLEAN_STEPS (each ablatable on/off):
  median3   : temporal median-of-3 on head-prob series + feature columns.
  lum_norm  : remove global exposure flicker: divide luma-sensitive columns
              (y_mean-driven: y_mean itself, brightness-coupledOpponent? no —
              implemented as: residualise each pooled ACTIVITY column against
              the window's per-frame y_mean via least squares) + divide head
              series? heads are nonlinear; instead standardise input exposure:
              documented limit — lum_norm acts on pooled features, head series
              detrended. Honest limit stated in report.
  gm_comp   : global-motion compensation. Re-decodes each window at 120-px
              gray, phase-correlation shift per consecutive pair; local motion
              = frame-diff energy minus global-shift energy. On/off compares
              pooling with raw diff vs compensated diff.
  detrend   : subtract least-squares linear trend from each series.
  hann_welch: temporal spectra of head outputs via Hann-windowed Welch
              (3 overlapping segments, mean periodogram) vs single raw FFT.
  robust    : median/MAD pooling vs mean/std.
  gauss     : spatial Gaussian 3x3 denoise before extraction (separate cache
              human_windows_gauss.npz) vs raw cache.
A step stays ONLY if it helps inside nested CV (inner LOCO on train clips
selects steps with delta>0; outer LOCO evaluates). CPU cost recorded.

Pooling per window (from cleaned series): mean, max, q10/q50/q90, frac>0.5,
temporal std, Welch band energies (0.15-1/1-3/3-8 Hz + Nyquist flicker),
motion (mean/std/frozen_frac), quality means, skin mean/max.
LOCO: logistic (StandardScaler, C=1, balanced) on pooled vector.
Metrics: AUROC + clip-bootstrap CI (resample 7 clips w/ replacement, B=1000),
balanced accuracy @0.5, per-reason recall. Reason tracking: HANDS_NOT_VISIBLE
vs pooled hands score, NO_ACTIVE_WORK vs pooled activity score (correlation).

Also: EG2-embedding-head comparison (read-only, if frames10k results exist —
never wait, never touch); noise-sigma per clip; label-noise audit
(AI flags + human review queue in segments.json schema under OUT);
retrain-without-flagged evaluated on HUMAN windows only.

Usage: python -m eval.frames_spectral.window_heads
Out: eval/results/frames_spectral/window_results.json,
     human_review_queue.json
"""

from __future__ import annotations

import glob
import json
import time

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from eval.frames_spectral.frame_features import FAMILIES, FEATURES

IN = "eval/results/frames_spectral"
OUT = "eval/results/frames_spectral/window_results.json"
EG2_DIR = "/home/parshu/projects/jadal-qa-wt/frames10k/eval/results/frames10k"
SEED = 0
B = 1000
THR = 0.5

FI = {f: i for i, f in enumerate(FEATURES)}
YCOL = FI["y_mean"]

CLEAN_DEFAULT = {"median3": True, "lum_norm": True, "gm_comp": True,
                 "detrend": True, "hann_welch": True, "robust": True,
                 "gauss": False}


# ----------------------------- head training -----------------------------

def train_final_heads():
    z = np.load(f"{IN}/egocentric_10k.npz", allow_pickle=True)
    X = z["X"].astype(np.float64)
    y_h = (z["hand"].astype(int) > 0).astype(int)
    y_w = z["work"].astype(int)
    heads = {}
    for name, y in (("hands", y_h), ("work", y_w)):
        sc = StandardScaler()
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                               random_state=SEED)
        m.fit(sc.fit_transform(X), y)
        heads[name] = (sc, m)
    return heads


# ----------------------------- cleaning -----------------------------

def med3(S):
    M = S.copy()
    M[1:-1] = np.median(np.stack([S[:-2], S[1:-1], S[2:]]), axis=0)
    return M


def detrend_series(S):
    T = S.shape[0]
    t = np.arange(T, dtype=np.float64)
    A = np.stack([t, np.ones(T)], axis=1)
    coef, *_ = np.linalg.lstsq(A, S, rcond=None)
    return S - A @ coef


def lum_residualise(F, cols):
    """Remove global-exposure-flicker coupling: residualise ACTIVITY columns
    of F (T,d) against per-frame y_mean."""
    y = F[:, YCOL]
    if float(y.std()) < 1e-9:
        return F
    A = np.stack([y, np.ones_like(y)], axis=1)
    G = F.copy()
    for c in cols:
        coef, *_ = np.linalg.lstsq(A, F[:, c], rcond=None)
        G[:, c] = F[:, c] - A @ coef
    return G


def welch_bands(s, fps, hann=True):
    """Band energies of series s via Hann-windowed Welch (3 segs) or raw FFT."""
    T = len(s)
    s = s - s.mean()
    if hann and T >= 12:
        L = T // 2
        starts = [0, (T - L) // 2, T - L]
        P = None
        for st in starts:
            seg = s[st:st + L] * np.hanning(L)
            sp = np.fft.rfft(seg)
            p = sp.real ** 2 + sp.conj().imag ** 2
            P = p if P is None else P + p
        P = P / len(starts)
        fr = np.fft.rfftfreq(L, d=1.0 / fps)
    else:
        sp = np.fft.rfft(s * (np.hanning(T) if hann else 1.0))
        P = sp.real ** 2 + sp.conj().imag ** 2
        fr = np.fft.rfftfreq(T, d=1.0 / fps)
    tot = P[1:].sum() + 1e-12

    def band(lo, hi):
        m = (fr[1:] >= lo) & (fr[1:] < hi)
        return float(P[1:][m].sum() / tot) if m.any() else 0.0

    return {"lo": band(0.15, 1.0), "mid": band(1.0, 3.0),
            "hi": band(3.0, 8.0), "flick": float(P[-1] / tot) if T > 2 else 0.0}


def pool_series(s, fps, cfg):
    """Pooling of one cleaned head-prob series -> dict of floats."""
    if cfg["robust"]:
        loc = float(np.median(s))
        scale = float(np.median(np.abs(s - loc)) / 0.6745)
    else:
        loc = float(s.mean())
        scale = float(s.std())
    w = welch_bands(s, fps, hann=cfg["hann_welch"])
    return {"mean": float(s.mean()), "median": loc, "std": scale,
            "max": float(s.max()),
            "q10": float(np.percentile(s, 10)), "q50": float(np.percentile(s, 50)),
            "q90": float(np.percentile(s, 90)),
            "frac_above": float(np.mean(s > THR)),
            **{f"spec_{k}": v for k, v in w.items()}}


# ----------------------------- global motion -----------------------------

_gm_cache: dict = {}


def global_motion(path, start_s, end_s):
    """Per-pair global shift magnitudes at 120-px gray via phase correlation.
    Cached per window id. Returns (diff_raw, gm_shift, diff_comp)."""
    key = (path, start_s, end_s)
    if key in _gm_cache:
        return _gm_cache[key]
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    grays = []
    while True:
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        ok, fr = cap.read()
        if not ok or pos >= end_s - 1e-9:
            break
        if pos >= start_s - 1e-9:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            h, w = g.shape
            nw = 120
            nh = max(1, int(round(h * nw / w)))
            grays.append(cv2.resize(g, (nw, nh),
                                    interpolation=cv2.INTER_AREA).astype(np.float64) / 255.0)
    cap.release()
    diffs, shifts = [], []
    prev = None
    hann = None
    for g in grays:
        if prev is not None:
            diffs.append(float(np.mean(np.abs(g - prev))))
            try:
                sh, _resp = cv2.phaseCorrelate(prev.astype(np.float32),
                                               g.astype(np.float32))
                shifts.append(float(np.hypot(sh[0], sh[1])))
            except Exception:
                shifts.append(0.0)
        prev = g
    diffs = np.asarray(diffs or [0.0])
    shifts = np.asarray(shifts or [0.0])
    # compensated local motion: raw diff minus scaled global shift energy
    comp = np.maximum(diffs - 0.02 * shifts, 0.0)
    _gm_cache[key] = (diffs, shifts, comp)
    return _gm_cache[key]


# ----------------------------- window feature build -----------------------------

def build_window_matrix(cfg, variant="raw"):
    z = np.load(f"{IN}/human_windows_{variant}.npz", allow_pickle=True)
    X = z["X"].astype(np.float64)
    L = z["lengths"].astype(int)
    fps = z["fps"].astype(float)
    ids = [str(s) for s in z["ids"].tolist()]
    clips = [str(s) for s in z["clips"].tolist()]
    labels = z["labels"].astype(int)
    # reasons/meta from gold manifest (read-only)
    with open("/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json") as f:
        man = json.load(f)
    meta = {it["id"]: it for it in man["items"]}
    heads = train_final_heads()
    rows, names = [], None
    zraw = np.load(f"{IN}/human_windows_raw.npz", allow_pickle=True) \
        if variant != "raw" else z
    Xr_all = zraw["X"].astype(np.float64)
    Lr_all = zraw["lengths"].astype(int)
    for i in range(len(ids)):
        T = int(L[i])
        F = X[i, :T]
        if T < 4:  # degenerate: pad stats
            F = np.tile(F.mean(axis=0, keepdims=True) if T else np.zeros((1, F.shape[1])), (4, 1))
            T = 4
        it = meta[ids[i]]
        if cfg["median3"]:
            F = med3(F)
        if cfg["lum_norm"]:
            act_cols = [FI[f] for fam in
                        ("skin", "opponent", "chroma", "fft", "dct", "wavelet")
                        for f in FAMILIES[fam]]
            F = lum_residualise(F, act_cols)
        if cfg["detrend"]:
            F = detrend_series(F)
        # head series on cleaned features
        series = {}
        for hname, (sc, m) in heads.items():
            series[hname] = m.predict_proba(sc.transform(F))[:, 1]
            if cfg["median3"]:
                series[hname] = med3(series[hname][:, None])[:, 0]
            if cfg["detrend"]:
                mu = series[hname].mean()
                series[hname] = detrend_series(series[hname][:, None])[:, 0] + mu
        # motion
        diffs, shifts, comp = global_motion(it["path"], it["start_s"], it["end_s"])
        mot = comp if cfg["gm_comp"] else diffs
        row = {}
        for hname, s in series.items():
            for k, v in pool_series(s, float(fps[i]) or 10.0, cfg).items():
                row[f"{hname}_{k}"] = v
        if cfg["robust"]:
            row["mot_loc"] = float(np.median(mot))
            row["mot_scale"] = float(np.median(np.abs(mot - np.median(mot))) / 0.6745)
        else:
            row["mot_loc"] = float(mot.mean())
            row["mot_scale"] = float(mot.std())
        row["mot_max"] = float(mot.max())
        row["frozen_frac"] = float(np.mean(mot < 1e-4))
        # QUALITY on RAW variant always (cleaning would erase the signal)
        Tr = int(Lr_all[i])
        Fr = Xr_all[i, :Tr] if Tr else F
        for f in ("lap_var", "sobel_energy", "edge_density", "noise_immerkaer",
                  "noise_mad_hh1", "jpeg_block", "under_frac", "over_frac",
                  "y_mean", "y_std", "lvar_mean"):
            row[f"q_{f}"] = float(Fr[:, FI[f]].mean())
        row["q_skin_mean"] = float(Fr[:, FI["skin_frac"]].mean())
        row["q_skin_max"] = float(Fr[:, FI["skin_frac"]].max())
        if names is None:
            names = sorted(row.keys())
        rows.append([row[k] for k in names])
    M = np.asarray(rows, dtype=np.float64)
    return M, names, ids, clips, labels


# ----------------------------- LOCO -----------------------------

def loco_scores(M, labels, clips):
    """Leave-one-clip-out oof P(usable). Returns oof array."""
    oof = np.zeros(len(labels))
    for c in sorted(set(clips)):
        tr = np.asarray([x != c for x in clips])
        te = ~tr
        sc = StandardScaler()
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                               random_state=SEED)
        m.fit(sc.fit_transform(M[tr]), labels[tr])
        oof[te] = m.predict_proba(sc.transform(M[te]))[:, 1]
    return oof


def clip_boot_ci(labels, oof, clips, rng, b=B):
    uclips = sorted(set(clips))
    vals = []
    lab = np.asarray(labels)
    idx_all = list(range(len(lab)))
    for _ in range(b):
        samp = rng.choice(uclips, len(uclips), replace=True)
        idx = []
        for c in samp:
            idx += [i for i, cc in enumerate(clips) if cc == c]
        try:
            v = roc_auc_score(lab[idx], oof[idx])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def summarize(labels, oof, clips, rng, reasons=None):
    auc = float(roc_auc_score(labels, oof))
    pred = (oof >= 0.5).astype(int)
    s = {"auroc": auc, "auroc_clipboot_ci": list(clip_boot_ci(labels, oof, clips, rng)),
         "bal_acc": float(balanced_accuracy_score(labels, oof >= 0.5)),
         "n": int(len(labels))}
    if reasons is not None:
        rec = {}
        for r in sorted(set(reasons)):
            idx = [i for i, x in enumerate(reasons) if x == r]
            if r == "":
                continue
            rec[r] = {"n": len(idx),
                      "recall_bad": float(np.mean(pred[idx] == 0))}
        s["per_reason_recall_bad"] = rec
    return s


# ----------------------------- main -----------------------------

def main():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    with open("/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json") as f:
        man = json.load(f)
    reasons_all = {it["id"]: it.get("reason", "") for it in man["items"]}
    out: dict = {"label_source": "HUMAN gold (single human labeller, IndEgo 480p H.264); "
                                   "frame heads trained on AI labels (gemini-2.5-flash, JPEG frames)",
                 "n_windows": len(man["items"]), "snapshot_utc": man["snapshot_utc"]}
    # nested-CV step selection: inner LOCO on train clips
    M_full, names, ids, clips, labels = build_window_matrix(CLEAN_DEFAULT, "raw")
    reasons = [reasons_all[i] for i in ids]
    out["n_features"] = len(names)
    uclips = sorted(set(clips))

    def inner_loco_auc(M, labels, clips, held):
        keep = [i for i, c in enumerate(clips) if c != held]
        Mk = M[keep]
        lk = labels[keep]
        ck = [clips[i] for i in keep]
        oof = loco_scores(Mk, lk, ck)
        try:
            return float(roc_auc_score(lk, oof))
        except Exception:
            return float("nan")

    base_inner = {c: inner_loco_auc(M_full, labels, clips, c) for c in uclips}
    # one-at-a-time ablation (outer, non-nested, for reporting) + nested select
    steps = [k for k in CLEAN_DEFAULT if k != "gauss"]
    out["cleaning_ablation"] = {}
    nested_keep_votes: dict = {k: 0 for k in steps}
    for step in steps:
        cfg = dict(CLEAN_DEFAULT)
        cfg[step] = not cfg[step]
        M, *_ = build_window_matrix(cfg, "raw")
        oof = loco_scores(M, labels, clips)
        s = summarize(labels, oof, clips, rng, reasons)
        # nested: does removing `step` hurt inner-LOCO on each outer-train set?
        for c in uclips:
            if inner_loco_auc(M, labels, clips, c) < base_inner[c]:
                nested_keep_votes[step] += 1  # removing hurt -> step helps
        out["cleaning_ablation"][f"no_{step}"] = s
    oof_full = loco_scores(M_full, labels, clips)
    out["full_pipeline"] = summarize(labels, oof_full, clips, rng, reasons)
    out["full_pipeline"]["nested_keep_votes_out_of_7"] = nested_keep_votes
    out["full_pipeline"]["kept_steps"] = [k for k, v in nested_keep_votes.items() if v >= 4]
    # gauss variant
    try:
        M_g, *_ = build_window_matrix({**CLEAN_DEFAULT, "gauss": True}, "gauss")
        out["cleaning_ablation"]["gauss_denoise"] = summarize(
            labels, loco_scores(M_g, labels, clips), clips, rng, reasons)
    except FileNotFoundError:
        out["cleaning_ablation"]["gauss_denoise"] = {"status": "missing-cache"}
    # reason tracking correlations
    hands_pool = M_full[:, names.index("hands_mean")]
    work_pool = M_full[:, names.index("work_mean")]
    bad = labels == 0
    out["reason_tracking"] = {
        "hands_score_lower_in_HANDS_NOT_VISIBLE": {
            "n": int(sum(1 for r in reasons if r == "HANDS_NOT_VISIBLE")),
            "mean_hands_bad_handreason": float(np.mean(
                [hands_pool[i] for i, r in enumerate(reasons) if r == "HANDS_NOT_VISIBLE"] or [float("nan")])),
            "mean_hands_usable": float(np.mean(hands_pool[labels == 1]))},
        "work_score_lower_in_NO_ACTIVE_WORK": {
            "n": int(sum(1 for r in reasons if r == "NO_ACTIVE_WORK")),
            "mean_work_bad_workreason": float(np.mean(
                [work_pool[i] for i, r in enumerate(reasons) if r == "NO_ACTIVE_WORK"] or [float("nan")])),
            "mean_work_usable": float(np.mean(work_pool[labels == 1]))}}
    # noise sigma per clip
    zr = np.load(f"{IN}/human_windows_raw.npz", allow_pickle=True)
    Xr = zr["X"].astype(np.float64)
    Lr = zr["lengths"].astype(int)
    noise = {}
    for i, c in enumerate(clips):
        d = {"immer": float(Xr[i, :Lr[i], FI["noise_immerkaer"]].mean()),
             "mad": float(Xr[i, :Lr[i], FI["noise_mad_hh1"]].mean())}
        noise.setdefault(c, []).append(d)
    out["noise_sigma_per_clip"] = {
        c: {"n": len(v), "immer_mean": float(np.mean([d["immer"] for d in v])),
            "mad_mean": float(np.mean([d["mad"] for d in v]))} for c, v in noise.items()}
    # EG2 comparison (read-only; never wait/touch)
    eg2 = {"status": "NOT RUN / missing"}
    try:
        cands = glob.glob(f"{EG2_DIR}/*.json") + glob.glob(f"{EG2_DIR}/*.npz")
        if cands:
            eg2 = {"status": f"files present but no per-window-score schema consumed: {cands[:5]}",
                   "note": "EG2 arm (#53 frames10k) not consumed; spectral LOCO stands alone."}
    except Exception as e:
        eg2 = {"status": f"unreadable: {e}"}
    out["eg2_comparison"] = eg2
    # label-noise audit
    try:
        zo = np.load(f"{IN}/frame_oof_logreg.npz", allow_pickle=True)
        ai_flags = {}
        for tname, key in (("hands_visible", "hands_visible"), ("active_labor", "active_labor")):
            p = zo[key].astype(float)
            y = (zo["hand"].astype(int) > 0).astype(int) if key == "hands_visible" \
                else zo["work"].astype(int)
            flag = np.where(np.abs(p - y) > 0.8)[0]
            ai_flags[tname] = {"n_flagged": int(len(flag)),
                               "frac": float(len(flag) / len(y)),
                               "idx": [int(i) for i in flag[:5000]]}
        out["ai_label_audit"] = ai_flags
        # retrain heads without flagged AI samples -> effect on HUMAN windows only
        flags = set(ai_flags["hands_visible"]["idx"]) | set(ai_flags["active_labor"]["idx"])
        out["ai_label_audit"]["n_union_flagged"] = len(flags)
        keep = np.asarray([i not in flags for i in range(len(zo["hand"]))])
        z10 = np.load(f"{IN}/egocentric_10k.npz", allow_pickle=True)
        X10 = z10["X"].astype(np.float64)
        heads_clean = {}
        for name, y in (("hands", (z10["hand"].astype(int) > 0).astype(int)),
                        ("work", z10["work"].astype(int))):
            sc = StandardScaler()
            m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                   random_state=SEED)
            m.fit(sc.fit_transform(X10[keep]), y[keep])
            heads_clean[name] = (sc, m)
        # quick re-pool with cleaned heads (raw variant, default cfg) -> LOCO
        import eval.frames_spectral.window_heads as _self  # noqa
        _orig = _self.train_final_heads
        _self.train_final_heads = lambda: heads_clean
        try:
            M_c, names_c, ids_c, clips_c, labels_c = _self.build_window_matrix(CLEAN_DEFAULT, "raw")
            oof_c = loco_scores(M_c, labels_c, clips_c)
            out["retrain_without_flagged_ai"] = summarize(labels_c, oof_c, clips_c, rng)
        finally:
            _self.train_final_heads = _orig
    except FileNotFoundError:
        out["ai_label_audit"] = {"status": "frame_oof_logreg.npz missing"}
    # human review queue (segments.json schema; NEVER edit gold)
    order = np.argsort(np.abs(oof_full - labels))[::-1]
    queue = []
    for i in order[:15]:
        it = man["items"][[j for j, x in enumerate(man["items"]) if x["id"] == ids[i]][0]]
        queue.append({"segment_id": ids[i], "clip": clips[i], "path": it["path"],
                      "start_s": it["start_s"], "end_s": it["end_s"],
                      "human_label": int(labels[i]), "human_reason": reasons[i],
                      "oof_p_usable": float(oof_full[i]),
                      "flag": "oof strongly disagrees with human label"})
    strong = [q for q in queue if abs(q["oof_p_usable"] - q["human_label"]) > 0.7]
    with open(f"{IN}/human_review_queue.json", "w") as f:
        json.dump({"n_flagged_strong": len(strong),
                   "note": "human labels NEVER edited; re-check with labelling tool",
                   "items": queue}, f, indent=1)
    out["human_label_audit"] = {"n_strong_disagree": len(strong),
                                "queue": f"{IN}/human_review_queue.json"}
    out["elapsed_s"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    fp = out["full_pipeline"]
    print(f"FULL LOCO: AUROC={fp['auroc']:.3f} CI={fp['auroc_clipboot_ci']} "
          f"balAcc={fp['bal_acc']:.3f}", flush=True)
    print("kept steps:", out["full_pipeline"]["kept_steps"], flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
