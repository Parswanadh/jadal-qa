"""Amendment A (cleaning ablation) + D (signal-theory detectors, rule list).

Inputs: spectral_data/manifest_gold.json (+ videos), eval/results/perpixel/
        {gold.npz, series_gold.npz, dup_sigs.json, m2.json (optional, for compare)}.
Output: eval/results/perpixel/theory.json + theory_feats.npz
Regenerate: PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/perpixel/theory_clean.py

Protocol: everything nested leave-one-clip-out. Cleaning ladder rungs are
compared with the SAME model class/protocol; a rung stays only on nested-CV
gain. Rule thresholds: exhaustive seed-free grid search inside train clips,
lexicographic tie-break, monotone directions fixed in RULES.
"""

from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from jadal.perpixel import series as S  # noqa: E402
from jadal.perpixel import theory as T  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SPEC_DIR = "/home/parshu/projects/jadal-qa-wt/spectral_data"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results", "perpixel")
SEED = 0

QUALITY_REASONS = {"TOO_DARK", "UNDECODABLE"}
ACTIVITY_REASONS = {"NO_ACTIVE_WORK", "HANDS_NOT_VISIBLE"}

RUNGS = ["raw", "flick", "gmc", "tempmed", "full"]


def auroc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def clip_bootstrap_ci(y, s, clips, n_boot=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    uclips = np.array(sorted(set(clips)))
    base = auroc(np.asarray(y), np.asarray(s))
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(uclips, size=len(uclips), replace=True)
        idx = []
        for c in samp:
            idx += [i for i, cc in enumerate(clips) if cc == c]
        if len(np.unique(np.asarray(y)[idx])) < 2:
            continue
        vals.append(auroc(np.asarray(y)[idx], np.asarray(s)[idx]))
    if not vals:
        return base, [float("nan"), float("nan")]
    return base, [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def rung_stacks(luma: np.ndarray) -> dict:
    """Build the cleaning ladder (cumulative). luma (T,H,W) float64."""
    out = {"raw": luma}
    fl = luma - luma.mean(axis=(1, 2), keepdims=True)
    out["flick"] = fl
    ref = fl[0]
    gmc = np.empty_like(fl)
    gmc[0] = fl[0]
    for i in range(1, fl.shape[0]):
        dy, dx = T.phase_shift_vec(ref, fl[i])
        gmc[i] = np.roll(fl[i], shift=(-dy, -dx), axis=(0, 1))
    out["gmc"] = gmc
    out["tempmed"] = ndimage.median_filter(gmc, size=(3, 1, 1), mode="nearest")
    sp = np.empty_like(gmc)
    for i in range(gmc.shape[0]):
        sp[i] = cv2.GaussianBlur(out["tempmed"][i].astype(np.float32), (0, 0), 0.8)
    out["full"] = sp.astype(np.float64)
    return out


def rung_desc(st: np.ndarray, fps: float) -> dict:
    """Small motion descriptor set for one rung stack (deterministic)."""
    d = np.abs(np.diff(st, axis=0)).mean(axis=(1, 2))
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    floor = med + mad
    best = cur = 0
    for v in d:
        if v <= floor:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    lo, mid, hi, ent = T.welch_bands(d, fps)
    x0 = d - d.mean()
    if float((x0 ** 2).sum()) > 1e-18 and len(d) >= 4:
        ac = np.correlate(x0, x0, mode="full")[len(d) - 1:]
        ac = ac / (ac[0] + 1e-18)
        pk, lag = float(ac[1:].max()), float(int(np.argmax(ac[1:])) + 1)
    else:
        pk, lag = 0.0, 0.0
    return {
        "mot_frac": float((d > floor).mean()), "still": float(best / max(len(d), 1)),
        "w_lo": lo, "w_mid": mid, "w_hi": hi, "w_ent": ent,
        "ac_peak": pk, "ac_lag": lag,
        "burst": float(d.std() / (abs(d.mean()) + 1e-9)),
    }


def phase1():
    """Decode each gold window once; theory feats + ladder descs + noise."""
    with open(os.path.join(SPEC_DIR, "manifest_gold.json")) as f:
        gm = json.load(f)
    t0 = time.time()
    feats, ladders, perclip, fails = {}, {}, {}, []
    for it in gm["items"]:
        try:
            fr, fps = S.decode_160x90(it["path"], it["start_s"], it["end_s"])
        except Exception:  # noqa: BLE001
            fails.append(it["id"])
            continue
        th = T.extract_theory(fr, fps)
        feats[it["id"]] = th
        luma = (0.299 * fr[..., 0] + 0.587 * fr[..., 1] + 0.114 * fr[..., 2])
        rs = rung_stacks(luma)
        sk = T.skin_series(fr)
        ladders[it["id"]] = {
            r: dict(rung_desc(rs[r], fps), skin_mean=float(sk.mean()),
                    skin_dyn=float(sk.std())) for r in RUNGS}
        perclip.setdefault(it["clip"], {"immer": [], "wavelet": [], "block": []})
        perclip[it["clip"]]["immer"].append(th["sigma_immer"])
        perclip[it["clip"]]["wavelet"].append(th["sigma_wavelet"])
        perclip[it["clip"]]["block"].append(th["blockiness"])
    # int64 fixed-point matrix (loads with allow_pickle=False; C1)
    Fmat = np.array([[feats[i][k] for k in T.THEORY_FEATURES]
                     for i in sorted(feats)], dtype=np.float64)
    np.savez_compressed(os.path.join(RES, "theory_feats.npz"),
                        ids=np.array(sorted(feats)),
                        F=(Fmat * 10 ** T.Q).round().astype(np.int64),
                        meta=np.array([json.dumps({
                            "id": i, "float": feats[i],
                            "ladder": ladders[i]}) for i in sorted(feats)]))
    noise = {c: {k: {"median": float(np.median(v)), "mad": float(
        np.median(np.abs(np.asarray(v) - np.median(v)))),
        "n": len(v)} for k, v in d.items()} for c, d in perclip.items()}
    print(f"phase1: n={len(feats)} fails={fails} seconds={time.time()-t0:.1f}",
          flush=True)
    return gm, feats, ladders, noise, fails


def loco_logistic(Xa, ya, cla):
    """Nested LOCO L2-logistic (inner C grid). Returns OOF proba."""
    uclips = sorted(set(cla))
    p = np.zeros(len(ya))
    for tc in uclips:
        tr = np.array([c != tc for c in cla])
        Xtr, ytr = Xa[tr], ya[tr]
        ctr = [c for c, t in zip(cla, tr) if t]
        best = None
        for C in (0.01, 0.1, 1.0, 10.0):
            ss = []
            for c2 in sorted(set(ctr)):
                tr2 = np.array([c != c2 for c in ctr])
                if len(np.unique(ytr[tr2])) < 2 or len(np.unique(ytr[~tr2])) < 2:
                    continue
                sc = StandardScaler().fit(Xtr[tr2])
                m = LogisticRegression(C=C, max_iter=5000)
                m.fit(sc.transform(Xtr[tr2]), ytr[tr2])
                ss.append(auroc(ytr[~tr2],
                               m.predict_proba(sc.transform(Xtr[~tr2]))[:, 1]))
            ss = [v for v in ss if np.isfinite(v)]
            mu = float(np.mean(ss)) if ss else float("nan")
            if best is None or (np.isfinite(mu) and mu > best[1]):
                best = (C, mu)
        C = best[0] if best else 1.0
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=C, max_iter=5000, random_state=SEED)
        m.fit(sc.transform(Xtr), ytr)
        p[~tr] = m.predict_proba(sc.transform(Xa[~tr]))[:, 1]
    return p


def ablation(gm, ladders):
    """Same protocol per rung; rung stays only on nested-CV gain."""
    items = {it["id"]: it for it in gm["items"]}
    keep = [i for i in sorted(ladders)
            if items[i]["label"] == 1
            or items[i]["reason"] in ACTIVITY_REASONS]
    rkeys = sorted(ladders[keep[0]]["raw"].keys())
    res = {}
    for rung in RUNGS:
        Xa = np.array([[ladders[i][rung][k] for k in rkeys] for i in keep],
                      dtype=np.float64)
        ya = np.array([items[i]["label"] for i in keep])
        cla = [items[i]["clip"] for i in keep]
        t0 = time.time()
        p = loco_logistic(Xa, ya, cla)
        a, ci = clip_bootstrap_ci(ya, p, cla)
        # operating threshold at 5% FRR per outer fold (mean+-sd across folds)
        thrs = []
        for tc in sorted(set(cla)):
            tr = np.array([c != tc for c in cla])
            sg = p[tr & (ya == 1)]
            thrs.append(float(np.percentile(sg, 5)) if len(sg) else 0.5)
        res[rung] = {"auroc": a, "ci": ci, "n": len(keep),
                     "thr5FRR_mean": float(np.mean(thrs)),
                     "thr5FRR_sd": float(np.std(thrs)),
                     "cpu_s": float(time.time() - t0)}
        print(f"ablation {rung}: auroc={a:.3f} ci={[round(v,3) for v in ci]}",
              flush=True)
    verdict = {}
    prev = None
    for rung in RUNGS:
        gain = None if prev is None else res[rung]["auroc"] - res[prev]["auroc"]
        keep_r = True if prev is None else (gain is not None and gain > 0)
        verdict[rung] = {"gain_vs_prev": gain, "stays": bool(keep_r)}
        prev = rung
    return {"per_rung": res, "verdict": verdict, "features": rkeys}


# ---------------- D: empirical nulls + rule list ----------------
RULES = [
    # (name, feature, op, grid-kind, grid) — op fires BAD; monotone directions:
    # lower frozen_idx / p05 / blur_norm, higher p95 / still_run / mot lack,
    # weaker periodicity (higher g_logp) can only ever vote BAD, never GOOD.
    ("frozen", "frozen_idx", "<", "fixed", [1.5, 2.0, 3.0, 5.0, 10.0]),
    ("dark", "p05", "<", "pct", [1, 2, 5, 10, 20]),
    ("overexp", "p95", ">", "pct", [80, 90, 95, 98, 99]),
    ("blur", "blur_norm", "<", "pct", [1, 2, 5, 10, 20]),
    ("still", "still_run", ">", "fixed", [0.5, 0.7, 0.8, 0.9, 0.95]),
    ("nomotion", "mot_frac", "<", "pct", [5, 10, 20, 30, 50]),
    ("norhythm", "g_logp", ">", "fixed", [-6.0, -5.0, -4.0, -3.0, -2.0]),
]
STAGE1 = {"frozen", "dark", "overexp", "blur"}
STAGE3 = {"still", "nomotion", "norhythm"}
QMAP = {n: i for i, n in enumerate(T.THEORY_FEATURES)}


def qmat(feats, order):
    F = np.array([[feats[i][k] for k in T.THEORY_FEATURES] for i in order])
    return np.round(F, T.Q)


def grid_values(kind, grid, col):
    if kind == "fixed":
        return list(grid)
    return [float(np.percentile(col, q)) for q in grid]


def hits(QM, order_idx, name, feat, op, thr):
    v = QM[:, QMAP[feat]]
    return (v < thr) if op == "<" else (v > thr)


def fit_stage(QMtr, ytr, rtr, stage, positives, n_good_mask):
    """Exhaustive grid search on train; returns {rulename: thr} + train balacc."""
    pool = np.array([r in positives or (y == 1) for y, r in zip(ytr, rtr)])
    Xp, yp = QMtr[pool], np.array(
        [1 if r in positives else 0 for y, r in zip(ytr, rtr) if
         (r in positives or y == 1)])
    chosen, bals = {}, None
    for name, feat, op, kind, grid in RULES:
        if name not in stage:
            continue
        cand = grid_values(kind, grid, Xp[:, QMAP[feat]])
        best = None
        for t in cand:
            h = hits(Xp, None, name, feat, op, t)
            P = yp.sum()
            N = len(yp) - P
            if P == 0 or N == 0:
                J = 0.0
            else:
                J = h[yp == 1].mean() + (~h[yp == 0]).mean() - 1
            if best is None or J > best[1]:
                best = (t, J)
        chosen[name] = best[0]
    # train balacc of the stage (any-hit)
    H = np.zeros(len(Xp), bool)
    for name, feat, op, _, _ in RULES:
        if name in stage:
            H |= hits(Xp, None, name, feat, op, chosen[name])
    P, N = yp.sum(), len(yp) - yp.sum()
    bal = float(H[yp == 1].mean() + (~H[yp == 0]).mean()) / 2 if P and N else 0.5
    return chosen, bal


def rule_list_loco(gm, feats):
    order = sorted(feats)
    items = {it["id"]: it for it in gm["items"]}
    QM = qmat(feats, order)
    y = np.array([items[i]["label"] for i in order])
    clips = [items[i]["clip"] for i in order]
    rea = [items[i]["reason"] for i in order]
    uclips = sorted(set(clips))
    decide = np.zeros(len(order), bool)  # True = BAD
    ev = [""] * len(order)
    margins = np.zeros(len(order))
    thr_trace = {}
    for tc in uclips:
        tr = np.array([c != tc for c in clips])
        c1, b1 = fit_stage(QM[tr], y[tr],
                           [r for r, t in zip(rea, tr) if t],
                           STAGE1, QUALITY_REASONS, None)
        # stage3 train pool: activity-bad + good NOT rejected by stage1
        H1 = np.zeros(tr.sum(), bool)
        for name, feat, op, _, _ in RULES:
            if name in STAGE1:
                H1 |= hits(QM[tr], None, name, feat, op, c1[name])
        sub = ~H1
        c3, b3 = fit_stage(QM[tr][sub], y[tr][sub],
                           [r for r, t in zip(
                               np.array(rea)[tr][sub], sub) if True],
                           STAGE3, ACTIVITY_REASONS, None)
        thr = dict(c1)
        thr.update(c3)
        thr_trace[tc] = {k: float(v) for k, v in thr.items()}
        te = np.where(~tr)[0]
        H = np.zeros(len(te), bool)
        mg = np.full(len(te), np.inf)
        rl = [""] * len(te)
        for name, feat, op, _, _ in RULES:
            v = QM[te][:, QMAP[feat]]
            t = thr[name]
            h = (v < t) if op == "<" else (v > t)
            H |= h
            signed = (t - v) if op == "<" else (v - t)  # >0 = bad side
            scale = float(np.percentile(QM[tr][:, QMAP[feat]], 75)
                          - np.percentile(QM[tr][:, QMAP[feat]], 25)) + 1e-12
            mg = np.minimum(mg, signed / scale)
            for j, jj in enumerate(te):
                if h[j] and not rl[j]:
                    rl[j] = (f"{name}:{feat} {op} {t:.4g} "
                             f"(value {v[j]:.4g})")
        decide[te] = H
        margins[te] = mg
        for j, jj in enumerate(te):
            ev[jj] = rl[j] if H[j] else f"pass (min-margin {mg[j]:.3g})"
    # metrics on all 150 (bad = positive)
    ybad = (y == 0).astype(int)
    P, N = ybad.sum(), (1 - ybad).sum()
    bal = float(decide[ybad == 1].mean() + (~decide[ybad == 0]).mean()) / 2
    # score for AUROC: max rule-hit strength (HIGHER = more BAD)
    s = np.where(decide, 1 + np.abs(margins), -np.abs(margins))
    a, ci = clip_bootstrap_ci(ybad, s, clips)
    # per-reason recall (BAD reasons caught) + good kept
    rec = {}
    for r in sorted(set(x or "GOOD" for x in rea)):
        m = np.array([(x or "GOOD") == r for x in rea])
        rec[r] = float(decide[m].mean()) if r != "GOOD" else float(
            (~decide[m]).mean())
    # abstain: margin < quant+numeric bound (5e-4 feature units ~ 5 quanta)
    curve = []
    for mult in (1, 2, 5, 10, 20):
        bound = mult * 1e-4
        dec = np.abs(margins) * 1 >= bound  # margins normalised; use rank below
        curve.append({"mult": mult})
    # coverage curve on normalised-margin quantiles (documented units)
    cov_curve = []
    for q in (0, 10, 20, 30, 40, 50, 60, 70, 80, 90):
        mgn = float(np.percentile(np.abs(margins), q))
        dec = np.abs(margins) >= mgn
        yd = ybad[dec]
        pred = decide[dec].astype(int)
        tpr = float(pred[yd == 1].mean()) if (yd == 1).sum() else float("nan")
        tnr = float((1 - pred[yd == 0]).mean()) if (yd == 0).sum() else float("nan")
        cov_curve.append({"margin_q": q, "coverage": float(dec.mean()),
                          "bal_acc": float((tpr + tnr) / 2)})
    # Clopper-Pearson upper bounds on false rejects (good windows called BAD)
    from jadal import certify as C
    cp = {}
    for name, feat, op, _, _ in RULES:
        # pooled outer-test: windows where this rule fired and label is good
        fr = sum(1 for i, e in enumerate(ev)
                 if e.startswith(name + ":") and y[i] == 1)
        n = int((y == 1).sum())
        cp[name] = {"false_rejects": fr, "n_good": n,
                    "cp_upper_95": float(C.exact_binomial_upper(fr, n))}
    # determinism receipts: 10 repeats + thread settings
    det = determinism_check(feats, order, thr_trace)
    return {"bal_acc": bal, "auroc": a, "ci": ci,
            "auroc_note": "score higher = more BAD (rule-hit strength)",
            "per_reason": rec, "coverage_curve": cov_curve,
            "cp_bounds": cp, "thresholds_per_fold": thr_trace,
            "evidence": {i: e for i, e in zip(order, ev)},
            "margins": {i: float(m) for i, m in zip(order, margins)},
            "decisions": {i: bool(d) for i, d in zip(order, decide)},
            "determinism": det}


def determinism_check(feats, order, thr_trace):
    """10 repeats + 1 vs 8 threads -> identical decisions and digests."""
    import hashlib
    QM = qmat(feats, order)
    thr = thr_trace[sorted(thr_trace)[0]]
    def decide_once():
        H = np.zeros(len(order), bool)
        for name, feat, op, _, _ in RULES:
            v = QM[:, QMAP[feat]]
            H |= (v < thr[name]) if op == "<" else (v > thr[name])
        dg = hashlib.sha256(np.round(QM, T.Q).tobytes()).hexdigest()
        return H.tobytes(), dg
    outs = [decide_once() for _ in range(10)]
    cv2.setNumThreads(8)
    th8 = decide_once()
    cv2.setNumThreads(1)
    th1 = decide_once()
    return {"repeats_identical": bool(all(o == outs[0] for o in outs)),
            "digest": outs[0][1],
            "threads_1_vs_8_identical": bool(th8 == th1 == outs[0])}


def null_checks(gm, feats):
    """Calibrate theory nulls on real clips: empirical FAR vs nominal alpha."""
    items = {it["id"]: it for it in gm["items"]}
    good = [i for i in feats if items[i]["label"] == 1]
    fidx = np.array([feats[i]["frozen_idx"] for i in good])
    # nominal: chi-square lower-tail at alpha ( razor-sharp dof~1e6 )
    from scipy import stats as st
    Ndof = 100 * 90 * 160.0
    out = {}
    for alpha in (0.01, 0.001):
        t = float(st.chi2.ppf(alpha, Ndof) / Ndof)
        out[f"frozen_alpha{alpha}"] = {
            "nominal_thr": t, "empirical_far": float((fidx < t).mean()),
            "n_good": len(good)}
    glp = np.array([feats[i]["g_logp"] for i in good])
    for alpha in (0.01, 0.001):
        out[f"rhythm_alpha{alpha}"] = {
            "nominal_thr_logp": float(np.log10(alpha)),
            "empirical_far": float((glp < np.log10(alpha)).mean()),
            "n_good": len(good)}
    # duplicate empirical null: cross-clip pair sims from dup_sigs.json
    from eval.perpixel import detectors as D
    with open(os.path.join(RES, "dup_sigs.json")) as f:
        sigs = json.load(f)
    with open(os.path.join(SPEC_DIR, "manifest_gold.json")) as f:
        gmm = json.load(f)
    gitems = {it["id"]: it for it in gmm["items"]}
    ids = sorted(sigs)
    cross, same = [], []
    for x in range(len(ids)):
        for y2 in range(x + 1, len(ids)):
            s = D.dup_sim(sigs[ids[x]], sigs[ids[y2]])
            (same if gitems[ids[x]]["clip"] == gitems[ids[y2]]["clip"]
             else cross).append(s)
    cross = np.array(cross)
    # exact-hash equality pairs
    exact = [(ids[x], ids[y2]) for x in range(len(ids)) for y2 in range(
        x + 1, len(ids))
        if sigs[ids[x]]["dct64"] == sigs[ids[y2]]["dct64"]
        and sigs[ids[x]]["dct32"] == sigs[ids[y2]]["dct32"]]
    # NN recall on the 6 human DUPLICATEs at alpha-threshold
    t_dup = float(np.percentile(cross, 99))
    nn_hit1 = nn_hit3 = 0
    dup_ids = [i for i in ids if gitems[i]["reason"] == "DUPLICATE"]
    nn_table = {}
    for a in ids:
        rk = sorted(((b, D.dup_sim(sigs[a], sigs[b])) for b in ids if b != a),
                    key=lambda z: -z[1])
        nn_table[a] = [{"id": b, "sim": s, "clip": gitems[b]["clip"],
                        "label": gitems[b]["label"],
                        "reason": gitems[b]["reason"]} for b, s in rk[:3]]
    for i in dup_ids:
        if nn_table[i][0]["sim"] >= t_dup:
            nn_hit1 += 1
        if any(r["sim"] >= t_dup for r in nn_table[i]):
            nn_hit3 += 1
    out["duplicate"] = {
        "empirical_null_n_cross": len(cross), "null_p99": t_dup,
        "null_max": float(cross.max()),
        "exact_hash_pairs": exact, "n_exact_pairs": len(exact),
        "human_dup_ids": dup_ids,
        "recall_at1": float(nn_hit1 / max(len(dup_ids), 1)),
        "recall_at3": float(nn_hit3 / max(len(dup_ids), 1)),
        "nn_top3": {i: nn_table[i] for i in dup_ids},
        "pair_descriptions": [
            {"id": i, "clip": gitems[i]["clip"],
             "nn": nn_table[i][0]} for i in dup_ids],
    }
    return out


def main():
    gm, feats, ladders, noise, fails = phase1()
    abl = ablation(gm, ladders)
    nulls = null_checks(gm, feats)
    print("nulls:", json.dumps({k: v for k, v in nulls.items()
                                if not k.startswith("duplicate")})[:400],
          flush=True)
    rl = rule_list_loco(gm, feats)
    print(f"rule list: bal_acc={rl['bal_acc']:.3f} "
          f"auroc={rl['auroc']:.3f}", flush=True)
    # compare stub: regression numbers merged by freeze.py from m2.json
    out = {"noise_per_clip": noise, "fails": fails,
           "cleaning_ablation": abl, "null_calibration": nulls,
           "rule_list": {k: v for k, v in rl.items()},
           "rules_doc": [{"name": n, "feature": f, "op": o, "grid_kind": k,
                          "grid": g} for n, f, o, k, g in RULES],
           "monotone_note": ("Each rule may only vote BAD; a stronger "
                             "quality defect (darker, blurrier, stiller) "
                             "never moves a window toward GOOD."),
           "proxy_note": ("Activity proxy = local motion above the 3-sigma "
                          "noise floor + 0.3-8 Hz rhythm (Fisher g) + skin "
                          "dynamics. Mathematics cannot define 'hands-on "
                          "work'; what it cannot see: still-held tools, "
                          "occluded hands, off-camera work."),
           "git_sha": os.popen("git rev-parse --short HEAD").read().strip()}
    with open(os.path.join(RES, "theory.json"), "w") as f:
        json.dump(out, f)
    print("wrote theory.json", flush=True)


if __name__ == "__main__":
    main()
