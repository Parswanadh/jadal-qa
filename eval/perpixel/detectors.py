"""M2 per-reason detectors + nested LOCO eval.

Inputs: eval/results/perpixel/{gold,synth}.npz, series_gold.npz, dup_sigs.json
Output: eval/results/perpixel/m2.json (+ m2_dup_pairs.json)
Regenerate: PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/perpixel/detectors.py

Protocols (all nested leave-one-clip-out; NOTHING chosen on the test clip):
 (i)  QUALITY: single-feature threshold rules; direction+threshold = max Youden
      on TRAIN clips only (quality positives = TOO_DARK/UNDECODABLE in train,
      negatives = good windows in train). Test-clip scores aggregated.
      Synth eval = transfer: rule refit on ALL gold, applied to synth (labelled).
 (ii) DUPLICATE: threshold chosen on held-out half of constructed pairs,
      checked on human labels (flag rate of the 6 DUPLICATE + NN table).
  (iii) ACTIVITY: good vs {NO_ACTIVE_WORK, HANDS_NOT_VISIBLE}; L2-logistic /
       HGB with feature subset + hparams + Platt calibration chosen INSIDE
       train clips (inner LOCO); tiny 1D-conv with PRE-SPECIFIED arch/epochs
       (conv, 40 — fixed capacity prior, no selection; its LOCO number is
       unbiased for that fixed procedure). Clip-bootstrap CIs.
       (Fix round 1 / C2: an earlier draft selected (kind, epochs) by max
       outer-LOCO AUROC — a leak, since every test clip voted on the
       selection. Removed; (conv,40) is now fixed up-front.)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from jadal.perpixel import series as S  # noqa: E402

SPEC_DIR = "/home/parshu/projects/jadal-qa-wt/spectral_data"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results", "perpixel")
SEED = 0
rng_global = np.random.RandomState(SEED)

QUALITY_REASONS = {"TOO_DARK", "UNDECODABLE"}
ACTIVITY_REASONS = {"NO_ACTIVE_WORK", "HANDS_NOT_VISIBLE"}

# candidate quality rules: (name, feature substring, direction); value low->bad '<'
QUALITY_CANDS = [
    ("dark_luma", "P_luma_mean_mean", "<"),
    ("dark_G", "G_luma_mean", "<"),
    ("dark_B", "B_brightness", "<"),
    ("dark_p05", "G_luma_p05", "<"),
    ("darkfrac", "G_dark_frac", ">"),
    ("under", "P_under_mean", ">"),
    ("flat_lap", "P_lap_var_mean", "<"),
    ("flat_dct", "P_dct_hf_mean", "<"),
    ("frozen", "P_motion_frac_above", "<"),
]


def load_npz(name):
    # C1: pickle-free loads (gold/synth arrays are plain numeric/unicode).
    z = np.load(os.path.join(RES, name), allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_series():
    """series_gold.npz is ragged (variable T) -> object array of ndarrays.

    Keep allow_pickle=True (unavoidable for ragged storage) but validate:
    every element must be a 2-D float ndarray with 8 columns. Plain ndarrays
    carry no executable payload; anything else raises instead of loading on.
    """
    z = np.load(os.path.join(RES, "series_gold.npz"), allow_pickle=True)
    S = z["S"]
    assert len(S) == len(z["ids"]), "series/ids length mismatch"
    for s in S:
        assert isinstance(s, np.ndarray) and s.ndim == 2 and s.shape[1] == 8, \
            f"unexpected series element {type(s)}"
    return z


def youden_thr(v, y):
    """Best threshold on (v, y) by Youden J. Returns (thr, direction). Direction
    fixed by caller; here threshold only."""
    order = np.argsort(v, kind="stable")
    vs, ys = v[order], y[order]
    P, N = ys.sum(), len(ys) - ys.sum()
    if P == 0 or N == 0:
        return float(vs[len(vs) // 2]), 0.0
    # sweep midpoints; predict bad(=1) for v<=thr
    tp = P - np.cumsum(ys) + ys  # careful below; use straightforward loop-free
    # simpler: cumulative positives above each split
    cpos = np.cumsum(ys)
    best_J, best_t = -1, float(vs[0])
    for i in range(1, len(vs)):
        if vs[i] == vs[i - 1]:
            continue
        t = (vs[i - 1] + vs[i]) / 2
        pred = (v <= t).astype(int)
        TP = int((pred * y).sum())
        TN = int(((1 - pred) * (1 - y)).sum())
        J = TP / P + TN / N - 1
        if J > best_J:
            best_J, best_t = J, t
    return float(best_t), float(best_J)


def auroc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def clip_bootstrap_ci(y, s, clips, n_boot=2000, seed=SEED):
    """Clip-level resampled AUROC CI (resample clips with replacement)."""
    rng = np.random.RandomState(seed)
    uclips = np.array(sorted(set(clips)))
    base = auroc(np.asarray(y), np.asarray(s))
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(uclips, size=len(uclips), replace=True)
        idx = [i for i, c in enumerate(clips) if c in samp]
        # count multiplicity of clips
        idx = []
        for c in samp:
            idx += [i for i, cc in enumerate(clips) if cc == c]
        if len(np.unique(np.asarray(y)[idx])) < 2:
            continue
        vals.append(auroc(np.asarray(y)[idx], np.asarray(s)[idx]))
    if not vals:
        return base, [float("nan"), float("nan")]
    return base, [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ------------------------------------------------------------- (i) QUALITY
def run_quality(gold):
    X, names = gold["X"].astype(np.float64), [str(n) for n in gold["names"]]
    ids = [str(i) for i in gold["ids"]]
    clips = [str(c) for c in gold["clips"]]
    y_all = gold["y"].astype(int)
    reasons = [str(r) for r in gold["reasons"]]
    is_q = np.array([r in QUALITY_REASONS for r in reasons])
    is_good = (y_all == 1)
    uclips = sorted(set(clips))
    # nested LOCO: rule choice + threshold inside train
    test_score = np.zeros(len(ids))  # P(bad-capture)
    test_rule = [""] * len(ids)
    for tc in uclips:
        tr = np.array([c != tc for c in clips])
        # train pool: quality positives + good negatives (train clips only)
        pool = tr & (is_q | is_good)
        yt = is_q[pool].astype(int)
        best = None
        for rname, feat, direction in QUALITY_CANDS:
            j = names.index(feat)
            v = X[pool, j]
            vv = -v if direction == ">" else v
            thr, J = youden_thr(vv, yt)
            if best is None or J > best[4]:
                best = (rname, feat, direction, thr, J)
        rname, feat, direction, thr, J = best
        j = names.index(feat)
        vte = X[~tr, j]
        # score = signed distance beyond threshold (higher = more bad)
        s = (thr - vte) if direction == "<" else (vte - thr)
        # scale by train IQR for comparability
        scale = float(np.percentile(X[pool, j], 75) - np.percentile(X[pool, j], 25)) + 1e-12
        test_score[~tr] = s / scale
        for i in np.where(~tr)[0]:
            test_rule[i] = f"{rname}:{feat} {direction} {thr:.4g}"
    # gold eval: quality(7) vs good(100)
    mask = is_q | is_good
    aq, ciq = clip_bootstrap_ci(is_q[mask].astype(int), test_score[mask],
                                [c for c, m in zip(clips, mask) if m])
    # operating point: threshold on test scores with train-only... report fixed:
    # sensitivity at ~5% false-alarm on good (threshold from pooled test scores,
    # reported as descriptive; honest LOCO numbers are AUROC + rule table)
    sg = test_score[is_good]
    thr5 = float(np.percentile(sg, 95))
    sens5 = float((test_score[is_q] >= thr5).mean())
    # per-window rule table for the 7 quality windows
    qtable = [{"id": i, "clip": c, "reason": r, "score": float(s), "rule": ru}
              for i, c, r, s, ru in zip(ids, clips, reasons, test_score, test_rule)
              if r in QUALITY_REASONS]
    # transfer to synth: refit best rule on ALL gold, apply per degradation
    return {"test_score": test_score, "test_rule": test_rule,
            "gold_quality_auroc": aq, "gold_quality_ci": ciq,
            "sens_at_5pctFAR": sens5, "qtable": qtable,
            "n_quality": int(is_q.sum())}


def quality_synth_transfer(gold, synth, qres):
    Xg, names = gold["X"].astype(np.float64), [str(n) for n in gold["names"]]
    yg = gold["y"].astype(int)
    rg = [str(r) for r in gold["reasons"]]
    is_q = np.array([r in QUALITY_REASONS for r in rg])
    pool = is_q | (yg == 1)
    best = None
    for rname, feat, direction in QUALITY_CANDS:
        j = names.index(feat)
        vv = -Xg[pool, j] if direction == ">" else Xg[pool, j]
        thr, J = youden_thr(vv, is_q[pool].astype(int))
        if best is None or J > best[-1]:
            best = (rname, feat, direction, thr, J)
    rname, feat, direction, thr, J = best
    j = names.index(feat)
    Xs = synth["X"].astype(np.float64)
    s = (thr - Xs[:, j]) if direction == "<" else (Xs[:, j] - thr)
    scale = float(np.percentile(Xg[pool, j], 75) - np.percentile(Xg[pool, j], 25)) + 1e-12
    s = s / scale
    degs = [str(r) for r in synth["reasons"]]
    sg = s[np.array([d == "clean" for d in degs])]
    thr5 = float(np.percentile(sg, 95))
    per = {}
    for d in sorted(set(degs)):
        m = np.array([x == d for x in degs])
        per[d] = {"n": int(m.sum()), "det_at_5pctFAR": float((s[m] >= thr5).mean()),
                  "auroc_vs_clean": auroc(
                      np.array([1 if x == d else 0 for x in degs
                                if x in (d, "clean")]),
                      np.array([v for v, x in zip(s, degs) if x in (d, "clean")]))}
    return {"rule": f"{rname}:{feat} {direction} {thr:.4g} (refit all-gold, transfer)",
            "per_degradation": per}


# ------------------------------------------------------------- (ii) DUPLICATE
def _ham(a, b):
    return (int(a, 16) ^ int(b, 16)).bit_count()


def dup_sim(A, B):
    import numpy as np
    t1 = 1 - _ham(A["dct64"], B["dct64"]) / 64
    t2 = 1 - _ham(A["dct32"], B["dct32"]) / 64
    ta = np.asarray(A["temp"], dtype=np.float64)
    tb = np.asarray(B["temp"], dtype=np.float64)
    na, nb = float((ta ** 2).sum() ** 0.5), float((tb ** 2).sum() ** 0.5)
    cos = 0.5 if na == 0 or nb == 0 else float(np.clip((ta * tb).sum() / (na * nb), 0, 1))
    return float(0.25 * t1 + 0.25 * t2 + 0.5 * cos)


def run_duplicates():
    import numpy as np
    with open(os.path.join(RES, "dup_sigs.json")) as f:
        sigs = json.load(f)
    with open(os.path.join(SPEC_DIR, "manifest_dups.json")) as f:
        md = json.load(f)
    files = md["files"]
    dsigs = {}
    for name, rel in files.items():
        p = rel if os.path.isabs(rel) else os.path.join(SPEC_DIR, rel)
        from jadal.spectral import dup
        try:
            dsigs[name] = dup.signature(p)
        except Exception:  # noqa: BLE001
            pass
    pairs = md["pairs"]
    sims = []
    for pr in pairs:
        if pr["a"] in dsigs and pr["b"] in dsigs:
            sims.append((pr, dup_sim(dsigs[pr["a"]], dsigs[pr["b"]])))
    # threshold on stratified half (seed 0), test on other half
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(sims))
    half = len(sims) // 2
    tr, te = [sims[i] for i in idx[:half]], [sims[i] for i in idx[half:]]
    ytr = np.array([1 if p["is_duplicate"] else 0 for p, _ in tr])
    str_ = np.array([s for _, s in tr])
    bestJ, bestT = -1, 0.5
    for t in np.linspace(0.3, 0.99, 140):
        pred = (str_ >= t).astype(int)
        P, N = ytr.sum(), len(ytr) - ytr.sum()
        J = pred[ytr == 1].sum() / P + (1 - pred[ytr == 0]).sum() / N - 1
        if J > bestJ:
            bestJ, bestT = J, t
    yte = np.array([1 if p["is_duplicate"] else 0 for p, _ in te])
    ste = np.array([s for _, s in te])
    pred = (ste >= bestT).astype(int)
    TP = int((pred * yte).sum())
    FP = int((pred * (1 - yte)).sum())
    FN = int(((1 - pred) * yte).sum())
    auc = auroc(yte, ste)
    # gold NN: for each of 150, nearest neighbour among other 149
    with open(os.path.join(SPEC_DIR, "manifest_gold.json")) as f:
        gm = json.load(f)
    gitems = {it["id"]: it for it in gm["items"]}
    ids = sorted(sigs.keys())
    nn = {}
    for i, a in enumerate(ids):
        best = None
        for b in ids:
            if b == a:
                continue
            s = dup_sim(sigs[a], sigs[b])
            if best is None or s > best[1]:
                best = (b, s)
        # top-3
        ranked = sorted(((b, dup_sim(sigs[a], sigs[b])) for b in ids if b != a),
                        key=lambda t: -t[1])[:3]
        nn[a] = {"nn1": {"id": best[0], "sim": best[1],
                         "clip": gitems[best[0]]["clip"],
                         "label": gitems[best[0]]["label"],
                         "reason": gitems[best[0]]["reason"]},
                 "top3": [{"id": b, "sim": s, "clip": gitems[b]["clip"],
                           "label": gitems[b]["label"]} for b, s in ranked],
                 "self": {"clip": gitems[a]["clip"], "label": gitems[a]["label"],
                          "reason": gitems[a]["reason"]}}
    dup_ids = [i for i in ids if gitems[i]["reason"] == "DUPLICATE"]
    rec1 = float(np.mean([nn[i]["nn1"]["sim"] >= bestT for i in dup_ids]))
    rec3 = float(np.mean([any(t["sim"] >= bestT for t in nn[i]["top3"]) for i in dup_ids]))
    flagged = [i for i in ids if nn[i]["nn1"]["sim"] >= bestT]
    prec = float(len([i for i in flagged if gitems[i]["reason"] == "DUPLICATE"])
                 / max(len(flagged), 1))
    return {"constructed": {"threshold": float(bestT), "n_test": len(te),
                            "precision": float(TP / max(TP + FP, 1)),
                            "recall": float(TP / max(TP + FN, 1)),
                            "auroc": auc},
            "human": {"dup_ids": dup_ids, "recall_at1": rec1, "recall_at3": rec3,
                      "precision_top": prec, "n_flagged": len(flagged)},
            "nn": nn}


# ------------------------------------------------------------- (iii) ACTIVITY
def inner_select_logistic(Xtr, ytr, ctr):
    """Inner LOCO over train clips: pick C. Returns (C, inner_auc)."""
    ucl = sorted(set(ctr))
    if len(ucl) < 2:
        return 1.0, float("nan")
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        ss = []
        for c in ucl:
            tr = np.array([x != c for x in ctr])
            if len(np.unique(ytr[tr])) < 2 or len(np.unique(ytr[~tr])) < 2:
                continue
            sc = StandardScaler().fit(Xtr[tr])
            m = LogisticRegression(C=C, max_iter=5000)
            m.fit(sc.transform(Xtr[tr]), ytr[tr])
            ss.append(auroc(ytr[~tr], m.predict_proba(sc.transform(Xtr[~tr]))[:, 1]))
        ss = [v for v in ss if np.isfinite(v)]
        mu = float(np.mean(ss)) if ss else float("nan")
        if best is None or (np.isfinite(mu) and mu > best[1]):
            best = (C, mu)
    return best if best else (1.0, float("nan"))


def run_activity(gold):
    Xn = gold["X"].astype(np.float64)
    names = [str(n) for n in gold["names"]]
    ids = [str(i) for i in gold["ids"]]
    clips = [str(c) for c in gold["clips"]]
    y_all = gold["y"].astype(int)
    reasons = [str(r) for r in gold["reasons"]]
    is_act_bad = np.array([r in ACTIVITY_REASONS for r in reasons])
    mask = (y_all == 1) | is_act_bad  # 137 windows
    Xa, ya = Xn[mask], y_all[mask]
    ida = [i for i, m in zip(ids, mask) if m]
    cla = [c for c, m in zip(clips, mask) if m]
    rea = [r or "GOOD" for r, m in zip(reasons, mask) if m]
    uclips = sorted(set(cla))
    p_lr = np.zeros(len(ida))
    p_hgb = np.zeros(len(ida))
    for tc in uclips:
        tr = np.array([c != tc for c in cla])
        Xtr, ytr = Xa[tr], ya[tr]
        Xte = Xa[~tr]
        C, _ = inner_select_logistic(Xtr, ytr, [c for c, t in zip(cla, tr) if t])
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=C, max_iter=5000, random_state=SEED)
        m.fit(sc.transform(Xtr), ytr)
        # Platt calibration on inner-LOCO out-of-fold train scores
        uin = sorted(set(c for c, t in zip(cla, tr) if t))
        oof = np.zeros(len(Xtr))
        for c2 in uin:
            tr2 = np.array([c != c2 for c in [c for c, t in zip(cla, tr) if t]])
            if len(np.unique(ytr[tr2])) < 2:
                oof[~tr2] = 0.5
                continue
            sc2 = StandardScaler().fit(Xtr[tr2])
            m2 = LogisticRegression(C=C, max_iter=5000, random_state=SEED)
            m2.fit(sc2.transform(Xtr[tr2]), ytr[tr2])
            oof[~tr2] = m2.predict_proba(sc2.transform(Xtr[~tr2]))[:, 1]
        cal = LogisticRegression(C=1.0).fit(oof.reshape(-1, 1), ytr)
        raw = m.predict_proba(sc.transform(Xte))[:, 1]
        p_lr[~tr] = cal.predict_proba(raw.reshape(-1, 1))[:, 1]
        # HGB: inner-LOCO over small grid
        uin2 = uin
        best = None
        for lr_ in (0.05, 0.1):
            for leaf in (7, 15):
                ss = []
                for c2 in uin2:
                    cltr = [c for c, t in zip(cla, tr) if t]
                    tr2 = np.array([c != c2 for c in cltr])
                    if len(np.unique(ytr[tr2])) < 2 or len(np.unique(ytr[~tr2])) < 2:
                        continue
                    h = HistGradientBoostingClassifier(
                        max_iter=100, learning_rate=lr_, max_leaf_nodes=leaf,
                        min_samples_leaf=10, random_state=SEED)
                    h.fit(Xtr[tr2], ytr[tr2])
                    ss.append(auroc(ytr[~tr2], h.predict_proba(Xtr[~tr2])[:, 1]))
                ss = [v for v in ss if np.isfinite(v)]
                mu = float(np.mean(ss)) if ss else float("nan")
                if best is None or (np.isfinite(mu) and mu > best[1]):
                    best = ((lr_, leaf), mu)
        (lr_, leaf), _ = best if best else ((0.05, 7), 0)
        h = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=lr_, max_leaf_nodes=leaf,
            min_samples_leaf=10, random_state=SEED)
        h.fit(Xtr, ytr)
        p_hgb[~tr] = h.predict_proba(Xte)[:, 1]
    out = {"ids": ida, "clips": cla, "reasons": rea, "y": ya.astype(int).tolist(),
           "p_lr": [float(v) for v in p_lr], "p_hgb": [float(v) for v in p_hgb]}
    for k in ("lr", "hgb"):
        s = np.array(out[f"p_{k}"])
        a, ci = clip_bootstrap_ci(ya, s, cla)
        out[f"auroc_{k}"] = a
        out[f"ci_{k}"] = ci
    # fixed 5% false-reject on good: per-reason recall
    sg = p_lr[np.array(out["y"]) == 1]
    thr = float(np.percentile(sg, 5))
    out["thr_5pctFRR"] = thr
    rec = {}
    for r in sorted(set(rea)):
        m = np.array([x == r for x in rea])
        if r == "GOOD":
            rec[r] = float((p_lr[m] >= thr).mean())  # kept fraction
        else:
            rec[r] = float((p_lr[m] < thr).mean())  # rejected fraction
    out["per_reason_at_5pctFRR_lr"] = rec
    # abstain curve: coverage vs balanced accuracy of decided (margin on |p-.5|)
    curve = []
    for q in (0, 10, 20, 30, 40, 50, 60, 70, 80, 90):
        mgn = float(np.percentile(np.abs(p_lr - 0.5), q))
        dec = np.abs(p_lr - 0.5) >= mgn
        yd, sd = np.array(out["y"])[dec], p_lr[dec]
        pred = (sd >= 0.5).astype(int)
        tpr = float(pred[yd == 1].mean()) if (yd == 1).sum() else float("nan")
        tnr = float((1 - pred[yd == 0]).mean()) if (yd == 0).sum() else float("nan")
        curve.append({"margin_q": q, "margin": mgn, "coverage": float(dec.mean()),
                      "bal_acc": float((tpr + tnr) / 2)})
    out["abstain_curve_lr"] = curve
    # per-clip AUROC (LR)
    pcl = {}
    for c in uclips:
        m = np.array([x == c for x in cla])
        if len(np.unique(np.array(out["y"])[m])) == 2:
            pcl[c] = auroc(np.array(out["y"])[m], p_lr[m])
        else:
            pcl[c] = float("nan")
    out["per_clip_auroc_lr"] = pcl
    return out


def run_series_model(gold):
    """Tiny 1D-conv vs GRU on per-frame series, nested LOCO, 3 seeds."""
    import torch
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    z = load_series()
    s_ids = [str(i) for i in z["ids"]]
    Slist = z["S"]
    ids = [str(i) for i in gold["ids"]]
    clips = [str(c) for c in gold["clips"]]
    y_all = gold["y"].astype(int)
    reasons = [str(r) for r in gold["reasons"]]
    is_act_bad = np.array([r in ACTIVITY_REASONS for r in reasons])
    mask = (y_all == 1) | is_act_bad
    keep = [i for i, m in zip(ids, mask) if m]
    smap = {i: np.asarray(s, dtype=np.float64) for i, s in zip(s_ids, Slist)}
    T = 100
    Xa = np.stack([S.fit_length(smap[i], T) for i in keep])
    # standardise per channel on train only inside folds (store global mu/sd per fold)
    ya = np.array([y for y, m in zip(y_all, mask) if m])
    cla = [c for c, m in zip(clips, mask) if m]
    uclips = sorted(set(cla))

    def make_model(kind):
        if kind == "conv":
            return torch.nn.Sequential(
                torch.nn.Conv1d(8, 16, 5, padding=2), torch.nn.ReLU(),
                torch.nn.Conv1d(16, 16, 5, padding=2), torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(),
                torch.nn.Linear(16, 1))
        else:
            class G(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gru = torch.nn.GRU(8, 16, batch_first=True)
                    self.fc = torch.nn.Linear(16, 1)

                def forward(self, x):
                    _, h = self.gru(x)
                    return self.fc(h[-1])
            return G()

    def run_fold(kind, epochs, seed):
        torch.manual_seed(seed)
        ps = np.zeros(len(keep))
        for tc in uclips:
            tr = np.array([c != tc for c in cla])
            mu, sd = Xa[tr].mean((0, 1)), Xa[tr].std((0, 1)) + 1e-9
            Xtr = torch.tensor((Xa[tr] - mu) / sd, dtype=torch.float32)
            ytr = torch.tensor(ya[tr], dtype=torch.float32)
            Xte = torch.tensor((Xa[~tr] - mu) / sd, dtype=torch.float32)
            net = make_model(kind)
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
            loss = torch.nn.BCEWithLogitsLoss()
            net.train()
            for _ in range(epochs):
                opt.zero_grad()
                o = (net(Xtr.transpose(1, 2)) if kind == "conv"
                     else net(Xtr)).squeeze(1)
                l = loss(o, ytr)
                l.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                o = (net(Xte.transpose(1, 2)) if kind == "conv"
                     else net(Xte)).squeeze(1)
                ps[~tr] = torch.sigmoid(o).numpy()
        return ps

    # C2: (kind, epochs) = PRE-SPECIFIED ("conv", 40). Selecting by max over
    # outer-LOCO scores leaks every test clip into the choice, so there is no
    # selection here at all; the LOCO AUROC below is unbiased for this fixed
    # procedure. (An earlier draft did max-select; removed in fix round 1.)
    best = ("conv", 40)
    inner = {"preselected": list(best)}
    P = {}
    for seed in (0, 1, 2):
        P[seed] = run_fold(best[0], best[1], seed)
    pm = np.mean([P[s] for s in P], axis=0)
    psd = np.std([P[s] for s in P], axis=0)
    a, ci = clip_bootstrap_ci(ya, pm, cla)
    # determinism: repeat seed 0 twice
    d1, d2 = run_fold(best[0], best[1], 0), run_fold(best[0], best[1], 0)
    return {"arch": best[0], "epochs": best[1], "inner": {str(k): v for k, v in inner.items()},
            "auroc_mean": a, "ci": ci,
            "auroc_seeds": [auroc(ya, P[s]) for s in P],
            "mean_abs_run_sd": float(np.abs(psd).mean()),
            "determinism_maxdiff": float(np.abs(d1 - d2).max())}


def main():
    gold = load_npz("gold.npz")
    synth = load_npz("synth.npz")
    qres = run_quality(gold)
    print("quality LOCO auroc:", round(qres["gold_quality_auroc"], 3), flush=True)
    qsyn = quality_synth_transfer(gold, synth, qres)
    dres = run_duplicates()
    print("dups constructed:", dres["constructed"], "human:", dres["human"], flush=True)
    ares = run_activity(gold)
    print("activity LR:", round(ares["auroc_lr"], 3), "HGB:",
          round(ares["auroc_hgb"], 3), flush=True)
    sres = run_series_model(gold)
    print("series:", sres["arch"], sres["epochs"], round(sres["auroc_mean"], 3), flush=True)
    out = {"quality": {k: v for k, v in qres.items() if k != "test_score"},
           "quality_synth": qsyn, "duplicates": dres,
           "activity": ares, "series": sres}
    with open(os.path.join(RES, "m2.json"), "w") as f:
        json.dump(out, f)
    np.savez_compressed(os.path.join(RES, "m2_quality_scores.npz"),
                        ids=gold["ids"], scores=qres["test_score"])
    print("wrote m2.json", flush=True)


if __name__ == "__main__":
    main()
