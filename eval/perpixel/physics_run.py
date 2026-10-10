"""Physics features on all 150 gold windows + ablation vs theory/M1 table.

Output: eval/results/perpixel/physics.json + physics_feats.npz
Regenerate: PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/perpixel/physics_run.py
Protocol: nested LOCO throughout; families compared with the SAME head.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from jadal.perpixel import physics as P  # noqa: E402
from jadal.perpixel import series as S  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SPEC_DIR = "/home/parshu/projects/jadal-qa-wt/spectral_data"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results", "perpixel")
SEED = 0

QUALITY_REASONS = {"TOO_DARK", "UNDECODABLE"}
ACTIVITY_REASONS = {"NO_ACTIVE_WORK", "HANDS_NOT_VISIBLE"}


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


def fill_nan_med(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isfinite(Xtr), Xtr, med), np.where(
        np.isfinite(Xte), Xte, med)


def loco_lr(X, y, clips, verbose=""):
    ucl = sorted(set(clips))
    p = np.zeros(len(y))
    for tc in ucl:
        tr = np.array([c != tc for c in clips])
        Xtr, ytr = X[tr], y[tr]
        ctr = [c for c, t in zip(clips, tr) if t]
        best = None
        for C in (0.01, 0.1, 1.0, 10.0):
            ss = []
            for c2 in sorted(set(ctr)):
                tr2 = np.array([c != c2 for c in ctr])
                if len(np.unique(ytr[tr2])) < 2 or len(np.unique(ytr[~tr2])) < 2:
                    continue
                A, B = fill_nan_med(Xtr[tr2], Xtr[~tr2])
                sc = StandardScaler().fit(A)
                m = LogisticRegression(C=C, max_iter=5000)
                m.fit(sc.transform(A), ytr[tr2])
                ss.append(auroc(ytr[~tr2], m.predict_proba(sc.transform(B))[:, 1]))
            ss = [v for v in ss if np.isfinite(v)]
            mu = float(np.mean(ss)) if ss else float("nan")
            if best is None or (np.isfinite(mu) and mu > best[1]):
                best = (C, mu)
        C = best[0] if best else 1.0
        A, B = fill_nan_med(Xtr, X[~tr])
        sc = StandardScaler().fit(A)
        m = LogisticRegression(C=C, max_iter=5000, random_state=SEED)
        m.fit(sc.transform(A), ytr)
        p[~tr] = m.predict_proba(sc.transform(B))[:, 1]
    a, ci = clip_bootstrap_ci(y, p, clips)
    print(f"{verbose}: auroc={a:.3f} ci={[round(v,3) for v in ci]}", flush=True)
    return p, a, ci


def main():
    gm = json.load(open(os.path.join(SPEC_DIR, "manifest_gold.json")))
    t0 = time.time()
    feats, fails = {}, []
    for it in gm["items"]:
        try:
            fr, fps = S.decode_160x90(it["path"], it["start_s"], it["end_s"])
            feats[it["id"]] = P.extract_physics(fr, fps, it["path"],
                                                it["start_s"], it["end_s"])
        except Exception as e:  # noqa: BLE001
            fails.append({"id": it["id"], "err": str(e)[:120]})
    print(f"extract: n={len(feats)} fails={len(fails)} "
          f"s={time.time()-t0:.1f}", flush=True)
    order = sorted(feats)
    PX = np.array([[feats[i].get(k, float("nan")) for k in P.PHYSICS_FEATURES]
                   for i in order])
    np.savez_compressed(os.path.join(RES, "physics_feats.npz"),
                        ids=np.array(order), X=PX,
                        names=np.array(P.PHYSICS_FEATURES))
    items = {it["id"]: it for it in gm["items"]}
    y = np.array([items[i]["label"] for i in order])
    clips = [items[i]["clip"] for i in order]
    rea = [items[i]["reason"] for i in order]
    # NaN report (spec: <20 blocks -> NaN, never guessed)
    nanrep = {k: float(np.isnan(PX[:, j]).mean())
              for j, k in enumerate(P.PHYSICS_FEATURES)}
    print("nan_frac:", {k: v for k, v in nanrep.items() if v > 0}, flush=True)
    out = {"n": len(order), "fails": fails, "nan_frac": nanrep,
           "feature_units": ("a unitless; b DN^2; snr dB; bytes; MAD DN; "
                             "Tenengrad DN^2 + norm ratio; flow px/frame; "
                             "dhash bits; clip fraction; entropy bits")}
    # --- quality task: quality(7) vs good(100)
    is_q = np.array([r in QUALITY_REASONS for r in rea])
    m = is_q | (y == 1)
    out["quality"] = {}
    pq, aq, cq = loco_lr(PX[m], is_q[m].astype(int),
                         [c for c, mm in zip(clips, m) if mm], "phys quality")
    out["quality"]["physics_only"] = {"auroc": aq, "ci": cq}
    # --- activity task: good vs NO_ACTIVE_WORK/HANDS_NOT_VISIBLE
    is_a = np.array([r in ACTIVITY_REASONS for r in rea])
    m2 = (y == 1) | is_a
    Xph = PX[m2]
    ya = y[m2]
    ca = [c for c, mm in zip(clips, m2) if mm]
    _, aa, ca_ = loco_lr(Xph, ya, ca, "phys activity")
    out["activity"] = {"physics_only": {"auroc": aa, "ci": ca_}}
    # --- combined: M1 121 + theory 18 + physics 16 (needs theory_feats.npz)
    try:
        z = np.load(os.path.join(RES, "gold.npz"), allow_pickle=True)
        zids = [str(i) for i in z["ids"]]
        zmap = {i: z["X"][j].astype(np.float64) for j, i in enumerate(zids)}
        zt = np.load(os.path.join(RES, "theory_feats.npz"), allow_pickle=True)
        tids = [str(i) for i in zt["ids"]]
        from jadal.perpixel import theory as _T
        def _trow(j):
            r = zt["F"][j]
            if isinstance(r, dict):
                return np.array([r[k] for k in _T.THEORY_FEATURES]) / 1e4
            return np.asarray(r, dtype=np.float64)
        tmap = {i: _trow(j) for j, i in enumerate(tids)}
        Xc = np.array([np.concatenate([zmap[i], tmap[i], PX[j]])
                       for j, i in enumerate(order)])
        _, ac, cc = loco_lr(Xc[m2], ya, ca, "combined activity")
        out["activity"]["m1+theory+physics"] = {"auroc": ac, "ci": cc}
        Xth = np.array([np.concatenate([tmap[i], PX[j]])
                       for j, i in enumerate(order)])
        _, ath, cth = loco_lr(Xth[m2], ya, ca, "theory+physics activity")
        out["activity"]["theory+physics"] = {"auroc": ath, "ci": cth}
    except FileNotFoundError:
        out["activity"]["combined_note"] = "theory_feats.npz not ready; rerun"
    # --- per-feature single LOCO AUROC (ablation table; sign noted)
    fam = {}
    for j, k in enumerate(P.PHYSICS_FEATURES):
        v = PX[m2][:, j]
        if np.isnan(v).mean() > 0.5:
            fam[k] = {"auroc": float("nan"), "note": ">50% NaN"}
            continue
        vv = np.where(np.isfinite(v), v, np.nanmedian(v[np.isfinite(v)]))
        # OOF single-feature = rank score, both directions; report best + sign
        a1 = auroc(ya, vv)
        a2 = auroc(ya, -vv)
        fam[k] = {"auroc": float(max(a1, a2)),
                  "direction": "+bad" if a1 >= a2 else "-bad (inverse!)"}
    out["activity"]["per_feature"] = fam
    with open(os.path.join(RES, "physics.json"), "w") as f:
        json.dump(out, f)
    print("wrote physics.json", flush=True)


if __name__ == "__main__":
    main()
