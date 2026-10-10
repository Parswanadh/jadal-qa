"""Independent recompute of the IndEgo per-pixel arms quoted in FACTS 17:40:
  1D-conv on 8 per-frame series 0.782 (0.63-0.86); boosted trees 0.758;
  physics-only logistic 0.715.

CPU only. Re-implements eval/perpixel/detectors.py:run_series_model and
physics_run.py:loco_lr semantics (imported code is NOT used for the models).
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results", "perpixel"))
OUT = os.environ.get("OUT", os.path.join(os.getcwd(), "recompute_perpixel.json"))
SEED = 0
ACTIVITY_REASONS = {"NO_ACTIVE_WORK", "HANDS_NOT_VISIBLE"}


def auroc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def clip_boot(y, s, clips, n_boot=2000, seed=0):
    """Proper cluster bootstrap (duplicates all windows of a resampled clip)."""
    rng = np.random.RandomState(seed)
    uclips = sorted(set(clips))
    idx_by = {c: np.where(np.asarray(clips) == c)[0] for c in uclips}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uclips, size=len(uclips), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        if len(np.unique(np.asarray(y)[idx])) < 2:
            continue
        vals.append(roc_auc_score(np.asarray(y)[idx], np.asarray(s)[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------- physics LR (physics_run.loco_lr semantics) ----------------
def fill_nan_med(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return (np.where(np.isfinite(Xtr), Xtr, med), np.where(np.isfinite(Xte), Xte, med))


def loco_lr(X, y, clips):
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
    return p


# ---------------- conv1d / GRU (detectors.run_series_model semantics) -------
def run_series(gold):
    import torch
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    z = np.load(os.path.join(RES, "series_gold.npz"), allow_pickle=True)
    smap = {str(i): np.asarray(s, dtype=np.float64) for i, s in zip(z["ids"], z["S"])}
    ids = [str(i) for i in gold["ids"]]
    clips = [str(c) for c in gold["clips"]]
    y_all = gold["y"].astype(int)
    reasons = [str(r) for r in gold["reasons"]]
    mask = (y_all == 1) | np.array([r in ACTIVITY_REASONS for r in reasons])
    keep = [i for i, m in zip(ids, mask) if m]
    T = 100
    Xa = np.stack([np.resize(smap[i], (T, 8)) for i in keep])
    ya = np.array([y for y, m in zip(y_all, mask) if m])
    cla = [c for c, m in zip(clips, mask) if m]
    uclips = sorted(set(cla))

    def make(kind):
        if kind == "conv":
            return torch.nn.Sequential(
                torch.nn.Conv1d(8, 16, 5, padding=2), torch.nn.ReLU(),
                torch.nn.Conv1d(16, 16, 5, padding=2), torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(),
                torch.nn.Linear(16, 1))
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
            net = make(kind)
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
            loss = torch.nn.BCEWithLogitsLoss()
            net.train()
            for _ in range(epochs):
                opt.zero_grad()
                o = (net(Xtr.transpose(1, 2)) if kind == "conv" else net(Xtr)).squeeze(1)
                loss(o, ytr).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                o = (net(Xte.transpose(1, 2)) if kind == "conv" else net(Xte)).squeeze(1)
                ps[~tr] = torch.sigmoid(o).numpy()
        return ps

    inner = {}
    for kind in ("conv", "gru"):
        for ep in (20, 40):
            inner[(kind, ep)] = auroc(ya, run_fold(kind, ep, SEED))
    best = max(inner, key=lambda k: np.nan_to_num(inner[k], nan=-1))
    P = {s: run_fold(best[0], best[1], s) for s in (0, 1, 2)}
    pm = np.mean([P[s] for s in P], axis=0)
    a = auroc(ya, pm)
    lo, hi = clip_boot(ya, pm, cla)
    return {"n": int(len(ya)), "arch": best[0], "epochs": int(best[1]),
            "inner": {f"{k[0]},{k[1]}": round(v, 6) for k, v in inner.items()},
            "oof_auroc_mean_over_seeds": a,
            "auroc_seeds": [auroc(ya, P[s]) for s in (0, 1, 2)],
            "clip_boot95": [lo, hi]}


def main():
    gold = np.load(os.path.join(RES, "gold.npz"), allow_pickle=True)
    out = {}

    # 1) HGB + LR: re-derive metric from the stored OOF probabilities
    m2 = json.load(open(os.path.join(RES, "m2.json")))
    act = m2["activity"]
    y = np.asarray(act["y"])
    cl = np.asarray(act["clips"])
    out["m1_logistic_stored_oof"] = {
        "n": int(len(y)), "n_good": int(y.sum()), "n_bad": int((1 - y).sum()),
        "auroc": auroc(y, act["p_lr"]), "clip_boot95": clip_boot(y, act["p_lr"], cl),
        "stored_auroc": act["auroc_lr"], "stored_ci": act["ci_lr"]}
    out["hgb_stored_oof"] = {
        "n": int(len(y)), "n_good": int(y.sum()), "n_bad": int((1 - y).sum()),
        "auroc": auroc(y, act["p_hgb"]), "clip_boot95": clip_boot(y, act["p_hgb"], cl),
        "stored_auroc": act["auroc_hgb"], "stored_ci": act["ci_hgb"]}

    # 2) physics-only LR recomputed from cached features
    z = np.load(os.path.join(RES, "physics_feats.npz"), allow_pickle=True)
    pids = [str(i) for i in z["ids"]]
    PX = z["X"].astype(np.float64)
    rea = np.asarray([str(r) for r in gold["reasons"]])
    y_all = gold["y"].astype(int)
    gids = [str(i) for i in gold["ids"]]
    order = {i: j for j, i in enumerate(gids)}
    perm = np.array([order[i] for i in pids])
    PXg, reag, yg = PX[perm], rea[perm], y_all[perm]
    clg = np.asarray([str(c) for c in gold["clips"]])[perm]
    m = (yg == 1) | np.array([r in ACTIVITY_REASONS for r in reag])
    p = loco_lr(PXg[m], yg[m], list(clg[m]))
    out["physics_only_lr_recomputed"] = {
        "n": int(m.sum()), "n_good": int(yg[m].sum()), "n_bad": int((1 - yg[m]).sum()),
        "auroc": auroc(yg[m], p), "clip_boot95": clip_boot(yg[m], p, list(clg[m])),
        "stored_auroc": 0.7154054054054054}

    # 3) conv1d/GRU nested LOCO retrained from series
    sys.stderr.write("training series models...\n")
    out["conv1d_series_recomputed"] = run_series(gold)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
