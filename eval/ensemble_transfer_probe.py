"""EXPLORATORY transfer probe: why the IndEgo ensemble scores 0.415 OOD.

Inputs: eval/results/ensemble/{ood_frames.npz, ood_pixel.npz, ood_scores.json},
gold caches (segments.npz, features.json, manifest_gold.json),
frames10k lean_10k heads (AI labels, same dataset as OOD).
Writes eval/results/ensemble/transfer_probe.json. Tunes nothing frozen.

(a) lean_10k frame heads (active_labor, hands_ok) on every OOD frame -> pooled
    per window (mean/max/p25/p75/frac>0.5/var) -> AUROC vs human labels.
(b) sign table: 26 pixel feats + frozen-ens prob + pooled head means:
    IndEgo pooled-AUC sign vs OOD oriented/free AUC.
(c) leave-one-window-out (each window ~ own factory) feasibility: single best
    feature, frozen-ens prob, [p_e,p_g] two-view logistic. UPPER BOUNDS (they
    use OOD labels); estimate of what ~100 more labels could enable.
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ensemble")
HEADS = "/home/parshu/projects/jadal-qa-wt/frames10k/eval/results/frames10k/egocentric_10k/frame_heads.lean_10k.json"
LABEL_MAP = "/home/parshu/projects/jadal-qa-wt/ood_labels/labels.json"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "eval", "ensemble"))
from eval_ood import load_labels_file


def wboot_ci(y, p, rng, B=2000):
    from sklearn.metrics import roc_auc_score
    outs = []
    n = len(y)
    for _ in range(B):
        b = rng.choice(n, n, replace=True)
        if len(set(y[b])) < 2:
            continue
        outs.append(roc_auc_score(y[b], p[b]))
    import numpy as np
    return [float(np.percentile(outs, 2.5)), float(np.percentile(outs, 97.5))]


def pool_row(v):
    import numpy as np
    return [float(v.mean()), float(v.max()), float(np.percentile(v, 25)),
            float(np.percentile(v, 75)), float((v > 0.5).mean()), float(v.var())]


def main():
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    out = {"exploratory": True, "label": "uses OOD human labels; nothing frozen tuned"}

    z = np.load(os.path.join(R, "ood_frames.npz"))
    F, ids, cnt = z["F"].astype(np.float64), [str(i) for i in z["ids"]], z["counts"].astype(int)
    px = np.load(os.path.join(R, "ood_pixel.npz"))
    pxids, X, names = [str(i) for i in px["ids"]], px["X"].astype(np.float64), [str(i) for i in px["names"]]
    labels = load_labels_file(LABEL_MAP)
    keep = [i for i in ids if i in labels and i in set(pxids)]
    y = np.array([labels[i] for i in keep])
    Fi = np.array([ids.index(i) for i in keep])
    Xi = np.array([pxids.index(i) for i in keep])
    out["n"] = len(keep)
    out["n_good"], out["n_bad"] = int(y.sum()), int((y == 0).sum())

    # (a) lean_10k heads per frame -> pooled
    H = json.load(open(HEADS))
    assert H["dim"] == 768
    out["frame_heads"] = {}
    pooled = {}
    for hn, h in H["heads"].items():
        mu, sc = np.asarray(h["mean"]), np.asarray(h["scale"])
        w = np.asarray(h["coef"])
        P = np.full((len(keep), F.shape[1]), np.nan)
        for r, fi in enumerate(Fi):
            E = (F[fi, :cnt[ids.index(keep[r])]] - mu) / sc
            P[r, :len(E)] = 1 / (1 + np.exp(-(E @ w + h["intercept"])))
        rows = np.array([pool_row(P[r, :cnt[ids.index(keep[r])]]) for r in range(len(keep))])
        pooled[hn] = rows
        for j, pn in enumerate(["mean", "max", "p25", "p75", "frac>0.5", "var"]):
            try:
                a = float(roc_auc_score(y, rows[:, j]))
            except ValueError:
                a = None
            out["frame_heads"].setdefault(hn, {})[pn] = {
                "auroc": a, "ci95": wboot_ci(y, rows[:, j], rng) if a is not None else None}
    # reference: same heads pooled on IndEgo gold (pooled, non-LOCO)
    from eval.ensemble_nested_loco import load as gold_load
    gids, gclip, gy, greason, Xe, Xg, gnames, gman = gold_load()
    gz = np.load("/home/parshu/projects/jadal-qa-wt/frames10k/eval/results/frames10k/human_segments/segments.npz",
                 allow_pickle=True)
    gE, gcnt = gz["F"].astype(np.float64), gz["counts"].astype(int)
    gids2 = [str(i) for i in gz["ids"]]
    out["frame_heads_gold_pooled_ref"] = {}
    for hn, h in H["heads"].items():
        mu, sc = np.asarray(h["mean"]), np.asarray(h["scale"])
        w = np.asarray(h["coef"])
        P = np.full((len(gids2), gE.shape[1]), np.nan)
        for r in range(len(gids2)):
            E = (gE[r, :gcnt[r]] - mu) / sc
            P[r, :gcnt[r]] = 1 / (1 + np.exp(-(E @ w + h["intercept"])))
        rows = np.array([pool_row(P[r, :gcnt[r]]) for r in range(len(gids2))])
        gm = {i["id"]: i["label"] for i in
              json.load(open("/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"))["items"]}
        gyy = np.array([gm[i] for i in gids2])
        for j, pn in enumerate(["mean", "max", "p25", "p75", "frac>0.5", "var"]):
            try:
                a = float(roc_auc_score(gyy, rows[:, j]))
            except ValueError:
                a = None
            out["frame_heads_gold_pooled_ref"].setdefault(hn, {})[pn] = a

    # (b) sign table
    GF = json.load(open("/home/parshu/projects/jadal-qa-wt/gpu-pipeline/eval/results/gpu/features.json"))["windows"]
    gm = {i["id"]: i["label"] for i in
          json.load(open("/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"))["items"]}
    gk = [i for i in GF if i in gm]
    gX = np.nan_to_num(np.array([[GF[i]["full"][n] for n in names] for i in gk]))
    gy2 = np.array([gm[i] for i in gk])
    sc_scores = {s["id"]: s["usable_prob"] for s in
                 json.load(open(os.path.join(R, "ood_scores.json")))["scores"] if "error" not in s}
    p_ood = np.array([sc_scores[i] for i in keep])
    ens_row = pool_row  # noqa
    table = []
    feats_ood = {"frozen_ens_prob": p_ood,
                 "head_work_mean": pooled["active_labor"][:, 0],
                 "head_hands_mean": pooled["hands_ok"][:, 0]}
    feats_gold = {}
    for j, n in enumerate(names):
        feats_ood[n] = X[Xi, j]
        feats_gold[n] = gX[:, j]
    # gold frozen-ens probs: nested OOF
    oof = np.load(os.path.join(R, "oof_probs.npz"), allow_pickle=True)
    oofids = [str(i) for i in oof["ids"]]
    # map gold window order gk -> oof: oof ids are manifest ids
    feats_gold["frozen_ens_prob"] = np.array([float(oof["p_ens"][oofids.index(i)]) for i in gk])
    feats_gold["head_work_mean"] = pooled and np.array(
        [float(np.mean(1 / (1 + np.exp(-(((gE[r, :gcnt[r]] - np.asarray(H["heads"]["active_labor"]["mean"])) / np.asarray(H["heads"]["active_labor"]["scale"])) @ np.asarray(H["heads"]["active_labor"]["coef"]) + H["heads"]["active_labor"]["intercept"]))))) for r in range(len(gids2))])
    feats_gold["head_hands_mean"] = pooled and np.array(
        [float(np.mean(1 / (1 + np.exp(-(((gE[r, :gcnt[r]] - np.asarray(H["heads"]["hands_ok"]["mean"])) / np.asarray(H["heads"]["hands_ok"]["scale"])) @ np.asarray(H["heads"]["hands_ok"]["coef"]) + H["heads"]["hands_ok"]["intercept"]))))) for r in range(len(gids2))])
    feats_ood["head_hands_mean"] = pooled["hands_ok"][:, 0]
    for feat, vo in feats_ood.items():
        vg = feats_gold.get(feat)
        try:
            ag = float(roc_auc_score(gy2, vg)) if vg is not None else None
        except ValueError:
            ag = None
        try:
            ao = float(roc_auc_score(y, vo))
        except ValueError:
            ao = None
        if ag is None or ao is None:
            oriented, free, agree, ci = None, None, None, None
        else:
            oriented = ao if ag >= 0.5 else 1 - ao
            free = max(ao, 1 - ao)
            agree = bool((ao - 0.5) * (ag - 0.5) > 0)
            vs = vo if ag >= 0.5 else -vo  # CI on the IndEgo-oriented direction
            ci = [round(v, 3) for v in wboot_ci(y, vs, rng)]
        table.append({"feature": feat,
                      "indego_pooled_auc": round(ag, 3) if ag is not None else None,
                      "ood_oriented_auc": round(oriented, 3) if oriented is not None else None,
                      "ood_free_auc": round(free, 3) if free is not None else None,
                      "ood_ci95_oriented": ci,
                      "sign_agrees": agree})
    table.sort(key=lambda r: (r["ood_free_auc"] is None, -(r["ood_free_auc"] or 0)))
    out["sign_table"] = table

    # (c) leave-one-window-out feasibility (upper bounds)
    out["lowo"] = {}
    Xall = np.column_stack([feats_ood[r["feature"]] for r in table if r["ood_free_auc"] is not None])
    # single best free-auc feature (selected on same 60: optimistic, stated)
    best = table[0]["feature"]
    for tag, Xa in [("single_best:" + best, feats_ood[best].reshape(-1, 1)),
                    ("frozen_ens_prob", p_ood.reshape(-1, 1))]:
        p = np.zeros(len(y))
        for i in range(len(y)):
            tr = np.array([j for j in range(len(y)) if j != i])
            sc = StandardScaler().fit(Xa[tr])
            m = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced")
            m.fit(sc.transform(Xa[tr]), y[tr])
            p[i] = float(m.predict_proba(sc.transform(Xa[i:i + 1]))[0, 1])
        a = float(roc_auc_score(y, p))
        out["lowo"][tag] = {"auroc": round(a, 3), "ci95": [round(v, 3) for v in wboot_ci(y, p, rng)],
                            "note": "UPPER BOUND: model selected/fit with OOD labels"}
    # two-view LOO on frozen head probs applied to OOD
    from jadal.ensemble.scorer import load_config, ensemble_prob
    cfg, _ = load_config()
    order = cfg["views"]["pixel"]["feature_names"]
    prow, pcol = [], []
    for r in range(len(keep)):
        em = F[Fi[r], :cnt[ids.index(keep[r])]].mean(axis=0)
        em = em / max(np.linalg.norm(em), 1e-12)
        _, pe, pg = ensemble_prob(em, {n: X[Xi[r], j] for j, n in enumerate(order)}, cfg)
        prow.append(pe)
        pcol.append(pg)
    Xa = np.column_stack([prow, pcol])
    p = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.array([j for j in range(len(y)) if j != i])
        sc = StandardScaler().fit(Xa[tr])
        m = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced")
        m.fit(sc.transform(Xa[tr]), y[tr])
        p[i] = float(m.predict_proba(sc.transform(Xa[i:i + 1]))[0, 1])
    a = float(roc_auc_score(y, p))
    out["lowo"]["two_view_frozen_heads_LOWO"] = {
        "auroc": round(a, 3), "ci95": [round(v, 3) for v in wboot_ci(y, p, rng)],
        "note": "heads frozen (IndEgo); only 2-param logistic uses OOD labels LOO"}
    json.dump(out, open(os.path.join(R, "transfer_probe.json"), "w"), indent=1)
    print(f"n={out['n']} good={out['n_good']} bad={out['n_bad']}")
    for hn, rows in out["frame_heads"].items():
        print(hn, {k: v["auroc"] for k, v in rows.items()})
    print("top signs:", [(r["feature"], r["indego_pooled_auc"], r["ood_oriented_auc"],
                          r["ood_free_auc"], r["sign_agrees"]) for r in table[:12]])
    print("lowo:", {k: v["auroc"] for k, v in out["lowo"].items()})


if __name__ == "__main__":
    main()
