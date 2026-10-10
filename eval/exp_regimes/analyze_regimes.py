"""E4.01 diagnosis + E4.02 regime-vs-global + E4.03 gap control (CPU-only).

Inputs: eval/results/exp_regimes/desc_*.jsonl, cached 26-feats, EG2 npz,
        ood labels, frozen ensemble ood_scores.json.
Outputs: eval/results/exp_regimes/{diagnosis,regime_vs_global,gap_control}.json
         + figs/*.png
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "exp_regimes")
WT = "/home/parshu/projects/jadal-qa-wt"

KEYS = ["width", "height", "fps", "bitrate_kbps", "bytes_per_frame",
        "sharp_lapvar", "sharp_norm", "noise_sigma", "edge_density",
        "luma_mean", "luma_p05", "luma_p50", "luma_p95", "dark_frac",
        "bright_frac", "clip_frac", "blockiness", "chroma_spread",
        "shift_mean", "shift_max", "static_frac", "temp_energy", "dup_frac"]


def load_all():
    recs = []
    for fn in ("desc_indego.jsonl", "desc_ood.jsonl"):
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            continue
        for line in open(p):
            recs.append(json.loads(line))
    return [r for r in recs if r["ok"]]


def grp(recs, s):
    return [r for r in recs if r["set"] == s]


def cohens_d(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    v = ((n1 - 1) * a.var() + (n2 - 1) * b.var()) / max(n1 + n2 - 2, 1)
    return float((a.mean() - b.mean()) / np.sqrt(v + 1e-12))


def summarize(vals):
    v = np.asarray(vals, float)
    return {"n": len(v), "mean": round(float(v.mean()), 4),
            "median": round(float(np.median(v)), 4),
            "p10": round(float(np.percentile(v, 10)), 4),
            "p90": round(float(np.percentile(v, 90)), 4)}


def diagnosis(recs):
    ind = grp(recs, "indego")
    ood = grp(recs, "ood1_source") or grp(recs, "ood1_transcode")
    ego = grp(recs, "ego10k_sample")
    table = []
    for k in KEYS:
        a = [r["desc"][k] for r in ind]
        b = [r["desc"][k] for r in ood]
        row = {"descriptor": k, "indego": summarize(a), "ood60_src": summarize(b),
               "cohens_d_ind_vs_oodsrc": round(cohens_d(a, b), 3)}
        if ego:
            c = [r["desc"][k] for r in ego]
            row["ego10k_sample"] = summarize(c)
            row["cohens_d_ind_vs_ego10k"] = round(cohens_d(a, c), 3)
        table.append(row)
    table.sort(key=lambda r: -abs(r["cohens_d_ind_vs_oodsrc"]))
    # codec mix
    def mix(rs):
        from collections import Counter
        return dict(Counter(r["desc"]["codec"] for r in rs))
    out = {"n_indego": len(ind), "n_ood": len(ood), "n_ego10k": len(ego),
           "codec_mix": {"indego": mix(ind), "ood_src": mix(ood),
                         "ego10k": mix(ego) if ego else {}},
           "ranked_table": table,
           "cmd": "eval/exp_regimes/analyze_regimes.py --part diagnosis"}
    json.dump(out, open(os.path.join(RES, "diagnosis.json"), "w"), indent=1)
    return out


def eg2_clusters():
    """KMeans (k=6, seed 0) on IndEgo EG2 window means; assign OOD-60."""
    from sklearn.cluster import KMeans
    di = np.load(f"/workspace/jadal_pod_bundle/caches/frames10k/segments.npz",
                 allow_pickle=True)
    F = di["F"].astype(np.float64)
    Mi = F.mean(axis=1)
    nrm = Mi / (np.linalg.norm(Mi, axis=1, keepdims=True) + 1e-12)
    km = KMeans(n_clusters=6, random_state=0, n_init=10).fit(nrm)
    do = np.load(f"/workspace/jadal_pod_bundle/caches/ensemble/ood_frames.npz",
                 allow_pickle=True)
    Fo = do["F"].astype(np.float64)
    Mo = Fo.mean(axis=1)
    nrmo = Mo / (np.linalg.norm(Mo, axis=1, keepdims=True) + 1e-12)
    co = km.predict(nrmo)
    ci = km.labels_
    from collections import Counter
    hi = Counter(ci.tolist())
    ho = Counter(co.tolist())
    # chi-square distance between cluster histograms
    pi = np.array([hi.get(k, 0) for k in range(6)]) / len(ci)
    po = np.array([ho.get(k, 0) for k in range(6)]) / len(co)
    js = float(0.5 * np.sum((pi - po) ** 2 / (pi + po + 1e-12)))
    out = {"k": 6, "indego_hist": pi.round(3).tolist(),
           "ood60_hist": po.round(3).tolist(),
           "sym_chi2": round(js, 4),
           "cmd": "KMeans(k=6,seed=0) on L2-normalised EG2 window means"}
    json.dump(out, open(os.path.join(RES, "eg2_clusters.json"), "w"), indent=1)
    return out


# ---------- E4.02 ----------
FEAT26 = ["temp_b0", "temp_b1", "temp_b2", "temp_b3", "frozen_frac", "flicker_idx",
          "radial_slope", "hf_ratio", "dct_energy", "pca_resid", "pca_top1_share",
          "luma_mean", "luma_p05", "luma_p50", "luma_p95", "dark_frac", "bright_frac",
          "motion_energy", "global_shift", "skin_frac", "illum_r", "illum_g",
          "illum_b", "rppg_pulse", "dup_hash_mean", "zncc_self"]


def load_feat_matrices():
    feat = json.load(open(f"/workspace/jadal_pod_bundle/caches/gpu/features.json"))["windows"]
    di = np.load(f"/workspace/jadal_pod_bundle/caches/frames10k/segments.npz",
                 allow_pickle=True)
    ids = [str(x) for x in di["ids"]]
    F = di["F"].astype(np.float64)
    Eg = F.mean(axis=1)
    Eg = Eg / (np.linalg.norm(Eg, axis=1, keepdims=True) + 1e-12)
    clips = {str(c) for c in di["clips"]}
    return feat, ids, Eg, {str(c): str(x) for c, x in zip(di["clips"], di["clips"])}, clips


def loco_auroc(X, y, groups, C=0.01):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    proba = np.zeros(len(y))
    for g in sorted(set(groups)):
        te = np.array(groups) == g
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
        m.fit(sc.transform(X[tr]), y[tr])
        proba[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return float(roc_auc_score(y, proba)), proba


def regime_vs_global(recs):
    import warnings
    warnings.filterwarnings("ignore")
    feat, ids, Eg, _, _ = load_feat_matrices()
    dmap = {r["window_id"]: r["desc"] for r in grp(recs, "indego")}
    # align: gold ids g000.. map to EG2 ids order (both length 150, same order assumed; verify by label match)
    m = json.load(open(f"/workspace/jadal_pod_bundle/labels/manifest_gold.json"))
    items = m["items"]
    assert len(items) == 150 and len(ids) == 150
    y = np.array([it["label"] for it in items])
    di = np.load(f"/workspace/jadal_pod_bundle/caches/frames10k/segments.npz",
                 allow_pickle=True)
    assert (np.array(di["labels"]).astype(int) == y).all(), "EG2/gold order mismatch"
    X26 = np.array([[feat[it["id"]]["full"][k] for k in FEAT26] for it in items])
    groups = [it["clip"] for it in items]
    results = {}
    splitters = {"sharpness": "sharp_lapvar", "lighting": "luma_mean", "motion": "shift_mean"}
    for view, X in (("feat26", X26), ("eg2", Eg)):
        g_auc, g_proba = loco_auroc(X, y, groups)
        results[view] = {"global_loco_auroc": round(g_auc, 4), "regimes": {}}
        for rname, dkey in splitters.items():
            vals = np.array([dmap[it["id"]][dkey] for it in items])
            t1, t2 = np.percentile(vals, [100 / 3, 200 / 3])
            for ri, name in enumerate(["lo", "mid", "hi"]):
                sel = (vals <= t1) if ri == 0 else ((vals <= t2) if ri == 1 else (vals > t2))
                yy = y[sel]
                if yy.min() == yy.max():
                    results[view]["regimes"][f"{rname}_{name}"] = {
                        "n": int(sel.sum()), "skip": "single-class regime"}
                    continue
                XX, gg = X[sel], np.array(groups)[sel]
                if len(set(gg)) < 2:
                    results[view]["regimes"][f"{rname}_{name}"] = {
                        "n": int(sel.sum()), "skip": "single-clip regime"}
                    continue
                r_auc, r_proba = loco_auroc(XX, yy, gg)
                # global model's proba restricted to regime
                from sklearn.metrics import roc_auc_score
                g_on_r = float(roc_auc_score(yy, g_proba[sel]))
                results[view]["regimes"][f"{rname}_{name}"] = {
                    "n": int(sel.sum()), "n_bad": int((yy == 0).sum()),
                    "regime_model_loco_auroc": round(r_auc, 4),
                    "global_model_on_regime_auroc": round(g_on_r, 4),
                    "delta": round(r_auc - g_on_r, 4)}
    results["cmd"] = "nested LOCO: regime models trained only on train-clip windows of that regime"
    json.dump(results, open(os.path.join(RES, "regime_vs_global.json"), "w"), indent=1)
    return results


# ---------- E4.03 ----------
def gap_control(recs):
    t = grp(recs, "ood1_transcode")
    s = grp(recs, "ood1_source")
    tm = {r["window_id"]: r["desc"] for r in t}
    sm = {r["window_id"]: r["desc"] for r in s}
    common = sorted(set(tm) & set(sm))
    gap = []
    for k in KEYS:
        a = np.array([tm[i][k] for i in common])
        b = np.array([sm[i][k] for i in common])
        gap.append({"descriptor": k,
                    "transcode_median": round(float(np.median(a)), 4),
                    "source_median": round(float(np.median(b)), 4),
                    "median_ratio_src_over_tc": round(float(np.median(b) / (np.median(a) + 1e-12)), 3)})
    # frozen ensemble baseline score distribution vs labels
    ens = json.load(open(f"/workspace/jadal_pod_bundle/caches/ensemble/ood_scores.json"))
    lab = json.load(open(f"/workspace/jadal_pod_bundle/labels/ood_labels.json"))["latest"]
    ys, ps = [], []
    for sc in ens["scores"]:
        v = lab.get(sc["id"], {}).get("verdict")
        if v in ("good", "bad"):
            ys.append(1 if v == "good" else 0)
            ps.append(sc["usable_prob"])
    from sklearn.metrics import roc_auc_score
    ys = np.array(ys)
    ps = np.array(ps)
    base_auc = float(roc_auc_score(ys, ps))
    # per-view collapse: correlate p_pixel vs descriptors on source windows
    pmap = {sc["id"]: sc for sc in ens["scores"]}
    corr = {}
    for k in ["shift_mean", "sharp_lapvar", "noise_sigma", "luma_mean", "temp_energy"]:
        xs = np.array([sm[i][k] for i in common if i in pmap])
        pp = np.array([pmap[i]["p_pixel"] for i in common if i in pmap])
        corr[k] = round(float(np.corrcoef(xs, pp)[0, 1]), 3)
    out = {"n_paired": len(common),
           "transcode_vs_source": sorted(gap, key=lambda r: -abs(np.log10(abs(r["median_ratio_src_over_tc"]) + 1e-9))),
           "frozen_ensemble_on_ood": {
               "n": len(ys), "n_bad": int((ys == 0).sum()),
               "auroc": round(base_auc, 4),
               "mean_p_good": round(float(ps[ys == 1].mean()), 4),
               "mean_p_bad": round(float(ps[ys == 0].mean()), 4),
               "note": "baseline ood_scores.json used 300 pixel frames/20 EG2 frames per 10 s window"},
           "p_pixel_vs_source_descriptor_corr": corr,
           "cmd": "paired transcode-vs-source descriptors + frozen ensemble score audit"}
    json.dump(out, open(os.path.join(RES, "gap_control.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    recs = load_all()
    print(f"loaded {len(recs)} descriptor records", flush=True)
    if a.part in ("all", "diagnosis"):
        d = diagnosis(recs)
        print("top separators:", [(r["descriptor"], r["cohens_d_ind_vs_oodsrc"]) for r in d["ranked_table"][:8]], flush=True)
    if a.part in ("all", "eg2"):
        e = eg2_clusters()
        print("eg2 cluster hists:", e["indego_hist"], e["ood60_hist"], "sym_chi2:", e["sym_chi2"], flush=True)
    if a.part in ("all", "regimes"):
        r = regime_vs_global(recs)
        print(json.dumps(r, indent=1)[:2000], flush=True)
    if a.part in ("all", "gap"):
        g = gap_control(recs)
        print(json.dumps(g["frozen_ensemble_on_ood"], indent=1), flush=True)
        print("p_pixel corr:", g["p_pixel_vs_source_descriptor_corr"], flush=True)
