"""M6 ablation: does the inverse-PCA residual (and illumination / rPPG rows)
carry signal? LOCO logistic AUC per feature group + clip-bootstrap CIs.

CPU-only. Reads eval/results/gpu/features.json. Writes eval/results/gpu/ablation.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FEATS = os.path.join(HERE, "..", "results", "gpu", "features.json")
OUT = os.path.join(HERE, "..", "results", "gpu", "ablation.json")
GOLD = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"

GROUPS = {
    "full_26": None,
    "no_pca_resid": ["pca_resid"],
    "pca_only": ["pca_resid", "pca_top1_share"],
    "illum_only": ["illum_r", "illum_g", "illum_b"],
    "rppg_only": ["rppg_pulse"],
    "skin_only": ["skin_frac"],
    "temporal_only": ["temp_b0", "temp_b1", "temp_b2", "temp_b3", "frozen_frac",
                      "flicker_idx", "motion_energy", "global_shift"],
    "no_illum_no_rppg": ["illum_r", "illum_g", "illum_b", "rppg_pulse"],
}


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    feats = json.load(open(FEATS))["windows"]
    gold = {it["id"]: it for it in json.load(open(GOLD))["items"]}
    names = ["temp_b0", "temp_b1", "temp_b2", "temp_b3", "frozen_frac", "flicker_idx",
             "radial_slope", "hf_ratio", "dct_energy", "pca_resid", "pca_top1_share",
             "luma_mean", "luma_p05", "luma_p50", "luma_p95", "dark_frac", "bright_frac",
             "motion_energy", "global_shift", "skin_frac", "illum_r", "illum_g",
             "illum_b", "rppg_pulse", "dup_hash_mean", "zncc_self"]
    ids = [i for i in gold if i in feats and "full" in feats[i]]
    X = np.stack([[feats[i]["full"][k] for k in names] for i in ids])
    y = np.array([gold[i]["label"] for i in ids])
    clips = sorted({gold[i]["clip"] for i in ids})
    clip_of = np.array([gold[i]["clip"] for i in ids])
    rng = np.random.default_rng(1)
    rows = {}
    for gname, drop_or_keep in GROUPS.items():
        if gname == "full_26":
            cols = list(range(len(names)))
        elif gname.startswith("no_"):
            cols = [j for j, n in enumerate(names) if n not in drop_or_keep]
        else:
            cols = [j for j, n in enumerate(names) if n in drop_or_keep]
        Xs = (X[:, cols] - X[:, cols].mean(0)) / (X[:, cols].std(0) + 1e-12)
        p = np.zeros(len(ids))
        for c in clips:
            te = clip_of == c
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xs[~te], y[~te])
            p[te] = clf.predict_proba(Xs[te])[:, 1]
        auc = float(roc_auc_score(y, p))
        boots = []
        for _ in range(200):
            cs = rng.choice(clips, size=len(clips), replace=True)
            sel = np.concatenate([np.where(clip_of == c)[0] for c in cs])
            if len(set(y[sel])) == 2:
                boots.append(roc_auc_score(y[sel], p[sel]))
        rows[gname] = {"auc": auc, "ci95": [float(np.percentile(boots, 2.5)),
                                            float(np.percentile(boots, 97.5))],
                       "n_feats": len(cols)}
    res = {"device": "RTX 4070 Laptop, NOT a Pi, NOT a datacenter GPU",
           "regenerate": "/home/parshu/envs/jadal/bin/python eval/gpu/ablate.py",
           "n": len(ids), "protocol": "leave-one-clip-out, clip-bootstrap CI (200)",
           "verdict": ("inverse-PCA residual and illumination rows are ablated honestly; "
                       "no hyperspectral claim (metamerism limit, see rgb_to_spectral.md)."),
           "rows": rows}
    json.dump(res, open(OUT, "w"), indent=1)
    for k, v in rows.items():
        print(f"{k:16s} auc={v['auc']:.3f} ci=[{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
