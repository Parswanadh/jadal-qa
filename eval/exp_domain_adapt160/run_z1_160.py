#!/usr/bin/env python3
"""Z1 (domain-adaptation) re-run on the 160 OOD labels.

Same protocol as `eval/exp_adapt/run_adapt.py` (Z1.01-Z1.03, PR #121), but the
out-of-domain set is the merged 160 human-labelled Egocentric-10K windows
(138 good / 22 bad, 47 factories) instead of the old 60 (51 good / 9 bad), and
the primary CI is the *factory-grouped* bootstrap (honest) with the window
bootstrap kept for continuity.

The 10 fps-effective view is used for all 160 windows: the q2 (100) source
transcodes are natively 10 fps, so a native-30fps column is only defined for the
old q1=60 windows and is not recomputed here (that is stated, not hidden).

Outputs eval/results/exp_domain_adapt160/Z1.01_160.json .. Z1.03_160.json.
CPU only, fixed seeds.
Run: OMP_NUM_THREADS=2 python3 eval/exp_domain_adapt160/run_z1_160.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from eval.exp_domain_adapt160 import adaptlib as A  # noqa: E402

OUT = os.path.join(REPO, "eval", "results", "exp_domain_adapt160")
BUNDLE = A.BUNDLE
PIXEL160 = os.environ.get(
    "JADAL_PIXEL160",
    os.path.join(os.environ.get("JADAL_WT", "wt"), 'm-fewshot160/eval/results/exp_fewshot/inputs/ood160_pixel.npz'))
N_BOOT = 2000
SEED = 0
DEV = "CPU (numpy/sklearn); GPU not used"


def load_labels160():
    d = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))
    w = d["windows"]
    return {"ids": [x["segment_id"] for x in w],
            "y": np.asarray([1 if x["verdict"] == "good" else 0 for x in w], int),
            "factory": np.asarray([x["factory"] for x in w]),
            "batch": np.asarray([x["batch"] for x in w])}


def load_ood160_pixel():
    z = np.load(PIXEL160, allow_pickle=True)
    lab = load_labels160()
    order = {i: k for k, i in enumerate(z["ids"])}
    ids = lab["ids"]
    assert all(i in order for i in ids), "labels160 ids missing from ood160_pixel"
    return {"ids": ids, "y": lab["y"], "factory": lab["factory"],
            "names": [str(x) for x in z["names"]],
            "X_10fps": np.asarray([z["X10"][order[i]] for i in ids], float)}


def group_ci(y, s, groups, n_boot=N_BOOT, seed=SEED):
    return list(A.clip_boot_ci(y, s, groups, n_boot=n_boot, seed=seed))


def window_ci(y, s, n_boot=N_BOOT, seed=SEED):
    return list(A.boot_ci(y, s, n_boot=n_boot, seed=seed))


def row(method, indego_oof, y_in, ood_scores, y_ood, groups, note):
    return {"method": method,
            "indego_auroc": A.auroc(y_in, indego_oof),
            "indego_balacc": A.balacc(y_in, indego_oof),
            "ood160_auroc": A.auroc(y_ood, ood_scores),
            "ood160_balacc": A.balacc(y_ood, ood_scores),
            "ood160_ci95_factory_grouped": group_ci(y_ood, ood_scores, groups),
            "ood160_ci95_window_optimistic": window_ci(y_ood, ood_scores),
            "note": note}


def main():
    os.makedirs(OUT, exist_ok=True)
    g = A.load_gold()
    o = load_ood160_pixel()
    y_in, clips = g["y"], g["clips"]
    y_ood, fac = o["y"], o["factory"]
    Xood = o["X_10fps"]
    n_good, n_bad = int((y_ood == 1).sum()), int((y_ood == 0).sum())
    n_fac = len(set(fac.tolist()))
    sha = A.git_sha()

    head = {
        "n_ood": int(len(y_ood)), "n_good_ood": n_good, "n_bad_ood": n_bad,
        "n_factories": n_fac,
        "n_indego": int(len(y_in)),
        "label_source": {
            "indego": "human (labels/manifest_gold.json, single labeller, n=150: 100 good / 50 bad)",
            "ood": ("human (labels/ood160_labels.json, single labeller, n=160: 138 good / 22 bad, "
                    "47 factories; merge q1=60 + q2=100)"),
        },
        "protocol": {
            "indego": "leave-one-clip-out (LOCO) over 7 IndEgo clips; clip-bootstrap CI (2000)",
            "ood": ("head fit on IndEgo (all 7 clips); OOD-160 evaluation only. "
                    "FACTORY-GROUPED bootstrap CI (honest, n_boot=2000, seed=0); "
                    "window bootstrap kept as the optimistic comparison. 10 fps-effective view "
                    "for all 160 (q2 transcodes are natively 10 fps)."),
        },
        "supersedes": {"branch": "m/domain-adapt", "pr": 121, "n_ood": 60},
        "device": DEV, "git_sha": sha,
        "command": "OMP_NUM_THREADS=2 python3 eval/exp_domain_adapt160/run_z1_160.py",
    }

    # ---------------- Z1.01 stable-feature selection ----------------
    fs = {}
    for fn in ("ood_scores.json", "ood_scores2.json"):
        d = json.load(open(os.path.join(BUNDLE, "caches", "ensemble", fn)))
        for x in d["scores"]:
            if "error" not in x:
                fs[x["id"]] = float(x["usable_prob"])
    frozen = np.asarray([fs[i] for i in o["ids"]], float)

    rows = [{"method": "FROZEN ensemble_v1 (PoC prior)", "indego_auroc": 0.85,
             "ood160_auroc": A.auroc(y_ood, frozen),
             "ood160_ci95_factory_grouped": group_ci(y_ood, frozen, fac),
             "ood160_ci95_window_optimistic": window_ci(y_ood, frozen),
             "note": "no fitting; IndEgo in-domain ensemble baseline ~0.85 (nested LOCO)"}]
    arms = [
        ("raw · all-26", None, "raw"),
        ("raw · stable-8 (sharpness/light)", "stable", "raw"),
        ("raw · sharpness-3", "sharp", "raw"),
    ]
    for name, sub, method in arms:
        fi = None if sub is None else (A.idx_of(A.STABLE) if sub == "stable"
                                       else A.idx_of(A.SHARPNESS))
        oof, sc = A.loco_transfer(g["X26"], y_in, clips, Xood, method=method, feat_idx=fi)
        rows.append(row(f"{name} · OOD=10fps", oof, y_in, sc, y_ood, fac,
                        "head trained on IndEgo (C by inner LOCO); zero-shot transfer"))
    z101 = {**head, "experiment": "Z1.01_160",
            "title": "Stable-feature selection for zero-shot OOD transfer (160)",
            "features": {"all26": A.FEATURE_NAMES, "stable8": A.STABLE,
                         "sharpness3": A.SHARPNESS},
            "eg2_768_note": ("EG2-768 window means are NOT recomputed for the q2=100 windows "
                             "(needs a GPU EG2 embed); the old n=60 row (0.546) is unchanged "
                             "and omitted from the 160 table."),
            "caveat": ("OOD n_bad=22, 47 factories (~1 window/factory); factory-grouped CI is "
                       "the honest one, window bootstrap is optimistic. Sign-stable means IndEgo "
                       "sign == OOD sign (E5.03), not a quality measure."),
            "rows": rows}
    json.dump(z101, open(os.path.join(OUT, "Z1.01_160.json"), "w"), indent=1)

    # ---------------- Z1.02 alignment + per-clip norm ----------------
    rows2 = []
    for method in ("raw", "domain_std", "domain_whiten", "coral_s2t", "coral_t2s"):
        oof, sc = A.loco_transfer(g["X26"], y_in, clips, Xood, method=method)
        rows2.append(row(f"{method} · OOD=10fps", oof, y_in, sc, y_ood, fac,
                         "unsupervised alignment (no OOD labels); CORAL target = unlabelled OOD-160"))
    for mode in ("z", "rank"):
        oof, sc = A.per_clip_transfer(g["X26"], y_in, clips, Xood, mode=mode, ood_norm="global")
        rows2.append(row(f"perclip_{mode} · OOD=10fps (global target norm)", oof, y_in, sc,
                         y_ood, fac,
                         "IndEgo z/rank-normed within each clip (pallet/pallet2 shift removed); "
                         "OOD block normalised globally (~1 window/factory)"))
    for method in ("domain_whiten", "coral_s2t"):
        oof, sc = A.loco_transfer(g["X26"], y_in, clips, Xood, method=method,
                                  feat_idx=A.idx_of(A.STABLE))
        rows2.append(row(f"stable8+{method} · OOD=10fps", oof, y_in, sc, y_ood, fac,
                         "sign-stable features + unsupervised alignment"))
    z102 = {**head, "experiment": "Z1.02_160",
            "title": "Unsupervised alignment + per-clip normalisation (160)",
            "alignment": {"methods": ["raw", "domain_std", "domain_whiten (ZCA per domain)",
                                      "coral_s2t", "coral_t2s"],
                          "no_labels_used": True,
                          "target_distribution": "unlabelled OOD-160 features (160)"},
            "caveat": ("OOD n_bad=22 exploratory. Per-clip norm fitted on IndEgo clips; OOD has "
                       "~1 window/factory so per-factory norm is degenerate -> OOD block "
                       "normalised globally. Native-30fps view not recomputed for q2."),
            "rows": rows2}
    json.dump(z102, open(os.path.join(OUT, "Z1.02_160.json"), "w"), indent=1)

    # ---------------- Z1.03 few-shot curve + self-training ----------------
    curve = []
    for k in (5, 12, 24, 48, 80):
        res = A.grouped_cv_fewshot(Xood, y_ood, fac, k=k, n_repeats=60, seed=0)
        aurocs = np.asarray([r["auroc"] for r in res])
        curve.append({"k": k, "view": "10fps", "repeats": int(len(res)),
                      "auroc_mean": float(aurocs.mean()) if len(aurocs) else None,
                      "auroc_std": float(aurocs.std()) if len(aurocs) else None,
                      "frac_gt_0.6": float((aurocs > 0.6).mean()) if len(aurocs) else None,
                      "n_train_mean": float(np.mean([r["n_train"] for r in res])) if res else None,
                      "min_n_bad_train": int(min(r["n_bad_train"] for r in res)) if res else None,
                      "class_weight": "balanced"})
        print(f"  [Z1.03] few-shot k={k:2d} AUROC {curve[-1]['auroc_mean']} "
              f"std {curve[-1]['auroc_std']} repeats {len(res)} "
              f"min_n_bad_train {curve[-1]['min_n_bad_train']}")

    # self-training on the unlabelled Ego10K corpus (1546) -> evaluate on 160
    corpus = A.load_e8_corpus()
    Xc = corpus["X"]
    st_rows = []
    for method in ("raw", "domain_std"):
        fs_t, ft = A._fit_transformers(method, g["X26"], Xc)
        Xtr = fs_t(g["X26"])
        Xc_t = ft(Xc)
        c = A.pick_C(Xtr, y_in, clips)
        base = A.fit_head(Xtr, y_in, c)
        p = base.predict_proba(Xc_t)[:, 1]
        for tau in (0.8, 0.9):
            hi = (p >= tau) | (p <= 1 - tau)
            n_hi = int(hi.sum())
            n_pos = int((p[hi] >= 0.5).sum())
            if n_hi < 20 or len(set((p[hi] >= 0.5).astype(int).tolist())) < 2:
                st_rows.append({"method": method, "tau": tau, "n_pseudo": n_hi,
                                "pseudo_pos": n_pos, "skipped": True,
                                "note": "too few / single-class pseudo-labels"})
                continue
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            Xaug = np.vstack([Xtr, Xc_t[hi]])
            yaug = np.concatenate([y_in, (p[hi] >= 0.5).astype(int)])
            w = np.concatenate([np.ones(len(y_in)), np.full(n_hi, 0.5)])
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=c, max_iter=5000, class_weight="balanced"))
            clf.fit(Xaug, yaug, **{"logisticregression__sample_weight": w})
            sc = clf.predict_proba(ft(Xood))[:, 1]
            st_rows.append({"method": f"self-train·{method}", "tau": tau, "view": "10fps",
                            "n_pseudo": n_hi, "pseudo_pos": n_pos,
                            "ood160_auroc": A.auroc(y_ood, sc),
                            "ood160_balacc": A.balacc(y_ood, sc),
                            "ood160_ci95_factory_grouped": group_ci(y_ood, sc, fac),
                            "source_head_ood160_auroc": A.auroc(
                                y_ood, base.predict_proba(ft(Xood))[:, 1]),
                            "note": "pseudo-labels from Ego10K corpus (no human labels); "
                                    "weighted 0.5"})
            print(f"  [Z1.03] self-train {method} tau={tau} n_pseudo={n_hi} "
                  f"AUROC {st_rows[-1]['ood160_auroc']:.3f} "
                  f"(base {st_rows[-1]['source_head_ood160_auroc']:.3f})")

    best_zero = max(z101["rows"] + z102["rows"], key=lambda r: r["ood160_auroc"])
    best_k = max([c for c in curve if c["auroc_mean"] is not None],
                 key=lambda c: c["auroc_mean"])
    z103 = {**head, "experiment": "Z1.03_160",
            "title": "Few-shot calibration curve + honest self-training (160)",
            "fewshot_protocol": ("factory-grouped: whole factories are either train or test; "
                                 "k = sampled labelled windows; C by inner grouped CV; test = all "
                                 "remaining factories; POST-HOC (fits on OOD labels)"),
            "self_train_protocol": ("head trained on IndEgo; pseudo-label high-confidence "
                                    "unlabelled Ego10K corpus windows (E8, n=1546); add once "
                                    "with weight 0.5; evaluate on OOD-160 (no OOD labels used)"),
            "fewshot_curve": curve, "self_training": st_rows,
            "recommendation": {
                "best_zero_shot": {"method": best_zero["method"],
                                   "ood160_auroc": best_zero["ood160_auroc"],
                                   "ood160_ci95_factory_grouped":
                                       best_zero["ood160_ci95_factory_grouped"]},
                "best_fewshot": {"k": best_k["k"], "view": best_k["view"],
                                 "ood160_auroc_mean": best_k["auroc_mean"],
                                 "ood160_auroc_std": best_k["auroc_std"]}},
            "caveat": ("n_bad=22, 47 factories (~1 window/factory): few-shot and OOD estimates "
                       "are exploratory. The few-shot curve is POST-HOC; self-training uses no "
                       "OOD labels.")}
    json.dump(z103, open(os.path.join(OUT, "Z1.03_160.json"), "w"), indent=1)

    # ---------------- stdout ----------------
    print(f"n_ood={len(y_ood)} good={n_good} bad={n_bad} factories={n_fac} (old 60, n_bad=9)")
    for r in z101["rows"] + z102["rows"]:
        print(f"  {r['method']:46s} IndEgo {r['indego_auroc']:.3f} "
              f"OOD160 {r['ood160_auroc']:.3f} {r['ood160_ci95_factory_grouped']}")
    print("few-shot:", [(c["k"], round(c["auroc_mean"], 3)) for c in curve])
    print("self-train:", [(r.get("method"), r.get("tau"), round(r.get("ood160_auroc", 0), 3))
                          for r in st_rows if not r.get("skipped")])
    print("best zero-shot:", best_zero["method"], best_zero["ood160_auroc"])
    print("best few-shot:", best_k["k"], best_k["auroc_mean"])


if __name__ == "__main__":
    main()
