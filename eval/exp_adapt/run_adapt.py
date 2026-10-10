#!/usr/bin/env python3
"""Z1 OOD error reduction: stable features, unsupervised alignment, per-clip
normalisation, few-shot curve, self-training.

Outputs eval/results/exp_adapt/Z1.01..03.json + a figure. CPU only, fixed seeds.

Run: OMP_NUM_THREADS=2 python3 eval/exp_adapt/run_adapt.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.exp_adapt import adaptlib as A  # noqa: E402


def _wr(name, rep):
    os.makedirs(A.OUT, exist_ok=True)
    rep["git_sha"] = A.git_sha()
    with open(os.path.join(A.OUT, name), "w") as f:
        json.dump(rep, f, indent=1)
    print("wrote", os.path.join(A.OUT, name), flush=True)


def _score(y, s):
    lo, hi = A.boot_ci(y, s)
    return {"auroc": A.auroc(y, s), "ci95": [lo, hi], "balacc": A.balacc(y, s)}


PROTO = {
    "indego": "leave-one-clip-out (LOCO) over 7 IndEgo clips; clip-bootstrap CI (2000)",
    "ood": "head fit on IndEgo (all 7 clips); OOD-60 = 60 human-labelled Egocentric-10K "
           "windows (51 good / 9 bad), ~1 window/factory; window-bootstrap CI (optimistic, "
           "exploratory)",
}
LABEL_SRC = {
    "indego": "human (labels/manifest_gold.json, single labeller, n=150: 100 good / 50 bad)",
    "ood": "human (labels/ood_labels.json, single labeller, n=60: 51 good / 9 bad, n_bad=9 exploratory)",
}


def _row(method, indego_oof, y_in, ood_scores, y_ood, note):
    r = {"method": method,
         "indego_auroc": A.auroc(y_in, indego_oof),
         "indego_balacc": A.balacc(y_in, indego_oof),
         "ood_auroc": A.auroc(y_ood, ood_scores),
         "ood_balacc": A.balacc(y_ood, ood_scores),
         "note": note}
    lo, hi = A.boot_ci(y_ood, ood_scores)
    r["ood_ci95"] = [lo, hi]
    return r


# ------------------------------------------------------------------ Z1.01
def z1_01():
    g = A.load_gold(); o = A.load_ood_pixel()
    y_in, clips = g["y"], g["clips"]; y_ood = o["y"]
    fs = A.load_frozen_ood_scores()
    lab = {i: int(v) for i, v in zip(o["ids"], y_ood)}
    ids = [i for i in fs if i in lab]
    frozen_auroc = A.auroc([lab[i] for i in ids], [fs[i] for i in ids])

    rows = []
    # frozen PoC prior (no fitting)
    rows.append({"method": "FROZEN ensemble_v1 (PoC prior)", "indego_auroc": 0.85,
                 "ood_auroc": frozen_auroc,
                 "ood_ci95": list(A.boot_ci([lab[i] for i in ids], [fs[i] for i in ids])),
                 "note": "no fitting; IndEgo in-domain ensemble baseline ~0.85 (nested LOCO)"})
    arms = [
        ("raw · all-26", None, "raw", "none"),
        ("raw · stable-8 (sharpness/light)", "stable", "raw", "none"),
        ("raw · sharpness-3", "sharp", "raw", "none"),
    ]
    for name, sub, method, ood_norm in arms:
        fi = None if sub is None else (A.idx_of(A.STABLE) if sub == "stable" else A.idx_of(A.SHARPNESS))
        for view in ("10fps", "native"):
            Xood = o["X_10fps"] if view == "10fps" else o["X_native"]
            oof, sc = A.loco_transfer(g["X26"], y_in, clips, Xood, method=method,
                                      feat_idx=fi, ood_norm=ood_norm)
            rows.append(_row(f"{name} · OOD={view}", oof, y_in, sc, y_ood,
                             "head trained on IndEgo (C by inner LOCO); zero-shot transfer"))
    # EG2 embedding head (768-d window means)
    ge = A.load_gold_eg2(); oe = A.load_ood_eg2()
    oof, sc = A.loco_transfer(ge["means"], y_in, clips, oe["means"], method="raw")
    rows.append(_row("raw · EG2-768 window means · OOD=native", oof, y_in, sc, y_ood,
                     "EG2 embedding transfer; EG2 fps-independent but domain-shifted"))

    rep = {"experiment": "Z1.01", "title": "Stable-feature selection for zero-shot OOD transfer",
           "n_indego": int(len(y_in)), "n_ood": int(len(y_ood)),
           "n_bad_ood": int((y_ood == 0).sum()),
           "label_source": LABEL_SRC, "protocol": PROTO,
           "features": {"all26": A.FEATURE_NAMES, "stable8": A.STABLE, "sharpness3": A.SHARPNESS},
           "frozen_poc_auroc": frozen_auroc,
           "caveat": "OOD n_bad=9, ~1 window/factory -> exploratory; window-bootstrap CI optimistic. "
                     "Sign-stable here means IndEgo sign == OOD-10fps sign (E5.03), not a quality measure.",
           "device": "CPU (numpy/sklearn); GPU not used",
           "rows": rows,
           "command": "OMP_NUM_THREADS=2 python3 eval/exp_adapt/run_adapt.py"}
    _wr("Z1.01.json", rep)
    for r in rows:
        print(f"  [Z1.01] {r['method']:44s} IndEgo {r['indego_auroc']:.3f} "
              f"OOD {r['ood_auroc']:.3f} {[round(c,3) for c in r['ood_ci95']]}")
    return rep


# ------------------------------------------------------------------ Z1.02
def z1_02():
    g = A.load_gold(); o = A.load_ood_pixel()
    y_in, clips, y_ood = g["y"], g["clips"], o["y"]
    Xood10, XoodN = o["X_10fps"], o["X_native"]
    rows = []
    for view, Xood in (("10fps", Xood10), ("native", XoodN)):
        for method in ("raw", "domain_std", "domain_whiten", "coral_s2t", "coral_t2s"):
            oof, sc = A.loco_transfer(g["X26"], y_in, clips, Xood, method=method)
            rows.append(_row(f"{method} · OOD={view}", oof, y_in, sc, y_ood,
                             "unsupervised alignment (no OOD labels); CORAL target = unlabelled OOD-60"))
        # per-clip normalisation on IndEgo, OOD block normalised (global)
        for mode in ("z", "rank"):
            oof, sc = A.per_clip_transfer(g["X26"], y_in, clips, Xood, mode=mode, ood_norm="global")
            rows.append(_row(f"perclip_{mode} · OOD={view} (global target norm)", oof, y_in, sc, y_ood,
                             "IndEgo z/rank-normed within each clip (pallet/pallet2 shift removed)"))
        # stable-8 + domain_whiten (best-of-both), OOD global target norm
        for method in ("domain_whiten", "coral_s2t"):
            oof, sc = A.loco_transfer(g["X26"], y_in, clips, Xood, method=method,
                                      feat_idx=A.idx_of(A.STABLE))
            rows.append(_row(f"stable8+{method} · OOD={view}", oof, y_in, sc, y_ood,
                             "sign-stable features + unsupervised alignment"))
    # Ego10K unlabelled corpus (1546 windows) as the CORAL target on the 26 feats
    corpus = A.load_e8_corpus()
    n_fac = len(set(corpus["factory"].tolist()))
    Xc = corpus["X"]
    for method in ("coral_s2t", "coral_t2s"):
        # train in IndEgo space; corpus is the unlabelled target distribution
        fs, ft = A._fit_transformers(method, g["X26"], Xc)
        Xtr = fs(g["X26"])
        c = A.pick_C(Xtr, y_in, clips)
        clf = A.fit_head(Xtr, y_in, c)
        oof = np.full(len(y_in), np.nan)
        for held in sorted(set(clips.tolist())):
            te = np.where(clips == held)[0]; tr = np.where(clips != held)[0]
            f2, _ = A._fit_transformers(method, g["X26"][tr], Xc)
            cc = A.pick_C(f2(g["X26"][tr]), y_in[tr], clips[tr])
            cl = A.fit_head(f2(g["X26"][tr]), y_in[tr], cc)
            oof[te] = cl.predict_proba(f2(g["X26"][te]))[:, 1]
        for view, Xood in (("10fps", Xood10), ("native", XoodN)):
            sc = clf.predict_proba(ft(Xood))[:, 1]
            rows.append(_row(f"{method}→Ego10K-corpus · OOD={view}", oof, y_in, sc, y_ood,
                             f"unlabelled Ego10K corpus (n=1546, {n_fac} factories, native 30fps) "
                             "as target; fps confound vs IndEgo ~10fps"))
    rep = {"experiment": "Z1.02",
           "title": "Unsupervised alignment + per-clip normalisation for zero-shot OOD transfer",
           "n_indego": int(len(y_in)), "n_ood": int(len(y_ood)), "n_bad_ood": int((y_ood == 0).sum()),
           "label_source": LABEL_SRC, "protocol": PROTO,
           "alignment": {"methods": ["raw", "domain_std", "domain_whiten (ZCA per domain)",
                                     "coral_s2t", "coral_t2s"],
                         "no_labels_used": True,
                         "target_distributions": ["unlabelled OOD-60 features (60)",
                                                  "unlabelled Ego10K E8 corpus (1546)"]},
           "caveat": "OOD n_bad=9 exploratory. Per-clip norm is fitted on IndEgo clips; OOD has "
                     "~1 window/factory so per-factory norm is degenerate -> OOD block normalised "
                     "globally. Ego10K corpus is native ~30fps vs IndEgo ~10fps (fps confound).",
           "device": "CPU (numpy/sklearn); GPU not used",
           "rows": rows,
           "command": "OMP_NUM_THREADS=2 python3 eval/exp_adapt/run_adapt.py"}
    _wr("Z1.02.json", rep)
    for r in rows:
        print(f"  [Z1.02] {r['method']:46s} IndEgo {r['indego_auroc']:.3f} "
              f"OOD {r['ood_auroc']:.3f} {[round(c,3) for c in r['ood_ci95']]}")
    return rep


# ------------------------------------------------------------------ Z1.03
def z1_03():
    g = A.load_gold(); o = A.load_ood_pixel()
    y_in, clips, y_ood = g["y"], g["clips"], o["y"]
    fac = o["factory"]
    XoodN = o["X_native"]

    # ---- few-shot calibration curve (post-hoc: fits on OOD labels, grouped) ----
    curve = []
    for k in (5, 12, 24):
        for view, Xood in (("native", XoodN), ("10fps", o["X_10fps"])):
            res = A.grouped_cv_fewshot(Xood, y_ood, fac, k=k, n_repeats=60, seed=0)
            aurocs = np.asarray([r["auroc"] for r in res])
            curve.append({"k": k, "view": view, "repeats": int(len(res)),
                          "auroc_mean": float(aurocs.mean()) if len(aurocs) else None,
                          "auroc_std": float(aurocs.std()) if len(aurocs) else None,
                          "frac_gt_0.6": float((aurocs > 0.6).mean()) if len(aurocs) else None,
                          "n_train_mean": float(np.mean([r["n_train"] for r in res])) if res else None,
                          "min_n_bad_train": int(min(r["n_bad_train"] for r in res)) if res else None,
                          "class_weight": "balanced"})
            print(f"  [Z1.03] few-shot k={k:2d} view={view:7s} "
                  f"AUROC {curve[-1]['auroc_mean']} std {curve[-1]['auroc_std']} "
                  f"repeats {len(res)} min_n_bad_train {curve[-1]['min_n_bad_train']}")

    # ---- self-training on the unlabelled Ego10K corpus (1546, native) ----
    corpus = A.load_e8_corpus()
    Xc = corpus["X"]
    st_rows = []
    for method in ("raw", "domain_std"):
        fs, ft = A._fit_transformers(method, g["X26"], Xc)
        Xtr = fs(g["X26"]); Xc_t = ft(Xc)
        c = A.pick_C(Xtr, y_in, clips)
        base = A.fit_head(Xtr, y_in, c)
        p = base.predict_proba(Xc_t)[:, 1]
        for tau in (0.8, 0.9):
            hi = (p >= tau) | (p <= 1 - tau)
            n_hi = int(hi.sum()); n_pos = int((p[hi] >= 0.5).sum())
            if n_hi < 20 or len(set((p[hi] >= 0.5).astype(int).tolist())) < 2:
                st_rows.append({"method": method, "tau": tau, "n_pseudo": n_hi,
                                "pseudo_pos": n_pos, "skipped": True,
                                "note": "too few / single-class pseudo-labels"})
                continue
            # pseudo-labels are binary; weight 0.5 so they cannot dominate IndEgo
            Xaug = np.vstack([Xtr, Xc_t[hi]]); yaug = np.concatenate([y_in, (p[hi] >= 0.5).astype(int)])
            w = np.concatenate([np.ones(len(y_in)), np.full(n_hi, 0.5)])
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=c, max_iter=5000, class_weight="balanced"))
            clf.fit(Xaug, yaug, **{"logisticregression__sample_weight": w})
            for view, Xood in (("native", XoodN), ("10fps", o["X_10fps"])):
                sc = clf.predict_proba(ft(Xood))[:, 1]
                a = A.auroc(y_ood, sc)
                st_rows.append({"method": f"self-train·{method}", "tau": tau, "view": view,
                                "n_pseudo": n_hi, "pseudo_pos": n_pos,
                                "ood_auroc": a, "ood_balacc": A.balacc(y_ood, sc),
                                "ood_ci95": list(A.boot_ci(y_ood, sc)),
                                "source_head_ood_auroc": A.auroc(
                                    y_ood, base.predict_proba(ft(Xood))[:, 1]),
                                "note": "pseudo-labels from Ego10K corpus (no human labels); "
                                        "weighted 0.5; fps confound IndEgo~10 vs corpus~30"})
                print(f"  [Z1.03] self-train {method} tau={tau} view={view} "
                      f"n_pseudo={n_hi} AUROC {a:.3f} (base {st_rows[-1]['source_head_ood_auroc']:.3f})")

    # ---- recommendation: best honest zero-shot + best few-shot ----
    z101 = json.load(open(os.path.join(A.OUT, "Z1.01.json")))
    z102 = json.load(open(os.path.join(A.OUT, "Z1.02.json")))
    best_zero = max(z101["rows"] + z102["rows"], key=lambda r: r["ood_auroc"])
    best_k = max([c for c in curve if c["auroc_mean"] is not None], key=lambda c: c["auroc_mean"])
    rep = {"experiment": "Z1.03",
           "title": "Few-shot calibration curve + honest self-training",
           "n_indego": int(len(y_in)), "n_ood": int(len(y_ood)), "n_bad_ood": int((y_ood == 0).sum()),
           "label_source": LABEL_SRC, "protocol": PROTO,
           "fewshot_protocol": "factory-grouped: whole factories are either train or test; "
                               "k = sampled labelled windows; C by inner grouped CV; test = all "
                               "remaining factories; POST-HOC (fits on OOD labels)",
           "self_train_protocol": "head trained on IndEgo; pseudo-label high-confidence unlabelled "
                                  "Ego10K corpus windows (E8, n=1546); add once with weight 0.5; "
                                  "re-evaluate on OOD-60 (no OOD labels used for training)",
           "fewshot_curve": curve, "self_training": st_rows,
           "recommendation": {
               "best_zero_shot": {"method": best_zero["method"], "ood_auroc": best_zero["ood_auroc"],
                                  "ood_ci95": best_zero["ood_ci95"]},
               "best_fewshot": {"k": best_k["k"], "view": best_k["view"],
                                "ood_auroc_mean": best_k["auroc_mean"],
                                "ood_auroc_std": best_k["auroc_std"]}},
           "caveat": "n_bad=9, ~1 window/factory: few-shot and OOD estimates are exploratory/optimistic. "
                     "The few-shot curve is POST-HOC; self-training uses no OOD labels.",
           "device": "CPU (numpy/sklearn); GPU not used",
           "command": "OMP_NUM_THREADS=2 python3 eval/exp_adapt/run_adapt.py"}
    _wr("Z1.03.json", rep)
    return rep


def make_figure(z101, z102, z103):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print("figure skipped:", e)
        return
    os.makedirs(A.FIGS, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    rows = z101["rows"] + z102["rows"]
    rows = sorted(rows, key=lambda r: r["ood_auroc"])
    names = [r["method"].replace(" · ", "\n") for r in rows]
    axes[0].barh(range(len(rows)), [r["ood_auroc"] for r in rows], color="#357",
                 xerr=[[r["ood_auroc"] - r["ood_ci95"][0] for r in rows],
                       [r["ood_ci95"][1] - r["ood_auroc"] for r in rows]], capsize=2)
    axes[0].axvline(0.5, color="k", ls="--", lw=1)
    axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels(names, fontsize=6)
    axes[0].set_xlabel("OOD-60 AUROC"); axes[0].set_title("zero-shot OOD AUROC (95% window boot)")
    curve = z103["fewshot_curve"]
    for view in ("native", "10fps"):
        cs = sorted([c for c in curve if c["view"] == view], key=lambda c: c["k"])
        axes[1].errorbar([c["k"] for c in cs], [c["auroc_mean"] for c in cs],
                         yerr=[c["auroc_std"] for c in cs], marker="o", capsize=3, label=view)
    axes[1].axhline(0.5, color="k", ls="--", lw=1)
    axes[1].set_xlabel("k labelled OOD windows (factory-grouped)")
    axes[1].set_ylabel("OOD AUROC (mean±std)"); axes[1].legend()
    axes[1].set_title("few-shot curve (POST-HOC)")
    st = [r for r in z103["self_training"] if not r.get("skipped") and r.get("view") == "native"]
    if st:
        axes[2].bar(range(len(st)), [r["ood_auroc"] for r in st], color="#a53")
        axes[2].axhline(0.5, color="k", ls="--", lw=1)
        axes[2].set_xticks(range(len(st)))
        axes[2].set_xticklabels([f"{r['method']}\nτ={r['tau']}" for r in st], fontsize=6)
        axes[2].set_ylabel("OOD AUROC"); axes[2].set_title("self-training vs chance")
    fig.tight_layout()
    p = os.path.join(A.FIGS, "z1_ood_error_reduction.png")
    fig.savefig(p, dpi=140); print("figure ->", p)


def main():
    print("=== Z1.01 ===", flush=True); r1 = z1_01()
    print("=== Z1.02 ===", flush=True); r2 = z1_02()
    print("=== Z1.03 ===", flush=True); r3 = z1_03()
    make_figure(r1, r2, r3)


if __name__ == "__main__":
    main()
