#!/usr/bin/env python3
"""O2: E5.04 final + few-shot adaptation curve + E5.03 refresh (160 OOD labels).

O2.01  E5.04 final: frozen ensemble_v1 usable_prob on the 160 human-labelled OOD
       windows -> AUROC / balanced accuracy with a FACTORY-GROUPED bootstrap CI
       (the honest cluster bootstrap; window bootstrap is optimistic). Verdict
       adopt / maybe / drop (was inconclusive: labels2 not delivered).
O2.02  Few-shot adaptation curve: train a small logistic / calibrator on
       k in {10, 20, 40, 80, 160} labelled windows (factory-grouped CV) ->
       OOD AUROC / balanced accuracy / error vs k. The "error per human label"
       curve. Recipes: frozen usable_prob Platt calibrator; two-view logistic
       on [p_eg2, p_pixel]; 26 per-pixel-feature logistic.
O2.03  E5.03 refresh: per-feature signed AUROC against the 160 labels vs the
       old 60 -> updated stable / flip / chance table + transferable subset.

All CV is factory-grouped (StratifiedGroupKFold / leave-one-factory-out): a
factory never appears in both train and test.

Run:  OMP_NUM_THREADS=2 python3 eval/exp_fewshot/run_o2.py {o2_01,o2_02,o2_03,all}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from eval.exp_fewshot import o2lib as L  # noqa: E402

CURVE_KS = [10, 20, 40, 80, 160]
REPEATS = 20


def _write(name, rep):
    os.makedirs(L.OUT, exist_ok=True)
    rep["git_sha"] = L.git_sha()
    p = os.path.join(L.OUT, name)
    with open(p, "w") as f:
        json.dump(rep, f, indent=1)
    print("wrote", p, flush=True)


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


# ----------------------------------------------------------------- O2.01
def o2_01():
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    lab = L.load_labels160()
    scores = L.load_frozen_scores()
    ids = [i for i in lab["ids"] if i in scores]
    order = {sid: k for k, sid in enumerate(lab["ids"])}
    y = np.asarray([lab["y"][order[i]] for i in ids], int)
    fac = np.asarray([lab["factory"][order[i]] for i in ids])
    batch = np.asarray([lab["batch"][order[i]] for i in ids])
    p = np.asarray([scores[i]["usable_prob"] for i in ids], float)
    pe = np.asarray([scores[i]["p_eg2"] for i in ids], float)
    pp = np.asarray([scores[i]["p_pixel"] for i in ids], float)

    auroc = float(roc_auc_score(y, p))
    ba = float(balanced_accuracy_score(y, p > 0.5))
    acc = float((y == (p > 0.5)).mean())
    g_m, g_lo, g_hi = L.group_boot_ci(y, p, fac, metric="auroc")
    b_m, b_lo, b_hi = L.group_boot_ci(y, p, fac, metric="balacc")
    w_m, w_lo, w_hi = L.boot_ci(y, p)

    # per-batch (descriptive)
    per_batch = {}
    for b in sorted(set(batch.tolist())):
        sel = batch == b
        if len(np.unique(y[sel])) >= 2:
            per_batch[b] = {"n": int(sel.sum()), "n_good": int(y[sel].sum()),
                            "n_bad": int((1 - y[sel]).sum()),
                            "auroc": float(roc_auc_score(y[sel], p[sel])),
                            "balacc": float(balanced_accuracy_score(y[sel], p[sel] > 0.5))}

    # verdict: the factory-grouped 95% CI must exclude 0.5 (honest cluster CI).
    # adopt  : CI lower > 0.5 AND AUROC >= 0.60
    # maybe  : CI lower > 0.5 (significant but weak)
    # drop   : CI includes 0.5 -> not a reliable OOD scorer
    if g_lo > 0.5 and auroc >= 0.60:
        verdict = "adopt"
    elif g_lo > 0.5:
        verdict = "maybe"
    else:
        verdict = "drop"

    rep = {
        "experiment": "O2.01 (updates E5.04)", "n": int(len(ids)),
        "n_good": int(y.sum()), "n_bad": int((1 - y).sum()),
        "n_factories": len(set(fac.tolist())),
        "label_source": "human, single labeller (labels/ood160_labels.json; "
                        "merge of ood_labels.json q1 60 + ood_labels2.json q2 100; "
                        "n_bad=22 now, was 9)",
        "protocol": "frozen ensemble_v1 usable_prob vs 160 human OOD labels; no tuning "
                    "(scorer frozen, config canonical sha asserted by E5.04). CIs: "
                    "FACTORY-GROUPED cluster bootstrap (primary) + window bootstrap (optimistic)",
        "device": "CPU (frozen scores from the pod bundle; no GPU scoring)",
        "cost": {"n_windows": int(len(ids)), "new_gpu_scoring": 0},
        "ood_auroc": auroc,
        "auroc_ci95_factory_grouped": [g_lo, g_hi], "auroc_group_boot_mean": g_m,
        "auroc_ci95_window_bootstrap_optimistic": [w_lo, w_hi],
        "balacc_at_0.5": ba, "balacc_ci95_factory_grouped": [b_lo, b_hi],
        "acc_at_0.5": acc,
        "mean_prob_good": float(p[y == 1].mean()), "mean_prob_bad": float(p[y == 0].mean()),
        "p_eg2_auroc": float(roc_auc_score(y, pe)), "p_pixel_auroc": float(roc_auc_score(y, pp)),
        "per_batch_descriptive": per_batch,
        "verdict": verdict,
        "verdict_rule": "adopt: factory-grouped CI lower > 0.5 AND AUROC >= 0.60; maybe: grouped-CI "
                        "lower > 0.5 (significant but weak); drop: grouped CI includes 0.5",
        "prior_E5.04": {"verdict": "inconclusive", "n": 0,
                        "reason": "ood_labels2.json was not delivered; descriptive only"},
        "bundle_reference": "docs/ood_eval_ensemble_v1_160.txt (window-bootstrap 0.566 "
                            "[0.433,0.693]); O2.01 recomputes with factory-grouped CI",
        "bytes_sha_guard": "frozen config sha 4da1068a (recorded by scorer; E5.04 asserts canonical match)",
        "command": "OMP_NUM_THREADS=2 python3 eval/exp_fewshot/run_o2.py o2_01",
    }
    print(f"  O2.01 n={rep['n']} good={rep['n_good']} bad={rep['n_bad']} "
          f"AUROC={auroc:.3f} groupedCI=[{g_lo:.3f},{g_hi:.3f}] BA={ba:.3f} -> {verdict}")
    _write("O2.01.json", rep)
    return rep


# ----------------------------------------------------------------- O2.02
def _arm_curve(name, X, y, fac, ks, repeats=REPEATS, single_feature=False):
    """Learning curve for one recipe: subsampling points + a full-160 grouped-CV point."""
    from sklearn.metrics import roc_auc_score

    curve = []
    sub_ks = [k for k in ks if k < len(y)]
    if sub_ks:
        curve += L.grouped_subsample_curve(X, y, fac, sub_ks, repeats=repeats, seed=0)
    if len(y) in ks:
        oof, folds = L.grouped_cv_oof(X, y, fac, n_splits=5, seed=0)
        m, lo, hi = L.group_boot_ci(y, oof, fac, metric="auroc")
        bm, blo, bhi = L.group_boot_ci(y, oof, fac, metric="balacc")
        nb = [f["n_bad_train"] for f in folds if not f.get("skipped")]
        auroc = float(roc_auc_score(y, oof))
        ba = L.bal_acc(y, oof)
        curve.append({
            "k_target": int(len(y)), "k_train_mean": float(np.mean([f["n_train"] for f in folds if not f.get("skipped")])),
            "repeats": 1, "dropped": 0, "auroc_mean": auroc, "auroc_std": 0.0,
            "balacc_mean": ba, "balacc_std": 0.0, "error_mean": float(1.0 - ba),
            "min_n_bad_train": int(min(nb)) if nb else None,
            "protocol": "all-%d grouped-CV OOF (StratifiedGroupKFold 5, factory groups)" % len(y),
            "auroc_ci95_factory_grouped": [lo, hi], "balacc_ci95_factory_grouped": [blo, bhi],
        })
    for c in curve:
        c.setdefault("protocol", "train ~k (factory-grouped) / test complement (factory-disjoint)")
    return {"recipe": name, "curve": curve}


def o2_02():
    from sklearn.metrics import roc_auc_score

    lab = L.load_labels160()
    scores = L.load_frozen_scores()
    pix = L.load_ood160_pixel()
    pidx = {str(i): k for k, i in enumerate(pix["ids"])}
    ids = [i for i in lab["ids"] if i in scores and i in pidx]
    order = {sid: k for k, sid in enumerate(lab["ids"])}
    y = np.asarray([lab["y"][order[i]] for i in ids], int)
    fac = np.asarray([lab["factory"][order[i]] for i in ids])
    P = np.asarray([[scores[i]["p_eg2"], scores[i]["p_pixel"]] for i in ids], float)
    p_use = np.asarray([scores[i]["usable_prob"] for i in ids], float)
    X_platt = _logit(p_use).reshape(-1, 1)
    X_2v = P
    X_pix = np.asarray([pix["X10"][pidx[i]] for i in ids], float)
    print(f"O2.02 n={len(ids)} good={int(y.sum())} bad={int((1-y).sum())} "
          f"factories={len(set(fac.tolist()))}")

    arms = {}
    # 0-label baseline: frozen score, fixed 0.5 threshold (no training at all)
    auroc0 = float(roc_auc_score(y, p_use))
    ba0 = L.bal_acc(y, p_use)
    arms["frozen_usable_prob_0labels"] = {
        "recipe": "frozen ensemble_v1 usable_prob, threshold fixed at 0.5 (no labels)",
        "curve": [{"k_target": 0, "k_train_mean": 0, "repeats": 1, "dropped": 0,
                   "auroc_mean": auroc0, "auroc_std": 0.0, "balacc_mean": ba0,
                   "balacc_std": 0.0, "error_mean": float(1.0 - ba0),
                   "min_n_bad_train": None, "protocol": "no training; frozen threshold"}],
    }
    arms["frozen_platt_calibrator"] = _arm_curve("frozen_platt_calibrator", X_platt, y, fac, CURVE_KS)
    arms["frozen_platt_calibrator"]["recipe"] = ("Platt calibrator (1 feature = logit(usable_prob)) "
                                                 "trained on k labels")
    arms["two_view_logistic"] = _arm_curve("two_view_logistic", X_2v, y, fac, CURVE_KS)
    arms["two_view_logistic"]["recipe"] = "logistic on frozen view scores [p_eg2, p_pixel]"
    arms["pixel26_logistic"] = _arm_curve("pixel26_logistic", X_pix, y, fac, CURVE_KS)
    arms["pixel26_logistic"]["recipe"] = "logistic on 26 per-pixel features (10 fps-effective)"

    # pick the best label-efficient recipe at the low end (k<=40) by mean error
    def mean_err_low(arm):
        vals = [c["error_mean"] for c in arm["curve"] if c.get("error_mean") is not None and c["k_target"] <= 40]
        return float(np.mean(vals)) if vals else 9.9
    ranked = sorted(arms.items(), key=lambda kv: mean_err_low(kv[1]))
    best_name = ranked[0][0]
    best = arms[best_name]
    full = [c for c in best["curve"] if c["k_target"] == 160]
    best_error_full = full[0]["error_mean"] if full else None
    best_auroc_full = full[0]["auroc_mean"] if full else None

    rep = {
        "experiment": "O2.02", "n": int(len(ids)), "n_good": int(y.sum()),
        "n_bad": int((1 - y).sum()), "n_factories": len(set(fac.tolist())),
        "label_source": "human, single labeller (labels/ood160_labels.json; 138 good / 22 bad)",
        "protocol": "factory-grouped CV only: StratifiedGroupKFold(5, groups=factory) for the "
                    "all-160 point; repeated (20x) factory-grouped subsampling (train ~k windows / "
                    "test the factory-disjoint complement) for k in {10,20,40,80}; inner "
                    "leave-one-factory-out on train picks C in {0.01,0.1,1,10}; class_weight=balanced",
        "ci_kind": "factory-grouped cluster bootstrap (2000) for the all-160 point",
        "device": "CPU (numpy/sklearn)",
        "cost": {"n_windows": int(len(ids)), "recipes": 4, "repeats_per_k": REPEATS},
        "ks": CURVE_KS,
        "arms": arms,
        "recommended_recipe": best_name,
        "recommended_measured": {"k": 160, "ood_auroc": best_auroc_full,
                                  "error": best_error_full,
                                  "error_per_label_note": "error = 1 - balanced accuracy @0.5"},
        "command": "OMP_NUM_THREADS=2 python3 eval/exp_fewshot/run_o2.py o2_02",
    }
    for nm, a in arms.items():
        pts = {c["k_target"]: (round(c["auroc_mean"], 3) if c["auroc_mean"] is not None else None,
                               round(c["error_mean"], 3) if c["error_mean"] is not None else None)
               for c in a["curve"]}
        print(f"  {nm:28s} k->(AUROC,err): {pts}")
    print(f"  recommended (low-k mean error): {best_name} full AUROC={best_auroc_full} err={best_error_full}")
    _write("O2.02.json", rep)
    return rep


# ----------------------------------------------------------------- O2.03
def o2_03():
    from sklearn.metrics import roc_auc_score

    gold = L.load_gold_indego()
    pix = L.load_ood160_pixel()
    lab = L.load_labels160()
    scores = L.load_frozen_scores()
    pidx = {str(i): k for k, i in enumerate(pix["ids"])}
    ids = [i for i in lab["ids"] if i in pidx]
    order = {sid: k for k, sid in enumerate(lab["ids"])}
    y160 = np.asarray([lab["y"][order[i]] for i in ids], int)
    fac160 = np.asarray([lab["factory"][order[i]] for i in ids])
    X160 = np.asarray([pix["X10"][pidx[i]] for i in ids], float)

    # old 60-window column (frozen batch-1 X_10fps + ood_labels.json)
    y60map = L.load_labels60()
    ids60 = [i for i in ids if i in y60map]
    y60 = np.asarray([y60map[i] for i in ids60], int)
    X60 = np.asarray([pix["X10"][pidx[i]] for i in ids60], float)

    rows, counts = [], {"stable": 0, "flip": 0, "chance": 0}
    old_counts = {"stable": 0, "flip": 0, "chance": 0}
    changed = []
    for j, name in enumerate(L.FEATURE_NAMES):
        # IndEgo (clip-bootstrap)
        ag = float(roc_auc_score(gold["y"], gold["X26"][:, j]))
        gm, glo, ghi = L.group_boot_ci(gold["y"], gold["X26"][:, j], gold["clips"], metric="auroc")
        # OOD 160 (factory-grouped bootstrap)
        a16 = float(roc_auc_score(y160, X160[:, j]))
        m16, lo16, hi16 = L.group_boot_ci(y160, X160[:, j], fac160, metric="auroc")
        # OOD 60 (old, window bootstrap)
        a60 = float(roc_auc_score(y60, X60[:, j]))
        m60, lo60, hi60 = L.boot_ci(y60, X60[:, j])
        cls = L.classify_transfer(ag, a16)
        cls60 = L.classify_transfer(ag, a60)
        counts[cls] += 1
        old_counts[cls60] += 1
        if cls != cls60:
            changed.append({"feature": name, "class_60": cls60, "class_160": cls,
                            "ood60_auroc": a60, "ood160_auroc": a16})
        rows.append({
            "feature": name,
            "indego_auroc": ag, "indego_ci95": [glo, ghi],
            "ood160_auroc": a16, "ood160_ci95_grouped": [lo16, hi16],
            "ood60_auroc": a60, "ood60_ci95_window": [lo60, hi60],
            "class": cls, "class_60": cls60, "class_changed_vs60": bool(cls != cls60),
            "indego_sign": "high-usable" if ag >= 0.5 else "low-usable",
            "ood160_sign": "high-usable" if a16 >= 0.5 else "low-usable",
        })
        print(f"  {name:16s} IndEgo {ag:.3f} | OOD160 {a16:.3f} [{lo16:.2f},{hi16:.2f}] "
              f"| OOD60 {a60:.3f} | {cls} (was {cls60})")

    rows.sort(key=lambda r: -abs(r["ood160_auroc"] - 0.5))
    # transferable subset: class stable AND informative in at least one domain (>=0.10)
    transferable = [r["feature"] for r in rows
                    if r["class"] == "stable"
                    and max(abs(r["indego_auroc"] - 0.5), abs(r["ood160_auroc"] - 0.5)) >= 0.10]

    rep = {
        "experiment": "O2.03 (refreshes E5.03)",
        "n": int(len(gold["y"]) + len(y160)),
        "n_indego": int(len(gold["y"])), "n_good_indego": int(gold["y"].sum()),
        "n_ood160": int(len(y160)), "n_good_ood160": int(y160.sum()),
        "n_bad_ood160": int((1 - y160).sum()), "n_ood60_old": int(len(y60)),
        "n_bad_ood60_old": int((1 - y60).sum()),
        "label_source": "human, single labeller each: IndEgo manifest_gold (150) + Egocentric-10K "
                        "ood160_labels.json (160, merge of 60 + 100); n_bad 22 now vs 9 before",
        "protocol": "signed AUROC per feature (label 1 = usable/good); IndEgo CI = clip-bootstrap "
                    "(7 clips); OOD160 CI = FACTORY-GROUPED bootstrap; OOD60 CI = window bootstrap "
                    "(optimistic). Same classification rule as E5.03 (classify_transfer).",
        "regime": "both decoded 160x90 with the frozen recipe; IndEgo native ~10fps; OOD "
                  "10 fps-effective (batch-1 every 3rd frame of the 30fps transcode, batch-2 native 10fps)",
        "device": "CPU (numpy)",
        "cost": {"features": 26, "indego_windows": 150, "ood160_windows": int(len(y160))},
        "classification_rule": "chance iff BOTH |AUROC-0.5| < 0.05; otherwise stable if both AUROCs "
                               "are on the same side of 0.5, flip if opposite",
        "counts_160": counts, "counts_60_old": old_counts,
        "class_changes_vs60": changed,
        "transferable_subset": transferable,
        "table_sorted_by_ood160_signal": rows,
        "command": "OMP_NUM_THREADS=2 python3 eval/exp_fewshot/run_o2.py o2_03",
    }
    print(f"  counts(160)={counts}  counts(60 old)={old_counts}  changed={len(changed)}")
    print(f"  transferable subset: {transferable}")
    _write("O2.03.json", rep)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["o2_01", "o2_02", "o2_03", "all"])
    a = ap.parse_args()
    cmds = ["o2_01", "o2_02", "o2_03"] if a.cmd == "all" else [a.cmd]
    for c in cmds:
        print(f"=== {c} ===", flush=True)
        globals()[c]()


if __name__ == "__main__":
    main()
