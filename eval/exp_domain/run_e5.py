#!/usr/bin/env python3
"""E5.00-E5.04 domain-adaptation experiments (CPU only; fixed seeds).

E5.00  reproduce the frozen ensemble OOD prior (usable_prob AUROC on the 60
       human-labelled OOD windows); the E5 report cites this number.
E5.01  heads trained on the 10k AI-labelled FACTORY frames (EG2 embeddings),
       pooled per IndEgo window -> LOCO AUROC vs human gold. Model-vs-model
       agreement, NOT ground-truth accuracy.
E5.02  few-shot on the 60 human OOD windows, factory-grouped CV; recipes:
       26 per-pixel feats, EG2 window means, two-view probability average;
       learning curve = how many labels before it beats chance.
E5.03  per-feature signed AUROC IndEgo (gold, 10 fps) vs OOD (native 30 fps
       and 10 fps effective) -- the what-transfers table.
E5.04  frozen ensemble scores on the 160 OOD windows; if labels2 is absent the
       confirmatory eval is logged inconclusive and only descriptive stats are
       emitted (no accuracy).

Run:  OMP_NUM_THREADS=1 python3 eval/exp_domain/run_e5.py {e5_00,e5_01,e5_02,e5_03,e5_04,all}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from eval.exp_domain import e5lib as L  # noqa: E402

OOD_LABELS2 = os.path.join(L.BUNDLE, "labels", "ood_labels2.json")
CONFIG_PATH = os.path.join(L.REPO, "configs", "ensemble_v1.json")
# Frozen sha recorded by the scorer in $JADAL_BUNDLE/caches/ensemble/ood_scores.json
# (raw bytes of configs/ensemble_v1.json on main / PR #79).
CONFIG_SHA256_EXPECTED = "4da1068a78d5d15021f4e167d6aa24fd2e0bb08aa88f83c83c2f167cc086fe31"
# Same content with the pod path/provenance keys stripped (remap_paths.py rewrites
# embed_model_dir / embed_command / pixel_command, which do not affect scoring).
CONFIG_CANONICAL_SHA256_EXPECTED = "f43654d784e2690a2c8e5a316e9f90a3e2da4615ca6bbd93f625c2f5b29d100d"


def _write(name, rep):
    os.makedirs(L.OUT, exist_ok=True)
    p = os.path.join(L.OUT, name)
    rep["git_sha"] = L.git_sha()
    rep["command"] = rep.get("command", "")
    with open(p, "w") as f:
        json.dump(rep, f, indent=1)
    print("wrote", p, flush=True)


# ----------------------------------------------------------------- E5.00
def e5_00():
    """Reproduce the frozen-ensemble OOD prior quoted in the E5 report.

    This is the only number in the E5 report that was previously cited from a
    report not present on this branch (`docs/reports/ensemble_views.md`, main /
    PR #79). Recomputed here from the frozen batch-1 scores so it traces to a
    committed results JSON.
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    scores = L.load_frozen_ood_scores()
    ref = L.load_ood()
    lab = {sid: int(v) for sid, v in zip(ref["ids"], ref["y"])}
    ids = [s["id"] for s in scores if s["id"] in lab]
    y = np.asarray([lab[i] for i in ids], int)
    p = np.asarray([next(s["usable_prob"] for s in scores if s["id"] == i) for i in ids], float)
    m, lo, hi = L.boot_ci(y, p, roc_auc_score)
    rep = {"experiment": "E5.00", "n": int(len(ids)), "n_good": int(y.sum()),
           "n_bad": int((1 - y).sum()),
           "label_source": "human (ood_labels.json, single labeller; OOD n_bad=9 exploratory)",
           "protocol": "frozen ensemble_v1 usable_prob vs human OOD labels; "
                       "window bootstrap CI (optimistic: ~1 window/factory)",
           "device": "CPU (numpy); frozen scores from the pod bundle",
           "cost": {"n_scored": int(len(ids))},
           "ood_auroc": float(roc_auc_score(y, p)), "boot_mean": m, "ci95": [lo, hi],
           "balacc_at_0.5": float(balanced_accuracy_score(y, p > 0.5)),
           "prior_reference": {"report": "docs/reports/ensemble_views.md (main / PR #79, NOT on "
                                          "this branch)",
                               "report_ci95": [0.210, 0.647],
                               "scores_in_bundle": "$JADAL_BUNDLE/caches/ensemble/ood_scores.json"},
           "command": "OMP_NUM_THREADS=1 python3 eval/exp_domain/run_e5.py e5_00"}
    print(f"  frozen ensemble OOD AUROC {rep['ood_auroc']:.4f} "
          f"ci95={[round(c, 3) for c in rep['ci95']]} (report quoted 0.210, 0.647)")
    _write("E5.00.json", rep)


# ----------------------------------------------------------------- E5.01
def e5_01():
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    d10 = L.load_10k_frames()
    E10, hand, work = d10["E"], d10["hand"], d10["work"]
    targets = {"active_labor": work.astype(int), "hands_ok": (hand >= 1).astype(int)}
    frame_cv, heads = {}, {}
    for name, y in targets.items():
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        oof = np.zeros(len(y))
        for tr, te in skf.split(E10, y):
            clf = L.make_clf(1.0, balanced=False)
            clf.fit(E10[tr], y[tr])
            oof[te] = clf.predict_proba(E10[te])[:, 1]
        fm, flo, fhi = L.boot_ci(y, oof, roc_auc_score)
        frame_cv[name] = {"n": int(len(y)), "pos_frac": float(y.mean()),
                          "oof_auroc": float(roc_auc_score(y, oof)),
                          "boot_mean": fm, "ci95": [flo, fhi],
                          "label_source": "AI (gemini-2.5-flash prompt; NOT human)",
                          "protocol": "stratified 5-fold, C=1 fixed, StandardScaler+logistic"}
        clf = L.make_clf(1.0, balanced=False)
        clf.fit(E10, y)
        heads[name] = {"mean": clf[0].mean_.tolist(), "scale": clf[0].scale_.tolist(),
                       "coef": clf[1].coef_.reshape(-1).tolist(),
                       "intercept": float(clf[1].intercept_[0])}
    print("frame-level CV (AI labels):", {k: round(v["oof_auroc"], 4) for k, v in frame_cv.items()})

    seg = L.load_gold_segments()
    F = seg["F"]
    mask = ~np.isnan(F[:, :, 0])
    Fn = np.nan_to_num(F)
    P = {}
    for name, h in heads.items():
        Xs = (Fn - np.asarray(h["mean"])) / np.asarray(h["scale"])
        P[name] = L.sigmoid(Xs @ np.asarray(h["coef"]) + h["intercept"])
    Pw = np.where(mask, P["active_labor"], np.nan)
    Ph = np.where(mask, P["hands_ok"], np.nan)
    y, clips = seg["y"], seg["clips"]

    arms = {}
    for nm, pp in (("work_prob_mean", Pw), ("hands_prob_mean", Ph),
                   ("both_prob_mean", (Pw + Ph) / 2.0)):
        s = np.nanmean(pp, axis=1)
        m, lo, hi = L.group_boot_ci(y, s, clips)
        arms[nm] = {"kind": "pooled frame-probability mean (no fitting)",
                    "oof_auroc": float(roc_auc_score(y, s)), "clip_boot_mean": m,
                    "ci95": [lo, hi], "balacc": float(balanced_accuracy_score(y, s > 0.5)),
                    "acc": float((y == (s > 0.5)).mean())}
    Fp = np.concatenate([L.pool_feats(Pw), L.pool_feats(Ph)], axis=1)
    oof = L.loco_cv(Fp, y, clips)
    m, lo, hi = L.group_boot_ci(y, oof, clips)
    arms["pooled_feats_logistic_loco"] = {
        "kind": "LEAVE-ONE-CLIP-OUT logistic on 12 pooled frame-head feats",
        "dim": 12, "oof_auroc": float(roc_auc_score(y, oof)), "clip_boot_mean": m,
        "ci95": [lo, hi], "balacc": float(balanced_accuracy_score(y, oof > 0.5)),
        "acc": float((y == (oof > 0.5)).mean())}
    for k, v in arms.items():
        print(f"  {k}: AUROC={v['oof_auroc']:.3f} ci95={[round(c,3) for c in v['ci95']]}")

    rep = {"experiment": "E5.01", "n": int(len(y)), "n_good": int(y.sum()),
           "n_bad": int((1 - y).sum()), "clips": sorted(set(clips.tolist())),
           "label_source": "head trained on AI labels (10k FACTORY frames, NOT human); "
                           "evaluated vs HUMAN IndEgo gold (single labeller)",
           "caveat": "the frame-level 0.911/0.957 is AI-vs-AI (model-vs-model), NOT ground-truth; "
                     "the window-level 0.789-0.826 IS scored against the human IndEgo gold labels "
                     "(single labeller) but the head itself was trained on AI labels",
           "protocol": "LOCO over 7 IndEgo clips; clip-bootstrap CI (2000); "
                       "frame heads C=1 fixed on 10k AI frames",
           "device": "CPU (numpy/sklearn); GPU not used",
           "cost": {"n_ai_frames": int(len(E10)), "n_indego_windows": int(len(y)),
                    "note": "reuses cached EG2 embeddings; no GPU encode"},
           "frame_level_cv_ai": frame_cv, "arms": arms,
           "baselines_cited": {"EG2 window-embedding logistic": 0.85,
                               "26 per-pixel feats logistic": 0.81,
                               "source": "experiments/EXPERIMENTS.md / nested_loco.json"},
           "command": "OMP_NUM_THREADS=1 python3 eval/exp_domain/run_e5.py e5_01"}
    _write("E5.01.json", rep)


# ----------------------------------------------------------------- E5.02
def e5_02():
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    ref = L.load_ood()
    pix = L.load_ood_pixel_selfdec()
    eg2 = L.load_ood_eg2_means()
    pidx = {str(i): k for k, i in enumerate(pix["ids"])}
    eidx = {str(i): k for k, i in enumerate(eg2["ids"])}
    ids = [i for i in ref["ids"] if i in pidx and i in eidx]
    ridx = {i: k for k, i in enumerate(ref["ids"])}
    y = np.asarray([ref["y"][ridx[i]] for i in ids], int)
    fac = np.asarray([ref["factory"][ridx[i]] for i in ids])
    Xpix = np.asarray([pix["X_native"][pidx[i]] for i in ids])
    Xeg2 = np.asarray([eg2["means"][eidx[i]] for i in ids], float)
    print(f"OOD n={len(ids)} good={int(y.sum())} bad={int((1-y).sum())} factories={len(set(fac.tolist()))}")

    recipes = {"pixel26_logistic": Xpix, "eg2_mean_logistic": Xeg2}
    oofs, folds_all = {}, {}
    for name, X in recipes.items():
        oof, folds = L.grouped_cv(X, y, fac, n_splits=5)
        oofs[name] = oof
        folds_all[name] = folds
    oofs["two_view_avg"] = (oofs["pixel26_logistic"] + oofs["eg2_mean_logistic"]) / 2.0
    folds_all["two_view_avg"] = folds_all["pixel26_logistic"]

    rep_recipes = {}
    for name, oof in oofs.items():
        m, lo, hi = L.boot_ci(y, oof, roc_auc_score)
        folds = folds_all.get(name, [])
        nb = [f["n_bad_train"] for f in folds if not f.get("skipped")]
        rep_recipes[name] = {
            "oof_auroc": float(roc_auc_score(y, oof)), "boot_mean": m, "ci95": [lo, hi],
            "balacc": float(balanced_accuracy_score(y, oof > 0.5)),
            "n_train_folds": len(folds),
            "smallest_n_bad_in_train_fold": int(min(nb)) if nb else None,
            "folds": folds}
        print(f"  {name}: AUROC={rep_recipes[name]['oof_auroc']:.3f} ci95={[round(c,3) for c in [lo,hi]]} "
              f"min n_bad train={rep_recipes[name]['smallest_n_bad_in_train_fold']}")

    # learning curve: how many labels to beat chance? grouped random subsampling.
    rng = np.random.RandomState(0)
    facs = sorted(set(fac.tolist()))
    curve = []
    for n_train in (12, 24, 36, 45):
        aucs, nbs, ntrs = [], [], []
        for _ in range(20):
            order = rng.permutation(facs)
            sel = []
            for f in order:
                sel.append(f)
                if sum(1 for i in range(len(ids)) if fac[i] in set(sel)) >= n_train:
                    break
            selset = set(sel)
            tr = np.asarray([i for i in range(len(ids)) if fac[i] in selset])
            te = np.asarray([i for i in range(len(ids)) if fac[i] not in selset])
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            clf = L.make_clf(1.0, balanced=True)
            clf.fit(Xpix[tr], y[tr])
            try:
                aucs.append(roc_auc_score(y[te], clf.predict_proba(Xpix[te])[:, 1]))
            except ValueError:
                pass
            nbs.append(int((y[tr] == 0).sum()))
            ntrs.append(int(len(tr)))
        aucs = np.asarray(aucs)
        curve.append({"n_train_target": n_train,
                      "n_train_mean": float(np.mean(ntrs)) if ntrs else None,
                      "repeats": int(len(aucs)),
                      "auroc_mean": float(aucs.mean()) if len(aucs) else None,
                      "auroc_std": float(aucs.std()) if len(aucs) else None,
                      "frac_above_0.6": float((aucs > 0.6).mean()) if len(aucs) else None,
                      "min_n_bad_train": int(min(nbs)) if nbs else None})
        print(f"  curve n_train~{n_train}: AUROC {curve[-1]['auroc_mean']} "
              f"(std {curve[-1]['auroc_std']}) frac>0.6={curve[-1]['frac_above_0.6']}")

    rep = {"experiment": "E5.02", "n": int(len(ids)), "n_good": int(y.sum()),
           "n_bad": int((1 - y).sum()), "n_factories": len(set(fac.tolist())),
           "label_source": "human (single labeller, ood_labels.json; exploratory, n_bad=9)",
           "protocol": "StratifiedGroupKFold(5, groups=factory); C in {0.01,0.1,1,10} "
                       "by inner grouped CV on train only; class_weight=balanced",
           "ci_kind": "window bootstrap (optimistic: ~1 window/factory)",
           "device": "CPU (numpy/sklearn)",
           "cost": {"n_ood_windows": int(len(ids)), "features": "26 per-pixel / 768 EG2 means"},
           "recipes": rep_recipes, "learning_curve": curve,
           "caveat": "n_bad=9 -> every estimate is unstable; smallest n_bad in any "
                     "training fold reported per recipe",
           "command": "OMP_NUM_THREADS=1 python3 eval/exp_domain/run_e5.py e5_02"}
    _write("E5.02.json", rep)


# ----------------------------------------------------------------- E5.03
def e5_03():
    from sklearn.metrics import roc_auc_score

    gold = L.load_gold()
    pix = L.load_ood_pixel_selfdec()
    yo = pix["y"]
    rows, n_stable, n_flip, n_chance = [], 0, 0, 0
    n_n30 = {"stable": 0, "flip": 0, "chance": 0}
    disagree_n30 = []
    for j, name in enumerate(L.FEATURE_NAMES):
        xg = gold["X26"][:, j]
        ag = float(roc_auc_score(gold["y"], xg))
        gm, glo, ghi = L.group_boot_ci(gold["y"], xg, gold["clips"])
        xn = pix["X_native"][:, j]
        an = float(roc_auc_score(yo, xn))
        nm, nlo, nhi = L.boot_ci(yo, xn, roc_auc_score)
        x10 = pix["X_10fps"][:, j]
        a10 = float(roc_auc_score(yo, x10))
        t10m, t10lo, t10hi = L.boot_ci(yo, x10, roc_auc_score)
        sg = "high-usable" if ag >= 0.5 else "low-usable"
        s10 = "high-usable" if a10 >= 0.5 else "low-usable"
        sn = "high-usable" if an >= 0.5 else "low-usable"
        # primary class uses the 10 fps-effective OOD column (matches IndEgo's ~10 fps);
        # class_native30 is the same rule applied to the native ~30 fps OOD column.
        cls = L.classify_transfer(ag, a10)
        cls_n30 = L.classify_transfer(ag, an)
        if cls == "chance":
            n_chance += 1
        elif cls == "stable":
            n_stable += 1
        else:
            n_flip += 1
        n_n30[cls_n30] += 1
        if cls_n30 != cls:
            disagree_n30.append(name)
        rows.append({"feature": name, "indego_auroc": ag, "indego_ci95": [glo, ghi],
                     "indego_sign": sg, "ood_native30_auroc": an, "ood_native30_ci95": [nlo, nhi],
                     "ood_native30_sign": sn, "ood_10fps_auroc": a10, "ood_10fps_ci95": [t10lo, t10hi],
                     "ood_10fps_sign": s10, "class": cls, "class_native30": cls_n30,
                     "sign_agrees_10fps": bool((ag - 0.5) * (a10 - 0.5) >= 0),
                     "class_agrees_native30_vs_10fps": bool(cls_n30 == cls)})
        print(f"  {name:16s} indego {ag:.3f} ({sg})  ood10 {a10:.3f} ({s10}) -> {cls} "
              f"| ood30 {an:.3f} ({sn}) -> {cls_n30}")

    rep = {"experiment": "E5.03", "n": int(len(gold["y"]) + len(yo)),
           "n_indego": int(len(gold["y"])), "n_good_indego": int(gold["y"].sum()),
           "n_ood": int(len(yo)), "n_good_ood": int(yo.sum()), "n_bad_ood": int((1 - yo).sum()),
           "label_source": "human (IndEgo manifest_gold n=150 + Egocentric-10K ood_labels n=60; "
                           "single labeller each; OOD n_bad=9 exploratory)",
           "label_source_indego": "human (manifest_gold, single labeller)",
           "label_source_ood": "human (ood_labels.json, single labeller; n_bad=9 exploratory)",
           "protocol": "signed AUROC per feature; label 1 = usable/good; "
                       "IndEgo CI = clip-bootstrap (7 clips), OOD CI = window bootstrap (optimistic)",
           "regime": "IndEgo 480p ~10fps vs OOD 480p transcode (~30fps native) subsampled to "
                     "10fps effective; both decoded 160x90",
           "device": "CPU (numpy)",
           "cost": {"features": 26, "indego_windows": 150, "ood_windows": 60},
           "classification_rule": "chance iff BOTH |AUROC-0.5| < 0.05 (tol); otherwise stable "
                                  "if both AUROCs are on the same side of 0.5, flip if opposite "
                                  "(e5lib.classify_transfer)",
           "counts": {"stable": n_stable, "flip": n_flip, "chance": n_chance},
           "counts_native30": n_n30,
           "native30_vs_10fps_class_disagreements": disagree_n30,
           "table": rows,
           "command": "OMP_NUM_THREADS=1 python3 eval/exp_domain/run_e5.py e5_03"}
    _write("E5.03.json", rep)


# ----------------------------------------------------------------- E5.04
def _load_batch2_scores():
    p = os.path.join(L.INPUTS, "ood_scores2.json")
    if os.path.exists(p):
        return json.load(open(p))["scores"], p
    return [], p


def _descriptive(scored, batch_of):
    probs = np.asarray([s["usable_prob"] for s in scored])
    verdicts = np.asarray([s["verdict"] for s in scored])
    pe = np.asarray([s["p_eg2"] for s in scored])
    pp = np.asarray([s["p_pixel"] for s in scored])
    out = {"n_scored": len(scored),
           "usable_prob": {"mean": float(probs.mean()), "median": float(np.median(probs)),
                           "p05": float(np.percentile(probs, 5)),
                           "p95": float(np.percentile(probs, 95)),
                           "min": float(probs.min()), "max": float(probs.max())},
           "verdicts": {v: int((verdicts == v).sum()) for v in sorted(set(verdicts.tolist()))},
           "p_eg2_mean": float(pe.mean()), "p_pixel_mean": float(pp.mean())}
    for b in sorted(set(batch_of.values())):
        sel = [s for s in scored if batch_of[s["id"]] == b]
        pr = np.asarray([s["usable_prob"] for s in sel])
        vv = np.asarray([s["verdict"] for s in sel])
        out[f"batch_{b}"] = {"n": len(sel), "usable_prob_mean": float(pr.mean()),
                             "accept": int((vv == "accept").sum()),
                             "review": int((vv == "review").sum()),
                             "reject": int((vv == "reject").sum())}
    return out


def e5_04():
    from sklearn.metrics import roc_auc_score

    cfg = json.load(open(CONFIG_PATH))
    config_sha = L.file_sha256(CONFIG_PATH)
    config_canonical_sha = L.canonical_config_sha256(cfg)
    assert config_canonical_sha == CONFIG_CANONICAL_SHA256_EXPECTED, (
        f"ensemble_v1 config content changed (path fields stripped): "
        f"{config_canonical_sha} != {CONFIG_CANONICAL_SHA256_EXPECTED}")

    s1 = L.load_frozen_ood_scores()
    s2, s2path = _load_batch2_scores()
    scored = {s["id"]: s for s in s1}
    batch_of = {s["id"]: "1(60)" for s in s1}
    for s in s2:
        scored.setdefault(s["id"], s)
        batch_of.setdefault(s["id"], "2(100)")
    ordered = [scored[k] for k in sorted(scored)]
    labels_present2 = os.path.exists(OOD_LABELS2)
    labels = L.load_labels_file(os.path.join(L.BUNDLE, "labels", "ood_labels.json"))
    n_labels2 = 0
    if labels_present2:
        lab2 = L.load_labels_file(OOD_LABELS2)
        n_labels2 = len(lab2)
        labels.update(lab2)

    rep = {"experiment": "E5.04", "n_scored": len(ordered),
           "n_batch1": len(s1), "n_batch2": len(s2),
           "labels2_present": bool(labels_present2), "n_labels2": int(n_labels2),
           "scores2_path": s2path, "config_sha256_expected": CONFIG_SHA256_EXPECTED,
           "config_sha256_observed": config_sha,
           "config_sha256_match": bool(config_sha == CONFIG_SHA256_EXPECTED),
           "config_canonical_sha256_expected": CONFIG_CANONICAL_SHA256_EXPECTED,
           "config_canonical_sha256_observed": config_canonical_sha,
           "config_assert": ("canonical content (path/provenance keys stripped) matches the frozen "
                             "config; raw sha differs only by pod path remapping"),
           "label_source": "NONE for the 160 scored windows (descriptive only, no accuracy); "
                           "ood_labels2.json NOT delivered (0 human labels)",
           "protocol": "frozen ensemble_v1 scores, no tuning; confirmatory OOD eval requires labels2",
           "device": "frozen scores cached (batch1: bundle caches; batch2: origin/feat/ensemble-views)",
           "cost": {"n_windows_scored": len(ordered)},
           "descriptive": _descriptive(ordered, batch_of),
           "command": "OMP_NUM_THREADS=1 python3 eval/exp_domain/run_e5.py e5_04 "
                      "(auto-runs on labels2 via eval/exp_domain/watch_e5_04.sh)"}

    if labels_present2:
        ids = [i for i in sorted(scored) if i in labels]
        y = np.asarray([labels[i] for i in ids])
        p = np.asarray([scored[i]["usable_prob"] for i in ids])
        rep["n"] = len(ids)
        rep["n_good"] = int(y.sum()); rep["n_bad"] = int((1 - y).sum())
        if len(ids) >= 2 and len(set(y.tolist())) == 2:
            rep["ood_auroc"] = float(roc_auc_score(y, p))
            m, lo, hi = L.boot_ci(y, p, roc_auc_score)
            rep["ci95"] = [lo, hi]
            rep["balacc"] = float(((p[y == 1] > 0.5).mean() + (p[y == 0] <= 0.5).mean()) / 2)
    else:
        rep["n"] = 0
        rep["labels_missing"] = True
        rep["verdict_note"] = ("confirmatory OOD eval INCONCLUSIVE: ood_labels2.json missing; "
                               "descriptive score distributions only -- NOT accuracy")
        rep["ci95"] = "n/a (no labels)"
    _write("E5.04.json", rep)
    print("  labels2 present:", labels_present2, "| n_scored:", len(ordered))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["e5_00", "e5_01", "e5_02", "e5_03", "e5_04", "all"])
    a = ap.parse_args()
    cmds = ["e5_00", "e5_01", "e5_02", "e5_03", "e5_04"] if a.cmd == "all" else [a.cmd]
    for c in cmds:
        print(f"=== {c} ===", flush=True)
        globals()[c]()


if __name__ == "__main__":
    main()
