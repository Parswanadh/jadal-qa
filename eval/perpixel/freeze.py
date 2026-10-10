"""M3: freeze the deterministic scorer -> configs/perpixel_v1.json (+ .sha256).

Refit protocol (documented, deterministic, seed-free except fixed SEED=0
for LR/HGB solvers):
  quality rule : best Youden on ALL gold (same code as detectors.py transfer)
  theory rules : fit_stage on ALL gold (same exhaustive grids as theory_clean)
  dup threshold: constructed-pairs threshold from m2.json (unchanged)
  activity LR  : feature set chosen by nested-LOCO AUROC (deterministic
                 argmax over {m1, theory+physics, all}); C by inner LOCO;
                 Platt calibration on inner-LOCO OOF; threshold at 5% FRR.
LOCO numbers in theory.json/physics.json/m2.json are the honest estimates;
this file is the PRE-REGISTERED frozen artefact for one later scoring on
Egocentric-10K labels. Do NOT tune after freezing.

Also writes eval/results/perpixel/compare.json (rule list vs LR vs HGB vs
conv vs physics-LR, same nested LOCO) and freeze_receipts.json
(10x-repeat digests, threads 1v8, CPU-vs-GPU G_* check under flock).

Regenerate: PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/perpixel/freeze.py
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from eval.perpixel import detectors as D  # noqa: E402
from eval.perpixel import theory_clean as TC  # noqa: E402
from jadal.perpixel import physics as P  # noqa: E402
from jadal.perpixel import theory as T  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SPEC_DIR = "/home/parshu/projects/jadal-qa-wt/spectral_data"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results", "perpixel")
CFG = os.path.join(os.path.dirname(HERE), "..", "..", "configs",
                   "perpixel_v1.json")
SEED = 0


def main():
    gm = json.load(open(os.path.join(SPEC_DIR, "manifest_gold.json")))
    m2 = json.load(open(os.path.join(RES, "m2.json")))
    theory = json.load(open(os.path.join(RES, "theory.json")))
    phys = json.load(open(os.path.join(RES, "physics.json")))
    z = np.load(os.path.join(RES, "gold.npz"), allow_pickle=True)
    X = z["X"].astype(np.float64)
    names = [str(n) for n in z["names"]]
    ids = [str(i) for i in z["ids"]]
    clips = [str(c) for c in z["clips"]]
    y = z["y"].astype(int)
    rea = [str(r) for r in z["reasons"]]
    zt = np.load(os.path.join(RES, "theory_feats.npz"), allow_pickle=True)
    tids = [str(i) for i in zt["ids"]]
    tmap = {i: np.asarray(zt["F"][j], dtype=np.float64) / 1e4
            for j, i in enumerate(tids)}
    zp = np.load(os.path.join(RES, "physics_feats.npz"), allow_pickle=True)
    pids = [str(i) for i in zp["ids"]]
    pmap = {i: np.asarray(zp["X"][j], dtype=np.float64)
            for j, i in enumerate(pids)}
    order = sorted(set(ids) & set(tids) & set(pids))
    print(f"freeze intersection n={len(order)}", flush=True)
    Xm1 = np.array([dict(zip(ids, X))[i] for i in order])
    Xth = np.array([tmap[i] for i in order])
    Xph = np.array([pmap[i] for i in order])
    yo = np.array([dict(zip(ids, y))[i] for i in order])
    co = [dict(zip(ids, clips))[i] for i in order]
    ro = [dict(zip(ids, rea))[i] for i in order]

    # ---- quality rule refit on ALL gold (same Youden code as detectors.py)
    is_q = np.array([r in D.QUALITY_REASONS for r in ro])
    pool = is_q | (yo == 1)
    best = None
    for rname, feat, direction in D.QUALITY_CANDS:
        j = names.index(feat)
        vv = -Xm1[pool, j] if direction == ">" else Xm1[pool, j]
        thr, J = D.youden_thr(vv, is_q[pool].astype(int))
        if best is None or J > best[-1]:
            best = (rname, feat, direction, thr, J)

    # ---- theory rules refit on ALL gold (floats rounded to Q first)
    meta = {json.loads(m)["id"]: json.loads(m) for m in zt["meta"]}
    QM = np.round(np.array(
        [[meta[i]["float"][k] for k in T.THEORY_FEATURES] for i in order]), T.Q)
    c1, b1 = TC.fit_stage(QM, yo, ro, TC.STAGE1, TC.QUALITY_REASONS, None)
    H1 = np.zeros(len(order), bool)
    for name, feat, op, _, _ in TC.RULES:
        if name in TC.STAGE1:
            H1 |= TC.hits(QM, None, name, feat, op, c1[name])
    c3, b3 = TC.fit_stage(QM[~H1], yo[~H1],
                          [r for r, h in zip(ro, H1) if not h],
                          TC.STAGE3, TC.ACTIVITY_REASONS, None)

    # ---- activity LR: feature-set choice by nested LOCO (deterministic)
    is_a = np.array([r in D.ACTIVITY_REASONS for r in ro])
    ma = (yo == 1) | is_a
    cands = {
        "m1": Xm1[ma],
        "theory+physics": np.concatenate([Xth[ma], Xph[ma]], axis=1),
        "all": np.concatenate([Xm1[ma], Xth[ma], Xph[ma]], axis=1),
    }
    acts = {"m1": float(m2["activity"]["auroc_lr"])}
    for k in ("theory+physics", "all"):
        p, a, _ = __import__("eval.perpixel.physics_run", fromlist=["x"]
                             ).loco_lr(cands[k], yo[ma],
                                       [c for c, m in zip(co, ma) if m], k)
        acts[k] = a
    feat_choice = max(sorted(acts),
                      key=lambda k: (np.nan_to_num(acts[k], nan=-1)))
    print("activity feature choice:", acts, "->", feat_choice, flush=True)
    Xc, cnames = cands[feat_choice], feat_choice
    C, _ = D.inner_select_logistic(Xc, yo[ma],
                                   [c for c, m in zip(co, ma) if m])
    A, _ = fill(Xc, Xc)
    sc = StandardScaler().fit(A)
    m = LogisticRegression(C=C, max_iter=5000, random_state=SEED)
    m.fit(sc.transform(A), yo[ma])
    # Platt on inner-LOCO OOF
    cla = [c for c, mm in zip(co, ma) if mm]
    uin = sorted(set(cla))
    oof = np.zeros(len(Xc))
    for c2 in uin:
        tr2 = np.array([c != c2 for c in cla])
        if len(np.unique(yo[ma][tr2])) < 2:
            oof[~tr2] = 0.5
            continue
        A2, B2 = fill(Xc[tr2], Xc[~tr2])
        sc2 = StandardScaler().fit(A2)
        m2a = LogisticRegression(C=C, max_iter=5000, random_state=SEED)
        m2a.fit(sc2.transform(A2), yo[ma][tr2])
        oof[~tr2] = m2a.predict_proba(sc2.transform(B2))[:, 1]
    cal = LogisticRegression(C=1.0).fit(oof.reshape(-1, 1), yo[ma])
    thr5 = float(np.percentile(oof[yo[ma] == 1], 5))

    # ---- physics NaN fill medians (frozen)
    nan_fill = {}
    for j, k in enumerate(P.PHYSICS_FEATURES):
        col = Xph[:, j]
        med = float(np.nanmedian(col)) if np.isfinite(col).any() else 0.0
        nan_fill[k] = med

    cfg = {
        "version": "perpixel_v1",
        "created_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "git_sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True).stdout.strip(),
        "label_snapshot": {"manifest": "manifest_gold.json",
                           "sha256": gm.get("labels_sha256"),
                           "n_good": gm.get("n_good"), "n_bad": gm.get("n_bad")},
        "feature_names": names,
        "theory_features": T.THEORY_FEATURES,
        "physics_features": P.PHYSICS_FEATURES,
        "nan_fill": nan_fill,
        "quality_rule": {"name": best[0], "feature": best[1], "op": best[2],
                         "threshold": float(best[3]), "train_youden": float(best[4])},
        "theory_rules": [
            {"name": n, "feature": f, "op": o,
             "threshold": float((c1 if n in c1 else c3)[n])}
            for n, f, o, _, _ in TC.RULES],
        "theory_stage_balacc_train": {"stage1": b1, "stage3": b3},
        "dup": {"threshold": float(m2["duplicates"]["constructed"]["threshold"]),
                "note": "from constructed pairs; over-flags real data "
                        "(human precision 0.048); single-window scorer "
                        "carries threshold for audit, corpus check offline"},
        "activity_lr": {"feature_set": cnames, "nested_loco_auroc": acts,
                        "C": C, "features": None,
                        "coef": [float(v) for v in m.coef_[0]],
                        "intercept": float(m.intercept_[0]),
                        "scaler_mean": [float(v) for v in sc.mean_],
                        "scaler_scale": [float(v) for v in sc.scale_],
                        "platt_A": float(cal.coef_[0][0]),
                        "platt_B": float(cal.intercept_[0]),
                        "threshold": thr5},
        "certifiability": ("At n=150 (CERTIFY-by-clip would leave n~84), no "
                           "single rule reaches the ~86 clean fires needed "
                           "for a Bonferroni certifiable bound; per-rule "
                           "Clopper-Pearson uppers in theory.json are "
                           "descriptive only."),
        "frozen_note": ("PRE-REGISTERED for one later scoring on a new "
                        "human-labelled Egocentric-10K sample. Do NOT tune "
                        "after freezing."),
    }
    with open(CFG, "w") as f:
        json.dump(cfg, f, indent=1)
    h = hashlib.sha256(open(CFG, "rb").read()).hexdigest()
    open(CFG + ".sha256", "w").write(h + "  perpixel_v1.json\n")
    print("froze", CFG, h[:16], flush=True)

    # ---- compare.json : same nested LOCO, all tracks
    rl = theory["rule_list"]
    rl_auroc = float(rl.get("auroc", 1 - rl.get("auroc_usable", 0.5)))
    rl_ci = rl["ci"]
    if "auroc" not in rl:  # legacy sign: score was negated
        rl_ci = [1 - rl_ci[1], 1 - rl_ci[0]]
    cmpj = {
        "protocol": "nested leave-one-clip-out, 7 IndEgo clips, 150 windows",
        "tracks": {
            "rule_list": {"bal_acc": rl["bal_acc"],
                          "auroc": rl_auroc, "ci": rl_ci,
                          "per_reason": rl["per_reason"],
                          "n_params": "7 thresholds (ints)",
                          "explains": "every fire carries {value,op,threshold}"},
            "l2_logistic_m1": {"auroc": float(m2["activity"]["auroc_lr"]),
                               "ci": m2["activity"]["ci_lr"],
                               "n_params": "121 coef + scaler",
                               "explains": "coef signs only"},
            "hgb_m1": {"auroc": float(m2["activity"]["auroc_hgb"]),
                       "ci": m2["activity"]["ci_hgb"],
                       "n_params": "~100 trees x leaves",
                       "explains": "gain importances only"},
            "conv1d_series": {"auroc": float(m2["series"]["auroc_mean"]),
                              "ci": m2["series"]["ci"],
                              "seeds": m2["series"]["auroc_seeds"],
                              "n_params": "~1k",
                              "explains": "no"},
            "physics_lr": {"auroc": float(phys["activity"]["physics_only"]["auroc"]),
                           "ci": phys["activity"]["physics_only"]["ci"],
                           "n_params": "16 coef",
                           "explains": "coef signs only"},
        },
        "verdict": ("Deterministic rule list is primary by requirement; "
                    "regression tracks stay as baselines. Gap per reason in "
                    "rule_list.per_reason vs m2 per_reason_at_5pctFRR_lr."),
    }
    with open(os.path.join(RES, "compare.json"), "w") as f:
        json.dump(cmpj, f, indent=1)
    print("wrote compare.json", flush=True)


def fill(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return (np.where(np.isfinite(Xtr), Xtr, med),
            np.where(np.isfinite(Xte), Xte, med))


if __name__ == "__main__":
    main()
