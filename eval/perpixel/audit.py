"""Amendment B: label-noise audit (confident-learning style, OOF) + retrain.

(i)  AI labels of 10k frames (frames10k shards: hand/work from gemini prompt):
     5-fold OOF L2-logistic on frozen EG2 embeddings; flag |oof - y| > 0.7.
(ii) 150 human labels: OOF from m2.json (activity p_lr, quality scores);
     flag strong disagreements. NEVER edits labels: writes a review queue
     in funnel/eval/gold/segments.json schema under eval/results/perpixel/.
Retrain activity-LR with/without flagged (drop, downweight x0.25) — always evaluated on
HUMAN windows only (never on cleaned AI labels).

Output: eval/results/perpixel/audit.json + review_queue.json
Regenerate: PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/perpixel/audit.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SPEC_DIR = "/home/parshu/projects/jadal-qa-wt/spectral_data"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results", "perpixel")
SHARDS = "/home/parshu/projects/jadal-qa-wt/frames10k/eval/results/frames10k/egocentric_10k"
SEED = 0
FLAG_GAP = 0.7  # |oof - label| above this -> review queue


def auroc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def oof_logistic(X, y, seed=SEED):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tri, tei in skf.split(X, y):
        best, bestm = 1.0, None
        for C in (0.01, 0.1, 1.0, 10.0):
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            ss = []
            for tri2, tei2 in inner.split(X[tri], y[tri]):
                sc = StandardScaler().fit(X[tri][tri2])
                m = LogisticRegression(C=C, max_iter=5000)
                m.fit(sc.transform(X[tri][tri2]), y[tri][tri2])
                ss.append(auroc(y[tri][tei2],
                               m.predict_proba(sc.transform(X[tri][tei2]))[:, 1]))
            mu = float(np.mean(ss))
            if bestm is None or mu > bestm:
                best, bestm = C, mu
        sc = StandardScaler().fit(X[tri])
        m = LogisticRegression(C=best, max_iter=5000, random_state=seed)
        m.fit(sc.transform(X[tri]), y[tri])
        oof[tei] = m.predict_proba(sc.transform(X[tei]))[:, 1]
    return oof


def audit_ai():
    paths = sorted(glob.glob(os.path.join(SHARDS, "shard_*.npz")))
    if not paths:
        return {"n": 0, "note": "no frames10k shards available"}
    E = np.concatenate([np.load(p)["emb"] for p in paths]).astype(np.float64)
    H = np.concatenate([np.load(p)["hand"] for p in paths]).astype(int)
    W = np.concatenate([np.load(p)["work"] for p in paths]).astype(int)
    IDS = np.concatenate([np.load(p)["ids"] for p in paths])
    out = {"n_frames": int(len(E)), "n_shards": len(paths),
           "frac_work": float(W.mean()), "frac_hands": float((H >= 1).mean())}
    for name, y in (("active_labor", W), ("hands_ok", (H >= 1).astype(int))):
        oof = oof_logistic(E, y)
        gap = np.abs(oof - y)
        fl = np.where(gap > FLAG_GAP)[0]
        out[name] = {"oof_auroc": auroc(y, oof), "n_flagged": int(len(fl)),
                     "flag_rate": float(len(fl) / len(y)),
                     "flagged_ids": [str(IDS[i]) for i in fl[:50]],
                     "note": ("flagged = OOF prob disagrees with AI label by "
                              "> 0.7; first 50 ids listed")}
        print(f"AI audit {name}: n={len(y)} oof_auc={out[name]['oof_auroc']:.3f} "
              f"flagged={len(fl)}", flush=True)
    return out


def audit_human():
    m2 = json.load(open(os.path.join(RES, "m2.json")))
    gm = json.load(open(os.path.join(SPEC_DIR, "manifest_gold.json")))
    items = {it["id"]: it for it in gm["items"]}
    q = np.load(os.path.join(RES, "m2_quality_scores.npz"), allow_pickle=True)
    qids = [str(i) for i in q["ids"]]
    qscore = {i: float(s) for i, s in zip(qids, q["scores"])}
    act = m2["activity"]
    pact = dict(zip(act["ids"], act["p_lr"]))
    flagged = []
    for i, it in items.items():
        if it["reason"] in ("NO_ACTIVE_WORK", "HANDS_NOT_VISIBLE"):
            p = pact.get(i)
            if p is None:
                continue
            if (it["label"] == 1 and p < 1 - FLAG_GAP) or \
               (it["label"] == 0 and p > FLAG_GAP):
                flagged.append({"id": i, "kind": "activity",
                                "label": it["label"], "reason": it["reason"],
                                "oof_prob_usable": round(p, 4)})
        elif it["reason"] in ("TOO_DARK", "UNDECODABLE"):
            s = qscore.get(i)
            if s is None:
                continue
            # quality score: higher = more bad; usable prob proxy = -s rank
            if (it["label"] == 0 and s < np.median(list(qscore.values())) - 1):
                flagged.append({"id": i, "kind": "quality",
                                "label": it["label"], "reason": it["reason"],
                                "oof_bad_score": round(s, 4)})
    # duplicates: descriptive check (signature NN sim of the 6 labels)
    dup_nn = []
    for i in m2["duplicates"]["human"]["dup_ids"]:
        nn1 = m2["duplicates"]["nn"][i]["nn1"]
        dup_nn.append({"id": i, "clip": items[i]["clip"],
                       "nn1": nn1,
                       "same_clip": bool(nn1["clip"] == items[i]["clip"])})
    queue = [{"segment_id": items[f["id"]]["id"], "clip": items[f["id"]]["clip"],
              "path": items[f["id"]]["path"],
              "start_s": items[f["id"]]["start_s"],
              "end_s": items[f["id"]]["end_s"],
              "audit_kind": f["kind"], "audit_label": f["label"],
              "audit_reason": f.get("reason", ""),
              "audit_oof": f.get("oof_prob_usable",
                                 f.get("oof_bad_score"))}
             for f in flagged]
    with open(os.path.join(RES, "review_queue.json"), "w") as f:
        json.dump(queue, f)
    return {"n_flagged": len(flagged), "flagged": flagged,
            "dup_nn_check": dup_nn}


def retrain_variants(flagged_ids):
    """Activity LR nested LOCO: all vs drop-flagged vs soft-flagged.

    Compact protocol (same skeleton as detectors.run_activity, LR only).
    Evaluated on HUMAN windows only.
    """
    z = np.load(os.path.join(RES, "gold.npz"), allow_pickle=True)
    X = z["X"].astype(np.float64)
    names = [str(n) for n in z["names"]]
    ids = [str(i) for i in z["ids"]]
    clips = [str(c) for c in z["clips"]]
    y_all = z["y"].astype(int)
    rea = [str(r) for r in z["reasons"]]
    keep = np.array([(y == 1 or r in ("NO_ACTIVE_WORK", "HANDS_NOT_VISIBLE"))
                     for y, r in zip(y_all, rea)])
    Xa, ya = X[keep], y_all[keep]
    ida = [i for i, m in zip(ids, keep) if m]
    cla = [c for c, m in zip(clips, keep) if m]
    drop = np.array([i not in flagged_ids for i in ida])
    m2 = json.load(open(os.path.join(RES, "m2.json")))
    pact = dict(zip(m2["activity"]["ids"], m2["activity"]["p_lr"]))

    def run(mask, weight_flagged=1.0):
        Xm = Xa[mask]
        ytrue = ya[mask]
        ym = ytrue.copy()
        cm = [c for c, m in zip(cla, mask) if m]
        im = [i for i, m in zip(ida, mask) if m]
        w = np.array([weight_flagged if i in flagged_ids else 1.0
                      for i in im])
        ucl = sorted(set(cm))
        p = np.zeros(len(ym))
        for tc in ucl:
            tr = np.array([c != tc for c in cm])
            sc = StandardScaler().fit(Xm[tr])
            mm = LogisticRegression(C=1.0, max_iter=5000, random_state=SEED)
            mm.fit(sc.transform(Xm[tr]), ym[tr],
                   sample_weight=w[tr])
            p[~tr] = mm.predict_proba(sc.transform(Xm[~tr]))[:, 1]
        return auroc(ytrue, p)

    # NOTE: dropped windows are excluded from train AND test; 'all' is m2's LR
    # number restricted to the same eval for comparability.
    out = {"n_all": int(keep.sum()), "n_dropped": int((~drop).sum())}
    out["auroc_drop"] = run(drop)
    out["auroc_downweight025"] = run(np.ones(len(ida), bool),
                                     weight_flagged=0.25)
    out["auroc_all_sameproto"] = run(np.ones(len(ida), bool), soft=False)
    out["auroc_all_m2_LR"] = float(m2["activity"]["auroc_lr"])
    return out


def main():
    ai = audit_ai()
    hum = audit_human()
    print(f"human flagged: {hum['n_flagged']}", flush=True)
    fl = {f["id"] for f in hum["flagged"]}
    retr = retrain_variants(fl)
    print("retrain:", retr, flush=True)
    out = {"ai_labels": ai, "human": hum, "retrain_on_human_only": retr,
           "flag_gap": FLAG_GAP,
           "honesty": ("Retrain evaluated on HUMAN windows only, never on "
                       "cleaned AI labels. Human labels file untouched.")}
    with open(os.path.join(RES, "audit.json"), "w") as f:
        json.dump(out, f)
    print("wrote audit.json + review_queue.json", flush=True)


if __name__ == "__main__":
    main()
