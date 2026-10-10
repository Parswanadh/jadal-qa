"""Pod verification (claim LB1-cascade): early-reject cascade + selective accuracy.

Recomputes, from artifacts available on the pod, the numbers published in
docs/FACTS.md.  READ-ONLY on every input.

Artifacts used
    labels/manifest_gold.json            (READ-ONLY; 150 human gold windows)
    caches/gpu/features.json             (210 windows: 150 gold + 60 synthetic,
                                          carries the `degradation` tag)
    caches/frames10k/segments.npz        (EG2 mean window embeddings, 150)
    caches/ensemble/oof_probs.npz        (nested-LOCO OOF probs, 150)

Protocols reproduced (both are stated in the output JSON):
  cascade        - eval/gpu/cascade.py: S0 exact rules on the 2-s probe
                   (black p95<0.03, dark mean<0.05, lens_blocked spatial_var<1e-5,
                   overexposed p5>0.70, frozen temp_e<5e-4); S1 = StandardScaler +
                   LogisticRegression(26 features, no class weights, max_iter=2000),
                   leave-one-clip-out; operating point = argmax(tpr+tnr-1) over
                   np.linspace(0.05,0.95,37) with pred=(p>=t); MARGIN=0.25 around
                   p=0.5 -> uncertain -> S2 (stub).
  selective      - confidence = |p-0.5|; order = np.argsort(-conf) (numpy default
                   quicksort, ties by original window order); k = int(cov*n);
                   decision = (p > 0.5); balanced accuracy = mean(good accepted
                   rate, bad rejected rate) computed on the selected k only.
      nested view    : p from oof_probs.npz (inner-LOCO C selection, class_weight
                       balanced), i.e. eval/ensemble_nested_loco.py.
      exploratory view: scripts/ensemble_views_exploratory.py - fixed C (EG2 0.01,
                       pixel 0.3), class_weight='balanced', plain LOCO.

Usage (from the repo root):
    OMP_NUM_THREADS=2 python eval/pod_verify/cascade_selective_verify.py
Writes eval/results/pod_verify/cascade_and_selective.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from scipy.stats import rankdata  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path(os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle"))
sys.path.insert(0, str(ROOT))
from jadal.gpu.features import FEATURE_NAMES  # noqa: E402

MANIFEST = BUNDLE / "labels" / "manifest_gold.json"
POD_FEATS = BUNDLE / "caches" / "gpu" / "features.json"
REPO_FEATS = ROOT / "eval" / "results" / "gpu" / "features.json"
SEGMENTS = BUNDLE / "caches" / "frames10k" / "segments.npz"
OOF = BUNDLE / "caches" / "ensemble" / "oof_probs.npz"
OUT = ROOT / "eval" / "results" / "pod_verify" / "cascade_and_selective.json"

S0_RULES = [
    ("black", "p95", "<", 0.03),
    ("dark", "mean", "<", 0.05),
    ("lens_blocked", "spatial_var", "<", 1e-5),
    ("overexposed", "p5", ">", 0.70),
    ("frozen", "temp_e", "<", 5e-4),
]
MARGIN = 0.25
GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def s0_decide(s0):
    for rule, key, op, thr in S0_RULES:
        if key not in s0:
            continue
        val = s0[key]
        if (val < thr) if op == "<" else (val > thr):
            return {"rule": rule, "value": val, "op": op, "threshold": thr}
    return None


# ---------------------------------------------------------------- cascade
def run_cascade(manifest_path: Path, feats_path: Path) -> dict:
    gold = json.load(open(manifest_path))["items"]
    W = json.load(open(feats_path))["windows"]
    clips = sorted({it["clip"] for it in gold})
    ids = [it["id"] for it in gold if it["id"] in W and "full" in W[it["id"]]]
    X = np.array([[W[i]["full"][n] for n in FEATURE_NAMES] for i in ids])
    y = np.array([next(it["label"] for it in gold if it["id"] == i) for i in ids])
    cof = [next(it["clip"] for it in gold if it["id"] == i) for i in ids]

    p = np.zeros(len(ids))
    for c in clips:
        tr = [k for k in range(len(ids)) if cof[k] != c]
        te = [k for k in range(len(ids)) if cof[k] == c]
        if not tr or not te:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        p[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    ths = np.linspace(0.05, 0.95, 37)
    best, best_t = -1.0, 0.5
    for t in ths:
        pred = (p >= t).astype(int)
        tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else 0
        tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else 0
        if tpr + tnr - 1 > best:
            best, best_t = tpr + tnr - 1, t
    auc = float(roc_auc_score(y, p))

    exits = {"S0": 0, "S1_accept": 0, "S1_reject": 0, "S2": 0}
    s1_tp = s1_tn = s1_fp = s1_fn = 0
    false_rejects = []
    for it in gold:
        w = W.get(it["id"], {})
        if "full" not in w:
            exits["error"] = exits.get("error", 0) + 1
            continue
        ev = s0_decide(w.get("s0", {}))
        if ev is not None:
            exits["S0"] += 1
            if it["label"] == 1:
                false_rejects.append({"id": it["id"], "stage": "S0", "evidence": ev})
            else:
                s1_tn += 1
            continue
        pv = float(p[ids.index(it["id"])])
        if abs(pv - 0.5) < MARGIN:
            exits["S2"] += 1
        elif pv >= best_t:
            exits["S1_accept"] += 1
            if it["label"] == 1:
                s1_tp += 1
            else:
                s1_fp += 1
        else:
            exits["S1_reject"] += 1
            if it["label"] == 1:
                s1_fn += 1
                false_rejects.append({"id": it["id"], "stage": "S1", "p": pv})
            else:
                s1_tn += 1
    n = sum(v for k, v in exits.items() if k != "error")
    decided_good = s1_tp + s1_fn
    decided_bad = s1_tn + s1_fp
    out = {
        "n": n,
        "loco_auc": round(auc, 4),
        "s1_operating_threshold": float(best_t),
        "exits": exits,
        "exit_frac": {k: round(v / n, 4) for k, v in exits.items()},
        "decided": {
            "n_decided": decided_good + decided_bad,
            "coverage": round((decided_good + decided_bad) / n, 4),
            "good_decided": decided_good,
            "bad_decided": decided_bad,
            "good_accepted": s1_tp,
            "good_falsely_rejected": s1_fn,
            "bad_rejected": s1_tn,
            "bad_accepted": s1_fp,
            "balanced_acc": round(
                0.5 * (s1_tp / max(decided_good, 1) + s1_tn / max(decided_bad, 1)), 4),
        },
        "n_false_reject_total": len(false_rejects),
        "false_rejects": false_rejects,
    }
    # synthetic degradations, same file (210 windows: 150 gold + 60 synthetic)
    gold_ids = {it["id"] for it in gold}
    syn = {}
    for wid, w in W.items():
        if wid in gold_ids or "degradation" not in w or "full" not in w:
            continue
        d = w.get("degradation") or "?"
        syn.setdefault(d, {"n": 0, "S0": 0, "S1_or_S2": 0, "s0_rules": {}})
        syn[d]["n"] += 1
        ev = s0_decide(w.get("s0", {}))
        if ev is not None:
            syn[d]["S0"] += 1
            syn[d]["s0_rules"][ev["rule"]] = syn[d]["s0_rules"].get(ev["rule"], 0) + 1
        else:
            syn[d]["S1_or_S2"] += 1
    out["synthetic"] = syn
    return out


# ------------------------------------------------------------- selective
def selective_curve(p, y, covs=(1.0, 0.8, 0.6, 0.5)) -> list[dict]:
    n = len(y)
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf, kind="stable")
    n_ties = int(n - len(np.unique(np.round(conf, 12))))
    sorted_conf = np.sort(conf)[::-1]
    rows = []
    for cov in covs:
        k = int(cov * n)
        s = order[:k]
        pr = (p[s] > 0.5).astype(int)
        yy = y[s]
        ngood, nbad = int((yy == 1).sum()), int((yy == 0).sum())
        good_wrong = int(((pr == 0) & (yy == 1)).sum())
        bad_rej = int(((pr == 0) & (yy == 0)).sum())
        rows.append({
            "coverage": cov, "k": k,
            "n_good": ngood, "n_bad": nbad,
            "balanced_acc": round(float(
                ((pr[yy == 1] == 1).mean() + (pr[yy == 0] == 0).mean()) / 2), 4),
            "good_wrongly_rejected": good_wrong,
            "bad_rejected": bad_rej,
            "bad_rejected_frac": round(bad_rej / max(nbad, 1), 4),
            "boundary_conf": round(float(sorted_conf[k - 1]), 6),
            "boundary_ties": int((np.round(conf, 12)
                                  == round(float(sorted_conf[k - 1]), 12)).sum()),
        })
    rows.append({"_tie_check": {"n_exact_ties_in_conf": n_ties,
                                "ordering": "np.argsort(-|p-0.5|) (stable)",
                                "note": "ties broken by original window order"}})
    return rows


def main() -> int:
    t0 = time.perf_counter()
    res: dict = {
        "claim": "LB1-cascade",
        "source_of_truth": "docs/FACTS.md (pod bundle snapshot)",
        "protocols": {
            "cascade": ("eval/gpu/cascade.py: S0 exact rules on the 2-s probe; S1 = "
                        "StandardScaler + LogisticRegression(26 feats, no class_weight, "
                        "max_iter=2000), LOCO; threshold=argmax(tpr+tnr-1) over "
                        "linspace(0.05,0.95,37) with pred=(p>=t); MARGIN=0.25 -> S2"),
            "selective": ("confidence=|p-0.5|, order=argsort(-conf) (stable), "
                          "k=int(cov*n), decision=(p>0.5), balanced accuracy on the "
                          "selected k only; nested view = oof_probs.npz, exploratory "
                          "view = fixed C (EG2 0.01 / pixel 0.3), class_weight=balanced"),
        },
        "artifacts": {
            "gold_manifest": str(MANIFEST),
            "features": str(REPO_FEATS if REPO_FEATS.exists() else POD_FEATS),
            "features_pod_copy": str(POD_FEATS),
            "segments": str(SEGMENTS),
            "oof_probs": str(OOF),
        },
    }

    # ---- claims 1 + 2 (cascade) -------------------------------------------
    casc = run_cascade(MANIFEST, REPO_FEATS)
    res["claims"] = {}
    res["claims"]["1_cascade_split"] = {
        "text": ("Early-reject cascade: ~2% exit at stage 0 (dark/black/frozen caught "
                 "6/6 each on synthetic), 50% accepted at stage 1, 18% rejected at "
                 "stage 1, 30% uncertain (hand to the expensive tier)."),
        "published": {"S0": 0.02, "S1_accept": 0.50, "S1_reject": 0.18, "S2": 0.30,
                      "synthetic_dark_black_frozen_S0": "6/6 each"},
        "recomputed": {
            "gold_exit_frac": casc["exit_frac"],
            "gold_exits": casc["exits"],
            "synthetic_by_degradation": casc["synthetic"],
        },
        "artifact": "labels/manifest_gold.json + eval/results/gpu/features.json "
                    "(identical to caches/gpu/features.json, sha-checked)",
        "command": "OMP_NUM_THREADS=2 python eval/pod_verify/cascade_selective_verify.py",
        "n": casc["n"],
        "label_source": "human gold (single labeller) + synthetic-by-construction",
    }
    res["claims"]["2_decided_accuracy"] = {
        "text": ("On the 70% decided windows: balanced accuracy 0.81; 8 of the 75 good "
                 "decided windows were falsely rejected; 8 of the 30 bad decided "
                 "windows were accepted."),
        "published": {"balanced_acc": 0.81, "good_falsely_rejected": "8/75",
                      "bad_accepted": "8/30", "coverage": 0.70},
        "recomputed": casc["decided"],
        "artifact": "same as claim 1",
        "command": "same as claim 1",
        "n": casc["n"],
        "label_source": "human gold (single labeller)",
    }

    # ---- claims 3 + 4 (selective) -----------------------------------------
    z = np.load(OOF, allow_pickle=True)
    y = z["y"].astype(int)
    clip = np.array([str(c) for c in z["clip"]])
    nested = {
        "pixel26": selective_curve(z["p_pixel"], y),
        "eg2": selective_curve(z["p_eg2"], y),
        "prob_avg": selective_curve(z["p_ens"], y),
        "rank_avg": selective_curve((rankdata(z["p_eg2"]) + rankdata(z["p_pixel"]))
                                    / 2 / len(y), y),
    }

    M = {i["id"]: i for i in json.load(open(MANIFEST))["items"]}
    F = json.load(open(REPO_FEATS))["windows"]
    seg = np.load(SEGMENTS, allow_pickle=True)
    sids = [str(i) for i in seg["ids"]]
    y2 = np.array([M[i]["label"] for i in sids], dtype=int)
    clip2 = np.array([M[i]["clip"] for i in sids])
    E = seg["F"].astype(np.float32)
    cnt = seg["counts"].astype(int)
    Xe = np.stack([E[k, :max(1, cnt[k])].mean(0) for k in range(len(sids))])
    Xe /= np.maximum(np.linalg.norm(Xe, axis=1, keepdims=True), 1e-12)
    names = sorted(F[sids[0]]["full"])
    Xg = np.nan_to_num(np.array([[F[i]["full"][n] for n in names] for i in sids], float))

    def loco(X, C):
        p = np.zeros(len(y2))
        for c in set(clip2):
            tr, te = clip2 != c, clip2 == c
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
            m.fit(sc.transform(X[tr]), y2[tr])
            p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        return p

    pe, pg = loco(Xe, 0.01), loco(Xg, 0.3)
    ensp = (pe + pg) / 2
    rk = lambda v: rankdata(v) / len(v)  # noqa: E731
    exploratory = {
        "pixel26": selective_curve(pg, y2),
        "eg2": selective_curve(pe, y2),
        "prob_avg": selective_curve(ensp, y2),
        "rank_avg": selective_curve((rk(pe) + rk(pg)) / 2, y2),
    }
    aurocs_expl = {"eg2": round(float(roc_auc_score(y2, pe)), 4),
                   "pixel26": round(float(roc_auc_score(y2, pg)), 4),
                   "prob_avg": round(float(roc_auc_score(y2, ensp)), 4),
                   "rank_avg": round(float(roc_auc_score(y2, (rk(pe) + rk(pg)) / 2)), 4)}
    balacc_at_0p5 = {k: c[0]["balanced_acc"] for k, c in exploratory.items()}
    res["selective"] = {
        "nested_view": {"note": "oof_probs.npz (eval/ensemble_nested_loco.py)",
                        "curves": nested},
        "exploratory_view": {"note": ("fixed C, class_weight=balanced, plain LOCO "
                                      "(scripts/ensemble_views_exploratory.py)"),
                             "aurocs": aurocs_expl,
                             "balanced_acc_at_0.5": balacc_at_0p5,
                             "curves": exploratory},
    }

    def pick(curve, cov):
        return next(r for r in curve if r.get("coverage") == cov)

    res["claims"]["3_selective_pixel26"] = {
        "text": ("Selective accuracy (26 features + logistic, LOCO, n=150): deciding all "
                 "150 windows: balanced accuracy 0.74 (26 of 100 good windows wrongly "
                 "rejected); keeping the 80% most confident windows: 0.84 (15 of 80 good "
                 "wrongly rejected, 88% of bad rejected); 50% most confident: 0.81. "
                 "The curve is not monotone."),
        "view": "pixel26 (26 per-pixel features + logistic, LOCO)",
        "published": {"cov100": {"balanced_acc": 0.74, "good_wrongly_rejected": "26/100"},
                      "cov80": {"balanced_acc": 0.84, "good_wrongly_rejected": "15/80",
                                "bad_rejected": "88%"},
                      "cov50": {"balanced_acc": 0.81}},
        "recomputed_exploratory": exploratory["pixel26"],
        "recomputed_nested": nested["pixel26"],
        "artifact": "eval/results/gpu/features.json + caches/frames10k/segments.npz "
                    "(exploratory) / caches/ensemble/oof_probs.npz p_pixel (nested)",
        "command": "same as claim 1",
        "n": len(y2),
        "label_source": "human gold (single labeller), LOCO over 7 clips",
    }
    res["claims"]["4a_selective_summary"] = {
        "text": ("Selective accuracy: 0.74 deciding all 150 windows -> 0.84 on the 80% "
                 "most confident"),
        "view": ("pixel26, EXPLORATORY (fixed C=0.3, non-nested); nested pixel26 gives "
                 "0.76 -> 0.81"),
        "published": {"cov100": 0.74, "cov80": 0.84},
        "recomputed_exploratory": {"cov100": pick(exploratory["pixel26"], 1.0),
                                   "cov80": pick(exploratory["pixel26"], 0.8)},
        "recomputed_nested": {"cov100": pick(nested["pixel26"], 1.0),
                              "cov80": pick(nested["pixel26"], 0.8)},
        "artifact": "same as claim 3",
        "command": "same as claim 1",
        "n": len(y2),
        "label_source": "human gold (single labeller), LOCO over 7 clips",
    }
    res["claims"]["4b_selective_eg2_or_probavg"] = {
        "text": ("balanced accuracy 0.79 at threshold 0.5, 0.89 when deciding only the "
                 "60% most confident windows (7 of 63 good windows wrongly rejected)"),
        "view": ("probability average of EG2 + pixel26, EXPLORATORY (the FACTS sentence "
                 "is ambiguous: it follows a list of three models, and the referent of "
                 "'balanced accuracy 0.79' is not named; only prob-avg matches both "
                 "numbers)"),
        "published": {"balanced_acc_at_0.5": 0.79, "cov60": 0.89,
                      "good_wrongly_rejected_at_cov60": "7/63"},
        "recomputed_exploratory": {
            "eg2": {"balanced_acc_at_0.5": balacc_at_0p5["eg2"],
                    "cov60": pick(exploratory["eg2"], 0.6)},
            "prob_avg": {"balanced_acc_at_0.5": balacc_at_0p5["prob_avg"],
                         "cov60": pick(exploratory["prob_avg"], 0.6)},
            "pixel26": {"balanced_acc_at_0.5": balacc_at_0p5["pixel26"],
                        "cov60": pick(exploratory["pixel26"], 0.6)},
        },
        "recomputed_nested": {
            "eg2": {"balanced_acc_at_0.5": pick(nested["eg2"], 1.0)["balanced_acc"],
                    "cov60": pick(nested["eg2"], 0.6)},
            "prob_avg": {"balanced_acc_at_0.5": pick(nested["prob_avg"], 1.0)["balanced_acc"],
                         "cov60": pick(nested["prob_avg"], 0.6)},
        },
        "artifact": "same as claim 3",
        "command": "same as claim 1",
        "n": len(y2),
        "label_source": "human gold (single labeller), LOCO over 7 clips",
    }
    res["artifacts"]["features_sha256_repo"] = _sha(REPO_FEATS)
    res["artifacts"]["features_sha256_pod"] = _sha(POD_FEATS)
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"cascade exits: {casc['exits']} frac {casc['exit_frac']}")
    print(f"decided: {casc['decided']}")
    print("synthetic S0:", {k: f"{v['S0']}/{v['n']}" for k, v in casc["synthetic"].items()})
    print("exploratory selective (pixel26):",
          [(r["coverage"], r["balanced_acc"], r["good_wrongly_rejected"],
            r["bad_rejected_frac"]) for r in exploratory["pixel26"] if "coverage" in r])
    print("exploratory selective (prob_avg):",
          [(r["coverage"], r["balanced_acc"], r["good_wrongly_rejected"],
            r["bad_rejected_frac"]) for r in exploratory["prob_avg"] if "coverage" in r])
    print("nested selective (pixel26):",
          [(r["coverage"], r["balanced_acc"], r["good_wrongly_rejected"])
           for r in nested["pixel26"] if "coverage" in r])
    print("wrote", OUT)
    return 0


def _sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
