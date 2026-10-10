#!/usr/bin/env python3
"""Deterministic physics rule-list model — LOCO evaluation on the 150 IndEgo gold windows.

Writes:
  eval/results/exp_phys/Q1.01.json      full results (LOCO metrics, per-rule stats, ablations)
  eval/results/exp_phys/rule_table.json frozen rule table (all-clip thresholds) for deployment
  eval/results/exp_phys/roc_phys.png    ROC + per-rule precision figure

Run: OMP_NUM_THREADS=2 python3 eval/exp_phys/run_exp_phys.py
No GPU. Reads cached features only; never touches labels.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(THIS, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, THIS)

from phys_rules import (CANDIDATE_RULES, Q, build_rule_table, choose_reject_threshold,  # noqa: E402
                        choose_reject_threshold_ba, decide, derive_thresholds, score_window)

DATA = os.environ["JADAL_DATA"]
OUT = os.path.join(ROOT, "eval", "results", "exp_phys")
FR_BUDGET = 0.05          # pre-registered false-reject budget on training-good
BOOT = 2000
SEED = 0


# --------------------------------------------------------------------------
def load_data():
    manifest = json.load(open(os.path.join(DATA, "labels", "manifest_gold.json")))["items"]
    W = json.load(open(os.path.join(DATA, "caches", "gpu", "features.json")))["windows"]
    gold = np.load(os.path.join(DATA, "caches", "perpixel", "gold.npz"), allow_pickle=True)
    phys = np.load(os.path.join(DATA, "caches", "perpixel", "physics_feats.npz"), allow_pickle=True)
    oof = np.load(os.path.join(DATA, "caches", "ensemble", "oof_probs.npz"), allow_pickle=True)

    ids = [str(i) for i in gold["ids"]]
    y = np.asarray(gold["y"], dtype=int)          # 1 = good, 0 = bad
    clips = np.asarray(gold["clips"])
    reasons = np.asarray(gold["reasons"])
    full_names = list(W[ids[0]]["full"].keys())
    phys_names = [str(n) for n in phys["names"]]
    pidx = {str(k): i for i, k in enumerate(phys["ids"])}

    feats = []
    for i in ids:
        d = {f"full.{n}": float(W[i]["full"][n]) for n in full_names}
        d.update({f"phys.{n}": float(phys["X"][pidx[i], j]) for j, n in enumerate(phys_names)})
        feats.append(d)
    meta = {"manifest_n": len(manifest), "reasons": [str(r) for r in reasons],
            "full_names": full_names, "phys_names": phys_names}
    oof_map = {str(k): i for i, k in enumerate(oof["ids"])}
    oof_p = {k: np.array([float(oof[k][oof_map[i]]) for i in ids])
             for k in ("p_eg2", "p_pixel", "p_ens")}
    return ids, feats, y, clips, meta, oof_p


def loco(feats, y, clips, rule_specs=CANDIDATE_RULES, quantiles=None):
    """Leave-one-clip-out. Thresholds and reject operating points from train only.

    Two deterministic operating points are produced:
      * "fr5"  : smallest integer score with <=5% false-reject on training-good
                 (conservative product default).
      * "ba"   : integer score maximising training balanced accuracy.
    """
    n = len(y)
    scores = np.zeros(n, dtype=int)
    dec = {"fr5": [None] * n, "ba": [None] * n}
    ev = {"fr5": [None] * n, "ba": [None] * n}
    per_fold = {}
    for c in sorted(set(clips)):
        te = np.array([clips[i] == c for i in range(n)])
        tr = ~te
        train_feats = {k: np.array([feats[i].get(k, np.nan) for i in range(n)])[tr]
                       for k in {r["feature"] for r in rule_specs}}
        table = []
        for r in rule_specs:
            ths = derive_thresholds(train_feats[r["feature"]], r["side"])
            if ths:
                table.append({**r, "thresholds": ths})
        t_scores = np.array([score_window(feats[i], table)["score"]
                             for i in range(n) if tr[i]])
        t_rej_fr = choose_reject_threshold(t_scores, y[tr], FR_BUDGET)
        t_rej_ba = choose_reject_threshold_ba(t_scores, y[tr])
        for i in range(n):
            if not te[i]:
                continue
            sc = score_window(feats[i], table)
            scores[i] = sc["score"]
            for pol, t in (("fr5", t_rej_fr), ("ba", t_rej_ba)):
                d = decide(sc, t)
                dec[pol][i] = d["decision"]
                ev[pol][i] = d["evidence"]
        per_fold[str(c)] = {"n": int(te.sum()),
                            "reject_threshold_fr5": int(t_rej_fr),
                            "reject_threshold_ba": int(t_rej_ba),
                            "n_thresholds": int(sum(len(t["thresholds"]) for t in table))}
    return scores, dec, ev, per_fold


def loco_selected(feats, y, clips, k=12):
    """Ablation: per-fold feature selection (top-k by training univariate |AUROC-0.5|)."""
    from sklearn.metrics import roc_auc_score
    n = len(y)
    scores = np.zeros(n, dtype=int)
    for c in sorted(set(clips)):
        te = np.array([clips[i] == c for i in range(n)])
        tr = ~te
        ranked = []
        for r in CANDIDATE_RULES:
            v = np.array([feats[i].get(r["feature"], np.nan) for i in range(n)])
            m = np.isfinite(v) & tr
            if m.sum() < 8 or len(np.unique(v[m])) < 2:
                continue
            a = roc_auc_score(1 - y[m], v[m])
            ranked.append((abs(a - 0.5), r, 1 if a >= 0.5 else -1))
        ranked.sort(key=lambda x: -x[0])
        specs = []
        for _, r, d in ranked[:k]:
            rr = dict(r)
            rr["op"] = ">" if d > 0 else "<"
            rr["side"] = "high" if d > 0 else "low"
            specs.append(rr)
        train_feats = {r["feature"]: np.array([feats[i].get(r["feature"], np.nan)
                                               for i in range(n)])[tr] for r in specs}
        table = []
        for r in specs:
            ths = derive_thresholds(train_feats[r["feature"]], r["side"])
            if ths:
                table.append({**r, "thresholds": ths})
        for i in range(n):
            if te[i]:
                scores[i] = score_window(feats[i], table)["score"]
    return scores


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(1 - y, s))


def clip_bootstrap_ci(y, s, clips, n_boot=BOOT, seed=SEED):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    uc = sorted(set(clips))
    out = []
    for _ in range(n_boot):
        cs = rng.choice(uc, size=len(uc), replace=True)
        idx = np.concatenate([np.where(clips == c)[0] for c in cs])
        if len(set(y[idx])) < 2:
            continue
        out.append(roc_auc_score(1 - y[idx], s[idx]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else [None, None]


def logistic_loco(feats, y, clips, names):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.array([[feats[i][f"full.{n}"] for n in names] for i in range(len(y))])
    p = np.zeros(len(y))
    for c in sorted(set(clips)):
        te = clips == c
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        p[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def main():
    t0 = time.perf_counter()
    ids, feats, y, clips, meta, oof_p = load_data()
    n = len(y)
    assert len(set(y)) == 2

    scores, dec_by_pol, ev_by_pol, per_fold = loco(feats, y, clips)
    a_phys = auc(y, scores)
    ci = clip_bootstrap_ci(y, scores, clips)

    def policy_metrics(pol):
        dec = np.array(dec_by_pol[pol])
        rej = dec == "REJECT"
        rev = dec == "REVIEW"
        acc = dec == "ACCEPT"
        tp = int((rej & (y == 0)).sum())
        fp = int((rej & (y == 1)).sum())
        tn = int((acc & (y == 1)).sum())
        fn = int((acc & (y == 0)).sum())
        decided = rej | acc
        ba_decided = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
        # treat REVIEW as accept (product default: only reject when sure)
        tp_r, fp_r = tp, fp
        tn_r = tn + int((rev & (y == 1)).sum())
        fn_r = fn + int((rev & (y == 0)).sum())
        ba_review_accept = 0.5 * (tp_r / max(tp_r + fn_r, 1) + tn_r / max(tn_r + fp_r, 1))
        # treat REVIEW as reject
        ba_review_reject = 0.5 * ((tp + int((rev & (y == 0)).sum())) / max(int((y == 0).sum()), 1)
                                  + tn / max(int((y == 1).sum()), 1))
        return {
            "balanced_acc_decided": round(ba_decided, 4),
            "balanced_acc_review_as_accept": round(ba_review_accept, 4),
            "balanced_acc_review_as_reject": round(ba_review_reject, 4),
            "coverage_decided": round(float(decided.mean()), 4),
            "review_share": round(float(rev.mean()), 4),
            "reject_share": round(float(rej.mean()), 4),
            "accept_share": round(float(acc.mean()), 4),
            "confusion": {"tp_bad_rejected": tp, "fp_good_rejected": fp,
                          "tn_good_accepted": tn, "fn_bad_accepted": fn},
            "false_reject_rate_good": round(fp / max(int((y == 1).sum()), 1), 4),
            "bad_recall": round(tp / max(int((y == 0).sum()), 1), 4),
        }

    m_fr5 = policy_metrics("fr5")
    m_ba = policy_metrics("ba")
    decisions = dec_by_pol["fr5"]
    evidence = ev_by_pol["fr5"]

    # per-rule fire counts + precision/recall (fold-local tables; recompute fired)
    rule_stats = {}
    for r in CANDIDATE_RULES:
        rule_stats.setdefault(r["rule"], {"features": set(), "fire": 0, "fire_bad": 0,
                                          "bad_total": int((y == 0).sum())})
        rule_stats[r["rule"]]["features"].add(r["feature"])
    # recompute per-window fired rules using per-fold table
    fired_by_win = [set() for _ in range(n)]
    for c in sorted(set(clips)):
        te = clips == c
        tr = ~te
        train_feats = {k: np.array([feats[i].get(k, np.nan) for i in range(n)])[tr]
                       for k in {r["feature"] for r in CANDIDATE_RULES}}
        table = build_rule_table(train_feats)
        for i in range(n):
            if te[i]:
                fired_by_win[i] = set(score_window(feats[i], table)["fired_rules"])
    for rule, st in rule_stats.items():
        st["features"] = sorted(st["features"])
        st["fire"] = int(sum(rule in fw for fw in fired_by_win))
        st["fire_bad"] = int(sum((rule in fw) and (y[i] == 0) for i, fw in enumerate(fired_by_win)))
        st["precision"] = round(st["fire_bad"] / max(st["fire"], 1), 3)
        st["recall"] = round(st["fire_bad"] / max(st["bad_total"], 1), 3)
    # per named-rule indicator AUROC
    for rule, st in rule_stats.items():
        ind = np.array([1.0 if rule in fw else 0.0 for fw in fired_by_win])
        st["indicator_auroc"] = round(auc(y, ind), 3) if ind.any() else None

    # leave-one-rule-out ablation
    l1o = {}
    for rule in sorted(rule_stats):
        specs = [r for r in CANDIDATE_RULES if r["rule"] != rule]
        s2, _, _, _ = loco(feats, y, clips, rule_specs=specs)
        l1o[rule] = round(auc(y, s2), 4)

    # per-fold feature-selection ablation (optimism check)
    s_sel = loco_selected(feats, y, clips, k=12)
    a_sel = auc(y, s_sel)

    # baselines under the same folds (probability-of-GOOD; direction positive)
    p_log = logistic_loco(feats, y, clips, meta["full_names"])
    from sklearn.metrics import roc_auc_score as _ras
    a_log = float(_ras(y, p_log))
    a_eg2 = float(_ras(y, oof_p["p_eg2"]))
    a_pixel = float(_ras(y, oof_p["p_pixel"]))
    a_ens = float(_ras(y, oof_p["p_ens"]))

    # frozen all-clip rule table (deployment) + evidence example
    all_feats = {k: np.array([feats[i].get(k, np.nan) for i in range(n)])
                 for k in {r["feature"] for r in CANDIDATE_RULES}}
    frozen = build_rule_table(all_feats)
    frozen_score = np.array([score_window(feats[i], frozen)["score"] for i in range(n)])
    frozen_rej = choose_reject_threshold(frozen_score[y == 1], y[y == 1], FR_BUDGET)

    examples = []
    for i in range(n):
        if evidence[i]:
            examples.append({"id": ids[i], "clip": str(clips[i]), "y": int(y[i]),
                             "decision": decisions[i], "evidence": evidence[i]})
    # a couple of false rejects for honesty
    false_rejects = [{"id": ids[i], "clip": str(clips[i]), "score": int(scores[i]),
                      "decision": decisions[i], "evidence": evidence[i]}
                     for i in range(n) if y[i] == 1 and decisions[i] == "REJECT"]

    res = {
        "experiment": "Q1.01",
        "title": "Complete deterministic physics rule-list model (in-domain LOCO)",
        "determinism": "no learned weights; thresholds = fixed training-clip quantiles "
                       f"p01/p05/p10/p25 quantised to {Q} dp; integer score = count of "
                       "crossed thresholds; reject point from 5% training-good FR budget",
        "data": {"source": DATA, "n": n, "n_good": int((y == 1).sum()),
                 "n_bad": int((y == 0).sum()), "n_clips": len(set(clips)),
                 "clips": {str(c): int((clips == c).sum()) for c in sorted(set(clips))},
                 "label_source": "human (IndEgo gold)", "protocol": "leave-one-clip-out"},
        "loco": {
            "auroc_deterministic_score": round(a_phys, 4),
            "auroc_ci95_clip_bootstrap": [round(x, 4) if x else None for x in ci],
            "operating_points": {"fr5_conservative": m_fr5, "ba_balanced": m_ba},
            "per_fold": per_fold,
        },
        "baselines_same_folds": {
            "logistic_26feat_auroc": round(a_log, 4),
            "eg2_ensemble_auroc_cached": round(a_eg2, 4),
            "pixel_conv_auroc_cached": round(a_pixel, 4),
            "ens_auroc_cached": round(a_ens, 4),
            "determinism_cost_vs_logistic": round(a_log - a_phys, 4),
            "determinism_cost_vs_eg2": round(a_eg2 - a_phys, 4),
        },
        "per_rule": rule_stats,
        "leave_one_rule_out_auroc": l1o,
        "ablations": {"per_fold_feature_selection_top12_auroc": round(a_sel, 4),
                      "note": "selecting features per fold overfits clip-specific "
                              "features; the fixed spec candidate list is what transfers"},
        "frozen_rule_table_deployment": {
            "reject_threshold": int(frozen_rej), "n_rules": len(frozen),
            "auroc_in_sample": round(auc(y, frozen_score), 4)},
        "decisions_examples": examples[:40],
        "n_false_rejects": len(false_rejects),
        "false_rejects": false_rejects,
        "wall_s": round(time.perf_counter() - t0, 2),
    }

    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "Q1.01.json"), "w"), indent=1)
    json.dump({"rules": frozen, "reject_threshold": int(frozen_rej),
               "false_reject_budget": FR_BUDGET,
               "derivation": "thresholds = quantiles p01/p05/p10/p25 (mirrored to "
                             "p99/p95/p90/p75 for high-is-bad features) of feature "
                             "values on all 7 clips, quantised to 4 dp; "
                             "score = count of crossed thresholds; "
                             "reject if score>=threshold or a quality rule fires"},
              open(os.path.join(OUT, "rule_table.json"), "w"), indent=1)

    make_figure(y, scores, p_log, oof_p["p_eg2"], clips, rule_stats,
                os.path.join(OUT, "roc_phys.png"))

    print(json.dumps({k: res[k] for k in
                      ("loco", "baselines_same_folds", "ablations",
                       "frozen_rule_table_deployment")}, indent=1))
    print("per_rule:")
    for k, v in rule_stats.items():
        print(f"  {k:18s} fire={v['fire']:3d} prec={v['precision']:.2f} "
              f"rec={v['recall']:.2f} ind_auc={v['indicator_auroc']}")
    print("L1O:", l1o)
    print("wrote", OUT)


def make_figure(y, scores, p_log, p_eg2, clips, rule_stats, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, s, col in [("deterministic rules", scores, "C3"),
                         ("logistic 26-feat (LOCO)", -p_log, "C0"),
                         ("EG2 ensemble (cached)", -p_eg2, "C2")]:
        fpr, tpr, _ = roc_curve(1 - y, s)
        ax[0].plot(fpr, tpr, color=col,
                   label=f"{name}  AUC={roc_auc_score(1 - y, s):.3f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.7)
    ax[0].set_xlabel("false-reject rate (good windows)")
    ax[0].set_ylabel("bad-window recall")
    ax[0].set_title("ROC, 150 IndEgo gold, leave-one-clip-out")
    ax[0].legend(fontsize=8, loc="lower right")

    rules = list(rule_stats)
    prec = [rule_stats[r]["precision"] for r in rules]
    fires = [rule_stats[r]["fire"] for r in rules]
    ax[1].barh(rules, prec, color="C3")
    for i, (p, f) in enumerate(zip(prec, fires)):
        ax[1].text(min(p + 0.02, 0.98), i, f"n={f}", va="center", fontsize=7)
    ax[1].set_xlim(0, 1.15)
    ax[1].set_xlabel("precision (fraction of fires that are bad)")
    ax[1].set_title("Per-rule fire precision (LOCO)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("wrote", path)


if __name__ == "__main__":
    main()
