#!/usr/bin/env python3
"""Q4.01: pure deterministic rules vs rules+logistic fallback (hybrid).

Modes
-----
(a) pure rules      : the packaged deterministic model (tools/phys) ACCEPT/REJECT/REVIEW.
(b) rules + fallback: REVIEW windows are resolved deterministically by the frozen
    ensemble_v1 logistic score (accept iff prob >= 0.5); ACCEPT/REJECT stay deterministic.

Evaluated on
  * gold (150 IndEgo windows, leave-one-clip-out, fold-local thresholds);
  * OOD-160 (160 Egocentric-10K windows, factory-grouped bootstrap CI).

Writes eval/results/exp_phys/Q4.01.json. CPU only, cached features.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from tools.phys import model  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE")
if not BUNDLE:
    raise SystemExit("JADAL_BUNDLE is required (path to the pod data bundle)")
RES = os.path.join(ROOT, "eval", "results", "exp_phys")
QUANTILES = (0.01, 0.05, 0.10, 0.25)
Q = 4
FR_BUDGET = 0.05
N_BOOT = 2000
SEED = 0
FROZEN = model.load_rules()


# ---------------------------------------------------------------- thresholds
def _quantise(v):
    return model.quantise(v, Q)


def derive_fold_table(feats, train_idx, base_rules):
    """Fold-local integer thresholds from TRAINING feature values only."""
    rules = []
    for r in base_rules["rules"]:
        vals = np.array([feats[i].get(r["feature"], np.nan) for i in train_idx], float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 8 or len(np.unique(vals)) < 2:
            continue
        qs = np.array(QUANTILES) if r["side"] == "low" else 1.0 - np.array(QUANTILES)
        th_f = [float(t) for t in np.quantile(vals, qs)]
        rules.append({**r, "thresholds": th_f,
                      "thresholds_q": [_quantise(t) for t in th_f]})
    return {"quantisation": Q, "reject_threshold": 1,
            "undecodable_features": base_rules["undecodable_features"],
            "undecodable_score": base_rules["undecodable_score"], "rules": rules}


def choose_reject_threshold(train_scores, train_y, budget=FR_BUDGET):
    good = np.asarray(train_scores)[np.asarray(train_y) == 1]
    if good.size == 0:
        return 1
    for s in range(1, int(np.max(train_scores)) + 2):
        if float((good >= s).mean()) <= budget:
            return s
    return int(np.max(train_scores)) + 1


def score_table(feats, table):
    """Vectorised-free deterministic scoring: (score, decision) per window."""
    scores, decisions = [], []
    for f in feats:
        sc = model.score_window(f, table)
        d = model.decide(sc, table)
        scores.append(sc["score"])
        decisions.append(d["decision"])
    return np.array(scores, int), np.array(decisions, dtype=object)


def gold_loco(feats, y, clips):
    scores = np.zeros(len(y), int)
    decisions = np.empty(len(y), dtype=object)
    per_fold = {}
    for c in sorted(set(clips)):
        te = np.array([clips[i] == c for i in range(len(y))])
        tr = np.where(~te)[0]
        table = derive_fold_table(feats, tr, FROZEN)
        t_scores = np.array([model.score_window(feats[i], table)["score"] for i in tr])
        table["reject_threshold"] = choose_reject_threshold(t_scores, y[tr])
        for i in np.where(te)[0]:
            sc = model.score_window(feats[i], table)
            scores[i] = sc["score"]
            decisions[i] = model.decide(sc, table)["decision"]
        per_fold[str(c)] = {"n": int(te.sum()),
                            "reject_threshold": int(table["reject_threshold"]),
                            "n_thresholds": int(sum(len(r["thresholds_q"])
                                                    for r in table["rules"]))}
    return scores, decisions, per_fold


# ---------------------------------------------------------------- metrics
def auroc(y, s):
    """AUROC of a BAD-score (higher = more likely bad)."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(1 - np.asarray(y), np.asarray(s)))


def auroc_good(y, p):
    """AUROC of a GOOD-probability (higher = more likely good)."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y), np.asarray(p)))


def bal_acc(y, pred_bad):
    y = np.asarray(y)
    pred_bad = np.asarray(pred_bad, bool)
    n_bad = max(int((y == 0).sum()), 1)
    n_good = max(int((y == 1).sum()), 1)
    recall_bad = float((pred_bad & (y == 0)).sum()) / n_bad
    tnr_good = float((~pred_bad & (y == 1)).sum()) / n_good
    return 0.5 * (recall_bad + tnr_good)


def mode_metrics(y, scores, decisions, prob):
    """Metrics for pure rules and for the hybrid (fallback on REVIEW)."""
    decisions = np.asarray(decisions, dtype=object)
    rev = decisions == "REVIEW"
    rej = decisions == "REJECT"
    acc = decisions == "ACCEPT"
    # pure
    pure = {
        "auroc_deterministic_score": round(auroc(y, scores), 4),
        "balacc_review_as_accept": round(bal_acc(y, rej), 4),
        "balacc_review_as_reject": round(bal_acc(y, rej | rev), 4),
        "balacc_decided_only": round(bal_acc(y[~rev], rej[~rev]), 4) if (~rev).any() else None,
        "review_share": round(float(rev.mean()), 4),
        "reject_share": round(float(rej.mean()), 4),
        "accept_share": round(float(acc.mean()), 4),
        "share_decided_deterministically": round(float((~rev).mean()), 4),
        "share_by_model": round(float(rev.mean()), 4),
    }
    # hybrid: REVIEW -> logistic
    hyb_rej = rej.copy()
    hyb_rej[rev] = np.asarray(prob)[rev] < 0.5
    hyb_scores = np.asarray(scores, float).copy()
    hyb_scores[rev] = hyb_scores[rev] + (0.5 - np.asarray(prob, float)[rev])
    hybrid = {
        "auroc_hybrid_score": round(auroc(y, hyb_scores), 4),
        "auroc_fallback_prob_only": round(auroc_good(y, prob), 4),
        "balacc": round(bal_acc(y, hyb_rej), 4),
        "review_share_before_fallback": round(float(rev.mean()), 4),
        "review_share_after_fallback": 0.0,
        "reject_share": round(float(hyb_rej.mean()), 4),
        "accept_share": round(float((~hyb_rej).mean()), 4),
        "share_decided_deterministically": round(float((~rev).mean()), 4),
        "share_by_model": round(float(rev.mean()), 4),
        "fallback_flips_to_reject": int((rev & (np.asarray(prob) < 0.5)).sum()),
        "fallback_flips_to_accept": int((rev & (np.asarray(prob) >= 0.5)).sum()),
    }
    return pure, hybrid


def boot_auroc_ba(y, s, groups, pred_bad=None, n_boot=N_BOOT, seed=SEED):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); s = np.asarray(s, float); groups = np.asarray(groups)
    pred = None if pred_bad is None else np.asarray(pred_bad, bool)
    ug = sorted(set(groups.tolist()))
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    # same RNG as the Q1 evaluation (numpy default_rng, seed 0)
    rng = np.random.default_rng(seed)
    a, b = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([idx_by[ug[p]] for p in pick])
        if len(set(y[idx].tolist())) < 2:
            continue
        a.append(roc_auc_score(1 - y[idx], s[idx]))
        b.append(bal_acc(y[idx], (s[idx] >= 0.5) if pred is None else pred[idx]))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4),
                    round(float(np.percentile(v, 97.5)), 4)] if v else [None, None]
    return {"auroc_ci95": ci(a), "balacc_ci95": ci(b)}


def paired_boot_auroc(y, s_a, s_b, groups, n_boot=N_BOOT, seed=SEED):
    """Paired group bootstrap on AUROC(a) - AUROC(b), shared resamples."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); s_a = np.asarray(s_a, float); s_b = np.asarray(s_b, float)
    groups = np.asarray(groups)
    ug = sorted(set(groups.tolist()))
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([idx_by[ug[p]] for p in pick])
        if len(set(y[idx].tolist())) < 2:
            continue
        d.append(roc_auc_score(1 - y[idx], s_a[idx]) - roc_auc_score(1 - y[idx], s_b[idx]))
    if not d:
        return {"delta": None, "ci95": [None, None], "n_boot": 0}
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": round(float(auroc(y, s_a) - auroc(y, s_b)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)], "n_boot": len(d)}


def paired_boot_ba(y, pred_a, pred_b, groups, n_boot=N_BOOT, seed=SEED):
    """Paired group bootstrap on balanced-accuracy(a) - balanced-accuracy(b)."""
    y = np.asarray(y); pa = np.asarray(pred_a, bool); pb = np.asarray(pred_b, bool)
    groups = np.asarray(groups)
    ug = sorted(set(groups.tolist()))
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        idx = np.concatenate([idx_by[ug[p]] for p in pick])
        d.append(bal_acc(y[idx], pa[idx]) - bal_acc(y[idx], pb[idx]))
    if not d:
        return {"delta": None, "ci95": [None, None], "n_boot": 0}
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": round(float(bal_acc(y, pa) - bal_acc(y, pb)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)], "n_boot": len(d)}


# ---------------------------------------------------------------- loaders
def load_gold():
    W = json.load(open(os.path.join(BUNDLE, "caches", "gpu", "features.json")))["windows"]
    gold = np.load(os.path.join(BUNDLE, "caches", "perpixel", "gold.npz"), allow_pickle=True)
    phys = np.load(os.path.join(BUNDLE, "caches", "perpixel", "physics_feats.npz"),
                   allow_pickle=True)
    oof = np.load(os.path.join(BUNDLE, "caches", "ensemble", "oof_probs.npz"), allow_pickle=True)
    ids = [str(i) for i in gold["ids"]]
    y = np.asarray(gold["y"], int)
    clips = np.asarray(gold["clips"])
    full_names = list(W[ids[0]]["full"].keys())
    phys_names = [str(n) for n in phys["names"]]
    pidx = {str(k): i for i, k in enumerate(phys["ids"])}
    feats = []
    for i in ids:
        d = {f"full.{n}": float(W[i]["full"][n]) for n in full_names}
        d.update({f"phys.{n}": float(phys["X"][pidx[i], j]) for j, n in enumerate(phys_names)})
        feats.append(d)
    omap = {str(k): j for j, k in enumerate(oof["ids"])}
    prob = np.array([float(oof["p_ens"][omap[i]]) for i in ids])
    return ids, feats, y, clips, prob


def load_ood():
    z = np.load(os.path.join(RES, "inputs", "ood_full.npz"), allow_pickle=True)
    ids = [str(i) for i in z["ids"]]
    names = [str(n) for n in z["names"]]
    X = np.asarray(z["X"], float)
    feats = [{names[j]: X[i, j] for j in range(len(names))} for i in range(len(ids))]
    y = np.asarray(z["y"], int)
    factory = np.asarray(z["factory"])
    probs = {}
    for fn in ("ood_scores.json", "ood_scores2.json"):
        d = json.load(open(os.path.join(BUNDLE, "caches", "ensemble", fn)))
        for s in d["scores"]:
            probs[s["id"]] = float(s["usable_prob"])
    prob = np.array([probs.get(i, np.nan) for i in ids])
    return ids, feats, y, factory, prob


def main():
    t0 = time.perf_counter()
    ids, feats, y, clips, prob = load_gold()
    scores, decisions, per_fold = gold_loco(feats, y, clips)
    g_pure, g_hyb = mode_metrics(y, scores, decisions, prob)
    g_rev = decisions == "REVIEW"
    g_rej = decisions == "REJECT"
    g_hyb_rej = g_rej.copy()
    g_hyb_rej[g_rev] = prob[g_rev] < 0.5
    g_hyb_score = np.where(g_rev, scores + (0.5 - prob), scores)
    g_ci = boot_auroc_ba(y, scores, clips, pred_bad=g_rej)
    g_hyb_ci = boot_auroc_ba(y, g_hyb_score, clips, pred_bad=g_hyb_rej)

    ood_ids, ood_feats, ood_y, factory, ood_prob = load_ood()
    o_scores, o_decisions = score_table(ood_feats, FROZEN)
    o_pure, o_hyb = mode_metrics(ood_y, o_scores, o_decisions, ood_prob)
    o_rev = o_decisions == "REVIEW"
    o_rej = o_decisions == "REJECT"
    o_hyb_rej = o_rej.copy()
    o_hyb_rej[o_rev] = ood_prob[o_rev] < 0.5
    o_hyb_score = np.where(o_rev, o_scores + (0.5 - ood_prob), o_scores)
    o_ci = boot_auroc_ba(ood_y, o_scores, factory, pred_bad=o_rej)
    o_hyb_ci = boot_auroc_ba(ood_y, o_hyb_score, factory, pred_bad=o_hyb_rej)

    # paired group bootstrap on the differences (shared resamples)
    g_paired_auroc = paired_boot_auroc(y, scores, -prob, clips)   # rules - ensemble
    g_paired_ba = paired_boot_ba(y, g_hyb_rej, g_rej, clips)      # hybrid - pure (review-as-accept)
    o_paired_auroc = paired_boot_auroc(ood_y, o_scores, -ood_prob, factory)
    o_paired_ba = paired_boot_ba(ood_y, o_hyb_rej, o_rej, factory)

    # fallback source hashes (bind the hybrid decision to a hashed file)
    gold_fallback_sha = model.sha256_file(
        os.path.join(BUNDLE, "caches", "ensemble", "oof_probs.npz"))
    ood_fallback_sha = {fn: model.sha256_file(
        os.path.join(BUNDLE, "caches", "ensemble", fn))
        for fn in ("ood_scores.json", "ood_scores2.json")}

    gold_per_window = [
        {"id": ids[i], "clip": str(clips[i]), "y": int(y[i]),
         "score": int(scores[i]), "decision": str(decisions[i]),
         "p_ens": round(float(prob[i]), 6),
         "hybrid_decision": "REJECT" if bool(g_hyb_rej[i]) else "ACCEPT"}
        for i in range(len(y))]
    ood_per_window = [
        {"id": ood_ids[i], "factory": str(factory[i]), "y": int(ood_y[i]),
         "score": int(o_scores[i]), "decision": str(o_decisions[i]),
         "usable_prob": round(float(ood_prob[i]), 6),
         "hybrid_decision": "REJECT" if bool(o_hyb_rej[i]) else "ACCEPT"}
        for i in range(len(ood_y))]

    out = {
        "experiment": "Q4.01",
        "title": "Deterministic physics model: runnable package + pure-vs-hybrid comparison",
        "package": {
            "path": "tools/phys",
            "rules_json_sha256": FROZEN["_sha256"],
            "code_sha256": model.code_sha256(),
            "reject_threshold": int(FROZEN["reject_threshold"]),
            "n_rules": len(FROZEN["rules"]),
            "quantisation": int(FROZEN["quantisation"]),
        },
        "fallback": {"model": "ensemble_v1 (frozen logistic, no refit)",
                     "rule": "REVIEW -> accept iff usable_prob >= 0.5",
                     "gold_score": "caches/ensemble/oof_probs.npz p_ens (out-of-fold)",
                     "gold_source_sha256": gold_fallback_sha,
                     "ood_score": "caches/ensemble/ood_scores{,2}.json usable_prob",
                     "ood_source_sha256": ood_fallback_sha,
                     "note": ("the hybrid decision depends on an unhashed-in-code cache; "
                              "these sha256 values bind the fallback source into the "
                              "result and into each receipt's fallback.source_sha256")},
        "leakage": {
            "candidate_rule_list_and_signs": ("tools/phys/rules.json is the Q1 rule table; "
                                              "its candidate rules, features and signs were "
                                              "chosen by the author with all 150 IndEgo "
                                              "labels visible, so the gold LOCO AUROC is an "
                                              "OPTIMISTIC in-domain estimate (see the Q1 "
                                              "report). Fold thresholds are nested."),
            "in_sample_deployment_table": ("the shipped rules.json is derived from all 7 "
                                           "clips and its in-sample AUROC (0.8239) is not a "
                                           "held-out number"),
            "significance": ("paired group-bootstrap CIs on the differences are reported "
                             "below; treat differences whose CI contains 0 as not "
                             "established"),
        },
        "gold_loco": {
            "n": int(len(y)), "n_good": int((y == 1).sum()), "n_bad": int((y == 0).sum()),
            "n_clips": len(set(clips.tolist())),
            "pure_rules": {**g_pure, **g_ci},
            "hybrid_rules_plus_fallback": {**g_hyb, **g_hyb_ci},
            "logistic_oof_auroc": round(auroc_good(y, prob), 4),
            "determinism_cost_auroc": round(auroc_good(y, prob) - g_pure["auroc_deterministic_score"], 4),
            "fallback_buys_balacc": round(g_hyb["balacc"] - g_pure["balacc_review_as_accept"], 4),
            "paired_clip_bootstrap": {
                "auroc_rules_minus_ensemble": g_paired_auroc,
                "balacc_hybrid_minus_pure_review_as_accept": g_paired_ba,
                "note": ("paired clip-grouped bootstrap, 2,000 shared resamples, seed 0; "
                         "AUROC delta = rules - ensemble (both higher=bad), BA delta = "
                         "hybrid - pure(review-as-accept)"),
            },
            "per_fold": per_fold,
            "per_window_scores": gold_per_window,
            "per_window_scores_note": ("one row per gold window: id, clip, y, score, "
                                       "decision, p_ens, hybrid_decision; every metric "
                                       "above is re-derivable from these rows"),
        },
        "ood160": {
            "n": int(len(ood_y)), "n_good": int((ood_y == 1).sum()),
            "n_bad": int((ood_y == 0).sum()),
            "n_factories": int(len(set(factory.tolist()))),
            "protocol": ("frozen IndEgo rule table applied read-only; no OOD fitting. "
                         "Factory-grouped bootstrap CI (n_boot=2000, seed=0) is the honest CI."),
            "pure_rules": {**o_pure, **o_ci},
            "hybrid_rules_plus_fallback": {**o_hyb, **o_hyb_ci},
            "logistic_oof_auroc": round(auroc_good(ood_y, ood_prob), 4),
            "determinism_cost_auroc": round(auroc_good(ood_y, ood_prob) - o_pure["auroc_deterministic_score"], 4),
            "fallback_buys_balacc": round(o_hyb["balacc"] - o_pure["balacc_review_as_accept"], 4),
            "paired_factory_bootstrap": {
                "auroc_rules_minus_ensemble": o_paired_auroc,
                "balacc_hybrid_minus_pure_review_as_accept": o_paired_ba,
                "note": ("paired factory-grouped bootstrap, 2,000 shared resamples, seed 0; "
                         "AUROC delta = rules - ensemble (both higher=bad), BA delta = "
                         "hybrid - pure(review-as-accept)"),
            },
            "per_window_scores": ood_per_window,
            "per_window_scores_note": ("one row per OOD window: id, factory, y, score, "
                                       "decision, usable_prob, hybrid_decision"),
        },
        "wall_s": round(time.perf_counter() - t0, 2),
    }
    with open(os.path.join(RES, "Q4.01.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps({"gold": {k: out["gold_loco"][k] for k in
                               ("pure_rules", "hybrid_rules_plus_fallback",
                                "logistic_oof_auroc", "determinism_cost_auroc",
                                "fallback_buys_balacc")},
                      "ood160": {k: out["ood160"][k] for k in
                                 ("pure_rules", "hybrid_rules_plus_fallback",
                                  "logistic_oof_auroc", "determinism_cost_auroc",
                                  "fallback_buys_balacc")}}, indent=1))
    print("wrote", os.path.join(RES, "Q4.01.json"))


if __name__ == "__main__":
    main()
