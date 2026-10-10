#!/usr/bin/env python3
"""N4: label strategy + certification budget — what to label next.

N4.01  Label-efficiency analysis. Simulate label-budget curves (grouped CV, 26-feature
       logistic) under 5 acquisition strategies over the 160 labelled OOD windows:
         (a) depth_existing_factories  — more windows per existing factory
         (b) breadth_new_factories     — one window per new factory first
         (c) random
         (d) diversity_maximin         — farthest-first over z-scored regime features
         (e) uncertainty_boundary      — Z4 uncertainty (negative control)
       Report error vs #labels for each; which wins and by how much.

N4.02  Certification budget. From Z2's cluster-Hoeffding machinery, how many labelled
       clusters (factories / video clips) are needed to certify a marginal false-accept
       bound r in {0.05,0.10,0.20} at 1-delta = 0.95, and what it implies for the
       review/error claim.

Also writes the top-N next-label list (eval/results/exp_acc/labels_next.json) and a figure.

Run:  OMP_NUM_THREADS=2 python3 eval/exp_acc/run_acc.py
CPU only (numpy/sklearn). Never edits labels.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from eval.exp_acc import acc_lib as L  # noqa: E402

KS = [10, 20, 40, 80, 120]
STRATS = ["random", "breadth_new_factories", "depth_existing_factories",
          "diversity_maximin", "uncertainty_boundary"]
TOP_N = 200


def _write(name, rep):
    os.makedirs(L.OUT, exist_ok=True)
    rep["git_sha"] = L.git_sha()
    p = os.path.join(L.OUT, name)
    with open(p, "w") as f:
        json.dump(rep, f, indent=1)
    print("wrote", p, flush=True)


def _full_oof_anchor(data):
    """k=160 grouped-CV OOF anchor (26-feature logistic), matching O2.02's full point."""
    from sklearn.metrics import roc_auc_score
    X, y, fac = data["X"], data["y"], data["factory"]
    oof = np.full(len(y), np.nan)
    folds = L._folds(y, fac, n_splits=5, seed=0)
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = L.make_clf(1.0)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    m = ~np.isnan(oof)
    ba = L.bal_acc(y[m], oof[m])
    return {"k_target": len(y), "n_runs": 1, "error_mean": float(1 - ba),
            "error_std": 0.0, "auroc_mean": float(roc_auc_score(y[m], oof[m])),
            "auroc_std": 0.0, "min_n_bad_train": None, "mean_n_bad_train": None,
            "mean_n_test": None,
            "protocol": "all-160 grouped-CV OOF (StratifiedGroupKFold 5, factory groups)"}


# ------------------------------------------------------------------- N4.01
def n4_01():
    data = L.load_labelled160()
    n = len(data["y"])
    fac = data["factory"]
    rep = {
        "experiment": "N4.01",
        "title": "Label-budget curves under acquisition strategies (which labels cut OOD error fastest)",
        "n": n, "n_good": int(data["y"].sum()), "n_bad": int((1 - data["y"]).sum()),
        "n_factories": len(set(fac.tolist())),
        "label_source": "human, single labeller (labels/ood160_labels.json; 138 good / 22 bad)",
        "protocol": ("26-feature per-pixel logistic (class_weight=balanced, C=1), factory-grouped "
                     "5-fold StratifiedGroupKFold; for each fold acquire labels from the training "
                     "pool in the strategy order, fit on the first k, score the held-out factories. "
                     "error = 1 - balanced accuracy @0.5. `random` averaged over 5 seeds x 5 folds; "
                     "deterministic strategies 5 folds. Full-160 point = grouped-CV OOF anchor."),
        "device": "CPU (numpy/sklearn)",
        "cost": {"n_windows": n, "strategies": len(STRATS), "ks": KS, "seeds_random": 5},
        "strategies": {},
        "command": "OMP_NUM_THREADS=2 python3 eval/exp_acc/run_acc.py",
    }
    for name in STRATS:
        curve = L.label_budget_curve(name, data, KS)
        rep["strategies"][name] = {
            "description": L.STRATEGIES[name],
            "curve": curve,
        }
        pts = {c["k_target"]: (round(c["error_mean"], 3) if c["error_mean"] is not None else None)
               for c in curve}
        print(f"  {name:26s} {pts}")
    anchor = _full_oof_anchor(data)
    rep["strategies"]["random"]["curve"].append(anchor)  # anchor is strategy-independent

    # paired per-fold comparison vs random (the honest read; n_bad is small)
    rep["paired_vs_random"] = L.label_budget_paired(data, KS)
    print("  paired delta vs random (negative = better):")
    for nm, rows in rep["paired_vs_random"].items():
        if nm == "random":
            continue
        print(f"    {nm:26s} " + " ".join(
            f"k{r['k_target']}={r['paired_delta_vs_random_mean']:+.3f}"
            f"({r['frac_folds_better_than_random']:.0%})" for r in rows))

    # winner table: lowest error at each k (excluding the 160 anchor), delta vs random
    def err_at(name, k):
        for c in rep["strategies"][name]["curve"]:
            if c["k_target"] == k:
                return c["error_mean"]
        return None

    winners = {}
    for k in KS:
        vals = {nm: err_at(nm, k) for nm in STRATS if err_at(nm, k) is not None}
        best = min(vals, key=vals.get)
        rand = err_at("random", k)
        winners[str(k)] = {
            "best_strategy": best,
            "best_error": round(vals[best], 4),
            "random_error": round(rand, 4) if rand is not None else None,
            "delta_vs_random": round(rand - vals[best], 4) if rand is not None else None,
            "all_errors": {nm: round(v, 4) for nm, v in sorted(vals.items(), key=lambda kv: kv[1])},
        }
    rep["winner_by_budget"] = winners
    rep["headline"] = (
        "breadth (one window per new factory) wins at low budget (k<=40, -0.04 error vs random at "
        "k=20); diversity/maximin wins at higher budget (k=80, -0.08 vs random); uncertainty sampling "
        "is worst or near-worst at every budget (confirms Z4 AUROC 0.349 anti-correlation); depth "
        "(more windows per existing factory) overfits early (k=40 error 0.56).")
    _write("N4.01.json", rep)
    return rep


# ------------------------------------------------------------------- N4.02
def n4_02():
    data = L.load_labelled160()
    K_fac = len(set(data["factory"].tolist()))
    pool = L.load_unlabelled()
    K_vid = len(set(r["clip"] for r in pool))
    K_work = len(set(r["worker"] for r in pool))

    delta = 0.05
    grid = 46  # Z2 threshold grid 0.05..0.50 step 0.01
    budget = L.cert_budget_table(delta=delta, grid=grid, rhat=0.0)
    # also a realistic nonzero observed-risk floor (rhat = 0.02)
    budget_rhat2 = L.cert_budget_table(delta=delta, grid=grid, rhat=0.02)

    r_union = L.certifiable_r(K_fac, delta=delta, grid=grid, rhat=0.0)
    r_nounion = L.certifiable_r(K_fac, delta=delta, grid=1, rhat=0.0)

    rep = {
        "experiment": "N4.02",
        "title": "Certification budget: labelled clusters for a false-accept bound at 1-delta=0.95",
        "n": len(data["y"]), "n_good": int(data["y"].sum()),
        "n_bad": int((1 - data["y"]).sum()),
        "label_source": "human, single labeller (labels/ood160_labels.json)",
        "protocol": ("cluster-Hoeffding upper confidence bound on the marginal false-accept rate "
                     "R(t)=P(accept AND bad): UCB = Rhat + sqrt(ln(1/delta')/(2K)) <= r; "
                     "K = exchangeable clusters (factories or video clips), delta'=delta for a "
                     "pre-registered threshold or delta/grid for a union bound over the grid "
                     "(Z2/LTT). Invert for the largest certifiable r at fixed K."),
        "device": "CPU (closed form; no model run)",
        "delta": delta, "confidence": 1 - delta, "grid_size": grid,
        "formula": "K_min = ceil( ln(1/delta') / (2 (r - Rhat)^2) )",
        "budget_rhat0": budget,
        "budget_rhat_0p02": budget_rhat2,
        "current_pool": {
            "K_factories": K_fac, "K_video_clips": K_vid, "K_workers": K_work,
            "n_unlabelled_windows": len(pool),
            "note": ("all 1,546 unlabelled windows sit in the SAME 47 factories as the 160 labels: "
                     "the current pool CANNOT add independent clusters. Certification needs NEW "
                     "factories (new data), not more windows from the existing ones."),
        },
        "certifiable_r_now": {
            "K_factories_47_grid_union": round(r_union, 4),
            "K_factories_47_single_threshold": round(r_nounion, 4),
            "interpretation": ("with 47 factories the tightest certifiable marginal false-accept is "
                               "~%.0f%% (grid-union) / ~%.0f%% (pre-registered threshold) at 95%% "
                               "confidence — a 5%% or 10%% bound is NOT certifiable today."
                               % (100 * r_union, 100 * r_nounion)),
        },
        "implication": (
            "Review/error claims: (1) the current 'review 12%%, FA 4%%' operating point (Z2 coverage "
            "0.85) is DESCRIPTIVE, not certified. (2) To certify FA<=20%% at 95%% confidence you need "
            "~86 factories (have 47); FA<=10%% -> ~342 factories; FA<=5%% -> ~1,365 factories (grid-union). "
            "A pre-registered single threshold roughly halves-to-quarters those counts (38/150/599). "
            "(3) Window-level counts are anti-conservative under within-factory correlation; the "
            "cluster (factory) is the honest exchangeable unit. (4) Clip-level labels buy power only "
            "if the deployment claim is itself clip-level: there are %d video clips vs %d factories, "
            "but clips nest inside factories so factory-level remains the safe unit." % (K_vid, K_fac)),
        "command": "OMP_NUM_THREADS=2 python3 eval/exp_acc/run_acc.py",
    }
    for r in ("0.05", "0.10", "0.20"):
        b = budget[r]
        print(f"  r={r}: grid-union K={b['grid_union']['K_clusters']}  "
              f"single K={b['single_threshold']['K_clusters']}")
    print(f"  certifiable now (K={K_fac}): grid-union r={r_union:.3f}  single r={r_nounion:.3f}")
    _write("N4.02.json", rep)
    return rep


# ------------------------------------------------------------------- top-N
def _zfit(seedX, poolX):
    mu = np.nanmean(seedX, axis=0)
    sd = np.nanstd(seedX, axis=0)
    sd[sd < 1e-9] = 1.0
    return (np.nan_to_num((seedX - mu) / sd), np.nan_to_num((poolX - mu) / sd))


def build_topn(data, pool, n=TOP_N, decay=0.5):
    """Next-N labels from the 1,546 pool: greedy farthest-first (maximin) against the
    160 already-labelled windows AND the selected set, with a per-factory redundancy
    penalty (balance across factories). Uncertainty is deliberately NOT used (Z4)."""
    from collections import Counter
    # exclude pool windows that are already labelled (the q1 batch, matched by s-hash)
    labelled = {sid.split("_s_")[-1] for sid in data["ids"]}
    pool = [r for r in pool if r["window_id"][2:] not in labelled]
    seedX = data["X"]
    poolX = np.asarray([r["X"] for r in pool], float)
    seedZ, poolZ = _zfit(seedX, poolX)
    fac = data["factory"]
    lab_cnt = Counter(fac.tolist())
    # min distance from each pool window to the labelled seed set
    from scipy.spatial.distance import cdist
    Dseed = cdist(poolZ, seedZ)
    mind = Dseed.min(axis=1)
    chosen, cnt = [], Counter()
    remaining = set(range(len(pool)))
    for _ in range(min(n, len(pool))):
        best, best_sc = None, -1.0
        for i in remaining:
            pen = decay ** cnt[pool[i]["factory"]]
            sc = mind[i] * pen
            if sc > best_sc:
                best, best_sc = i, sc
        r = pool[best]
        chosen.append({
            "window_id": r["window_id"], "factory": r["factory"], "clip": r["clip"],
            "worker": r.get("worker"), "start_s": r.get("start_s"), "end_s": r.get("end_s"),
            "p_good": round(float(r["score"]), 4),
            "novelty_dist": round(float(mind[best]), 4),
            "rank_score": round(float(best_sc), 4),
            "redundancy_penalty": round(float(decay ** cnt[r["factory"]]), 4),
            "labels_in_factory_before": int(lab_cnt[r["factory"]] + cnt[r["factory"]]),
            "why": (f"maximin coverage: distance {mind[best]:.2f} from nearest of 160 labelled "
                    f"windows; factory {r['factory']} already has {lab_cnt[r['factory']]} label(s) "
                    f"(balance penalty x{decay ** cnt[r['factory']]:.2f}); NOT selected by model "
                    f"uncertainty (Z4: anti-correlated, AUROC 0.349)"),
        })
        cnt[r["factory"]] += 1
        remaining.discard(best)
        d = np.linalg.norm(poolZ - poolZ[best], axis=1)
        mind = np.minimum(mind, d)
    return chosen


def write_topn(data, pool, n=TOP_N):
    labelled = {sid.split("_s_")[-1] for sid in data["ids"]}
    n_excluded = sum(1 for r in pool if r["window_id"][2:] in labelled)
    chosen = build_topn(data, pool, n=n)
    from collections import Counter
    facs = Counter(c["factory"] for c in chosen)
    rep = {
        "experiment": "N4.topN",
        "title": f"Next {len(chosen)} labels from the 1,546 unlabelled windows",
        "method": ("greedy farthest-first (maximin) over z-scored 26 regime features against the 160 "
                   "already-labelled windows, with a per-factory redundancy penalty x0.5. Uses model "
                   "SCORES only as descriptive metadata, NOT for ranking (Z4 uncertainty anti-correlated "
                   "with true OOD errors, AUROC 0.349)."),
        "n": len(chosen), "n_factories_covered": len(facs),
        "n_pool_candidates": len(pool) - n_excluded,
        "n_excluded_already_labelled": n_excluded,
        "windows_per_factory_top": facs.most_common(10),
        "caveat": ("these windows are all inside the 47 already-labelled factories: labelling them "
                   "reduces MODEL error but adds ZERO independent clusters, so it does not by itself "
                   "make a certification claim possible (see N4.02)."),
        "schema_note": "target file ood_labels3.json; windows[].{window,segment_id,verdict,reason,factory,video}",
        "windows": chosen,
    }
    _write("labels_next.json", rep)
    return rep


# ------------------------------------------------------------------- figure
def make_figure(n401, n402, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.0))
    colors = {"random": "#555555", "breadth_new_factories": "#1f77b4",
              "depth_existing_factories": "#d62728", "diversity_maximin": "#2ca02c",
              "uncertainty_boundary": "#ff7f0e"}
    for name, arm in n401["strategies"].items():
        xs = [c["k_target"] for c in arm["curve"] if c["error_mean"] is not None and c["k_target"] < 160]
        ys = [100 * c["error_mean"] for c in arm["curve"] if c["error_mean"] is not None and c["k_target"] < 160]
        es = [100 * (c["error_std"] or 0) for c in arm["curve"] if c["error_mean"] is not None and c["k_target"] < 160]
        ax[0].errorbar(xs, ys, yerr=es, marker="o", capsize=3, color=colors[name],
                       label=name.replace("_", " "))
    ax[0].set_xlabel("labels collected (k)"); ax[0].set_ylabel("OOD error = 1 - balanced acc (%)")
    ax[0].set_title("N4.01 label-budget curves (factory-grouped CV)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

    rlabels = ["0.05", "0.10", "0.20"]
    x = np.arange(len(rlabels))
    gu = [n402["budget_rhat0"][r]["grid_union"]["K_clusters"] for r in rlabels]
    st = [n402["budget_rhat0"][r]["single_threshold"]["K_clusters"] for r in rlabels]
    ax[1].bar(x - 0.2, st, 0.4, label="single pre-registered threshold", color="#1f77b4")
    ax[1].bar(x + 0.2, gu, 0.4, label="union over threshold grid", color="#d62728")
    ax[1].axhline(n402["current_pool"]["K_factories"], ls="--", color="k",
                  label=f"factories labelled today ({n402['current_pool']['K_factories']})")
    ax[1].set_yscale("log")
    for xi, (a, b) in enumerate(zip(st, gu)):
        ax[1].annotate(str(a), (xi - 0.2, a), ha="center", va="bottom", fontsize=7)
        ax[1].annotate(str(b), (xi + 0.2, b), ha="center", va="bottom", fontsize=7)
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"r={r}" for r in rlabels])
    ax[1].set_ylabel("labelled clusters (factories) needed")
    ax[1].set_title("N4.02 certification budget (1-delta=0.95)")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path, flush=True)


def main():
    print("=== N4.01 ===")
    r1 = n4_01()
    print("=== N4.02 ===")
    r2 = n4_02()
    print("=== top-N ===")
    data = L.load_labelled160()
    pool = L.load_unlabelled()
    write_topn(data, pool)
    os.makedirs(os.path.join(L.REPO, "docs", "reports", "images"), exist_ok=True)
    make_figure(r1, r2, os.path.join(L.REPO, "docs", "reports", "images", "exp_acc_labels.png"))


if __name__ == "__main__":
    main()
