#!/usr/bin/env python3
"""Z4 -- Clip-level sequential decisions + next-100 active labels.

Three sub-experiments (all CPU-only, cached data, no GPU):

  Z4.01  Clip-level aggregation.  From window scores build clip-level decisions with
         mean / median / q75 / q90 / rank-mean aggregators of the bad-score (1 - p_ens),
         calibrate the clip level with leave-one-clip-out (threshold + isotonic), report
         clip AUROC / balanced accuracy, and the *window-level review* (fraction of
         windows whose true label a human confirms) needed to reach a target clip error.

  Z4.02  Sequential test.  Wald-style SPRT on calibrated log-odds: process the windows
         of a clip one at a time in time order, stop when the cumulative evidence
         crosses an accept / reject bound; if the clip is exhausted first -> abstain.
         Sweep (A, B) -> clip error vs average windows processed vs abstain rate.

  Z4.03  Best cell + active labels.  Cheapest sequential policy with clip-verdict
         agreement >= 0.90 and abstain <= 0.40, plus the ranking of the 1,546
         (1,486 not-yet-labelled) Egocentric-10K windows by expected information for
         reducing OOD error (uncertainty near the boundary, penalised for clip/factory
         redundancy) -> top 100 window ids.

Data (read-only):
  labels/manifest_gold.json            150 IndEgo windows / 7 clips (human labels)
  caches/ensemble/oof_probs.npz        LOCO OOF ensemble probabilities
  /workspace/artifacts/windows_rescored.jsonl   1,546 Egocentric-10K windows (unlabelled)
  labels/ood_labels.json               60 labelled Egocentric-10K windows (EVAL ONLY)

Caveat: n = 7 in-domain clips; the OOD set has n_bad = 9.  Both are far too small for
certified bounds -- every number here is descriptive.

Regen:
  OMP_NUM_THREADS=2 python3 eval/exp_clip/run_clip.py --out eval/results/exp_clip
"""
import argparse, json, math, os
from collections import Counter, defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
MANIFEST = f"{BUNDLE}/labels/manifest_gold.json"
OOF = f"{BUNDLE}/caches/ensemble/oof_probs.npz"
OOD = f"{BUNDLE}/labels/ood_labels.json"
RESCORED = "/workspace/artifacts/windows_rescored.jsonl"

THRESH = 0.30          # clip verdict: bad fraction >= 0.30 -> reject
THRESH2 = 0.50
CLIP_ORDER = ["desk", "ecu_full", "ecu_late", "logistics4", "organise", "pallet", "pallet2"]


# --------------------------------------------------------------------------- data
def load_gold():
    m = json.load(open(MANIFEST))
    oof = np.load(OOF, allow_pickle=True)
    items = m["items"]
    assert list(oof["ids"]) == [it["id"] for it in items], "oof order != manifest"
    clip = np.array([it["clip"] for it in items])
    start = np.array([it["start_s"] for it in items], float)
    y = np.array([it["label"] for it in items], int)          # 1 = good
    p_ens = np.asarray(oof["p_ens"], float)                   # P(good)
    p_platt = np.asarray(oof["p_platt"], float)
    assert set(clip) == set(CLIP_ORDER)
    return m, clip, start, y, p_ens, p_platt


def clip_truth(clip, y):
    truth_frac, verdict = {}, {}
    for c in CLIP_ORDER:
        m = clip == c
        bf = float(np.mean(1 - y[m]))
        truth_frac[c] = bf
        verdict[c] = int(bf >= THRESH)          # 1 = reject (bad)
    return truth_frac, verdict


def clip_bootstrap_ci(vals, n_rep=2000, seed=11):
    rng = np.random.RandomState(seed)
    v = np.asarray(vals, float)
    if len(v) == 0:
        return [None, None]
    b = [float(np.mean(rng.choice(v, len(v), replace=True))) for _ in range(n_rep)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def bal_acc(labels, preds):
    labels = np.asarray(labels, int)
    preds = np.asarray(preds, int)
    tps = np.sum((preds == 1) & (labels == 1))
    tns = np.sum((preds == 0) & (labels == 0))
    pos, neg = np.sum(labels == 1), np.sum(labels == 0)
    sens = tps / pos if pos else float("nan")
    spec = tns / neg if neg else float("nan")
    return float((sens + spec) / 2), float(sens), float(spec)


# ------------------------------------------------------------------- Z4.01 agg
def aggregate(p_bad, clip, method):
    """Return clip -> aggregated bad-score (higher = more likely reject)."""
    ranks = None
    if method == "rankmean":
        order = np.argsort(np.argsort(p_bad))     # 0-based ranks
        ranks = order / max(len(p_bad) - 1, 1)
    out = {}
    for c in CLIP_ORDER:
        v = p_bad[clip == c]
        if method == "mean":
            out[c] = float(np.mean(v))
        elif method == "median":
            out[c] = float(np.median(v))
        elif method == "q75":
            out[c] = float(np.quantile(v, 0.75))
        elif method == "q90":
            out[c] = float(np.quantile(v, 0.90))
        elif method == "trimmed":
            vv = np.sort(v)[1:-1] if len(v) > 2 else v
            out[c] = float(np.mean(vv))
        elif method == "rankmean":
            out[c] = float(np.mean(ranks[clip == c]))
        else:
            raise ValueError(method)
    return out


def loco_threshold_ba(scores, truth, grid=None):
    """LOCO: fit threshold on the 6 other clips, predict the held-out clip."""
    clips = list(CLIP_ORDER)
    preds, chosen = {}, {}
    for c in clips:
        tr = [k for k in clips if k != c]
        sv = np.array([scores[k] for k in tr])
        tv = np.array([truth[k] for k in tr])
        if tv.min() == tv.max():                   # degenerate train set
            t = float(np.median(sv))
        else:
            cand = np.unique(np.concatenate([sv, [sv.max() + 1e-9]]))
            best, t = -1, float(np.median(sv))
            for tt in cand:
                ba, _, _ = bal_acc(tv, (sv >= tt).astype(int))
                if ba > best:
                    best, t = ba, float(tt)
        preds[c] = int(scores[c] >= t)
        chosen[c] = t
    labels = np.array([truth[k] for k in clips])
    predv = np.array([preds[k] for k in clips])
    ba, sens, spec = bal_acc(labels, predv)
    return {"loco_ba": ba, "sensitivity": sens, "specificity": spec,
            "n_correct": int(np.sum(predv == labels)), "n": len(clips),
            "predictions": preds, "thresholds": chosen}


def loco_isotonic_ba(scores, truth):
    """LOCO isotonic calibration of aggregate -> probability, threshold at 0.30
    on the *bad fraction* scale (i.e. predicted frac >= THRESH)."""
    clips = list(CLIP_ORDER)
    preds, pred_frac = {}, {}
    for c in clips:
        tr = [k for k in clips if k != c]
        sv = np.array([scores[k] for k in tr])
        tv = np.array([truth[k] for k in tr], float)     # binary 0/1 here
        if len(np.unique(sv)) < 2:
            pt = float(np.mean(tv))
        else:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(sv, tv)
            pt = float(iso.predict([scores[c]])[0])
        pred_frac[c] = pt
        preds[c] = int(pt >= THRESH)
    labels = np.array([truth[k] for k in clips])
    predv = np.array([preds[k] for k in clips])
    ba, sens, spec = bal_acc(labels, predv)
    return {"loco_ba": ba, "sensitivity": sens, "specificity": spec,
            "n_correct": int(np.sum(predv == labels)), "n": len(clips),
            "predictions": preds, "pred_frac": pred_frac}


def review_curve(p_ens, clip, y, per_clip=True):
    """Fraction of windows a human reviews (highest-uncertainty first) -> clip verdict
    error.  Reviewed windows contribute their TRUE label; the rest the model prob.
    per_clip=True spreads the review budget evenly over clips (a fairer proxy when clip
    sizes differ)."""
    unn = 1.0 - np.abs(2.0 * p_ens - 1.0)               # closeness to boundary
    truth_frac, verdict = clip_truth(clip, y)
    order = {c: np.where(clip == c)[0][np.argsort(-unn[clip == c])] for c in CLIP_ORDER}
    out = []
    qs = np.round(np.linspace(0, 1, 21), 3)
    for q in qs:
        errs, preds = [], {}
        review_mask = np.zeros(len(y), bool)
        for c in CLIP_ORDER:
            idx = order[c]
            k = int(round(q * len(idx)))
            review_mask[idx[:k]] = True
        for c in CLIP_ORDER:
            idx = np.where(clip == c)[0]
            est = np.where(review_mask[idx], (1 - y[idx]).astype(float), 1 - p_ens[idx])
            preds[c] = int(np.mean(est) >= THRESH)
        n_correct = sum(preds[c] == verdict[c] for c in CLIP_ORDER)
        out.append({"review_frac": float(q), "n_reviewed": int(review_mask.sum()),
                    "agreement": n_correct / len(CLIP_ORDER),
                    "error": 1 - n_correct / len(CLIP_ORDER)})
    return out


def min_review_for_error(curve, tol):
    for r in curve:
        if r["error"] <= tol + 1e-9:
            return r["review_frac"]
    return None


# ------------------------------------------------------------------- Z4.02 SPRT
def make_llr(q_bad, mode, theta0=0.15, theta1=0.45, p_thresh=0.5):
    """Per-window log-likelihood-ratio increment for the clip-level SPRT.

    mode='hard' : binarise the calibrated bad-probability at 0.5  (classic Wald SPRT)
    mode='soft' : use the calibrated probability directly (expected LLR)
    """
    lp = math.log(theta1 / theta0)
    ln = math.log((1 - theta1) / (1 - theta0))
    q = np.asarray(q_bad, float)
    if mode == "hard":
        yb = (q >= p_thresh).astype(float)          # 1 = bad window
        return yb * lp + (1 - yb) * ln
    q = np.clip(q, 1e-4, 1 - 1e-4)
    return q * lp + (1 - q) * ln


def seq_run(clip, start, llr, A, B):
    """Wald SPRT: cumulative S over windows in time order; stop reject if S >= A,
    accept if S <= -B, else abstain when the clip's windows are exhausted."""
    res = {}
    for c in CLIP_ORDER:
        idx = np.where(clip == c)[0]
        idx = idx[np.argsort(start[idx])]
        S, k, dec, trace = 0.0, 0, "abstain", []
        for i in idx:
            S += float(llr[i])
            k += 1
            trace.append(S)
            if S >= A:
                dec = "reject"; break
            if S <= -B:
                dec = "accept"; break
        res[c] = {"decision": dec, "k": int(k), "n": int(len(idx)), "S": float(S),
                  "trace": [float(x) for x in trace]}
    return res


def seq_metrics(res, truth):
    dec = [c for c in CLIP_ORDER if res[c]["decision"] in ("accept", "reject")]
    ab = [c for c in CLIP_ORDER if res[c]["decision"] == "abstain"]
    n = len(CLIP_ORDER)
    kfrac_all = np.mean([res[c]["k"] / res[c]["n"] for c in CLIP_ORDER])
    kfrac_dec = np.mean([res[c]["k"] / res[c]["n"] for c in dec]) if dec else None
    kmean_all = np.mean([res[c]["k"] for c in CLIP_ORDER])
    agree = (sum(int((res[c]["decision"] == "reject") == bool(truth[c])) for c in dec) / len(dec)
             if dec else None)
    wrong = [c for c in dec if int(res[c]["decision"] == "reject") != bool(truth[c])]
    return {
        "abstain_rate": len(ab) / n,
        "abstained": ab,
        "n_decided": len(dec),
        "decided": dec,
        "agreement_decided": agree,
        "wrong": wrong,
        "error_decided": (1 - agree) if agree is not None else None,
        # conservative: wrong + abstained count as not-agreeing
        "agreement_all_conservative": (len(dec) - len(wrong)) / n,
        "mean_windows_processed_frac": float(kfrac_all),
        "mean_windows_processed_frac_decided": float(kfrac_dec) if kfrac_dec is not None else None,
        "mean_windows_processed": float(kmean_all),
        "mean_windows_processed_decided": float(np.mean([res[c]["k"] for c in dec])) if dec else None,
    }


def seq_frontier(clip, start, llr, truth, grid_A, grid_B):
    rows = []
    for A in grid_A:
        for B in grid_B:
            res = seq_run(clip, start, llr, A, B)
            m = seq_metrics(res, truth)
            m.update({"A": float(A), "B": float(B),
                      "per_clip": {c: {"decision": res[c]["decision"], "k": res[c]["k"],
                                       "n": res[c]["n"]} for c in CLIP_ORDER}})
            rows.append(m)
    return rows


def pareto_frontier(rows, ykey="mean_windows_processed_frac", errkey="error_decided"):
    """Pareto: minimise processed fraction AND clip error (abstentions penalised by using
    conservative agreement when error_decided is None)."""
    pts = []
    for r in rows:
        if r["n_decided"] == 0:
            continue
        err = r[errkey] if r[errkey] is not None else 1.0
        pts.append((r[ykey], err, r))
    pts.sort(key=lambda t: (t[0], t[1]))
    front, best_err = [], 2.0
    for x, y, r in pts:
        if y < best_err - 1e-12:
            front.append(r)
            best_err = y
    return front


# --------------------------------------------------------------- Z4.03 active
def load_rescored():
    rows = []
    with open(RESCORED) as f:
        for line in f:
            d = json.loads(line)
            if d.get("type") == "header":
                continue
            rows.append(d)
    return rows


def ood_label_set():
    d = json.load(open(OOD))["latest"]
    return {k.split("_", 1)[1]: v["verdict"] for k, v in d.items()}


def rank_active(rows, exclude_ids, per_clip_cap=None, decay_fac=0.5, decay_clip=0.5, top=100):
    """Greedy diversified uncertainty sampling.

    raw utility = closeness-to-boundary  U = 4 p (1-p)   (max at p = 0.5)
                  + 0.5 * two-view disagreement |p_eg2 - p_pixel|   (epistemic)
    redundancy penalty = decay_fac ** (#already selected in same factory)
                       * decay_clip ** (#already selected in same clip)
    """
    cand = []
    for r in rows:
        if r["window_id"] in exclude_ids:
            continue
        p = float(r["score"])
        pe = float(r["features"]["p_eg2"])
        pp = float(r["features"]["p_pixel"])
        U = 4.0 * p * (1.0 - p)
        D = abs(pe - pp)
        raw = U + 0.5 * D
        cand.append({"window_id": r["window_id"], "clip": r["clip"], "factory": r["factory"],
                     "p_good": round(p, 4), "uncertainty": round(U, 4),
                     "view_disagreement": round(D, 4), "raw_info": round(raw, 4),
                     "verdict": r["verdict"]})
    by_id = {c["window_id"]: c for c in cand}
    selected = []
    cnt_fac, cnt_clip = Counter(), Counter()
    remaining = set(by_id)
    for _ in range(min(top, len(remaining))):
        best, best_sc = None, -1.0
        for wid in remaining:
            c = by_id[wid]
            pen = (decay_fac ** cnt_fac[c["factory"]]) * (decay_clip ** cnt_clip[c["clip"]])
            sc = c["raw_info"] * pen
            if sc > best_sc:
                best, best_sc = wid, sc
        c = by_id[best]
        pen = (decay_fac ** cnt_fac[c["factory"]]) * (decay_clip ** cnt_clip[c["clip"]])
        sel = dict(c)
        sel["rank_score"] = round(best_sc, 5)
        sel["redundancy_penalty"] = round(pen, 5)
        sel["rationale"] = (f"uncertainty U={c['uncertainty']:.3f} (p={c['p_good']:.2f}), "
                            f"view-gap={c['view_disagreement']:.2f}, "
                            f"factory {c['factory']} x{cnt_fac[c['factory']]+1}, "
                            f"clip x{cnt_clip[c['clip']]+1}")
        selected.append(sel)
        cnt_fac[c["factory"]] += 1
        cnt_clip[c["clip"]] += 1
        remaining.discard(best)
    return selected, cand


# ------------------------------------------------------------------- figures
def fig_aggregation(outdir, agg_report, p_bad, clip, truth, clip_verdict, review_by_method):
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    methods = [m for m in agg_report]
    x = np.arange(len(methods))
    axes[0].bar(x - 0.2, [agg_report[m]["auroc"] for m in methods], 0.4, label="clip AUROC")
    axes[0].bar(x + 0.2, [agg_report[m]["loco_threshold_ba"]["loco_ba"] for m in methods],
                0.4, label="LOCO thresh BA")
    axes[0].axhline(0.5, ls=":", color="grey")
    axes[0].set_xticks(x); axes[0].set_xticklabels(methods, rotation=30, ha="right")
    axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=8)
    axes[0].set_title("clip AUROC / BA by aggregator (n=7 clips)")

    for name in ["mean", "q75", "rankmean"]:
        p = np.array([truth[c] for c in CLIP_ORDER])
        s = np.array([agg_report[name]["scores"][c] for c in CLIP_ORDER])
        axes[1].scatter(p, s, s=28, label=name)
    axes[1].axhline(THRESH, ls="--", color="k", lw=1)
    axes[1].axvline(THRESH, ls="--", color="k", lw=1)
    axes[1].set_xlabel("clip true bad fraction"); axes[1].set_ylabel("aggregate bad-score")
    axes[1].legend(fontsize=8); axes[1].set_title("clip aggregate vs truth")

    for name, curve in review_by_method.items():
        axes[2].plot([r["review_frac"] for r in curve],
                     [r["error"] for r in curve], marker=".", label=name)
    axes[2].set_xlabel("window review fraction (per clip)")
    axes[2].set_ylabel("clip verdict error")
    axes[2].legend(fontsize=8); axes[2].set_title("review needed for clip error")
    fig.tight_layout()
    fig.savefig(f"{outdir}/clip_aggregation.png", bbox_inches="tight", dpi=110)
    plt.close(fig)


def fig_frontier(outdir, rows, best, every2, truth):
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    err = [r["error_decided"] if r["error_decided"] is not None else 1.0 for r in rows]
    xf = [r["mean_windows_processed_frac"] for r in rows]
    ab = [r["abstain_rate"] for r in rows]
    sc = axes[0].scatter(xf, err, c=ab, cmap="viridis", s=40, edgecolor="k", linewidth=0.3)
    plt.colorbar(sc, ax=axes[0], label="abstain rate")
    axes[0].scatter([every2[0]], [every2[1]], marker="*", s=200, c="red",
                    label="every-2 fixed (baseline)", zorder=5)
    if best:
        axes[0].scatter([best["mean_windows_processed_frac"]],
                        [best["error_decided"] if best["error_decided"] is not None else 1.0],
                        marker="D", s=90, facecolor="none", edgecolor="red", linewidth=2,
                        label="best cell", zorder=6)
    axes[0].set_xlabel("mean windows processed (fraction of clip)")
    axes[0].set_ylabel("clip verdict error (decided only)")
    axes[0].legend(fontsize=8); axes[0].set_title("SPRT frontier: error vs cost (colour=abstain)")

    # per-clip traces of the chosen policy -- recompute stored? use per_clip k/n
    labels = [c for c in CLIP_ORDER]
    kn = [best["per_clip"][c]["k"] / best["per_clip"][c]["n"] for c in labels] if best else []
    if best:
        cols = ["#d62728" if truth[c] else "#2ca02c" for c in labels]
        axes[1].bar(range(len(labels)), kn, color=cols)
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels([f"{c}\n{'BAD' if truth[c] else 'ok'}" for c in labels],
                                rotation=30, ha="right", fontsize=8)
        axes[1].axhline(best["mean_windows_processed_frac"], ls="--", color="k",
                        label=f"mean {best['mean_windows_processed_frac']:.2f}")
        axes[1].set_ylabel("windows processed fraction"); axes[1].legend(fontsize=8)
        axes[1].set_title("best cell: per-clip stop point")
    fig.tight_layout()
    fig.savefig(f"{outdir}/sequential_frontier.png", bbox_inches="tight", dpi=110)
    plt.close(fig)


def fig_active(outdir, selected, cand, ood_eval):
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    us = [c["uncertainty"] for c in cand]
    axes[0].hist(us, bins=30, color="#1f77b4")
    axes[0].axvline(np.mean([s["uncertainty"] for s in selected]), color="red",
                    label="top-100 mean")
    axes[0].set_xlabel("uncertainty 4p(1-p)"); axes[0].set_ylabel("windows")
    axes[0].legend(fontsize=8); axes[0].set_title("uncertainty over 1,486 candidates")

    fac = Counter(s["factory"] for s in selected)
    axes[1].bar(range(len(fac)), sorted(fac.values(), reverse=True), color="#ff7f0e")
    axes[1].set_xlabel("factory rank"); axes[1].set_ylabel("windows in top-100")
    axes[1].set_title(f"factory diversity: {len(fac)} factories covered")

    if ood_eval is not None:
        axes[2].hist(ood_eval["u_good"], bins=15, alpha=0.6, label="OOD good (n=51)",
                     color="#2ca02c", density=True)
        axes[2].hist(ood_eval["u_bad"], bins=8, alpha=0.6, label="OOD bad (n=9)",
                     color="#d62728", density=True)
        axes[2].set_xlabel("uncertainty U"); axes[2].legend(fontsize=8)
        axes[2].set_title(f"OOD-60 eval: U AUROC={ood_eval['auroc']:.2f}")
    fig.tight_layout()
    fig.savefig(f"{outdir}/active_labels.png", bbox_inches="tight", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/exp_clip")
    ap.add_argument("--imgdir", default="docs/experiments/exp_clip/images")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    m, clip, start, y, p_ens, p_platt = load_gold()
    truth_frac, clip_verdict = clip_truth(clip, y)
    p_bad = 1.0 - p_ens
    q_bad = 1.0 - p_platt

    meta = {
        "n_windows": int(len(y)), "n_good": int(np.sum(y)), "n_bad": int(np.sum(1 - y)),
        "n_clips": len(CLIP_ORDER),
        "label_source": "IndEgo human gold 150 windows / 7 clips (labels/manifest_gold.json)",
        "labels_sha256": m["labels_sha256"],
        "score": "caches/ensemble/oof_probs.npz p_ens (LOCO window AUROC 0.834)",
        "clip_truth_bad_frac": {c: round(truth_frac[c], 4) for c in CLIP_ORDER},
        "clip_verdict_at_0.30": {c: ("reject" if clip_verdict[c] else "accept") for c in CLIP_ORDER},
        "note": "4 reject / 3 accept at 0.30 -> clip verdicts are discriminative but thin (n=7).",
        "thresh": THRESH, "thresh_secondary": THRESH2,
    }

    # ---------------------------------------------------------------- Z4.01
    methods = ["mean", "median", "q75", "q90", "trimmed", "rankmean"]
    agg_report, review_by_method = {}, {}
    for meth in methods:
        s = aggregate(p_bad, clip, meth)
        sv = np.array([s[c] for c in CLIP_ORDER])
        tv = np.array([clip_verdict[c] for c in CLIP_ORDER])
        auroc = (float(roc_auc_score(tv, sv)) if 0 < tv.sum() < len(tv) else None)
        lt = loco_threshold_ba(s, clip_verdict)
        li = loco_isotonic_ba(s, clip_verdict)
        agg_report[meth] = {"scores": {c: round(s[c], 5) for c in CLIP_ORDER},
                            "auroc": auroc, "loco_threshold_ba": lt,
                            "loco_isotonic_ba": li}
    for meth in ["mean", "median", "q75", "rankmean"]:
        review_by_method[meth] = review_curve(p_ens, clip, y)
    # review thresholds for the strongest aggregator
    best_agg = max(methods, key=lambda k: (agg_report[k]["auroc"] or 0))
    rev_strong = review_by_method.get(best_agg, review_curve(p_ens, clip, y))
    z401 = {
        "meta": meta, "experiment": "Z4.01 clip-level aggregation + LOCO clip calibration",
        "protocol": ("aggregate window bad-score 1-p_ens per clip (mean/median/q75/q90/"
                     "trimmed/rankmean); clip truth = bad fraction >= 0.30; LOCO over 7 clips "
                     "for threshold + isotonic calibration; clip AUROC over 7 clips; "
                     "review curve = per-clip highest-uncertainty windows get TRUE labels"),
        "aggregators": agg_report,
        "best_aggregator_by_auroc": best_agg,
        "review_curve": rev_strong,
        "min_review_frac_for_error": {
            "<=0.143 (6/7)": min_review_for_error(rev_strong, 1 / 7 + 1e-9),
            "==0 (7/7)": min_review_for_error(rev_strong, 1e-9),
        },
        "caveat": "n=7 clips -> clip AUROC/BA and LOCO thresholds are coarse/noisy.",
    }
    json.dump(z401, open(f"{args.out}/Z4.01.json", "w"), indent=1)

    # ---------------------------------------------------------------- Z4.02
    theta0, theta1 = 0.15, 0.45
    grid_A = list(np.round(np.arange(0.5, 12.01, 0.2), 2))
    grid_B = list(np.round(np.arange(0.3, 4.51, 0.05), 2))
    coarse_A = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
    coarse_B = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    llr_hard = make_llr(p_bad, "hard", theta0, theta1)      # binarised p_ens at 0.5
    llr_soft = make_llr(q_bad, "soft", theta0, theta1)      # p_platt direct (expected LLR)
    rows_hard = seq_frontier(clip, start, llr_hard, clip_verdict, grid_A, grid_B)
    rows_soft = seq_frontier(clip, start, llr_soft, clip_verdict, grid_A, grid_B)
    for r in rows_hard:
        r["mode"] = "hard"
    for r in rows_soft:
        r["mode"] = "soft"
    rows = rows_hard + rows_soft

    def ok(r):
        return (r["agreement_decided"] is not None and r["agreement_decided"] >= 0.90
                and r["abstain_rate"] <= 0.40)
    cells = sorted([r for r in rows if ok(r)],
                   key=lambda r: (r["mean_windows_processed_frac"], -(r["agreement_decided"] or 0)))
    best = cells[0] if cells else None

    # robustness of the best cell's decided/abstained set
    best_robust = None
    if best:
        same = [r for r in cells if sorted(r["decided"]) == sorted(best["decided"])]
        best_robust = {"n_cells": len(same),
                       "A_range": [min(r["A"] for r in same), max(r["A"] for r in same)],
                       "B_range": [min(r["B"] for r in same), max(r["B"] for r in same)],
                       "note": ("the decided/abstained partition is stable over this range, but "
                                "the decide/abstain boundary sits between two clips' running "
                                "extrema (desk min -3.69 vs ecu_late -3.92) -> fragile at n=7.")}

    # rebuild chosen trace for the per-clip figure
    if best:
        llr_b = llr_hard if best["mode"] == "hard" else llr_soft
        best_res = seq_run(clip, start, llr_b, best["A"], best["B"])
        best = dict(best)
        best["per_clip"] = {c: {"decision": best_res[c]["decision"], "k": best_res[c]["k"],
                                "n": best_res[c]["n"], "S": round(best_res[c]["S"], 3)}
                            for c in CLIP_ORDER}

    # coarse reported grid rows (computed directly on the coarse grid)
    def coarse_rows(llr):
        out = []
        for A in coarse_A:
            for B in coarse_B:
                res = seq_run(clip, start, llr, A, B)
                m = seq_metrics(res, clip_verdict)
                out.append({"A": float(A), "B": float(B), "abstain_rate": m["abstain_rate"],
                            "agreement_decided": m["agreement_decided"],
                            "error_decided": m["error_decided"],
                            "mean_windows_processed_frac": m["mean_windows_processed_frac"],
                            "n_decided": m["n_decided"], "wrong": m["wrong"],
                            "abstained": m["abstained"]})
        return out

    # theta sensitivity for the best-cell shape (hard mode)
    theta_sens = []
    for (t0, t1) in [(0.10, 0.40), (0.15, 0.45), (0.20, 0.50), (0.25, 0.55)]:
        ll = make_llr(p_bad, "hard", t0, t1)
        rr = seq_frontier(clip, start, ll, clip_verdict, grid_A, grid_B)
        cc = sorted([r for r in rr if ok(r)],
                    key=lambda r: r["mean_windows_processed_frac"])
        theta_sens.append({"theta0": t0, "theta1": t1,
                           "n_cells_ok": len(cc),
                           "best_frac": cc[0]["mean_windows_processed_frac"] if cc else None,
                           "best_agree": cc[0]["agreement_decided"] if cc else None,
                           "best_abstain": cc[0]["abstain_rate"] if cc else None})
    z402 = {
        "meta": meta, "experiment": "Z4.02 sequential SPRT on calibrated log-odds",
        "protocol": ("Wald SPRT over the clip's windows in time order. Per-window LLR "
                     "increment, then cumulative S; stop reject if S>=A, accept if S<=-B, "
                     "abstain if the clip is exhausted first. Primary mode='hard': binarise "
                     "the calibrated ensemble bad-probability (1-p_ens) at 0.5, giving "
                     "increments log(t1/t0) for a bad window and log((1-t1)/(1-t0)) for a good "
                     "one (classic Wald SPRT). Secondary mode='soft': expected LLR from "
                     "q=1-p_platt. Clip error = 1 - agreement on decided clips; cost = mean "
                     "windows processed (fraction of the clip)."),
        "theta0": theta0, "theta1": theta1,
        "grid_A": [grid_A[0], grid_A[-1], 0.1], "grid_B": [grid_B[0], grid_B[-1], 0.02],
        "n_cells": len(rows),
        "frontier_hard_coarse": coarse_rows(llr_hard),
        "frontier_soft_coarse": coarse_rows(llr_soft),
        "frontier_pareto_best": [
            {"A": r["A"], "B": r["B"], "mode": r["mode"], "abstain_rate": r["abstain_rate"],
             "agreement_decided": r["agreement_decided"],
             "mean_windows_processed_frac": r["mean_windows_processed_frac"]}
            for r in pareto_frontier([r for r in rows if r["n_decided"] > 0])],
        "best_cell": best,
        "best_cell_robustness": best_robust,
        "theta_sensitivity": theta_sens,
        "soft_mode_finding": ("the soft expected-LLR SPRT never reaches the accept bound on any "
                              "clip: every clip's mean (1-p_platt) is positive, so good clips "
                              "only abstain or (wrongly) drift to reject. The clip-level model "
                              "probability is miscalibrated (a sign/offset shift, cf. the pallet "
                              "inversion) -- binarising at 0.5 is what recovers the accept side. "
                              "This is a calibration limitation, not a sequential-rule one."),
        "baseline_every2": {"mean_windows_processed_frac": 0.5146, "error_decided": 0.0,
                            "agreement_decided": 1.0, "source": "E3.01 (label-oracle mode)"},
        "caveat": ("7 clips; the frontier is a coarse staircase and the best cell's accept "
                   "bound separates desk (running min -3.69) from ecu_late (-3.92) -- a "
                   "1-clip-wide margin. Treat the cell as illustrative, not deployable. "
                   "abstain = send the clip to a human."),
    }
    json.dump(z402, open(f"{args.out}/Z4.02.json", "w"), indent=1)

    # ---------------------------------------------------------------- Z4.03
    rows_resc = load_rescored()
    ood = ood_label_set()
    ood_ids = set(ood.keys())
    all_ids = set(r["window_id"] for r in rows_resc)
    assert ood_ids <= all_ids, "OOD ids not in rescored set"
    selected, cand = rank_active(rows_resc, exclude_ids=ood_ids, top=100)

    # ---- evaluation only (never fit): does U predict badness on OOD-60?
    by_id = {r["window_id"]: r for r in rows_resc}
    u_good, u_bad, ood_u = [], [], {}
    for wid, v in ood.items():
        r = by_id[wid]
        p = float(r["score"])
        U = 4.0 * p * (1.0 - p)
        ood_u[wid] = (U, v)
        (u_bad if v == "bad" else u_good).append(U)
    u_good = np.array(u_good); u_bad = np.array(u_bad)
    ood_auroc = None
    if len(u_bad) and len(u_good):
        ulab = np.array([1] * len(u_bad) + [0] * len(u_good))
        uall = np.concatenate([u_bad, u_good])
        ood_auroc = float(roc_auc_score(ulab, uall))
    # capture: fraction of the 9 OOD-bad inside the top-25% most uncertain OOD windows
    ranked_ood = sorted(ood_u.items(), key=lambda kv: -kv[1][0])
    top25 = [wid for wid, _ in ranked_ood[:max(1, int(round(0.25 * len(ranked_ood))))]]
    cap = sum(1 for wid in top25 if ood[wid] == "bad")
    ood_eval = {"n_good": len(u_good), "n_bad": len(u_bad),
                "u_good": u_good.tolist(), "u_bad": u_bad.tolist(),
                "auroc": ood_auroc,
                "top25_capture_bad": int(cap), "top25_capture_bad_of": int(len(u_bad)),
                "note": "evaluation-only sanity check; labels NOT used for fitting/ranking"}

    # candidate distribution
    cand_clip = Counter(c["clip"] for c in cand)
    sel_clip = Counter(s["clip"] for s in selected)
    sel_fac = Counter(s["factory"] for s in selected)
    z403 = {
        "meta": {"n_windows_total": len(rows_resc), "n_ood_labelled_excluded": len(ood_ids),
                 "n_candidates": len(cand),
                 "source": "artifacts/windows_rescored.jsonl (descriptive, OOD, unlabelled)",
                 "score": "ensemble_v1 score = mean(p_eg2, p_pixel); router review band [0.25,0.75]"},
        "experiment": "Z4.03 best sequential cell + next-100 active labels",
        "best_cell": best,
        "active_labeling": {
            "method": ("greedy diversified uncertainty sampling: raw = 4p(1-p) "
                       "(closeness to the decision boundary) + 0.5*|p_eg2 - p_pixel| "
                       "(two-view epistemic disagreement); redundancy penalty = "
                       "0.5^(#picked in same factory) * 0.5^(#picked in same clip)"),
            "n_excluded_already_labelled": len(ood_ids),
            "top100": selected,
            "coverage": {"factories_covered": len(sel_fac), "clips_covered": len(sel_clip),
                         "max_per_factory": max(sel_fac.values()),
                         "all_factories": len(set(r["factory"] for r in rows_resc))},
        },
        "ood_eval": ood_eval,
        "caveat": ("OOD-60 has n_bad=9 -> the U-vs-bad check is descriptive only. "
                   "Ranking is model-uncertainty based, not fitted on OOD labels."),
    }
    json.dump(z403, open(f"{args.out}/Z4.03.json", "w"), indent=1)

    # ---------------------------------------------------------------- figures
    every2 = [0.5146, 0.0]
    fig_aggregation(args.imgdir, agg_report, p_bad, clip, truth_frac, clip_verdict, review_by_method)
    fig_frontier(args.imgdir, rows, best, every2, clip_verdict)
    fig_active(args.imgdir, selected, cand, ood_eval if ood_eval["auroc"] is not None else None)

    # copy top-100 to a plain file too
    with open(f"{args.out}/top100_active_labels.json", "w") as f:
        json.dump(selected, f, indent=1)

    print(f"Z4.01 best_agg={best_agg} auroc={agg_report[best_agg]['auroc']}")
    print(f"Z4.02 cells_ok={len(cells)} best={None if not best else (best['A'], best['B'], best['mean_windows_processed_frac'], best['agreement_decided'], best['abstain_rate'])}")
    print(f"Z4.03 selected={len(selected)} factories={len(sel_fac)} ood_u_auroc={ood_auroc}")
    print(f"wrote {args.out}/Z4.0[1-3].json + top100_active_labels.json and {args.imgdir}/*.png")


if __name__ == "__main__":
    main()
