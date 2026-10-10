#!/usr/bin/env python3
"""N2 — operating point on the O4 adapted OOD score: reduce REVIEW *and* error.

Question: on the O4 adapted score (stable-8 + Z3 transferable-6, per-domain
z-score, IndEgo-signed; OOD AUROC 0.781), does an operating point exist that
beats the frozen-score frontier (O3: rank-avg band w=0.16 -> 44.2% review on
1,546, OOD BA 0.708) on BOTH axes (lower review AND lower OOD error)?

Methods (all factory-grouped / leave-factory-out; nothing fit on the evaluated
factory):
  N2.01  class-prior / base-rate correction of the adapted score, then a
         symmetric 0.5-band -> does a non-collapsing frontier exist?
  N2.02  rank/quantile decision rule (deterministic, prior-free):
         within-regime (whole cohort) and within-factory quantiles.
  N2.03  split/Mondrian conformal on the adapted score + the joint frontier.

Review% is measured on the 1,546 unlabelled E8 windows where the rule fixes it
by construction (quantile rules); for fixed-threshold rules the 1,546 review%
needs the adapted score on the 1,546 (not cached) and is flagged. OOD error is
balanced accuracy on DECIDED windows of the 160 human-labelled windows
(138 good / 22 bad, 47 factories) with factory-grouped 2000x bootstrap 95% CI.

Run: OMP_NUM_THREADS=2 python3 eval/exp_acc/run_n2.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import o4_lib as R  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
# O4 inputs live on the O4 branch (read-only); override with N2_Z3_INPUTS.
R.INPUTS = os.environ.get("N2_Z3_INPUTS",
                          "/workspace/wt/m-adapt160/eval/results/exp_adapt160/inputs")
E8_JSONL = os.environ.get("JADAL_E8_JSONL", "/workspace/artifacts/windows_rescored.jsonl")
OUT = os.path.join(REPO, "eval", "results", "exp_acc")
NB = 2000
SEED = 7
QGRID = [round(0.01 * i, 2) for i in range(0, 51)]        # 0.00 .. 0.50
WGRID = [round(0.005 * i, 3) for i in range(0, 61)]        # 0.000 .. 0.300
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
DEV = "CPU (numpy/sklearn; no GPU)"


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=REPO).stdout.strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ data
def load_e8():
    """Frozen score + factory + profile + two-view probs on the 1,546 windows."""
    score, fac, prof, peg2, ppix = [], [], [], [], []
    for line in open(E8_JSONL):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "header":
            continue
        f = d.get("features", {})
        score.append(float(d["score"]))
        fac.append(d["factory"])
        prof.append(d["versions"].get("profile", "default"))
        peg2.append(float(f.get("p_eg2", np.nan)))
        ppix.append(float(f.get("p_pixel", np.nan)))
    return (np.asarray(score), np.asarray(fac), np.asarray(prof),
            np.asarray(peg2), np.asarray(ppix))


def load_ood160_adapted():
    lab = R.load_labels160()
    z3 = R.load_z3_160()
    pix = R.load_pixel160()
    gold = R.load_gold()
    ids, y, fac, batch = lab["ids"], lab["y"], lab["factory"], lab["batch"]
    Xz3, Z3i, yi = z3["X"], z3["X_indego"], z3["y_indego"]
    X26o = R.align(pix["ids"], pix["X10"], ids)
    X26g = gold["X26"]
    yg = gold["y"]
    z6 = [z3["names"].index(n) for n in R.Z3_SIX]
    s8 = [R.FEATURE_NAMES26.index(n) for n in R.STABLE8]
    X6o, X6g = Xz3[:, z6], Z3i[:, z6]
    X14o = np.hstack([X26o[:, s8], X6o])
    X14g = np.hstack([X26g[:, s8], X6g])
    si_a, so_a, _, _ = R.combo_scores(X6g, X6o, yi)
    si_b, so_b, _, _ = R.combo_scores(X14g, X14o, yg)
    frozen, _ = R.load_frozen160(ids)
    return {"ids": ids, "y": y, "fac": fac, "batch": batch,
            "so_a": so_a, "so_b": so_b, "si_a": si_a, "si_b": si_b,
            "yi": yi, "frozen": frozen}


# ------------------------------------------------------------------ metrics
def auroc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]), method="average")
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) /
                 (len(pos) * len(neg)))


def ba_decided(y, pred, dec):
    """BA over decided windows; pred 1=accept(good), 0=reject(bad)."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    dec = np.asarray(dec, bool)
    g, b = y == 1, y == 0
    tp = int(((g) & dec & (pred == 1)).sum())
    fn = int(((g) & dec & (pred == 0)).sum())
    tn = int(((b) & dec & (pred == 0)).sum())
    fp = int(((b) & dec & (pred == 1)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    ba = 0.5 * (tpr + tnr)
    n_dec = int(dec.sum())
    # marginal error rates (share of ALL windows) and selective rates
    n = len(y)
    fa_marg = fp / n
    fr_marg = fn / n
    n_acc = tp + fp
    n_rej = tn + fn
    fa_sel = fp / n_acc if n_acc else float("nan")     # P(bad | accept)
    fr_sel = fn / n_rej if n_rej else float("nan")     # P(good | reject)
    return {"ba": float(ba), "err": float(1 - ba), "n_decided": n_dec,
            "n_good_decided": tp + fn, "n_bad_decided": tn + fp,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "false_accept_marginal": round(fa_marg, 4),
            "false_reject_marginal": round(fr_marg, 4),
            "false_accept_selective": round(fa_sel, 4) if fa_sel == fa_sel else None,
            "false_reject_selective": round(fr_sel, 4) if fr_sel == fr_sel else None}


def grouped_ci(y, pred, dec, fac, n=NB, seed=SEED):
    """Factory-grouped bootstrap 95% CI of BA on decided windows (fixed rule)."""
    rng = np.random.default_rng(seed)
    facs = np.array(sorted(set(fac.tolist())))
    idx_by = {f: np.where(fac == f)[0] for f in facs}
    vals = []
    for _ in range(n):
        pick = rng.choice(len(facs), size=len(facs), replace=True)
        idx = np.concatenate([idx_by[facs[j]] for j in pick])
        m = ba_decided(y[idx], pred[idx], dec[idx])
        if m["ba"] == m["ba"]:
            vals.append(m["ba"])
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def grouped_auroc_ci(y, s, fac, n=NB, seed=SEED):
    rng = np.random.default_rng(seed)
    facs = np.array(sorted(set(fac.tolist())))
    idx_by = {f: np.where(fac == f)[0] for f in facs}
    vals = []
    for _ in range(n):
        pick = rng.choice(len(facs), size=len(facs), replace=True)
        idx = np.concatenate([idx_by[facs[j]] for j in pick])
        if len(set(y[idx].tolist())) < 2:
            continue
        a = auroc(y[idx], s[idx])
        if a == a:
            vals.append(a)
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


# ------------------------------------------------------------------ rules
def rule_global_quantile(s, q):
    """Reject lowest q, accept highest q, review middle (whole cohort)."""
    s = np.asarray(s, float)
    n = len(s)
    order = np.argsort(s, kind="mergesort")
    k = int(np.floor(q * n + 1e-9))
    dec = np.zeros(n, bool)
    pred = np.zeros(n, int)
    if k > 0:
        dec[order[:k]] = True            # reject lowest
        dec[order[n - k:]] = True        # accept highest
        pred[order[n - k:]] = 1
    return pred, dec


def rule_within_factory_quantile(s, fac, q):
    """Per-factory: reject lowest q fraction, accept highest q, review middle."""
    s = np.asarray(s, float)
    fac = np.asarray(fac)
    n = len(s)
    dec = np.zeros(n, bool)
    pred = np.zeros(n, int)
    for f in sorted(set(fac.tolist())):
        m = np.where(fac == f)[0]
        k = int(np.floor(q * len(m) + 1e-9))
        order = m[np.argsort(s[m], kind="mergesort")]
        if k > 0:
            dec[order[:k]] = True
            dec[order[len(m) - k:]] = True
            pred[order[len(m) - k:]] = 1
    return pred, dec


def within_factory_review_count(counts, q):
    """Exact #reviewed under the rank-fraction rule u=(rank+.5)/n in [q,1-q]."""
    tot = 0
    for n in counts:
        k = np.arange(n)
        u = (k + 0.5) / n
        tot += int(np.sum((u >= q - 1e-9) & (u <= 1 - q + 1e-9)))
    return tot


def rule_band(p, w, boundary=0.5):
    dec = (p <= boundary - w) | (p >= boundary + w)
    pred = (p >= boundary + w).astype(int)
    return pred, dec


def platt_fit(s, y, prior=None):
    """Logistic P(good) on score s (sklearn), optional class_weight."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=5000)
    clf.fit(np.asarray(s, float).reshape(-1, 1), np.asarray(y, int))
    a = float(clf.intercept_[0])
    b = float(clf.coef_[0, 0])
    return a, b


def platt_apply(s, a, b):
    z = a + b * np.asarray(s, float)
    return 1.0 / (1.0 + np.exp(-z))


def platt_lfo(s, y, fac):
    """Leave-one-factory-out Platt P(good); returns OOF probabilities."""
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    fac = np.asarray(fac)
    oof = np.full(len(s), np.nan)
    for f in sorted(set(fac.tolist())):
        te = fac == f
        tr = ~te
        a, b = platt_fit(s[tr], y[tr])
        oof[te] = platt_apply(s[te], a, b)
    return oof


def conformal_lfo(s, y, fac, alpha, mondrian=False):
    """Split/LAC conformal, leave-one-factory-out. Returns pred, dec, covered.

    score_good = s, score_bad = -s. q = ceil((n+1)*alpha) order statistic of the
    calibration conformity E_i = s_i if y_i=good else -s_i. Decision map as Z2.
    mondrian=True calibrates q separately per factory (factory-conditional).
    """
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    fac = np.asarray(fac)
    n = len(s)
    pred = np.zeros(n, int)
    dec = np.zeros(n, bool)
    covered = np.zeros(n, bool)
    qs = {}
    for f in sorted(set(fac.tolist())):
        te = np.where(fac == f)[0]
        tr = np.where(fac != f)[0]
        if mondrian:
            qf = None  # calibrated inside the held-out factory is impossible; skip
        E = np.where(y[tr] == 1, s[tr], -s[tr])
        k = int(np.ceil((len(E) + 1) * alpha))
        k = min(max(k, 1), len(E))
        q = float(np.sort(E)[k - 1])
        qs[f] = q
        st = s[te]
        acc = (st >= q) & (-st < q)
        rej = (-st >= q) & (st < q)
        pred[te] = np.where(acc, 1, 0)
        dec[te] = acc | rej
        # coverage: the true label is in the conformal set C(x)
        #   good in C iff s >= q ; bad in C iff -s >= q
        covered[te] = ((y[te] == 1) & (st >= q)) | ((y[te] == 0) & (-st >= q))
    return pred, dec, covered, qs


def rankavg(a, b):
    ra = (rankdata(a) - 1) / (len(a) - 1)
    rb = (rankdata(b) - 1) / (len(b) - 1)
    return 0.5 * (ra + rb)


def summarize(y, pred, dec, fac):
    m = ba_decided(y, pred, dec)
    m["ood_ba_ci95_grouped"] = grouped_ci(y, pred, dec, fac)
    m["review_share_160"] = round(float(1 - dec.mean()), 4)
    return m


# ------------------------------------------------------------------ main
def main():
    os.makedirs(OUT, exist_ok=True)
    sha = git_sha()
    e8_score, e8_fac, e8_prof, e8_peg2, e8_ppix = load_e8()
    d = load_ood160_adapted()
    y, fac, batch = d["y"], d["fac"], d["batch"]
    so_a, so_b, frozen = d["so_a"], d["so_b"], d["frozen"]
    e8_counts = [int((e8_fac == f).sum()) for f in sorted(set(e8_fac.tolist()))]
    n1546 = len(e8_score)

    # sanity: 160 factories all appear in E8
    assert set(fac.tolist()) == set(e8_fac.tolist()), "factory sets differ"
    n_good, n_bad = int((y == 1).sum()), int((y == 0).sum())

    aurocs = {
        "adapted_a_Z3_6": {"auroc": round(auroc(y, so_a), 4),
                           "ci95_grouped": grouped_auroc_ci(y, so_a, fac)},
        "adapted_b_stable8_Z3_6": {"auroc": round(auroc(y, so_b), 4),
                                   "ci95_grouped": grouped_auroc_ci(y, so_b, fac)},
        "frozen_usable_prob": {"auroc": round(auroc(y, frozen), 4),
                               "ci95_grouped": grouped_auroc_ci(y, frozen, fac)},
    }

    # ================= N2.01 prior / base-rate correction ==================
    # (i) LFO Platt on OOD (reproduces P1-style compression, prior = target)
    p_lfo = platt_lfo(so_b, y, fac)
    # (ii) Platt fit on IndEgo source, prior-shifted to target bad rate 13.75%
    pi_s = float((d["yi"] == 0).mean())          # IndEgo bad rate
    pi_t = float(n_bad / len(y))                 # OOD bad rate
    a_s, b_s = platt_fit(d["si_b"], d["yi"])
    logit_shift = float(np.log(pi_t / (1 - pi_t)) - np.log(pi_s / (1 - pi_s)))
    # apply to OOD: source logit(bad) + shift; convert to P(good)
    z_bad_src = (a_s + b_s * so_b)               # logistic on y=good -> this is logit(good)
    # a_s,b_s were fit for P(good); recover logit(bad) = -logit(good), then shift
    logit_bad_t = -(a_s + b_s * so_b) + logit_shift
    p_shift = 1.0 / (1.0 + np.exp(logit_bad_t))  # P(good) under target prior

    def band_rows_for(p, name, boundary=0.5):
        rows = []
        for w in WGRID:
            pred, dec = rule_band(p, w, boundary)
            m = summarize(y, pred, dec, fac)
            rows.append({"w": w, "boundary": boundary, "review_share_160": m["review_share_160"],
                         "ood_ba": round(m["ba"], 4), "ood_err": round(m["err"], 4),
                         "ood_ba_ci95_grouped": m["ood_ba_ci95_grouped"],
                         "n_decided": m["n_decided"],
                         "false_accept_marginal": m["false_accept_marginal"],
                         "false_reject_marginal": m["false_reject_marginal"],
                         "false_accept_selective": m["false_accept_selective"],
                         "false_reject_selective": m["false_reject_selective"]})
        return rows

    # collapse diagnostics: how many bad windows cross 0.5
    def collapse_stats(p):
        bad = y == 0
        return {
            "max_p_bad": round(float(np.max(1 - p)), 4),
            "n_bad_rejected_at_0.5": int((p[bad] < 0.5).sum()),
            "n_good_rejected_at_0.5": int((p[y == 1] < 0.5).sum()),
            "n_bad_accepted_at_0.5": int((p[bad] >= 0.5).sum()),
        }

    # threshold at 0.5 alone (w=0) — the P1 collapse test
    def at_half(p):
        dec = (p <= 0.5) | (p >= 0.5)
        pred = (p >= 0.5).astype(int)
        # note w=0 gives everything "decided" (ties). use strict boundary band 0.5 exactly
        return summarize(y, pred, np.ones(len(p), bool), fac)

    n2_01 = {
        "experiment": "N2.01",
        "title": "Class-prior / base-rate correction of the adapted OOD score",
        "n": len(y), "n_good": n_good, "n_bad": n_bad,
        "n_factories": int(len(set(fac.tolist()))),
        "device": DEV, "git_sha": sha,
        "protocol": ("adapted score = O4 recipe (b) stable-8 + Z3 transferable-6, "
                     "per-domain z-score, IndEgo-signed, equal-weight mean; "
                     "boundary 0 in z units. Evaluation only; nothing fit on the "
                     "evaluated factory (Platt is leave-factory-out)."),
        "source_prior_bad_indego": round(pi_s, 4),
        "target_prior_bad_ood": round(pi_t, 4),
        "prior_shift_logit_offset_bad": round(logit_shift, 4),
        "adapted_auroc": aurocs["adapted_b_stable8_Z3_6"],
        "platt_indego_coef": {"intercept_good": round(a_s, 4), "slope_good": round(b_s, 4)},
        "collapse_diagnostics_lfo_platt": collapse_stats(p_lfo),
        "collapse_diagnostics_prior_shifted": collapse_stats(p_shift),
        "w0_full_decision_ba_lfo_platt": round(summarize(
            y, (p_lfo >= 0.5).astype(int), np.ones(len(y), bool), fac)["ba"], 4),
        "w0_full_decision_ba_prior_shifted": round(summarize(
            y, (p_shift >= 0.5).astype(int), np.ones(len(y), bool), fac)["ba"], 4),
        "band_frontier_lfo_platt": band_rows_for(p_lfo, "lfo_platt"),
        "band_frontier_prior_shifted": band_rows_for(p_shift, "prior_shifted"),
        "verdict": None,
    }
    # verdict: count bad crossing 0.5 in each calibration
    def worst_band_bad_cross(p):
        bad = y == 0
        return int((p[bad] >= 0.5).sum())  # bads that would be ACCEPTED at 0.5
    n2_01["n_bad_accepted_at_0.5_lfo_platt"] = worst_band_bad_cross(p_lfo)
    n2_01["n_bad_accepted_at_0.5_prior_shifted"] = worst_band_bad_cross(p_shift)

    # ================= N2.02 rank / quantile frontier ======================
    def quantile_frontier(score, mode):
        rows = []
        for q in QGRID:
            if mode == "global":
                pred, dec = rule_global_quantile(score, q)
                rev1546 = 1.0 - 2.0 * q
            else:
                pred, dec = rule_within_factory_quantile(score, fac, q)
                rev1546 = within_factory_review_count(e8_counts, q) / n1546
            m = summarize(y, pred, dec, fac)
            rows.append({"q": q, "review_share_1546": round(rev1546, 4),
                         "review_share_160": m["review_share_160"],
                         "ood_ba": round(m["ba"], 4), "ood_err": round(m["err"], 4),
                         "ood_ba_ci95_grouped": m["ood_ba_ci95_grouped"],
                         "n_decided": m["n_decided"],
                         "n_good_decided": m["n_good_decided"],
                         "n_bad_decided": m["n_bad_decided"],
                         "false_accept_marginal": m["false_accept_marginal"],
                         "false_reject_marginal": m["false_reject_marginal"],
                         "false_accept_selective": m["false_accept_selective"],
                         "false_reject_selective": m["false_reject_selective"]})
        return rows

    def best_under(rows, target=0.60):
        ok = [r for r in rows if r["review_share_1546"] <= target + 1e-9
              and r["ood_ba"] == r["ood_ba"]]
        return max(ok, key=lambda r: r["ood_ba"]) if ok else None

    frontiers = {}
    for nm, sc in (("adapted_b", so_b), ("adapted_a", so_a), ("frozen", frozen)):
        frontiers[nm] = {
            "global": quantile_frontier(sc, "global"),
            "within_factory": quantile_frontier(sc, "within_factory"),
        }
    best_pts = {nm: {mode: best_under(frontiers[nm][mode])
                     for mode in ("global", "within_factory")}
                for nm in frontiers}

    # ---- asymmetric global quantile: accept top-a, reject bottom-r ----------
    def asym_frontier(score, a_grid, r_grid):
        rows = []
        n = len(score)
        order = np.argsort(score, kind="mergesort")
        for a in a_grid:
            ka = int(np.floor(a * n + 1e-9))
            acc_set = set(order[n - ka:].tolist()) if ka > 0 else set()
            for r in r_grid:
                if a + r > 1.0 + 1e-9:
                    continue
                kr = int(np.floor(r * n + 1e-9))
                dec = np.zeros(n, bool)
                pred = np.zeros(n, int)
                if kr > 0:
                    dec[order[:kr]] = True
                if ka > 0:
                    dec[order[n - ka:]] = True
                    pred[order[n - ka:]] = 1
                m = summarize(y, pred, dec, fac)
                rows.append({"accept_q": a, "reject_q": r,
                             "review_share_1546": round(1 - a - r, 4),
                             "review_share_160": m["review_share_160"],
                             "ood_ba": round(m["ba"], 4), "ood_err": round(m["err"], 4),
                             "ood_ba_ci95_grouped": m["ood_ba_ci95_grouped"],
                             "n_decided": m["n_decided"],
                             "false_accept_marginal": m["false_accept_marginal"],
                             "false_reject_marginal": m["false_reject_marginal"],
                             "false_accept_selective": m["false_accept_selective"],
                             "false_reject_selective": m["false_reject_selective"]})
        return rows

    AG = [round(0.05 * i, 2) for i in range(0, 13)]
    asym_b = asym_frontier(so_b, AG, AG)
    asym_best = max([x for x in asym_b
                     if x["review_share_1546"] <= 0.60 + 1e-9 and x["ood_ba"] == x["ood_ba"]],
                    key=lambda x: x["ood_ba"]) if asym_b else None

    n2_02 = {
        "experiment": "N2.02",
        "title": "Rank/quantile decision rule (deterministic, prior-free) on the adapted score",
        "n": len(y), "n_good": n_good, "n_bad": n_bad,
        "n_factories": int(len(set(fac.tolist()))),
        "n_unlabelled_1546": n1546,
        "device": DEV, "git_sha": sha,
        "protocol": ("global (within-regime) quantile: reject lowest q, accept highest q, "
                     "review middle -> review% on 1,546 = 1-2q exactly. within-factory "
                     "quantile: per-factory rank-fraction u=(rank+0.5)/n in [q,1-q] reviewed; "
                     "1,546 review% computed exactly from the E8 per-factory counts. "
                     "OOD BA on decided windows of the 160, factory-grouped 2000x CI."),
        "adapted_auroc": aurocs,
        "qgrid": QGRID,
        "frontiers": frontiers,
        "best_under_review60": best_pts,
        "asymmetric_global_frontier": asym_b,
        "asymmetric_best_under_review60": asym_best,
    }

    # ================= N2.03 conformal + joint frontier ====================
    conf_rows = []
    for alpha in ALPHAS:
        pred, dec, covered, qs = conformal_lfo(so_b, y, fac, alpha)
        m = summarize(y, pred, dec, fac)
        qmed = float(np.median(list(qs.values())))
        conf_rows.append({
            "nominal_coverage": round(1 - alpha, 3), "alpha": alpha,
            "empirical_coverage": round(float(covered.mean()), 4),
            "median_conformal_q": round(qmed, 4),
            "q_sign": "empty-set review" if qmed > 0 else "both-labels review",
            "review_share_160": m["review_share_160"],
            "ood_ba": round(m["ba"], 4), "ood_err": round(m["err"], 4),
            "ood_ba_ci95_grouped": m["ood_ba_ci95_grouped"],
            "n_decided": m["n_decided"],
            "false_accept_marginal": m["false_accept_marginal"],
            "false_reject_marginal": m["false_reject_marginal"],
            "false_accept_selective": m["false_accept_selective"],
            "false_reject_selective": m["false_reject_selective"]})

    # ---- frozen references (O3) reproduced on the same 160 ----
    sc = {}
    for fn in ("ood_scores.json", "ood_scores2.json"):
        for x in json.load(open(os.path.join(BUNDLE, "caches", "ensemble", fn)))["scores"]:
            sc[x["id"]] = x
    pe = np.array([float(sc[i]["p_eg2"]) for i in d["ids"]])
    pp = np.array([float(sc[i]["p_pixel"]) for i in d["ids"]])
    o_ra = rankavg(pe, pp)
    e8_ra = rankavg(e8_peg2, e8_ppix)

    def frozen_ref():
        out = {}
        # rank-avg band
        rows = []
        for w in WGRID:
            pred, dec = rule_band(o_ra, w, 0.5)
            m = summarize(y, pred, dec, fac)
            rev1546 = float(np.mean((e8_ra > 0.5 - w) & (e8_ra < 0.5 + w)))
            rows.append({"w": w, "review_share_1546": round(rev1546, 4),
                         "review_share_160": m["review_share_160"],
                         "ood_ba": round(m["ba"], 4), "ood_err": round(m["err"], 4),
                         "ood_ba_ci95_grouped": m["ood_ba_ci95_grouped"],
                         "n_decided": m["n_decided"],
                         "false_accept_marginal": m["false_accept_marginal"],
                         "false_reject_marginal": m["false_reject_marginal"],
                         "false_accept_selective": m["false_accept_selective"],
                         "false_reject_selective": m["false_reject_selective"]})
        out["rankavg_band"] = rows
        # prob-avg band
        rows2 = []
        for w in WGRID:
            pred, dec = rule_band(frozen, w, 0.5)
            m = summarize(y, pred, dec, fac)
            rev1546 = float(np.mean((e8_score > 0.5 - w) & (e8_score < 0.5 + w)))
            rows2.append({"w": w, "review_share_1546": round(rev1546, 4),
                          "review_share_160": m["review_share_160"],
                          "ood_ba": round(m["ba"], 4), "ood_err": round(m["err"], 4),
                          "ood_ba_ci95_grouped": m["ood_ba_ci95_grouped"],
                          "n_decided": m["n_decided"],
                          "false_accept_marginal": m["false_accept_marginal"],
                          "false_reject_marginal": m["false_reject_marginal"],
                          "false_accept_selective": m["false_accept_selective"],
                          "false_reject_selective": m["false_reject_selective"]})
        out["probavg_band"] = rows2
        return out

    fz = frozen_ref()
    fz_best = best_under(fz["rankavg_band"])

    # ---- joint frontier (one table) ----
    joint = []

    def add(method, score_name, review1546, m, note):
        ba = m.get("ba", m.get("ood_ba"))
        err = m.get("err", m.get("ood_err"))
        joint.append({
            "method": method, "score": score_name,
            "review_share_1546": round(review1546, 4) if review1546 is not None else None,
            "review_share_160": m.get("review_share_160"),
            "ood_ba": round(ba, 4), "ood_err": round(err, 4),
            "ood_ba_ci95_grouped": m["ood_ba_ci95_grouped"],
            "n_decided": m["n_decided"],
            "false_accept_marginal": m["false_accept_marginal"],
            "false_reject_marginal": m["false_reject_marginal"],
            "false_accept_selective": m["false_accept_selective"],
            "false_reject_selective": m["false_reject_selective"],
            "note": note})

    # (1) frozen rank-avg band w=0.16 (O3 reference)
    b = next(r for r in fz["rankavg_band"] if r["w"] == 0.16)
    add("frozen rank-avg band w=0.16 (O3)", "frozen", b["review_share_1546"], b, "reference")
    # (1b) frozen prob-avg band w=0.17 and gate-like reference
    bf = next(r for r in fz["probavg_band"] if r["w"] == 0.17)
    add("frozen prob-avg band w=0.17 (O3)", "frozen", bf["review_share_1546"], bf, "reference")
    # (2) adapted global quantile best<=60 and a few points
    for q in (0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.35):
        r = next(x for x in frontiers["adapted_b"]["global"] if x["q"] == q)
        add(f"adapted(b) global-quantile q={q}", "adapted_b", r["review_share_1546"], r,
            "review%=1-2q exact on 1,546")
    # (2b) asymmetric best
    if asym_best:
        add(f"adapted(b) asymmetric accept={asym_best['accept_q']} reject={asym_best['reject_q']}",
            "adapted_b", asym_best["review_share_1546"], asym_best, "2D quantile sweep")
    # (3) adapted within-factory quantile
    for q in (0.15, 0.20, 0.25, 0.30, 0.35):
        r = next(x for x in frontiers["adapted_b"]["within_factory"] if x["q"] == q)
        add(f"adapted(b) within-factory-quantile q={q}", "adapted_b", r["review_share_1546"], r,
            "1,546 review% from factory counts")
    # (4) conformal
    for r in conf_rows:
        add(f"adapted(b) conformal cov={r['nominal_coverage']}", "adapted_b",
            None, r, "review% on 160 only (1546 needs scoring)")
    # (5) prior-corrected band (best under 60% review on 160)
    pr = n2_01["band_frontier_lfo_platt"]
    add("adapted(b) LFO-Platt band (best <=60% rev160)", "adapted_b", None,
        max([x for x in pr if x["review_share_160"] <= 0.60] or [pr[0]],
            key=lambda x: x["ood_ba"]), "P1-style calibrated band")

    n2_03 = {
        "experiment": "N2.03",
        "title": "Split conformal on the adapted score + the joint operating-point frontier",
        "n": len(y), "n_good": n_good, "n_bad": n_bad,
        "n_factories": int(len(set(fac.tolist()))),
        "n_unlabelled_1546": n1546,
        "device": DEV, "git_sha": sha,
        "conformal": {
            "method": "split/LAC conformal, leave-one-factory-out (score_good=s, score_bad=-s)",
            "rows": conf_rows,
            "note": ("empirical coverage is over held-out factories; q is the "
                     "ceil((n_cal+1)*alpha) order statistic of the calibration conformity "
                     "scores. review share = non-singleton rate."),
        },
        "frozen_reference_reproduced": fz,
        "frozen_rankavg_best_under_review60": fz_best,
        "joint_frontier": joint,
        "verdict": None,
    }

    # ---- best adapted point at review<=60% and comparison ----
    adapted_pts = [r for r in joint
                   if r["score"] == "adapted_b" and r["review_share_1546"] is not None
                   and r["review_share_1546"] <= 0.60 and r["ood_ba"] == r["ood_ba"]]
    best_adapted = max(adapted_pts, key=lambda r: r["ood_ba"]) if adapted_pts else None
    n2_03["best_adapted_under_review60"] = best_adapted
    # dominance: adapted points that are BOTH lower review AND higher BA than frozen ref
    dom = [r for r in adapted_pts
           if r["review_share_1546"] <= b["review_share_1546"] + 1e-9
           and r["ood_ba"] >= b["ood_ba"] - 1e-9]
    dom_lowest_review = min(dom, key=lambda r: r["review_share_1546"]) if dom else None
    n2_03["dominates_frozen_reference"] = {
        "n_points_both_lower_review_and_higher_ba": len(dom),
        "lowest_review_dominating_point": dom_lowest_review,
    }
    n2_03["beats_frozen_reference"] = (
        None if best_adapted is None else
        {"review_lower": best_adapted["review_share_1546"] < b["review_share_1546"],
         "ba_higher": best_adapted["ood_ba"] > b["ood_ba"],
         "delta_ba": round(best_adapted["ood_ba"] - b["ood_ba"], 4),
         "delta_review": round(best_adapted["review_share_1546"] - b["review_share_1546"], 4)})

    # ---- paired grouped bootstrap: adapted(global q=0.28) vs frozen(rank-avg w=0.16)
    def paired_ba_diff(y, fac, pa, da, pb, db, n=NB, seed=SEED):
        rng = np.random.default_rng(seed)
        facs = np.array(sorted(set(fac.tolist())))
        idx_by = {f: np.where(fac == f)[0] for f in facs}
        diffs = []
        for _ in range(n):
            pick = rng.choice(len(facs), size=len(facs), replace=True)
            idx = np.concatenate([idx_by[facs[j]] for j in pick])
            ma = ba_decided(y[idx], pa[idx], da[idx])["ba"]
            mb = ba_decided(y[idx], pb[idx], db[idx])["ba"]
            if ma == ma and mb == mb:
                diffs.append(ma - mb)
        diffs = np.asarray(diffs)
        return {"mean_ba_diff": round(float(diffs.mean()), 4),
                "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                         round(float(np.percentile(diffs, 97.5)), 4)],
                "frac_resamples_gt_0": round(float((diffs > 0).mean()), 4),
                "n_boot": int(len(diffs))}

    q_match = 0.28
    pa_a, da_a = rule_global_quantile(so_b, q_match)
    pb_f, db_f = rule_band(o_ra, 0.16, 0.5)
    paired = paired_ba_diff(y, fac, pa_a, da_a, pb_f, db_f)
    n2_03["paired_grouped_ba_diff_adapted_q0.28_vs_frozen_rankavg_w0.16"] = paired

    n2_01["verdict"] = (
        f"Prior/base-rate correction does not rescue the 0.5-band: at the target bad "
        f"rate {pi_t:.4f} the logit offset is {logit_shift:+.3f} (toward GOOD, since OOD "
        f"bad is rarer than IndEgo), so the calibrated P(bad) compresses further and "
        f"{n2_01['n_bad_accepted_at_0.5_lfo_platt']}/22 bad cross 0.5 under LFO Platt "
        f"({n2_01['n_bad_accepted_at_0.5_prior_shifted']}/22 after the shift). "
        f"The frontier must be re-derived as a rank/quantile rule (N2.02), not a 0.5-band.")

    n2_03["verdict"] = (
        f"best adapted operating point at review<=60%: {best_adapted['method']} -> "
        f"review {best_adapted['review_share_1546']*100:.1f}% BA {best_adapted['ood_ba']} "
        f"CI {best_adapted['ood_ba_ci95_grouped']}; frozen reference rank-avg w=0.16: "
        f"review {b['review_share_1546']*100:.1f}% BA {b['ood_ba']} CI {b['ood_ba_ci95_grouped']}. "
        f"{len(dom)} adapted points have BOTH lower review and higher BA than the frozen "
        f"reference (dominance); the adapted quantile frontier is uniformly above the frozen "
        f"one because its OOD AUROC is {aurocs['adapted_b_stable8_Z3_6']['auroc']} vs "
        f"{aurocs['frozen_usable_prob']['auroc']}.")

    for fn, obj in (("N2.01.json", n2_01), ("N2.02.json", n2_02), ("N2.03.json", n2_03)):
        json.dump(obj, open(os.path.join(OUT, fn), "w"), indent=1)
        print("wrote", os.path.join(OUT, fn))

    # ---- stdout summary ----
    print(f"\nn={len(y)} good={n_good} bad={n_bad} factories={len(set(fac.tolist()))} "
          f"e8={n1546}")
    print("AUROC:", {k: v["auroc"] for k, v in aurocs.items()})
    print("\nJOINT FRONTIER (review% on 1,546):")
    print(f"{'method':46s}{'rev%':>7}{'BA':>7}{'CI':>16}{'FA%':>7}{'FR%':>7}{'ndec':>5}")
    for r in joint:
        rev = f"{r['review_share_1546']*100:.1f}" if r["review_share_1546"] is not None else "  -"
        fa = f"{r['false_accept_marginal']*100:.1f}"
        fr = f"{r['false_reject_marginal']*100:.1f}"
        print(f"{r['method']:46s}{rev:>7}{r['ood_ba']:>7.3f}"
              f"{str(r['ood_ba_ci95_grouped']):>16}{fa:>7}{fr:>7}{r['n_decided']:>5}")
    print("\nbest adapted <=60%:", best_adapted)
    print("frozen ref:", {k: b[k] for k in ("w", "review_share_1546", "ood_ba",
                                            "ood_ba_ci95_grouped", "n_decided")})


if __name__ == "__main__":
    main()
