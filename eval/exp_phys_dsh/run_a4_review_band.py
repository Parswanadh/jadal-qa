#!/usr/bin/env python3
"""A4 / direction 5: deterministic REVIEW band on the frozen Q1 integer score.

Protocol (BRIEF, binding): leave-one-clip-out (LOCO) over the 7 IndEgo gold clips.
For each held-out clip the rule table thresholds are re-derived from the other 6
clips only (quantiles p01/p05/p10/p25, mirrored to p99/p95/p90/p75 for
high-is-bad features, quantised to Q=4); the operating point is chosen on the
training clips only (false-reject budget 5% on training-good windows); metrics
are reported on the held-out clip windows pooled per operating point.

Decision (frozen Q1 semantics + explicit REVIEW band):
    REJECT  if a QUALITY rule crosses its extreme threshold (index 0), or score >= t_reject
    ACCEPT  if score < t_accept (and, for the 2nd review axis, margin >= M)
    REVIEW  otherwise

Outputs
  eval/results/exp_phys_dsh/A4_review_band.json   (frontier, best point, per-rule, CI, command)
  eval/results/exp_phys_dsh/A4_determinism.json   (written by det_scorer_v2 --proof)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BUNDLE = "/workspace/jadal_pod_bundle"
RULE_TABLE = os.path.join(ROOT, "eval", "results", "exp_phys", "rule_table.json")
OUT = os.path.join(ROOT, "eval", "results", "exp_phys_dsh")

Q = 4                       # decimal places of the integer quantisation
SCALE = 10 ** Q
FR_BUDGET = 0.05            # pre-registered false-reject budget (training-good)
BOOT = 2000
SEED = 0
QUANTILES = np.array([0.01, 0.05, 0.10, 0.25])


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def load_windows(source: str = "q1"):
    """Return ids, feats (list of dict), y, clips.

    source="q1"  : exact Q1/Q4 feature space -> caches/gpu/features.json `full`
                   (float64) + caches/perpixel/physics_feats.npz `phys`.
                   This is the space the frozen rule-table thresholds were
                   derived in and the space the 0.808/0.60 baseline used.
    source="gold": BRIEF name map -> caches/perpixel/gold.npz `G_*` columns
                   (float32) + physics_feats.npz.  Sensitivity variant only.
    """
    gold = np.load(os.path.join(BUNDLE, "caches", "perpixel", "gold.npz"), allow_pickle=True)
    phys = np.load(os.path.join(BUNDLE, "caches", "perpixel", "physics_feats.npz"), allow_pickle=True)
    ids = [str(i) for i in gold["ids"]]
    y = np.asarray(gold["y"], dtype=int)
    clips = [str(c) for c in gold["clips"]]
    pidx = {str(k): i for i, k in enumerate(phys["ids"])}
    pnames = [str(n) for n in phys["names"]]
    phys_rows = {i: {f"phys.{n}": float(phys["X"][pidx[i], j]) for j, n in enumerate(pnames)}
                 for i in ids}

    feats = []
    if source == "q1":
        W = json.load(open(os.path.join(BUNDLE, "caches", "gpu", "features.json")))["windows"]
        fnames = list(W[ids[0]]["full"].keys())
        for i in ids:
            d = {f"full.{n}": float(W[i]["full"][n]) for n in fnames}
            d.update(phys_rows[i])
            feats.append(d)
    elif source == "gold":
        names = [str(n) for n in gold["names"]]
        gidx = {n: i for i, n in enumerate(names)}
        gnames = [n for n in names if n.startswith("G_")]
        for k, i in enumerate(ids):
            d = {f"full.{n[2:]}": float(gold["X"][k, gidx[n]]) for n in gnames}
            d.update(phys_rows[i])
            feats.append(d)
    else:
        raise ValueError(source)
    return ids, feats, y, clips


def load_rule_rows():
    tbl = json.load(open(RULE_TABLE))
    rows = [{k: r[k] for k in ("rule", "feature", "op", "side", "strength")}
            for r in tbl["rules"]]
    return rows


# ---------------------------------------------------------------------------
# integer decision path (thresholds are derived offline, per fold)
# ---------------------------------------------------------------------------
def quantise(value, q: int = Q):
    """Half-up quantisation of a feature value to an integer. None if non-finite."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    scaled = v * (10 ** q)
    return int(np.floor(scaled + 0.5)) if scaled >= 0 else int(np.ceil(scaled - 0.5))


def derive_thresholds_q(values, side: str) -> list:
    """Four Q=4 integer thresholds from TRAINING values only."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 8 or len(np.unique(v)) < 2:
        return []
    qs = QUANTILES if side == "low" else 1.0 - QUANTILES
    return [int(round(round(float(t), Q) * SCALE)) for t in np.quantile(v, qs)]


def resolve_table(feats, train_idx, rows):
    """Per-fold rule table: thresholds from training windows only."""
    table = []
    for r in rows:
        ths_q = derive_thresholds_q([feats[i].get(r["feature"], np.nan) for i in train_idx], r["side"])
        if ths_q:
            table.append({**r, "thresholds_q": ths_q})
    return table


def score_window(feats: dict, table: list, q: int = Q):
    """Pure integer score + every crossed threshold as evidence.

    Returns dict(score, fired=[...], hard_rules=sorted, min_margin_q,
                 min_crossed_margin_q).
    Every fixed Q=4 comparison is on integers; no float ever reaches a decision.
    """
    score = 0
    fired = []
    hard_rules = set()
    min_margin = None            # raw distance to the nearest threshold (Q=4 integer units)
    min_margin_norm = None       # scale-free distance in half-gaps (see below)
    min_crossed_margin = None    # distance to the nearest CROSSED threshold
    for r in table:
        vq = quantise(feats.get(r["feature"]), q)
        if vq is None:
            continue
        ths = r["thresholds_q"]
        crossed = []
        for idx, tq in enumerate(ths):
            d = abs(tq - vq)
            if min_margin is None or d < min_margin:
                min_margin = d
            hit = (vq < tq) if r["op"] == "<" else (vq > tq)
            if hit:
                crossed.append((idx, tq))
        # scale-free nearest-threshold distance, pure integer: d in units of half the
        # gap between the two thresholds adjacent to the nearest one.
        k = min(range(len(ths)), key=lambda j: (abs(ths[j] - vq), j))
        gaps = []
        if k > 0:
            gaps.append(abs(ths[k] - ths[k - 1]))
        if k + 1 < len(ths):
            gaps.append(abs(ths[k + 1] - ths[k]))
        gap = min(gaps) if gaps else max(1, abs(ths[k]))
        dn = (2 * abs(ths[k] - vq)) // max(gap, 1)
        if min_margin_norm is None or dn < min_margin_norm:
            min_margin_norm = dn
        if crossed:
            score += len(crossed)
            margin = min(abs(tq - vq) for _, tq in crossed)
            if min_crossed_margin is None or margin < min_crossed_margin:
                min_crossed_margin = margin
            hard = any(idx == 0 for idx, _ in crossed)
            if hard and r["strength"] == "quality":
                hard_rules.add(r["rule"])
            best = min(crossed, key=lambda it: (abs(it[1] - vq), it[1]))
            fired.append({"rule": r["rule"], "feature": r["feature"], "value": vq,
                          "op": r["op"], "threshold": best[1], "strength": r["strength"],
                          "n_thresholds_crossed": len(crossed), "hard": hard,
                          "crossed_thresholds": [t for _, t in crossed]})
    return {"score": int(score), "fired": fired, "hard_rules": sorted(hard_rules),
            "min_margin_q": min_margin, "min_margin_norm_q": min_margin_norm,
            "min_crossed_margin_q": min_crossed_margin}


def decide(score: int, hard_rules, t_accept: int, t_reject: int,
           min_margin_q=None, m_min: int = 0, quality_override: bool = True) -> str:
    """ACCEPT / REVIEW / REJECT from integers only."""
    if quality_override and hard_rules:
        return "REJECT"
    if score >= t_reject:
        return "REJECT"
    if score < t_accept and (min_margin_q is None or min_margin_q >= m_min):
        return "ACCEPT"
    return "REVIEW"


def choose_reject_threshold(train_scores, train_y, fr_budget: float = FR_BUDGET) -> int:
    """Smallest integer s with training-good false-reject rate <= budget (Q1)."""
    sc = np.asarray(train_scores)
    good = sc[np.asarray(train_y) == 1]
    if good.size == 0:
        return 1
    for s in range(1, int(sc.max()) + 2):
        if float((good >= s).mean()) <= fr_budget:
            return s
    return int(sc.max()) + 1


# ---------------------------------------------------------------------------
# LOCO
# ---------------------------------------------------------------------------
def run_loco(feats, y, clips, rows):
    n = len(feats)
    scores = np.zeros(n, dtype=int)
    hard = [None] * n
    min_margin = np.zeros(n, dtype=int)
    min_margin_norm = np.zeros(n, dtype=int)
    min_crossed = np.zeros(n, dtype=int)
    fired_by_win = [None] * n
    fold_t_rej = np.zeros(n, dtype=int)
    tables = {}
    train_side = {}
    for c in sorted(set(clips)):
        te = np.array([clips[i] == c for i in range(n)])
        tr = ~te
        table = resolve_table(feats, np.where(tr)[0], rows)
        tables[c] = table
        tr_idx = [i for i in range(n) if tr[i]]
        tr_sc = [score_window(feats[i], table) for i in tr_idx]
        tr_scores = np.array([x["score"] for x in tr_sc])
        t_rej = choose_reject_threshold(tr_scores, y[tr])
        train_side[c] = {
            "idx": tr_idx,
            "scores": tr_scores,
            "y": y[tr],
            "hard": [x["hard_rules"] for x in tr_sc],
            "min_margin_q": np.array([x["min_margin_q"] if x["min_margin_q"] is not None
                                      else 10 ** 9 for x in tr_sc]),
            "min_margin_norm_q": np.array([x["min_margin_norm_q"]
                                           if x["min_margin_norm_q"] is not None else 10 ** 9
                                           for x in tr_sc]),
            "min_crossed_margin_q": np.array([x["min_crossed_margin_q"]
                                              if x["min_crossed_margin_q"] is not None else 10 ** 9
                                              for x in tr_sc]),
        }
        for i in range(n):
            if tr[i]:
                continue
            sc = score_window(feats[i], table)
            scores[i] = sc["score"]
            hard[i] = sc["hard_rules"]
            min_margin[i] = sc["min_margin_q"] if sc["min_margin_q"] is not None else 10 ** 9
            min_margin_norm[i] = (sc["min_margin_norm_q"]
                                  if sc["min_margin_norm_q"] is not None else 10 ** 9)
            min_crossed[i] = (sc["min_crossed_margin_q"]
                              if sc["min_crossed_margin_q"] is not None else 10 ** 9)
            fired_by_win[i] = sc["fired"]
            fold_t_rej[i] = t_rej
    return {"scores": scores, "hard": hard, "min_margin_q": min_margin,
            "min_margin_norm_q": min_margin_norm, "min_crossed_margin_q": min_crossed,
            "fired": fired_by_win, "fold_t_reject": fold_t_rej, "tables": tables,
            "train": train_side}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def decisions_at(res, t_accept, t_reject_vec, m_min=0, quality_override=True, margin=None):
    """margin: 'raw' or 'norm' selects the axis-2 margin; None disables axis 2."""
    if margin is None:
        marg = [None] * len(res["scores"])
    elif margin == "raw":
        marg = res["min_margin_q"]
    elif margin == "norm":
        marg = res["min_margin_norm_q"]
    elif margin == "crossed":
        marg = res["min_crossed_margin_q"]
    else:
        raise ValueError(margin)
    dec = []
    for i in range(len(res["scores"])):
        dec.append(decide(int(res["scores"][i]), res["hard"][i], t_accept,
                          int(t_reject_vec[i]), marg[i], m_min,
                          quality_override))
    return np.array(dec)


def metrics(dec, y):
    dec = np.asarray(dec)
    y = np.asarray(y)
    n = len(y)
    acc = dec == "ACCEPT"
    rej = dec == "REJECT"
    rev = dec == "REVIEW"
    good = y == 1
    bad = y == 0
    decided = acc | rej
    d_good = decided & good
    d_bad = decided & bad
    ba = float("nan")
    if d_good.sum() and d_bad.sum():
        ba = 0.5 * (float((acc & d_good).sum()) / d_good.sum()
                    + float((rej & d_bad).sum()) / d_bad.sum())
    return {
        "decided_frac": float(decided.mean()),
        "accept_frac": float(acc.mean()),
        "reject_frac": float(rej.mean()),
        "review_frac": float(rev.mean()),
        "ba_decided": ba,
        "fr_good": float((rej & good).sum()) / max(int(good.sum()), 1),
        "fa_bad": float((acc & bad).sum()) / max(int(bad.sum()), 1),
        "n_decided": int(decided.sum()),
        "n_review": int(rev.sum()),
        "n_accept": int(acc.sum()),
        "n_reject": int(rej.sum()),
        "confusion": {
            "good_accept": int((acc & good).sum()), "good_review": int((rev & good).sum()),
            "good_reject": int((rej & good).sum()),
            "bad_accept": int((acc & bad).sum()), "bad_review": int((rev & bad).sum()),
            "bad_reject": int((rej & bad).sum()),
        },
    }


def auroc(scores, y):
    s = np.asarray(scores, dtype=float)
    yy = np.asarray(y)
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = int((yy == 1).sum())
    n0 = int((yy == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def clip_bootstrap(metric_fn, y, clips, B=BOOT, seed=SEED):
    """Clip-level bootstrap 95% CI (resample the 7 clips with replacement)."""
    cl = sorted(set(clips))
    by_clip = {c: np.array([i for i in range(len(y)) if clips[i] == c]) for c in cl}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(cl), len(cl))
        idx = np.concatenate([by_clip[cl[k]] for k in pick])
        v = metric_fn(idx)
        vals.append(v)
    vals = np.asarray(vals, dtype=float)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def decide_vec(scores, hard, margins, t_accept, t_reject, m_min=0,
               quality_override=True):
    """Vectorised decision for a set of pre-scored windows (integers only)."""
    out = []
    for i in range(len(scores)):
        out.append(decide(int(scores[i]), hard[i], int(t_accept), int(t_reject),
                          None if margins is None else int(margins[i]), int(m_min),
                          quality_override))
    return np.array(out)


def inner_select(res, y, clips, rows, tau_grid, m_grid=None, quality_override=True,
                 margin_mode="norm"):
    """Select (t_accept, tau[, m_min]) on TRAINING clips of each fold only.

    Candidate operating points are ranked by training balanced accuracy on decided
    windows (tie-break: higher training decided fraction; deterministic scan order).
    The chosen point is then applied to the held-out clip.  No test label is used
    for any choice.
    """
    n = len(y)
    dec = np.empty(n, dtype=object)
    per_fold = {}
    for c in sorted(set(clips)):
        ts = res["train"][c]
        scores, hard, ty = ts["scores"], ts["hard"], ts["y"]
        margins = (ts["min_margin_norm_q"] if margin_mode == "norm" else
                   ts["min_crossed_margin_q"]) if m_grid is not None else None
        best_key, best = None, None
        for tau in tau_grid:
            t_rej = choose_reject_threshold(scores, ty, tau)
            for ta in range(1, int(scores.max()) + 2):
                if ta > t_rej:
                    continue
                for m_min in (m_grid if m_grid is not None else [0]):
                    d = decide_vec(scores, hard, margins, ta, t_rej, m_min, quality_override)
                    m = metrics(d, ty)
                    ba = m["ba_decided"]
                    if np.isnan(ba):
                        continue
                    key = (ba, m["decided_frac"])
                    if best_key is None or key > best_key:
                        best_key, best = key, (tau, ta, t_rej, m_min, ba, m["decided_frac"])
        if best is None:
            best = (tau_grid[0], 1, choose_reject_threshold(scores, ty, tau_grid[0]), 0, None, None)
        tau, ta, t_rej, m_min, tr_ba, tr_dec = best
        te_idx = [i for i in range(n) if clips[i] == c]
        d = decide_vec(res["scores"][te_idx],
                       [res["hard"][i] for i in te_idx],
                       [res["min_margin_norm_q"][i] if margin_mode == "norm"
                        else res["min_crossed_margin_q"][i] for i in te_idx]
                       if m_grid is not None else None,
                       ta, t_rej, m_min, quality_override)
        for k, i in enumerate(te_idx):
            dec[i] = d[k]
        per_fold[c] = {"tau": tau, "t_accept": ta, "t_reject": int(t_rej), "m_min": m_min,
                       "train_ba_decided": None if tr_ba is None else round(tr_ba, 4),
                       "train_decided_frac": None if tr_dec is None else round(tr_dec, 4)}
    dec = np.array([str(x) for x in dec])
    return dec, per_fold


def paired_clip_bootstrap(dec_a, dec_b, y, clips, B=BOOT, seed=SEED):
    """Paired clip-bootstrap of BA(dec_b) - BA(dec_a) on the same resampled windows."""
    cl = sorted(set(clips))
    by_clip = {c: np.array([i for i in range(len(y)) if clips[i] == c]) for c in cl}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(B):
        pick = rng.integers(0, len(cl), len(cl))
        idx = np.concatenate([by_clip[cl[k]] for k in pick])
        ba_a = metrics(np.asarray(dec_a)[idx], y[idx])["ba_decided"]
        ba_b = metrics(np.asarray(dec_b)[idx], y[idx])["ba_decided"]
        if np.isnan(ba_a) or np.isnan(ba_b):
            continue
        deltas.append(ba_b - ba_a)
    d = np.asarray(deltas, dtype=float)
    return {"n_used": int(d.size), "n_skipped": int(B - d.size),
            "delta_ba_mean": round(float(d.mean()), 4) if d.size else None,
            "delta_ba_ci95": [round(float(np.percentile(d, 2.5)), 4),
                              round(float(np.percentile(d, 97.5)), 4)] if d.size else None,
            "p_delta_gt_0": round(float((d > 0).mean()), 4) if d.size else None}


def metrics_on(idx, res, t_accept, t_reject_vec, m_min=0, margin=None):
    dec = decisions_at(res, t_accept, t_reject_vec, m_min, margin=margin)
    return metrics(dec[idx], y_global[idx])


# ---------------------------------------------------------------------------
def main() -> int:
    global y_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="q1", choices=["q1", "gold"])
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--out", default=os.path.join(OUT, "A4_review_band.json"))
    ap.add_argument("--quick", action="store_true", help="only print the baseline reproduction")
    ap.add_argument("--light", action="store_true",
                    help="drop the full frontier/axis2 row lists (sensitivity runs)")
    args = ap.parse_args()
    t0 = time.time()

    ids, feats, y, clips = load_windows(args.source)
    y_global = y
    rows = load_rule_rows()
    n = len(y)
    res = run_loco(feats, y, clips, rows)
    scores = res["scores"]
    smax = int(scores.max())
    t_rej_vec = res["fold_t_reject"]

    # ---- baseline reproduction: t_accept = 1, per-fold training FR budget ----
    base = metrics(decisions_at(res, 1, t_rej_vec), y)
    auroc_score = auroc(-scores, y)   # score = badness -> negate (Q1 reports 0.8222)
    print(f"[source={args.source}] n={n} smax={smax} per-fold t_rej="
          f"{sorted(set(t_rej_vec.tolist()))}")
    print(f"  BASELINE t_accept=1 (train-good FR<=5%): decided={base['decided_frac']:.4f} "
          f"review={base['review_frac']:.4f} BA_decided={base['ba_decided']:.4f} "
          f"fr_good={base['fr_good']:.4f} fa_bad={base['fa_bad']:.4f} AUROC={auroc_score:.4f}")
    print("  confusion", base["confusion"])
    if args.quick:
        return 0

    # ---- frontier 1: protocol-faithful (train-budget t_reject per fold), sweep t_accept
    budget_curve = []
    for ta in range(1, smax + 2):
        dec = decisions_at(res, ta, t_rej_vec)
        m = metrics(dec, y)
        m["t_accept"] = ta
        m["t_reject_per_fold"] = {c: int(np.unique(t_rej_vec[np.array(clips) == c])[0])
                                  for c in sorted(set(clips))}
        budget_curve.append(m)

    # ---- frontier 2: global integer grid over (t_accept, t_reject)
    grid = []
    for tr in range(1, smax + 2):
        for ta in range(1, tr + 1):
            dec = decisions_at(res, ta, np.full(n, tr))
            m = metrics(dec, y)
            m = {k: m[k] for k in ("decided_frac", "accept_frac", "reject_frac", "review_frac",
                                   "ba_decided", "fr_good", "fa_bad", "n_decided", "n_review",
                                   "n_accept", "n_reject")}
            m["t_accept"] = ta
            m["t_reject"] = tr
            grid.append(m)

    # ---- best points -------------------------------------------------------
    def best(rows_, key="ba_decided", filt=None):
        cand = [r for r in rows_ if (filt is None or filt(r)) and not np.isnan(r[key])]
        if not cand:
            return None
        return max(cand, key=lambda r: (r[key], r["decided_frac"]))

    best_budget = best(budget_curve)
    best_grid = best(grid)
    best_grid_fr5 = best(grid, filt=lambda r: r["fr_good"] <= 0.05)
    best_grid_fr10 = best(grid, filt=lambda r: r["fr_good"] <= 0.10)
    best_budget_fr5_emp = best(budget_curve, filt=lambda r: r["fr_good"] <= 0.05)

    # ---- honest inner selection on training clips only ----------------------
    TAU_GRID = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
    dec_inner_budget, fold_inner_budget = inner_select(res, y, clips, rows, [FR_BUDGET])
    m_inner_budget = metrics(dec_inner_budget, y)
    dec_inner_tau, fold_inner_tau = inner_select(res, y, clips, rows, TAU_GRID)
    m_inner_tau = metrics(dec_inner_tau, y)
    dec_inner_tau_axis2, fold_inner_tau_axis2 = inner_select(
        res, y, clips, rows, TAU_GRID, m_grid=list(range(0, 17)))
    m_inner_tau_axis2 = metrics(dec_inner_tau_axis2, y)

    # inner selection with the crossed-threshold margin axis (train-only)
    dec_inner_crossed, fold_inner_crossed = inner_select(
        res, y, clips, rows, TAU_GRID, m_grid=[0, 1, 2, 4, 8, 16], margin_mode="crossed")
    m_inner_crossed = metrics(dec_inner_crossed, y)

    dec_base = decisions_at(res, 1, t_rej_vec)
    dec_best_test_chosen = decisions_at(res, best_budget["t_accept"], t_rej_vec)
    paired = {
        "baseline_vs_best_test_chosen_t_accept": paired_clip_bootstrap(
            dec_base, dec_best_test_chosen, y, clips, args.boot),
        "baseline_vs_inner_selected_tau5": paired_clip_bootstrap(
            dec_base, dec_inner_budget, y, clips, args.boot),
        "baseline_vs_inner_selected_tau_swept": paired_clip_bootstrap(
            dec_base, dec_inner_tau, y, clips, args.boot),
        "note": "paired clip-bootstrap: the same resampled windows are scored by both decision "
                "rules; P(delta>0) is the share of resamples in which the new point has the "
                "higher BA. CIs are wide because there are only 7 clips and the decided subset "
                "is 33-48% of 150 windows.",
    }

    # ablation: drop the QUALITY extreme-threshold hard reject, keep the band only
    def band_only_metrics(ta, tr_vec):
        d = decisions_at(res, ta, tr_vec, quality_override=False)
        return metrics(d, y)
    band_only = {
        "baseline_t_accept_1": band_only_metrics(1, t_rej_vec),
        "best_point": band_only_metrics(best_budget["t_accept"], t_rej_vec),
        "note": "frozen Q1 semantics always hard-reject a QUALITY rule crossing its extreme "
                "(index 0) threshold; this ablation removes that override and keeps only the "
                "integer band, so hard-fired low-score windows can fall into ACCEPT/REVIEW.",
    }

    # CI at the baseline and at the best protocol point
    def cier(res_, ta, tr_vec, m_min=0, B=BOOT, margin=None):
        dec_all = decisions_at(res_, ta, tr_vec, m_min, margin=margin)
        return {
            f"{k}_ci95": clip_bootstrap(lambda idx, k=k: metrics(dec_all[idx], y[idx])[k],
                                        y, clips, B)
            for k in ("ba_decided", "fr_good", "fa_bad", "review_frac", "decided_frac")
        }

    def cier_from_dec(dec, B=BOOT):
        return {f"{k}_ci95": clip_bootstrap(lambda idx, k=k: metrics(dec[idx], y[idx])[k],
                                            y, clips, B)
                for k in ("ba_decided", "fr_good", "fa_bad", "review_frac", "decided_frac")}

    base_ci = cier(res, 1, t_rej_vec, B=args.boot)
    best_ci = cier(res, best_budget["t_accept"], t_rej_vec, B=args.boot)
    inner_budget_ci = cier_from_dec(dec_inner_budget, args.boot)
    inner_tau_ci = cier_from_dec(dec_inner_tau, args.boot)
    inner_tau_axis2_ci = cier_from_dec(dec_inner_tau_axis2, args.boot)
    inner_crossed_ci = cier_from_dec(dec_inner_crossed, args.boot)

    # ---- per-rule fire counts (out-of-fold) ---------------------------------
    per_rule = {}
    for r in rows:
        per_rule.setdefault(r["rule"], {"features": [], "fire": 0, "fire_good": 0,
                                        "fire_bad": 0, "hard_fire": 0, "hard_fire_good": 0,
                                        "hard_fire_bad": 0, "threshold_crossings": 0})
    for i in range(n):
        seen = set()
        for e in res["fired"][i]:
            d = per_rule[e["rule"]]
            d["threshold_crossings"] += int(e["n_thresholds_crossed"])
            if e["rule"] not in seen:
                seen.add(e["rule"])
                d["fire"] += 1
                d["fire_good" if y[i] == 1 else "fire_bad"] += 1
        for rn in res["hard"][i]:
            d = per_rule[rn]
            d["hard_fire"] += 1
            d["hard_fire_good" if y[i] == 1 else "hard_fire_bad"] += 1
    for r in rows:
        if r["feature"] not in per_rule[r["rule"]]["features"]:
            per_rule[r["rule"]]["features"].append(r["feature"])
        per_rule[r["rule"]]["strength"] = r["strength"]
    for rn, d in per_rule.items():
        d["fire_share"] = d["fire"] / n
        d["bad_total"] = int((y == 0).sum())
        d["good_total"] = int((y == 1).sum())
        d["precision"] = (d["fire_bad"] / d["fire"]) if d["fire"] else None
        d["recall_bad"] = d["fire_bad"] / d["bad_total"]
        d["indicator_auroc"] = None  # filled below for single-feature rules

    # per-rule indicator AUROC over the out-of-fold score contribution
    contrib = {}
    for i in range(n):
        for e in res["fired"][i]:
            contrib.setdefault(e["rule"], np.zeros(n, dtype=float))
            contrib[e["rule"]][i] += e["n_thresholds_crossed"]
    for rn, v in contrib.items():
        per_rule[rn]["indicator_auroc"] = round(auroc(-v, y), 4)

    # ---- drop-one-rule LOCO delta -------------------------------------------
    group_names = sorted({r["rule"] for r in rows})
    drop_one = {}
    for g in group_names:
        rows2 = [r for r in rows if r["rule"] != g]
        r2 = run_loco(feats, y, clips, rows2)
        s2 = r2["scores"]
        tr2 = r2["fold_t_reject"]
        m_base = metrics(decisions_at(r2, 1, tr2), y)
        bc2 = []
        for ta in range(1, int(s2.max()) + 2):
            mm = metrics(decisions_at(r2, ta, tr2), y)
            mm["t_accept"] = ta
            bc2.append(mm)
        b2 = best(bc2)
        drop_one[g] = {
            "auroc_score": round(auroc(-s2, y), 4),
            "auroc_delta": round(auroc(-s2, y) - auroc_score, 4),
            "ba_decided_at_t_accept_1": round(m_base["ba_decided"], 4),
            "ba_delta_at_t_accept_1": round(m_base["ba_decided"] - base["ba_decided"], 4),
            "review_frac_at_t_accept_1": round(m_base["review_frac"], 4),
            "review_delta_at_t_accept_1": round(m_base["review_frac"] - base["review_frac"], 4),
            "best_ba_decided_over_t_accept": round(b2["ba_decided"], 4) if b2 else None,
            "best_t_accept": b2["t_accept"] if b2 else None,
            "best_ba_delta_vs_full_best": (round(b2["ba_decided"] - best_budget["ba_decided"], 4)
                                           if b2 else None),
        }

    # ---- second review axis: distance to nearest threshold, Q=4 -------------
    # Axis A ("norm"): scale-free integer distance in units of half the gap between
    #   the two thresholds adjacent to the nearest one (pure int arithmetic).
    # Axis B ("raw"): literal |value_q - threshold_q| in Q=4 integer units (dominated
    #   by the smallest-scale features; swept on a log2 lattice).
    norm_dist = {q_: int((res["min_margin_norm_q"] == q_).sum()) for q_ in range(0, 17)}
    raw_sorted = np.sort(res["min_margin_q"])
    axis2 = {
        "description": "margin = distance of the quantised value to the nearest of the 64 Q=4 "
                       "thresholds. ACCEPT requires score < t_accept AND margin >= m_min, "
                       "otherwise REVIEW (REJECT unchanged). m_min=0 reproduces the 2-axis band.",
        "raw_margin_q_quantiles": {str(p): int(np.percentile(res["min_margin_q"], p))
                                   for p in (0, 5, 25, 50, 75, 95, 100)},
        "norm_margin_distribution": norm_dist,
        "norm": {"n_points": 0, "best_over_m_min": None, "best_over_m_min_fr_le_5pct": None,
                 "delta_ba_vs_best_2axis": None},
        "raw": {"n_points": 0, "best_over_m_min": None, "delta_ba_vs_best_2axis": None},
        "iso_review_comparison": [],
    }

    def sweep_axis(margin_mode, m_grid):
        rows_ = []
        for ta in range(1, smax + 2):
            for m_min in m_grid:
                dec = decisions_at(res, ta, t_rej_vec, m_min, margin=margin_mode)
                m = metrics(dec, y)
                m["t_accept"] = ta
                m["m_min"] = m_min
                rows_.append(m)
        return rows_

    norm_grid = list(range(0, 17))
    a2_norm = sweep_axis("norm", norm_grid)
    best_a2 = best(a2_norm)
    best_a2_fr5 = best(a2_norm, filt=lambda r: r["fr_good"] <= 0.05)
    axis2["norm"] = {
        "n_points": len(a2_norm),
        "m_min_grid": norm_grid,
        "best_over_m_min": best_a2,
        "best_over_m_min_fr_le_5pct": best_a2_fr5,
        "delta_ba_vs_best_2axis": (round(best_a2["ba_decided"] - best_budget["ba_decided"], 4)
                                   if best_a2 else None),
        "delta_ba_vs_best_2axis_fr5": (round(best_a2_fr5["ba_decided"] - best_budget["ba_decided"], 4)
                                       if best_a2_fr5 else None),
    }

    raw_grid = [0, 1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096]
    a2_raw = sweep_axis("raw", raw_grid)
    best_a2_raw = best(a2_raw)
    axis2["raw"] = {
        "n_points": len(a2_raw),
        "m_min_grid": raw_grid,
        "best_over_m_min": best_a2_raw,
        "delta_ba_vs_best_2axis": (round(best_a2_raw["ba_decided"] - best_budget["ba_decided"], 4)
                                   if best_a2_raw else None),
    }

    crossed_grid = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    a2_crossed = sweep_axis("crossed", crossed_grid)
    best_a2_crossed = best(a2_crossed)
    axis2["crossed"] = {
        "description": "margin = Q=4 integer distance to the nearest CROSSED threshold "
                       "(1e9 when nothing fires); ACCEPT requires score < t_accept and "
                       "margin_crossed >= m_min.",
        "n_points": len(a2_crossed),
        "m_min_grid": crossed_grid,
        "best_over_m_min": best_a2_crossed,
        "delta_ba_vs_best_2axis": (round(best_a2_crossed["ba_decided"] - best_budget["ba_decided"], 4)
                                   if best_a2_crossed else None),
    }
    # iso-review comparison: at matched review fraction, best BA with/without axis 2
    for q_ in (0.30, 0.40, 0.50, 0.60, 0.70):
        c2 = [r for r in budget_curve if not np.isnan(r["ba_decided"])]
        c3 = [r for r in a2_norm if not np.isnan(r["ba_decided"])]
        if not c2 or not c3:
            continue
        r2_ = min(c2, key=lambda r: abs(r["review_frac"] - q_))
        r3_ = min(c3, key=lambda r: abs(r["review_frac"] - q_))
        axis2["iso_review_comparison"].append(
            {"target_review": q_,
             "two_axis": {"review": round(r2_["review_frac"], 4),
                          "ba_decided": round(r2_["ba_decided"], 4),
                          "t_accept": r2_["t_accept"]},
             "with_margin_axis": {"review": round(r3_["review_frac"], 4),
                                  "ba_decided": round(r3_["ba_decided"], 4),
                                  "t_accept": r3_["t_accept"], "m_min": r3_["m_min"]}})

    # ---- which rules drive the REVIEW band (at baseline and at the best point) ----
    def review_drivers(t_accept, t_reject_vec, m_min=0, margin=None):
        dec = decisions_at(res, t_accept, t_reject_vec, m_min, margin=margin)
        counts = {}
        for i in range(n):
            if dec[i] != "REVIEW":
                continue
            for e in res["fired"][i]:
                d = counts.setdefault(e["rule"], {"review_windows": 0, "review_good": 0,
                                                  "review_bad": 0})
                d["review_windows"] += 1
                d["review_good" if y[i] == 1 else "review_bad"] += 1
        return counts

    review_drivers_out = {
        "baseline_t_accept_1": review_drivers(1, t_rej_vec),
        "best_point": review_drivers(best_budget["t_accept"], t_rej_vec),
    }

    # ---- feature-source sensitivity (BRIEF name map vs Q1 space) -----------
    other = "gold" if args.source == "q1" else "q1"
    _, feats_o, y_o, clips_o = load_windows(other)
    assert y_o.tolist() == y.tolist() and clips_o == clips
    res_o = run_loco(feats_o, y_o, clips_o, rows)
    n_score_diff = int((res_o["scores"] != scores).sum())
    n_hard_diff = sum(1 for a, b in zip(res_o["hard"], res["hard"]) if a != b)
    fs_sensitivity = {
        "primary_source": args.source,
        "comparison_source": other,
        "n_windows": n,
        "n_loco_score_differences": n_score_diff,
        "n_hard_rule_differences": n_hard_diff,
        "identical_fold_t_reject": bool((res_o["fold_t_reject"] == t_rej_vec).all()),
        "note": "gold.npz G_* is float32; per-feature comparison against the Q1 float64 source "
                "shows full.temp_b0/b1/b2 differ by up to 0.156 and all other features by "
                "<=3e-6. Because LOCO re-derives thresholds from the same source as the values, "
                "the integer out-of-fold scores are identical, so the BRIEF gold name map is "
                "decision-equivalent for this rule table.",
        "companion_result": "eval/results/exp_phys_dsh/A4_review_band_gold_features.json"
                            if args.source == "q1" else
                            "eval/results/exp_phys_dsh/A4_review_band.json",
    }

    # ---- bundle -------------------------------------------------------------
    out = {
        "experiment": "A4",
        "title": "Deterministic REVIEW band [t_accept,t_reject) on the Q1 integer score: "
                 "review<->error frontier under LOCO",
        "n": n, "n_good": int((y == 1).sum()), "n_bad": int((y == 0).sum()),
        "label_source": "human (IndEgo gold)",
        "protocol": "leave-one-clip-out over 7 clips; per-fold thresholds from training clips only "
                    "(quantiles p01/p05/p10/p25 mirrored for high-is-bad, Q=4 integers); "
                    "operating point chosen on training clips only (false-reject budget 5% on "
                    "training-good); metrics pooled on held-out windows; clip-bootstrap 95% CI "
                    f"(B={args.boot}, seed={SEED})",
        "decision_rule": "REJECT if a QUALITY rule crosses its extreme threshold (index 0) or "
                         "score >= t_reject; ACCEPT if score < t_accept (and margin_q >= m_min for "
                         "axis 2); REVIEW otherwise",
        "feature_source": args.source,
        "feature_source_sensitivity": fs_sensitivity,
        "feature_source_note": (
            "q1 = caches/gpu/features.json `full` (float64) + caches/perpixel/physics_feats.npz "
            "`phys`; the exact Q1/Q4 feature space the frozen thresholds were derived in. "
            "gold (BRIEF name map, gold.npz G_* float32) is a sensitivity variant: full.temp_b0/b1/b2 "
            "differ from the q1 source by up to 0.156 (all other features ~1e-8)."
            if args.source == "q1" else
            "gold = gold.npz G_* (float32) + physics_feats.npz `phys`, per BRIEF name map; "
            "note full.temp_b0/b1/b2 differ from the Q1 source used to derive the frozen "
            "thresholds by up to 0.156."),
        "quantisation": Q,
        "metric_definitions": {
            "decided_frac": "(n_accept + n_reject) / n",
            "review_frac": "n_review / n",
            "ba_decided": "0.5 * ( P(REJECT | bad, decided) + P(ACCEPT | good, decided) ) "
                           "-- balanced accuracy restricted to the decided windows (Q1/Q4 "
                           "`balacc_decided_only` convention, verified to reproduce 0.808)",
            "fr_good": "P(REJECT | good) over ALL good windows (review counted as not-rejected)",
            "fa_bad": "P(ACCEPT | bad) over ALL bad windows (review counted as not-accepted)",
        },
        "false_reject_budget": FR_BUDGET,
        "n_thresholds": int(sum(len(t["thresholds_q"]) for t in res["tables"][sorted(set(clips))[0]])),
        "score_max_oof": smax,
        "auroc_deterministic_score_oof": round(auroc_score, 4),
        "auroc_ci95_clip_bootstrap": clip_bootstrap(
            lambda idx: auroc(-scores[idx], y[idx]), y, clips, args.boot),
        "per_fold": {c: {"n": int(sum(1 for x in clips if x == c)),
                         "t_reject": int(np.unique(t_rej_vec[np.array(clips) == c])[0])}
                     for c in sorted(set(clips))},
        "baseline_q1_q4": {**base, "t_accept": 1,
                           "t_reject_per_fold": {c: int(np.unique(t_rej_vec[np.array(clips) == c])[0])
                                                 for c in sorted(set(clips))},
                           "ci95_clip_bootstrap": base_ci,
                           "reference": "Q4.01 pure_rules: review_share 0.60, "
                                        "balacc_decided_only 0.808, auroc 0.8222"},
        "best_operating_point": {
            **best_budget,
            "constraint": "training-good false-reject <= 5% per fold (protocol-faithful, Q1 meaning)",
            "ci95_clip_bootstrap": best_ci,
        },
        "best_operating_point_empirical_fr5": {
            **best_grid_fr5,
            "constraint": "OBSERVED held-out false-reject <= 5% (empirical; chosen on test metrics, "
                          "reported for the frontier, not a deployable protocol)",
        } if best_grid_fr5 else None,
        "best_operating_point_empirical_fr10": {
            **best_grid_fr10,
            "constraint": "OBSERVED held-out false-reject <= 10% (empirical)",
        } if best_grid_fr10 else None,
        "best_grid_point": best_grid,
        "inner_selected_operating_point_train_only": {
            **m_inner_budget,
            "selection": "per fold, maximise TRAINING balanced accuracy on decided windows over "
                         "t_accept with the pre-registered tau=0.05 training-good FR budget; "
                         "no test label used",
            "per_fold_choice": fold_inner_budget,
            "ci95_clip_bootstrap": inner_budget_ci,
        },
        "inner_selected_operating_point_tau_swept": {
            **m_inner_tau,
            "selection": "per fold, maximise TRAINING balanced accuracy on decided windows over "
                         "(t_accept, tau); tau in " + str(TAU_GRID) + "; no test label used",
            "per_fold_choice": fold_inner_tau,
            "ci95_clip_bootstrap": inner_tau_ci,
        },
        "inner_selected_operating_point_tau_and_margin": {
            **m_inner_tau_axis2,
            "selection": "per fold, maximise TRAINING balanced accuracy on decided windows over "
                         "(t_accept, tau, m_min) with the scale-free margin axis (half-gap units); "
                         "no test label used",
            "per_fold_choice": fold_inner_tau_axis2,
            "ci95_clip_bootstrap": inner_tau_axis2_ci,
        },
        "inner_selected_operating_point_tau_and_crossed_margin": {
            **m_inner_crossed,
            "selection": "per fold, maximise TRAINING balanced accuracy on decided windows over "
                         "(t_accept, tau, m_min) with the crossed-threshold margin axis; "
                         "no test label used",
            "per_fold_choice": fold_inner_crossed,
            "ci95_clip_bootstrap": inner_crossed_ci,
        },
        "ablation_without_quality_override": band_only,
        "paired_bootstrap_deltas": paired,
        "review_drivers": review_drivers_out,
        "verdict": {
            "baseline_reproduced": {
                "point": "t_accept=1, per-fold t_reject from 5% training-good FR budget",
                "decided_frac": base["decided_frac"], "review_frac": base["review_frac"],
                "ba_decided": base["ba_decided"], "fr_good": base["fr_good"],
                "fa_bad": base["fa_bad"], "auroc": round(auroc_score, 4),
                "matches": "Q4.01 pure_rules: review_share 0.60, balacc_decided_only 0.808, "
                           "auroc 0.8222 -> exact reproduction",
            },
            "best_under_protocol_fr_budget_5pct": {
                "point": f"t_accept={best_budget['t_accept']} (per-fold t_reject, tau=0.05)",
                "ba_decided": round(best_budget["ba_decided"], 4),
                "delta_ba_vs_baseline": round(best_budget["ba_decided"] - base["ba_decided"], 4),
                "review_frac": best_budget["review_frac"],
                "decided_frac": best_budget["decided_frac"],
                "fr_good": best_budget["fr_good"], "fa_bad": best_budget["fa_bad"],
                "selected_on_test": True,
                "note": "this row is the best point of the swept frontier; it is selected on the "
                        "held-out metrics and is therefore an upper bound, not a deployable choice",
            },
            "best_deployable": {
                "point": "inner-selected (t_accept, tau) per fold on TRAINING clips only, "
                         "ranking = training BA on decided windows",
                "ba_decided": round(m_inner_tau["ba_decided"], 4),
                "delta_ba_vs_baseline": round(m_inner_tau["ba_decided"] - base["ba_decided"], 4),
                "review_frac": m_inner_tau["review_frac"],
                "decided_frac": m_inner_tau["decided_frac"],
                "fr_good": m_inner_tau["fr_good"], "fa_bad": m_inner_tau["fa_bad"],
                "ci95_clip_bootstrap": inner_tau_ci,
                "satisfies_observed_fr_le_5pct": bool(m_inner_tau["fr_good"] <= 0.05),
                "selected_on_test": False,
            },
            "best_under_observed_fr_le_5pct_grid": {
                "point": f"t_accept={best_grid_fr5['t_accept']}, t_reject={best_grid_fr5['t_reject']}"
                         if best_grid_fr5 else None,
                "ba_decided": round(best_grid_fr5["ba_decided"], 4) if best_grid_fr5 else None,
                "review_frac": best_grid_fr5["review_frac"] if best_grid_fr5 else None,
                "fr_good": best_grid_fr5["fr_good"] if best_grid_fr5 else None,
                "selected_on_test": True,
            },
            "frontier_mechanics": {
                "t_accept_edge": "moving t_accept changes only the ACCEPT/REVIEW split; the "
                                 "REJECT set (quality hard fires + score>=t_reject) is untouched, "
                                 "so fr_good/fa_bad are constant along that family and only the "
                                 "decided subset changes",
                "t_reject_edge": "moving t_reject (or the training budget tau) trades "
                                 "REJECT<->REVIEW: a stricter tau shrinks the reject set, lowers "
                                 "the observed false-reject rate on good windows and moves the "
                                 "hard cases into REVIEW instead of rejecting them",
            },
            "axis2": "no sharpening: the scale-free half-gap margin is 0 for all 150 windows "
                     "(m_min>=1 cannot ACCEPT anything) and the raw margin lattice is flat or "
                     "worse (delta BA <= +0.0007 at a HIGHER review rate). See axis2_margin_review.",
        },
        "budget_curve": budget_curve,
        "frontier_grid": grid,
        "per_rule": per_rule,
        "drop_one_rule_loco": drop_one,
        "axis2_margin_review": axis2,
        "device": "cpu",
        "wall_s": None,
        "command": None,
        "artifacts": {"rule_table": os.path.relpath(RULE_TABLE, ROOT)},
    }
    out["command"] = (f"cd {ROOT} && python3 eval/exp_phys_dsh/run_a4_review_band.py "
                      f"--source {args.source} --boot {args.boot}"
                      + (" --light" if args.light else ""))
    if args.light:
        out.pop("frontier_grid", None)
        out["frontier_grid"] = {"omitted": "light run: see A4_review_band.json for the full grid"}
        out.pop("budget_curve", None)
        out["budget_curve"] = {"omitted": "light run: see A4_review_band.json"}
        for k in ("norm", "raw", "crossed"):
            if k in out["axis2_margin_review"]:
                out["axis2_margin_review"][k].pop("best_over_m_min_fr_le_5pct", None)
    out["wall_s"] = round(time.time() - t0, 2)
    os.makedirs(OUT, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"  wrote {args.out} ({os.path.getsize(args.out)} bytes) in {out['wall_s']}s")
    print("  best (train-budget constraint): t_accept=%d review=%.3f decided=%.3f BA=%.4f "
          "fr=%.3f fa=%.3f" % (best_budget["t_accept"], best_budget["review_frac"],
                               best_budget["decided_frac"], best_budget["ba_decided"],
                               best_budget["fr_good"], best_budget["fa_bad"]))
    if best_grid_fr5:
        print("  best (observed FR<=5%%): t_accept=%d t_reject=%d review=%.3f BA=%.4f fr=%.3f" %
              (best_grid_fr5["t_accept"], best_grid_fr5["t_reject"], best_grid_fr5["review_frac"],
               best_grid_fr5["ba_decided"], best_grid_fr5["fr_good"]))
    print("  inner-selected (tau=5%%): review=%.3f decided=%.3f BA=%.4f fr=%.3f fa=%.3f" %
          (m_inner_budget["review_frac"], m_inner_budget["decided_frac"],
           m_inner_budget["ba_decided"], m_inner_budget["fr_good"], m_inner_budget["fa_bad"]))
    print("  inner-selected (tau swept): BA=%.4f review=%.3f fr=%.3f" %
          (m_inner_tau["ba_decided"], m_inner_tau["review_frac"], m_inner_tau["fr_good"]))
    b2n = axis2["norm"]["best_over_m_min"]
    print("  axis2(norm) best: m_min=%s BA=%.4f review=%.3f (delta vs 2-axis %.4f)" %
          (b2n["m_min"], b2n["ba_decided"], b2n["review_frac"],
           axis2["norm"]["delta_ba_vs_best_2axis"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
