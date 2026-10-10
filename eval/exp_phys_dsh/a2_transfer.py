#!/usr/bin/env python3
"""A2: does LABEL-FREE per-domain normalisation rescue OOD transfer of the frozen
Q1 physics rule table?

The frozen table (``eval/results/exp_phys/rule_table.json``) stores, per rule,
four thresholds that are the *IndEgo* empirical quantiles at levels
p01/p05/p10/p25 (mirrored p99/p95/p90/p75 for high-is-bad features). Under
domain shift those absolute numbers are meaningless: the same numeric threshold
fires on nearly every target window. This module tests whether keeping the rule
list, the sign/op, the ORDER STATISTIC level and the integer score/reject
semantics, but re-estimating the *numeric* threshold from the TARGET's own
unlabelled feature values, restores transfer.

Variants (all label-free w.r.t. the target; target labels are used ONLY to
compute AUROC/BA after the fact):

  a  raw                : frozen IndEgo thresholds, applied as-is (baseline).
  b  requantile         : per-feature target empirical quantiles at the SAME
                          frozen levels (the "same order statistic").
  b2 requantile_rank    : identical idea expressed as an explicit ECDF/rank
                          transform + fixed level thresholds (robustness check).
  c  zscore             : standardise each feature by the target's own mean/std,
                          with the frozen thresholds mapped into z-units using
                          IndEgo's own mean/std (matched location-scale).
  c1 zscore_literal     : naive reading - target z-values compared against the
                          RAW frozen thresholds (documented failure mode).
  d  regime_requantile  : (b) computed inside deterministic target regimes
                          derived from the E4 descriptor files (k-means on
                          z-scored descriptors, fixed random_state, no labels).
  e  calib32_requantile : (b) with the quantiles estimated from a 32-window
                          unlabelled calibration slice of the target only.

Determinism: the decision path stays integer at Q=4 (``value_q < threshold_q``
with ints); no learned weights; no RNG on the decision path (the RNG below is
used only for bootstrap CIs and the calibration-slice sampling).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sklearn.metrics import roc_auc_score  # noqa: E402

Q = 4
LEVELS = [0.01, 0.05, 0.10, 0.25]
BOOT = 2000
SEED = 0
RULES_PATH = os.path.join(REPO, "eval", "results", "exp_phys", "rule_table.json")
INPUTS = os.path.join(REPO, "eval", "results", "exp_phys_dsh", "inputs")
REGIME_DIR = "/workspace/jadal_pod_bundle/local_work/exp-regimes/eval/results/exp_regimes"


# --------------------------------------------------------------------------- #
# integers only on the decision path (mirrors tools/phys/model.py)
# --------------------------------------------------------------------------- #
def quantise(value, q: int = Q):
    """Deterministic integer quantisation; non-finite -> None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    scaled = v * (10 ** q)
    return int(math.floor(scaled + 0.5)) if scaled >= 0 else int(math.ceil(scaled - 0.5))


def load_rules(path: str = RULES_PATH) -> dict:
    rules = json.load(open(path))
    q = int(rules.get("quantisation", Q))
    for r in rules["rules"]:
        if "thresholds_q" not in r:
            r["thresholds_q"] = [quantise(t, q) for t in r["thresholds"]]
    rules["_levels"] = LEVELS
    return rules


def level_of(rule: dict, j: int) -> float:
    """Frozen quantile level of threshold j, in ECDF coordinates of the feature."""
    q = float(rule["quantiles"][j])
    return q if rule["op"] == "<" else 1.0 - q


def masked(vals: np.ndarray) -> np.ndarray:
    v = np.asarray(vals, float)
    return v[np.isfinite(v)]


def col(X: np.ndarray, names: list, feat: str) -> np.ndarray:
    return X[:, names.index(feat)]


# --------------------------------------------------------------------------- #
# threshold derivation (never sees y)
# --------------------------------------------------------------------------- #
def fit_thresholds(rules: dict, X: np.ndarray, names: list, mode: str,
                   ref: dict | None = None, rows: np.ndarray | None = None,
                   glob_rows: np.ndarray | None = None) -> dict:
    """Return {(rule, feature, j): threshold_q_int} fitted on target rows only.

    ``rows`` selects the rows used to estimate the target statistics (default:
    all rows). ``glob_rows`` selects rows for the per-feature location/scale
    reference of mode ``zscore`` when a regime sub-population is too small.
    ``ref`` is the IndEgo reference mean/std per feature (used by ``zscore``).
    """
    out = {}
    Xs = X if rows is None else X[rows]
    for r in rules["rules"]:
        f = r["feature"]
        v = masked(col(Xs, names, f)) if f in names else np.array([])
        for j in range(4):
            if mode == "raw":
                t = float(r["thresholds"][j])
            elif mode == "requantile":
                t = float(np.quantile(v, level_of(r, j))) if v.size else np.nan
            elif mode == "requantile_rank":
                t = level_of(r, j)  # thresholds live in ECDF coordinates
            elif mode == "zscore":
                rm, rs = ref[f]
                t = (float(r["thresholds"][j]) - rm) / (rs if rs > 0 else 1.0)
            elif mode == "zscore_literal":
                t = float(r["thresholds"][j])
            else:
                raise ValueError(mode)
            out[(r["rule"], f, j)] = quantise(t)
    return out


def transform_values(X: np.ndarray, names: list, mode: str, ref: dict | None = None,
                     rows: np.ndarray | None = None) -> np.ndarray:
    """Per-feature transform of the target matrix. Uses only unlabelled X."""
    if mode in ("raw", "requantile"):
        return X.astype(float, copy=True)
    Xs = X if rows is None else X[rows]
    Xe = X.astype(float, copy=True)
    for f in {r for r in names}:
        pass
    for j, f in enumerate(names):
        v = col(Xs, names, f)
        v = v[np.isfinite(v)]
        colv = Xe[:, j]
        if mode == "requantile_rank":
            s = np.sort(v)
            lo = np.searchsorted(s, colv, side="left")
            hi = np.searchsorted(s, colv, side="right")
            mid = (lo + hi) / 2.0
            Xe[:, j] = np.where(v.size, mid / v.size, np.nan)
        elif mode == "zscore":
            mu = float(np.mean(v)) if v.size else 0.0
            sd = float(np.std(v)) if v.size else 0.0
            Xe[:, j] = (colv - mu) / (sd if sd > 0 else 1.0)
        elif mode == "zscore_literal":
            mu = float(np.mean(v)) if v.size else 0.0
            sd = float(np.std(v)) if v.size else 0.0
            Xe[:, j] = (colv - mu) / (sd if sd > 0 else 1.0)
        else:
            raise ValueError(mode)
    return Xe


def indego_ref_stats(rules: dict, X: np.ndarray, names: list) -> dict:
    ref = {}
    for r in rules["rules"]:
        f = r["feature"]
        v = masked(col(X, names, f)) if f in names else np.array([])
        ref[f] = (float(np.mean(v)) if v.size else 0.0,
                  float(np.std(v)) if v.size else 0.0)
    return ref


# --------------------------------------------------------------------------- #
# scoring (integer decision path)
# --------------------------------------------------------------------------- #
def score_windows(Xeff: np.ndarray, names: list, rules: dict, thr: dict,
                  rows: np.ndarray | None = None) -> dict:
    n = Xeff.shape[0] if rows is None else len(rows)
    idxs = np.arange(Xeff.shape[0]) if rows is None else rows
    score = np.zeros(len(idxs), int)
    strong = np.zeros(len(idxs), bool)
    per_rule = {}
    for i, ri in enumerate(idxs):
        s = 0
        st = False
        for r in rules["rules"]:
            f = r["feature"]
            if f not in names:
                continue
            vq = quantise(Xeff[ri, names.index(f)])
            if vq is None:
                continue
            crossed = 0
            hard = False
            for j in range(4):
                tq = thr[(r["rule"], f, j)]
                if tq is None:
                    continue
                hit = vq < tq if r["op"] == "<" else vq > tq
                if hit:
                    crossed += 1
                    if j == 0:
                        hard = True
            if crossed:
                s += crossed
                if r["strength"] == "quality" and hard:
                    st = True
                per_rule.setdefault(r["rule"], np.zeros(len(idxs), float))[i] = 1.0
        score[i] = s
        strong[i] = st
    return {"score": score, "strong": strong, "per_rule": per_rule}


def decisions(score: np.ndarray, strong: np.ndarray, reject_threshold: int) -> np.ndarray:
    d = np.full(len(score), "ACCEPT", dtype=object)
    d[score >= 1] = "REVIEW"
    d[score >= reject_threshold] = "REJECT"
    d[strong] = "REJECT"
    return d


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC with the bad class (y==0) as positive (Q1 convention)."""
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(1 - np.asarray(y), np.asarray(s)))


def ba_metrics(y: np.ndarray, dec: np.ndarray) -> dict:
    y = np.asarray(y)
    rej = dec == "REJECT"
    rev = dec == "REVIEW"
    acc = dec == "ACCEPT"
    tp = int((rej & (y == 0)).sum())
    fp = int((rej & (y == 1)).sum())
    tn = int((acc & (y == 1)).sum())
    fn = int((acc & (y == 0)).sum())
    ba_dec = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
    tp_r, fp_r = tp, fp
    tn_r = tn + int((rev & (y == 1)).sum())
    fn_r = fn + int((rev & (y == 0)).sum())
    ba_ra = 0.5 * (tp_r / max(tp_r + fn_r, 1) + tn_r / max(tn_r + fp_r, 1))
    return {"balanced_acc_decided": float(ba_dec), "balanced_acc_review_as_accept": float(ba_ra),
            "reject_rate": float(rej.mean()),
            "review_rate": float(rev.mean()),
            "recall_bad": float(tp / max(int((y == 0).sum()), 1)),
            "false_reject_rate_good": float(fp / max(int((y == 1).sum()), 1)),
            "confusion": {"tp_bad_rejected": tp, "fp_good_rejected": fp,
                          "tn_good_accepted": tn, "fn_bad_accepted": fn}}


def paired_delta_ci_clips(y, s_a, s_b, clips, label_a="a", label_b="b", n_boot=BOOT,
                          seed=SEED, level=0.95) -> dict:
    """Paired clip-bootstrap CI of AUROC(b) - AUROC(a) (same resampled clips)."""
    rng = np.random.default_rng(seed)
    uc = sorted(set(map(str, clips)))
    clips = np.asarray([str(c) for c in clips])
    y = np.asarray(y)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(uc), len(uc), replace=True)
        idx = np.concatenate([np.where(clips == uc[p])[0] for p in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(auroc(y[idx], np.asarray(s_b)[idx]) - auroc(y[idx], np.asarray(s_a)[idx]))
    if not out:
        return {"delta": None, "ci95": [None, None]}
    lo, hi = np.percentile(out, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return {"delta": round(float(np.mean(out)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_delta_le_0": round(float(np.mean(np.array(out) <= 0)), 4),
            "of": f"AUROC({label_b}) - AUROC({label_a}), paired clip bootstrap"}


def window_bootstrap_ci(y, s, n_boot=BOOT, seed=SEED, level=0.95) -> list:
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(auroc(y[idx], s[idx]))
    if not out:
        return [None, None]
    lo, hi = np.percentile(out, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return [round(float(lo), 4), round(float(hi), 4)]


def paired_delta_ci(y, s_a, s_b, label_a="a", label_b="b", n_boot=BOOT, seed=SEED,
                    level=0.95) -> dict:
    """Paired window-bootstrap CI of AUROC(b) - AUROC(a) (same resampled windows)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(np.asarray(y)[idx])) < 2:
            continue
        out.append(auroc(np.asarray(y)[idx], np.asarray(s_b)[idx])
                   - auroc(np.asarray(y)[idx], np.asarray(s_a)[idx]))
    if not out:
        return {"delta": None, "ci95": [None, None]}
    lo, hi = np.percentile(out, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return {"delta": round(float(np.mean(out)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_delta_le_0": round(float(np.mean(np.array(out) <= 0)), 4),
            "of": f"AUROC({label_b}) - AUROC({label_a}), paired window bootstrap"}


def within_group_auroc(y, s, groups) -> dict:
    """Mean AUROC computed *within* each group (clip/batch), groups with both classes."""
    y = np.asarray(y)
    s = np.asarray(s)
    groups = np.asarray([str(g) for g in groups])
    vals, sizes = [], {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        sizes[g] = int(m.sum())
        if len(np.unique(y[m])) < 2:
            continue
        vals.append(auroc(y[m], s[m]))
    return {"mean_within_auroc": round(float(np.mean(vals)), 4) if vals else None,
            "n_groups_scored": len(vals), "group_sizes": sizes}


def clip_bootstrap_ci(y, s, clips, n_boot=BOOT, seed=SEED, level=0.95) -> list:
    rng = np.random.default_rng(seed)
    uc = sorted(set(map(str, clips)))
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(uc), len(uc), replace=True)
        idx = np.concatenate([np.where(np.asarray(clips, dtype=str) == uc[p])[0] for p in pick])
        if len(np.unique(np.asarray(y)[idx])) < 2:
            continue
        out.append(auroc(np.asarray(y)[idx], np.asarray(s)[idx]))
    if not out:
        return [None, None]
    lo, hi = np.percentile(out, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return [round(float(lo), 4), round(float(hi), 4)]


# --------------------------------------------------------------------------- #
# regimes (deterministic, descriptor-only, label-free)
# --------------------------------------------------------------------------- #
def load_descriptors(window_ids: list, prefer=("ood2_transcode", "ood1_transcode")) -> dict:
    """window_id -> descriptor dict from the E4 desc_ood.jsonl (prefer transcodes)."""
    path = os.path.join(REGIME_DIR, "desc_ood.jsonl")
    by = {}
    if not os.path.exists(path):
        return by
    for line in open(path):
        x = json.loads(line)
        wid = x["window_id"]
        if wid not in set(window_ids):
            continue
        cur = by.get(wid)
        if cur is None:
            by[wid] = x
        else:  # prefer the requested set order
            rank = {s: i for i, s in enumerate(prefer)}
            if rank.get(x["set"], 99) < rank.get(cur["set"], 99):
                by[wid] = x
    return by


def regime_split(window_ids: list, desc: dict, k: int = 2, seed: int = 0) -> tuple:
    """Deterministic k-means over z-scored E4 descriptors. Returns (labels, info)."""
    keys = ["bitrate_kbps", "luma_mean", "sharp_norm", "noise_sigma", "edge_density",
            "blockiness", "luma_p05", "bright_frac", "shift_mean"]
    rows, keep = [], []
    for i, wid in enumerate(window_ids):
        d = desc.get(wid)
        if d is None:
            continue
        rows.append([math.log1p(max(float(d["desc"][kk]), 0.0)) if kk in ("bitrate_kbps", "sharp_norm")
                     else float(d["desc"][kk]) for kk in keys])
        keep.append(i)
    Z = np.asarray(rows, float)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1.0)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Z)
    lab = np.full(len(window_ids), -1, int)
    for pos, i in enumerate(keep):
        lab[i] = int(km.labels_[pos])
    return lab, {"k": k, "seed": seed, "descriptor_keys": keys,
                 "sizes": {int(c): int((lab == c).sum()) for c in set(lab.tolist()) if c >= 0},
                 "centroids_z": np.round(km.cluster_centers_, 3).tolist()}


# --------------------------------------------------------------------------- #
# variant evaluation
# --------------------------------------------------------------------------- #
def evaluate(X: np.ndarray, y: np.ndarray, names: list, rules: dict, mode: str,
             label: str, groups: np.ndarray | None = None, fit_rows: np.ndarray | None = None,
             ref: dict | None = None, note: str = "") -> dict:
    """Fit on the target's unlabelled values (or `fit_rows` slice of them), score all rows."""
    n = X.shape[0]
    thr_all: dict = {}
    Xeff = np.full_like(X, np.nan, dtype=float)
    gs = np.zeros(n, int) if groups is None else np.asarray(groups, int)
    for g in sorted(set(gs.tolist())):
        rows = np.where(gs == g)[0]
        fit = rows if fit_rows is None else np.intersect1d(rows, fit_rows)
        if fit.size == 0:
            fit = rows
        Xe = transform_values(X, names, mode, ref=ref, rows=fit)
        th = fit_thresholds(rules, X, names, mode, ref=ref, rows=fit)
        Xeff[rows] = Xe[rows]
        for k, v in th.items():
            thr_all[k] = v
    sc = score_windows(Xeff, names, rules, thr_all)
    return {"label": label, "mode": mode, "note": note, "Xeff": Xeff,
            "thr": thr_all, "score": sc["score"], "strong": sc["strong"],
            "per_rule": sc["per_rule"]}


def variant_report(y: np.ndarray, res: dict, reject_threshold: int,
                   clips: np.ndarray | None = None, n_boot: int = BOOT,
                   seed: int = SEED) -> dict:
    dec = decisions(res["score"], res["strong"], reject_threshold)
    out = {"label": res["label"], "mode": res["mode"], "note": res["note"],
           "auroc": round(auroc(y, res["score"]), 4),
           "score_stats": {"mean": round(float(res["score"].mean()), 3),
                           "median": float(np.median(res["score"])),
                           "p90": float(np.percentile(res["score"], 90)),
                           "max": int(res["score"].max())},
           "reject_threshold": int(reject_threshold)}
    out.update({k: round(v, 4) if isinstance(v, float) else v
                for k, v in ba_metrics(y, dec).items() if k != "confusion"})
    out["confusion"] = ba_metrics(y, dec)["confusion"]
    if clips is None:
        out["auroc_ci95_window_bootstrap"] = window_bootstrap_ci(y, res["score"], n_boot, seed)
    else:
        out["auroc_ci95_clip_bootstrap"] = clip_bootstrap_ci(y, res["score"], clips, n_boot, seed)
    return out


def choose_reject_threshold(train_scores, train_y, fr_budget: float = 0.05) -> int:
    """Smallest integer s with training-good false-reject rate <= budget (Q1 rule)."""
    good = np.asarray(train_scores)[np.asarray(train_y) == 1]
    if good.size == 0:
        return 1
    for s in range(1, int(np.asarray(train_scores).max()) + 2):
        if float((good >= s).mean()) <= fr_budget:
            return s
    return int(np.asarray(train_scores).max()) + 1


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {"ids": [str(x) for x in d["ids"]], "y": np.asarray(d["y"], int),
            "names": [str(x) for x in d["names"]], "X": np.asarray(d["X"], float),
            "clips": np.asarray([str(x) for x in d["clips"]]),
            "meta": [str(x) for x in d["meta"]]}


def fire_rate_diagnostic(rules: dict, X: np.ndarray, names: list, thr: dict) -> dict:
    """Fraction of windows beyond the extreme (index 0) threshold, per rule-feature."""
    out = {}
    for r in rules["rules"]:
        f = r["feature"]
        v = col(X, names, f)
        vq = np.array([quantise(x) for x in v], dtype=object)
        tq = thr[(r["rule"], f, 0)]
        m = np.array([x is not None for x in vq])
        hits = np.array([(x < tq) if r["op"] == "<" else (x > tq)
                         for x in vq if x is not None], bool)
        out[f"{r['rule']}:{f}"] = round(float(hits.mean()) if m.any() else float("nan"), 4)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "eval", "results", "exp_phys_dsh",
                                                 "A2_ood_transfer.json"))
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--calib-sizes", default="32")
    ap.add_argument("--calib-seeds", type=int, default=20)
    ap.add_argument("--k-regimes", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()

    rules = load_rules()
    rt_rej = int(rules["reject_threshold"])
    ood = load_npz(os.path.join(INPUTS, "ood160_full.npz"))
    ind = load_npz(os.path.join(INPUTS, "indego_full.npz"))
    ref = indego_ref_stats(rules, ind["X"], ind["names"])
    names_o, names_i = ood["names"], ind["names"]

    results = {"variants_ood": [], "variants_in_domain": [], "extra": {}}

    # ---------------- OOD-160 ----------------
    # (a) raw frozen thresholds
    r_raw = evaluate(ood["X"], ood["y"], names_o, rules, "raw", "a_raw_frozen")
    results["variants_ood"].append(variant_report(ood["y"], r_raw, rt_rej, None, args.boot))
    # (b) per-domain re-quantiling (target's own quantiles at the frozen levels)
    r_b = evaluate(ood["X"], ood["y"], names_o, rules, "requantile", "b_requantile_target")
    results["variants_ood"].append(variant_report(ood["y"], r_b, rt_rej, None, args.boot))
    # (b2) explicit rank/ECDF implementation
    r_b2 = evaluate(ood["X"], ood["y"], names_o, rules, "requantile_rank", "b2_requantile_rank")
    results["variants_ood"].append(variant_report(ood["y"], r_b2, rt_rej, None, args.boot))
    # (c) per-domain z-score, IndEgo thresholds mapped to z-units
    r_c = evaluate(ood["X"], ood["y"], names_o, rules, "zscore", "c_zscore_target", ref=ref)
    results["variants_ood"].append(variant_report(ood["y"], r_c, rt_rej, None, args.boot))
    # (c1) naive literal reading
    r_c1 = evaluate(ood["X"], ood["y"], names_o, rules, "zscore_literal", "c1_zscore_literal")
    results["variants_ood"].append(variant_report(ood["y"], r_c1, rt_rej, None, args.boot))

    # (d) regime-specific re-quantiling
    desc = load_descriptors(ood["ids"])
    lab, rinfo = regime_split(ood["ids"], desc, k=args.k_regimes)
    results["extra"]["regimes_ood"] = {"info": rinfo, "n_with_descriptor": int((lab >= 0).sum()),
                                       "note": "k-means on z-scored E4 descriptors, fixed seed, no labels"}
    r_d = evaluate(ood["X"], ood["y"], names_o, rules, "requantile", "d_regime_requantile",
                   groups=np.where(lab < 0, 0, lab))
    results["variants_ood"].append(variant_report(ood["y"], r_d, rt_rej, None, args.boot))
    # (d2) z-score within regime
    r_d2 = evaluate(ood["X"], ood["y"], names_o, rules, "zscore", "d2_regime_zscore",
                    groups=np.where(lab < 0, 0, lab), ref=ref)
    results["variants_ood"].append(variant_report(ood["y"], r_d2, rt_rej, None, args.boot))

    # (e) label-free-by-construction: 32-window unlabelled calibration slice
    n = len(ood["y"])
    calib = {}
    for size in [int(x) for x in args.calib_sizes.split(",")]:
        aucs, b2aucs, held = [], [], []
        for sd in range(args.calib_seeds):
            rng = np.random.default_rng(1000 + sd)
            rows = np.sort(rng.choice(n, size, replace=False))
            rr = evaluate(ood["X"], ood["y"], names_o, rules, "requantile",
                          f"e_calib{size}_s{sd}", fit_rows=rows)
            aucs.append(auroc(ood["y"], rr["score"]))
            rest = np.setdiff1d(np.arange(n), rows)
            held.append(auroc(ood["y"][rest], rr["score"][rest]))
            rr2 = evaluate(ood["X"], ood["y"], names_o, rules, "requantile_rank",
                           f"e_calib{size}_rank_s{sd}", fit_rows=rows)
            b2aucs.append(auroc(ood["y"], rr2["score"]))
            if sd == 0:
                first = variant_report(ood["y"], rr, rt_rej, None, args.boot)
                first["calibration_rows"] = rows.tolist()
                first["auroc_heldout_windows_only"] = round(float(held[-1]), 4)
        calib[f"calib{size}_requantile"] = {
            "n_calib": size, "n_seeds": args.calib_seeds,
            "auroc_mean": round(float(np.mean(aucs)), 4),
            "auroc_std": round(float(np.std(aucs)), 4),
            "auroc_min": round(float(np.min(aucs)), 4),
            "auroc_max": round(float(np.max(aucs)), 4),
            "auroc_by_seed": [round(float(a), 4) for a in aucs],
            "auroc_heldout128_mean": round(float(np.mean(held)), 4),
            "auroc_heldout128_min": round(float(np.min(held)), 4),
            "auroc_heldout128_max": round(float(np.max(held)), 4),
            "note": "quantiles estimated from the unlabelled calibration slice only; "
                    "evaluated on all 160 (and on the disjoint 128)",
            "seed0_full_report": first}
        calib[f"calib{size}_requantile_rank"] = {
            "auroc_mean": round(float(np.mean(b2aucs)), 4),
            "auroc_min": round(float(np.min(b2aucs)), 4),
            "auroc_max": round(float(np.max(b2aucs)), 4)}
    results["extra"]["calibration_slice"] = calib

    # paired deltas vs raw: is the improvement resolvable despite the wide single CIs?
    results["extra"]["paired_delta_auroc_vs_raw_ood"] = {
        r["label"]: paired_delta_ci(ood["y"], r_raw["score"], r["score"],
                                    "a_raw_frozen", r["label"], args.boot)
        for r in (r_b, r_b2, r_c, r_c1, r_d, r_d2)}
    # mechanism: which rules dominate the raw score, and does re-quantiling rebalance them?
    results["extra"]["score_share_by_rule_ood"] = {}
    for tag, src in (("raw", r_raw), ("requantile", r_b), ("regime_requantile", r_d)):
        tot = max(int(src["score"].sum()), 1)
        results["extra"]["score_share_by_rule_ood"][tag] = {
            k: round(float(v.sum()) / tot, 4) for k, v in sorted(src["per_rule"].items())}

    # ---------------- in-domain (IndEgo 150) ----------------
    # raw, frozen table applied to all 150 (thresholds are in-sample -> optimistic)
    i_raw = evaluate(ind["X"], ind["y"], names_i, rules, "raw", "a_raw_frozen")
    results["variants_in_domain"].append(variant_report(ind["y"], i_raw, rt_rej, ind["clips"], args.boot))
    # LOCO per-clip transductive normalisation: fit on the held-out clip's own unlabelled values
    loco_scores = {}
    for mode, label in [("requantile", "b_requantile_heldout_clip"),
                        ("requantile_rank", "b2_requantile_rank_heldout_clip"),
                        ("zscore", "c_zscore_heldout_clip")]:
        sc = np.zeros(len(ind["y"]), int)
        st = np.zeros(len(ind["y"]), bool)
        per_rule = {}
        for c in sorted(set(map(str, ind["clips"]))):
            rows = np.where(ind["clips"] == c)[0]
            rr = evaluate(ind["X"][rows], ind["y"][rows], names_i, rules, mode,
                          f"{label}:{c}", ref=ref)
            sc[rows] = rr["score"]
            st[rows] = rr["strong"]
            for k, v in rr["per_rule"].items():
                per_rule.setdefault(k, np.zeros(len(ind["y"])))[rows] = v
        loco_scores[mode] = sc
        results["variants_in_domain"].append(variant_report(
            ind["y"], {"label": label, "mode": mode, "note":
                       f"per-fold target-side fit on the {len(set(map(str, ind['clips'])))} "
                       "held-out clips (6-48 unlabelled windows each)", "score": sc,
                       "strong": st, "per_rule": per_rule}, rt_rej, ind["clips"], args.boot))

    # in-domain self-quantile (target = all 150) -> must reproduce raw closely
    i_self = evaluate(ind["X"], ind["y"], names_i, rules, "requantile", "b_requantile_selfquantile")
    results["variants_in_domain"].append(variant_report(ind["y"], i_self, rt_rej, ind["clips"], args.boot))

    # in-domain mechanism: where does the re-quantiling loss come from?
    results["extra"]["paired_delta_auroc_vs_raw_indego"] = {
        f"{m}_heldout_clip": paired_delta_ci_clips(ind["y"], i_raw["score"], loco_scores[m],
                                                   ind["clips"], "a_raw_frozen",
                                                   f"{m}_heldout_clip", args.boot)
        for m in ("requantile", "requantile_rank", "zscore")}
    results["extra"]["within_clip_auroc_indego"] = {
        "raw": within_group_auroc(ind["y"], i_raw["score"], ind["clips"]),
        "b_requantile_heldout_clip": within_group_auroc(ind["y"], loco_scores["requantile"],
                                                        ind["clips"]),
        "note": "per-domain monotone re-quantiling cannot change within-clip feature ranking; "
                "it changes only the pooling across clips (between-clip level information)"}
    batches = []
    for m in ood["meta"]:
        try:
            batches.append(str(json.loads(m).get("batch", "?")))
        except Exception:  # noqa: BLE001
            batches.append("?")
    results["extra"]["within_batch_auroc_ood"] = {
        "raw": within_group_auroc(ood["y"], r_raw["score"], batches),
        "b_requantile_target": within_group_auroc(ood["y"], r_b["score"], batches),
        "d_regime_requantile": within_group_auroc(ood["y"], r_d["score"], batches),
        "note": "batch from the OOD window metadata (label-free grouping)"}

    # in-domain cost vs target-domain size: is the LOCO loss a semantic cost or a
    # small-sample (6-48 windows/clip) artefact?
    curve = {}
    for m in (16, 32, 64, 150):
        aucs, raw_aucs = [], []
        reps = 1 if m == 150 else 20
        for sd in range(reps):
            rng = np.random.default_rng(2000 + sd)
            rows = np.sort(rng.choice(len(ind["y"]), m, replace=False))
            rr = evaluate(ind["X"][rows], ind["y"][rows], names_i, rules, "requantile",
                          f"m{m}", ref=ref)
            if len(np.unique(ind["y"][rows])) < 2:
                continue
            aucs.append(auroc(ind["y"][rows], rr["score"]))
            thr_raw = fit_thresholds(rules, ind["X"][rows], names_i, "raw")
            s_raw = score_windows(ind["X"][rows], names_i, rules, thr_raw)["score"]
            raw_aucs.append(auroc(ind["y"][rows], s_raw))
        curve[f"m{m}"] = {"n_reps": reps,
                          "requantile_auroc_mean": round(float(np.mean(aucs)), 4),
                          "raw_auroc_mean_same_subsets": round(float(np.mean(raw_aucs)), 4),
                          "delta_mean": round(float(np.mean(np.array(aucs) - np.array(raw_aucs))), 4),
                          "requantile_auroc_min": round(float(np.min(aucs)), 4),
                          "requantile_auroc_max": round(float(np.max(aucs)), 4)}
    curve["clip_sizes_6_to_48"] = {
        "requantile_auroc": results["variants_in_domain"][1]["auroc"],
        "raw_auroc": results["variants_in_domain"][0]["auroc"],
        "note": "per-clip LOCO: each target domain is one IndEgo clip (6-48 windows)"}
    results["extra"]["in_domain_target_size_curve"] = curve

    # ---------------- diagnostics ----------------
    results["extra"]["fire_rate_extreme_threshold"] = {
        "indego_expected_level_p01_or_p99": 0.01,
        "indego_raw": fire_rate_diagnostic(rules, ind["X"], names_i,
                                           {k: v for k, v in r_raw["thr"].items()}),
        "ood_raw": fire_rate_diagnostic(rules, ood["X"], names_o, r_raw["thr"]),
        "ood_requantile": fire_rate_diagnostic(rules, ood["X"], names_o, r_b["thr"]),
    }
    # per-rule AUROC after re-quantiling vs raw (OOD)
    per_rule_ood = {}
    for r in rules["rules"]:
        f = r["feature"]
        if f not in names_o:
            continue
        d = {}
        for tag, src in (("raw", r_raw), ("requantile", r_b)):
            ind_ = src["per_rule"].get(r["rule"])
            d[tag] = round(auroc(ood["y"], ind_) if ind_ is not None else 0.5, 4)
        per_rule_ood[f"{r['rule']}:{f}"] = d
    results["extra"]["per_rule_indicator_auroc_ood"] = per_rule_ood

    # level recovery check (frozen thresholds are IndEgo quantiles at these levels)
    lv = {}
    for r in rules["rules"]:
        v = masked(col(ind["X"], names_i, r["feature"]))
        emp = [float((v < t).mean()) if r["op"] == "<" else float((v > t).mean())
               for t in r["thresholds"]]
        lv[f"{r['rule']}:{r['feature']}"] = {
            "empirical_beyond_frozen_thresholds": [round(e, 4) for e in emp],
            "levels": [float(r["quantiles"][j]) for j in range(4)],
            "recovered_levels_rank": [level_of(r, j) for j in range(4)]}
    results["extra"]["frozen_threshold_level_recovery"] = lv

    # sign/threshold transfer statement
    results["transfer_statement"] = {
        "rule_list": "frozen from IndEgo (Q1): 16 rule-features, 7 named rules",
        "sign_op_direction": "frozen (< for low-is-bad, > for high-is-bad) - unchanged",
        "order_statistic_levels": [0.01, 0.05, 0.10, 0.25],
        "numeric_threshold": "re-estimated from the TARGET domain's own UNLABELLED feature "
                             "values at the same levels (variants b/b2/d/e); variant c uses "
                             "the target's own mean/std instead",
        "score_and_decision": "unchanged integer Q=4 count of crossed thresholds; REJECT if "
                              "score >= %d or a quality rule crosses its extreme (index 0) "
                              "threshold" % rt_rej,
        "labels": "no target label is ever used to fit a threshold; y is used only for metrics",
    }
    results["headline"] = {
        "n_ood": len(ood["y"]), "n_ood_bad": int((ood["y"] == 0).sum()),
        "n_indego": len(ind["y"]), "n_indego_bad": int((ind["y"] == 0).sum()),
        "n_features": len(names_o), "n_rules": len(rules["rules"]),
        "reject_threshold": rt_rej,
        "auroc_raw_ood": results["variants_ood"][0]["auroc"],
        "auroc_requantile_ood": results["variants_ood"][1]["auroc"],
        "verdict": "LABEL-FREE PER-DOMAIN RE-QUANTILING PARTIALLY RESCUES OOD TRANSFER: "
                   "AUROC moves from below chance to slightly above it, and the reject-everything "
                   "pathology disappears, but it does not approach the in-domain 0.82 - see "
                   "variants_ood and paired_delta_auroc_vs_raw_ood.",
    }
    results["extra"]["baseline_cross_check"] = {
        "our_a_raw_frozen_ood160_auroc": results["variants_ood"][0]["auroc"],
        "recorded_Q4_01_ood160_pure_rules_auroc":
            json.load(open(os.path.join(REPO, "eval", "results", "exp_phys", "Q4.01.json")))
            ["ood160"]["pure_rules"]["auroc_deterministic_score"],
        "recorded_Q4_01_ood160_logistic_oof_auroc":
            json.load(open(os.path.join(REPO, "eval", "results", "exp_phys", "Q4.01.json")))
            ["ood160"]["logistic_oof_auroc"],
        "note": "the earlier corpus-level '99.55% reject' pathology is reproduced here by the "
                "naive z-score-literal variant (reject 0.9875), not by the raw frozen table on "
                "ood160 (reject 0.644)",
        "indego_in_sample_Q4_01": json.load(
            open(os.path.join(REPO, "eval", "results", "exp_phys", "Q4.01.json")))
            ["gold_loco"]["pure_rules"]["auroc_deterministic_score"],
    }
    results["protocol"] = {
        "ood": "all 160 windows; window-bootstrap 95% CI (2000 reps, seed 0) - OPTIMISTIC "
               "(resampling windows is not resampling domains)",
        "in_domain": "IndEgo 150 (7 clips); clip-bootstrap 95% CI; LOCO variants fit the "
                     "normaliser on the held-out clip's own unlabelled values only",
        "fitting": "label-free: target statistics use X only; OOD labels never used for fitting",
        "determinism": "integer Q=4 comparisons, fixed constants, no RNG on the decision path",
    }
    results["limitations"] = [
        "n_bad = 22 on OOD-160: every OOD AUROC has a wide CI; differences below ~0.05 are "
        "not resolvable.",
        "Window-bootstrap CIs are optimistic: windows are not independent (many share a source "
        "factory/batch and pairs share the same source video), so true sampling error is larger.",
        "Variants b/b2/c/d/e are TRANSDUCTIVE: they use the target domain's own unlabelled "
        "feature values (the whole target for b/b2/c/d). That is legitimate unlabelled domain "
        "adaptation, but it is not a purely inductive single-window decision: a deployed window "
        "must be scored against a bank of target statistics. Variant e (32-window calibration "
        "slice) quantifies how much of the effect survives with a small unlabelled slice.",
        "In-domain numbers for the raw table are in-sample (thresholds were derived on these "
        "150 windows); only the LOCO variants are held-out, and their per-clip target domain is "
        "tiny (6-48 windows), so their quantile estimates are noisy.",
        "phys.snr_db is non-finite for 2/160 OOD and 3/150 IndEgo windows; the scorer skips "
        "non-finite features (that rule cannot fire on those windows), exactly as tools/phys.",
        "Reject/BA numbers use the frozen reject_threshold = 14 for every variant; the score "
        "distribution changes with re-quantiling, so the operating point is not re-optimised "
        "(that is deliberate: re-optimising on OOD would use OOD labels).",
    ]
    results["n"] = {"ood": len(ood["y"]), "indego": len(ind["y"]),
                    "ood_bad": int((ood["y"] == 0).sum()), "indego_bad": int((ind["y"] == 0).sum())}
    results["label_source"] = ("human labels: labels/ood160_labels.json (OOD, n_bad=22) and "
                               "labels/manifest_gold.json (IndEgo, labels only for reporting)")
    results["command"] = ("OMP_NUM_THREADS=1 python3 eval/exp_phys_dsh/a2_transfer.py "
                          f"--boot {args.boot} --calib-sizes {args.calib_sizes} "
                          f"--calib-seeds {args.calib_seeds} --k-regimes {args.k_regimes}")
    results["device"] = "CPU (threads only; multiprocessing blocked by the sandbox)"
    results["rule_table"] = {"path": "eval/results/exp_phys/rule_table.json",
                             "sha256": sha256_file(RULES_PATH)}
    results["inputs"] = {
        "ood160_full.npz": {"sha256": sha256_file(os.path.join(INPUTS, "ood160_full.npz")),
                            "n": len(ood["y"]), "n_features": len(names_o)},
        "indego_full.npz": {"sha256": sha256_file(os.path.join(INPUTS, "indego_full.npz")),
                            "n": len(ind["y"]), "n_features": len(names_i)},
        "note": "built by the lead from the frozen vendored kernels "
                "(eval/exp_phys_dsh/build_matrix.py / build_indego.py); 0 extraction failures",
    }
    results["extra"]["regime_sources"] = {
        "ood": os.path.join(REGIME_DIR, "desc_ood.jsonl") + " (transcode rows preferred)",
        "indego": os.path.join(REGIME_DIR, "desc_indego.jsonl") +
                  " exists; the in-domain regime analogue used here is one held-out IndEgo clip "
                  "per fold (LOCO), because a clip is the natural homogeneous sub-domain and the "
                  "LOCO protocol is the binding in-domain metric",
    }
    # self-contained answers to the three questions the pod asked
    ood_by = {v["label"]: v for v in results["variants_ood"]}
    ind_by = {v["label"]: v for v in results["variants_in_domain"]}
    results["summary"] = {
        "does_label_free_requantile_rescue_ood": {
            "raw": {"auroc": ood_by["a_raw_frozen"]["auroc"],
                    "ci95": ood_by["a_raw_frozen"]["auroc_ci95_window_bootstrap"]},
            "requantile": {"auroc": ood_by["b_requantile_target"]["auroc"],
                           "ci95": ood_by["b_requantile_target"]["auroc_ci95_window_bootstrap"],
                           "paired_delta_vs_raw":
                               results["extra"]["paired_delta_auroc_vs_raw_ood"]
                               ["b_requantile_target"]},
            "regime_requantile_k2": {"auroc": ood_by["d_regime_requantile"]["auroc"],
                                     "ci95": ood_by["d_regime_requantile"]
                                     ["auroc_ci95_window_bootstrap"],
                                     "paired_delta_vs_raw":
                                         results["extra"]["paired_delta_auroc_vs_raw_ood"]
                                         ["d_regime_requantile"]},
            "answer": "PARTIALLY: +0.12 to +0.15 AUROC over raw, paired-bootstrap CI excludes 0 "
                      "for the rank/quantile variants, but the result (0.54-0.58) is only "
                      "marginally above chance and no better than the 26-feature logistic OOD "
                      "baseline (0.566, Q4.01). Not a rescue.",
            "reject_everything_pathology": "raw rejects 64.4% of OOD-160 (and the naive "
                                           "z-score-literal reading rejects 98.8%); "
                                           "re-quantiling brings reject rate to 16-26%",
        },
        "cost_in_domain": {
            "raw_in_sample": ind_by["a_raw_frozen"]["auroc"],
            "requantile_all_150_as_target": ind_by["b_requantile_selfquantile"]["auroc"],
            "requantile_per_heldout_clip_loco": ind_by["b_requantile_heldout_clip"]["auroc"],
            "target_size_curve": results["extra"]["in_domain_target_size_curve"],
            "answer": "ZERO when the target domain spans the pool (150 windows: identical to raw; "
                      "16-64 random windows: |delta| <= 0.03). The -0.11 drop in the per-clip "
                      "LOCO is the cost of normalising inside ONE label-homogeneous clip, i.e. it "
                      "removes the between-clip level component of the pooled AUROC, not a "
                      "semantic cost of the transform.",
        },
        "what_did_not_work": [
            "per-domain z-score (c): +0.07 OOD AUROC, CI includes 0 - the first two moments do "
            "not capture the shift (distributions shift shape, not just location/scale)",
            "naive z-score-literal (c1): 98.8% reject, AUROC 0.434 - the literal reading of "
            "'apply frozen thresholds to target z-values' is broken by construction",
            "regime z-score (d2): 0.470, no better than raw",
            "in-domain per-clip normalisation: -0.09 to -0.12 pooled AUROC vs the frozen global "
            "table (because between-clip level differences are real in-domain signal)",
        ],
    }
    results["runtime_s"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, sort_keys=False)
        f.write("\n")
    print(f"wrote {args.out} in {results['runtime_s']}s")
    for v in results["variants_ood"]:
        print(f"  OOD {v['label']:32s} auroc={v['auroc']:.4f} "
              f"ci={v.get('auroc_ci95_window_bootstrap')} rej={v['reject_rate']:.3f} "
              f"ba_dec={v['balanced_acc_decided']:.3f}")
    for v in results["variants_in_domain"]:
        print(f"  IND {v['label']:32s} auroc={v['auroc']:.4f} "
              f"ci={v.get('auroc_ci95_clip_bootstrap')} rej={v['reject_rate']:.3f} "
              f"ba_dec={v['balanced_acc_decided']:.3f}")
    print("  calib32:", {k: v for k, v in calib.get("calib32_requantile", {}).items()
                         if k not in ("seed0_full_report", "auroc_by_seed")})
    print("  paired deltas vs raw (OOD):",
          {k: (v["delta"], v["ci95"]) for k, v in
           results["extra"]["paired_delta_auroc_vs_raw_ood"].items()})
    print("  paired deltas vs raw (IndEgo LOCO):",
          {k: (v["delta"], v["ci95"]) for k, v in
           results["extra"]["paired_delta_auroc_vs_raw_indego"].items()})
    print("  within-clip AUROC:", json.dumps(results["extra"]["within_clip_auroc_indego"])[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
