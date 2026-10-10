#!/usr/bin/env python3
"""N3 — Accuracy fusion + routing: deterministic physics rules x adapted model x frozen ensemble.

Quality-vs-semantic split, deterministic regime router, and rank-average fusion.
CPU only, all inputs cached. No GPU. Evaluation only; no labels edited.

Scorers (all cached / frozen, nothing refit on OOD):
  * rules    — deterministic physics rule table (tools/phys), LOCO fold-local on gold,
               frozen table on OOD.  higher score = worse.
  * frozen   — ensemble_v1 two-view logistic: gold OOF p_ens, OOD usable_prob. higher = good.
  * adapted  — Z3 6 transferable features, per-domain z-score, IndEgo-signed mean.
               higher = good. (reproduces m/adapt160 recipe (a))

Sub-problems (label reasons):
  quality  = TOO_DARK, BLUR, OVEREXPOSED, FROZEN_STATIC, DUPLICATE, UNDECODABLE
  semantic = NO_ACTIVE_WORK, HANDS_NOT_VISIBLE

Writes eval/results/exp_acc/N3.01.json (split), N3.02.json (routing), N3.03.json (fusion).
Run: OMP_NUM_THREADS=2 python3 eval/exp_acc/run_acc_fusion.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "phys", ".."))
from tools.phys import model as physmodel  # noqa: E402
from jadal.regimes.router import choose_profile  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
INP = os.path.join(ROOT, "eval", "results", "exp_acc", "inputs")
OUT = os.path.join(ROOT, "eval", "results", "exp_acc")
N_BOOT = 2000
SEED = 0
Q = 4
QUANTILES = (0.01, 0.05, 0.10, 0.25)
FR_BUDGET = 0.05

QUALITY = {"TOO_DARK", "BLUR", "OVEREXPOSED", "FROZEN_STATIC", "DUPLICATE", "UNDECODABLE"}
SEMANTIC = {"NO_ACTIVE_WORK", "HANDS_NOT_VISIBLE"}
Z3_SIX = ["ct_fsize_std", "mo_hand_motion_excess", "inv_clip_frac",
          "inv_laplacian_var", "inv_hist_entropy", "inv_dct_hf_lf"]
FROZEN_RULES = physmodel.load_rules()
DEV = "CPU (numpy/sklearn; cached features only, no GPU)"


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=ROOT).stdout.strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ metrics
def auc(y, s):
    """AUROC of a bad-score (higher = more likely bad)."""
    return float(roc_auc_score(1 - np.asarray(y), np.asarray(s, float)))


def balacc(y, pred_bad):
    y = np.asarray(y)
    pred_bad = np.asarray(pred_bad, bool)
    n_bad = max(int((y == 0).sum()), 1)
    n_good = max(int((y == 1).sum()), 1)
    tpr = float((pred_bad & (y == 0)).sum()) / n_bad
    tnr = float((~pred_bad & (y == 1)).sum()) / n_good
    return 0.5 * (tpr + tnr)


def _groups_idx(groups):
    ug = sorted(set(np.asarray(groups).tolist()))
    return ug, {g: np.where(np.asarray(groups) == g)[0] for g in ug}


def grouped_boot(y, s, groups, fn, n_boot=N_BOOT, seed=SEED):
    ug, idx = _groups_idx(groups)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(ug), len(ug))
        ii = np.concatenate([idx[ug[p]] for p in pick])
        if len(np.unique(np.asarray(y)[ii])) < 2:
            continue
        vals.append(fn(np.asarray(y)[ii], np.asarray(s)[ii]))
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def paired_grouped_boot(y, s_a, s_b, groups, n_boot=N_BOOT, seed=SEED):
    ug, idx = _groups_idx(groups)
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(ug), len(ug))
        ii = np.concatenate([idx[ug[p]] for p in pick])
        if len(np.unique(np.asarray(y)[ii])) < 2:
            continue
        diffs.append(auc(np.asarray(y)[ii], np.asarray(s_a)[ii])
                     - auc(np.asarray(y)[ii], np.asarray(s_b)[ii]))
    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        return {}
    return {"mean": round(float(diffs.mean()), 4),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                     round(float(np.percentile(diffs, 97.5)), 4)],
            "frac_gt0": round(float((diffs > 0).mean()), 4)}


def rank01(x):
    """Rank-normalise to [0,1]; higher input -> higher output."""
    x = np.asarray(x, float)
    return (rankdata(x) - 1.0) / max(len(x) - 1, 1)


def platt_cv_good(y, s, groups):
    """OOF calibrated P(good) by logistic on the scalar score, grouped CV.

    class_weight='balanced' removes the base-rate so the [0.4,0.6] band measures
    genuine good/bad score overlap (a base-rate-robust REVIEW proxy) and 0.5 is
    the balanced-accuracy operating point.  AUROC is unaffected."""
    y = np.asarray(y)
    s = np.asarray(s, float).reshape(-1, 1)
    p = np.full(len(y), np.nan)
    for g in sorted(set(np.asarray(groups).tolist())):
        te = np.where(np.asarray(groups) == g)[0]
        tr = np.where(np.asarray(groups) != g)[0]
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(s[tr], y[tr])
        p[te] = clf.predict_proba(s[te])[:, 1]
    # fill any stragglers with a global fit
    if np.isnan(p).any():
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(s, y)
        p[np.isnan(p)] = clf.predict_proba(s[np.isnan(p)])[:, 1]
    return p


def review_metrics(y, p_good, band=(0.4, 0.6)):
    """Operating-point proxy: accept iff p>=0.5, REVIEW iff p in band."""
    y = np.asarray(y)
    p = np.asarray(p_good, float)
    rev = (p >= band[0]) & (p <= band[1])
    dec = ~rev
    pred_bad = p < 0.5
    if dec.any():
        acc = float(((np.asarray(y[dec]) == 1) == (~pred_bad[dec])).mean())
        err_dec = round(1.0 - acc, 4)
    else:
        err_dec = None
    out = {
        "review_share": round(float(rev.mean()), 4),
        "n_review": int(rev.sum()),
        "balacc_band_as_accept": round(balacc(y, pred_bad & ~rev), 4),
        "balacc_band_as_reject": round(balacc(y, pred_bad | rev), 4),
        "balacc_decided_only": round(balacc(y[dec], pred_bad[dec]), 4) if dec.any() else None,
        "err_decided_only": err_dec,
        "accept_share": round(float((~pred_bad).mean()), 4),
    }
    return out


# ------------------------------------------------------------------ loaders
def load_gold():
    m = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))
    items = m["items"]
    ids = [it["id"] for it in items]
    y = np.asarray([int(it["label"]) for it in items])
    clips = np.asarray([it["clip"] for it in items])
    reasons = np.asarray([it.get("reason", "") or "" for it in items])
    W = json.load(open(os.path.join(BUNDLE, "caches", "gpu", "features.json")))["windows"]
    phys = np.load(os.path.join(BUNDLE, "caches", "perpixel", "physics_feats.npz"),
                   allow_pickle=True)
    pnames = [str(n) for n in phys["names"]]
    pidx = {str(k): i for i, k in enumerate(phys["ids"])}
    full_names = list(W[ids[0]]["full"].keys())
    feats = []
    for i in ids:
        d = {f"full.{n}": float(W[i]["full"][n]) for n in full_names}
        d.update({f"phys.{n}": float(phys["X"][pidx[i], j]) for j, n in enumerate(pnames)})
        feats.append(d)
    oof = np.load(os.path.join(BUNDLE, "caches", "ensemble", "oof_probs.npz"),
                  allow_pickle=True)
    omap = {str(k): j for j, k in enumerate(oof["ids"])}
    frozen = np.asarray([float(oof["p_ens"][omap[i]]) for i in ids])
    z = np.load(os.path.join(INP, "z3_160.npz"), allow_pickle=True)
    znames = [str(n) for n in z["names"]]
    cols = [znames.index(c) for c in Z3_SIX]
    Xz = np.asarray(z["X_indego"], float)[:, cols]
    desc = _load_desc("desc_indego.jsonl")
    return dict(ids=ids, y=y, clips=clips, reasons=reasons, feats=feats,
                frozen=frozen, Xz=Xz, desc=desc)


def load_ood():
    d = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))
    w = d["windows"]
    ids = [x["segment_id"] for x in w]
    y = np.asarray([1 if x["verdict"] == "good" else 0 for x in w])
    factory = np.asarray([x["factory"] for x in w])
    batch = np.asarray([x["batch"] for x in w])
    reasons = np.asarray([x.get("reason", "") or "" for x in w])
    z = np.load(os.path.join(INP, "ood_full.npz"), allow_pickle=True)
    zn = [str(n) for n in z["names"]]
    Xf = np.asarray(z["X"], float)
    fmap = {str(k): i for i, k in enumerate(z["ids"])}
    feats = [{zn[j]: Xf[fmap[i], j] for j in range(len(zn))} for i in ids]
    probs = {}
    for fn in ("ood_scores.json", "ood_scores2.json"):
        dd = json.load(open(os.path.join(BUNDLE, "caches", "ensemble", fn)))
        for s in dd["scores"]:
            if "error" not in s:
                probs[s["id"]] = float(s["usable_prob"])
    frozen = np.asarray([probs[i] for i in ids])
    zz = np.load(os.path.join(INP, "z3_160.npz"), allow_pickle=True)
    znames = [str(n) for n in zz["names"]]
    cols = [znames.index(c) for c in Z3_SIX]
    Xo = np.asarray(zz["X"], float)[:, cols]
    desc = _load_desc("desc_ood.jsonl", only=set(ids))
    return dict(ids=ids, y=y, factory=factory, batch=batch, reasons=reasons,
                feats=feats, frozen=frozen, Xo=Xo, desc=desc)


def _load_desc(fn, only=None):
    out = {}
    for line in open(os.path.join(INP, fn)):
        r = json.loads(line)
        if only is not None and r["window_id"] not in only:
            continue
        if only is not None and "label_windows" not in r.get("src", ""):
            continue
        out[r["window_id"]] = r["desc"]
    return out


# ------------------------------------------------------------------ rules
def _quantise(v):
    return physmodel.quantise(v, Q)


def derive_fold_table(feats, train_idx):
    rules = []
    for r in FROZEN_RULES["rules"]:
        vals = np.array([feats[i].get(r["feature"], np.nan) for i in train_idx], float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 8 or len(np.unique(vals)) < 2:
            continue
        qs = np.array(QUANTILES) if r["side"] == "low" else 1.0 - np.array(QUANTILES)
        th = [float(t) for t in np.quantile(vals, qs)]
        rules.append({**r, "thresholds": th, "thresholds_q": [_quantise(t) for t in th]})
    return {"quantisation": Q, "reject_threshold": 1,
            "undecodable_features": FROZEN_RULES["undecodable_features"],
            "undecodable_score": FROZEN_RULES["undecodable_score"], "rules": rules}


def choose_reject_threshold(train_scores, train_y, budget=FR_BUDGET):
    good = np.asarray(train_scores)[np.asarray(train_y) == 1]
    if good.size == 0:
        return 1
    for s in range(1, int(np.max(train_scores)) + 2):
        if float((good >= s).mean()) <= budget:
            return s
    return int(np.max(train_scores)) + 1


def rules_scores(feats, table):
    """Return (total_score, quality_score, semantic_score, decisions)."""
    tot = np.zeros(len(feats), int)
    qual = np.zeros(len(feats), int)
    sem = np.zeros(len(feats), int)
    dec = np.empty(len(feats), dtype=object)
    for i, f in enumerate(feats):
        sc = physmodel.score_window(f, table)
        tot[i] = sc["score"]
        for e in sc["fired"]:
            n = int(e["n_thresholds_crossed"])
            if e["strength"] == "quality":
                qual[i] += n
            else:
                sem[i] += n
        dec[i] = physmodel.decide(sc, table)["decision"]
    return tot, qual, sem, dec


def gold_loco_rules(feats, y, clips):
    tot = np.zeros(len(y), int)
    qual = np.zeros(len(y), int)
    sem = np.zeros(len(y), int)
    dec = np.empty(len(y), dtype=object)
    for c in sorted(set(clips.tolist())):
        te = np.array([clips[i] == c for i in range(len(y))])
        tr = np.where(~te)[0]
        table = derive_fold_table(feats, tr)
        tr_scores = np.array([physmodel.score_window(feats[i], table)["score"] for i in tr])
        table["reject_threshold"] = choose_reject_threshold(tr_scores, y[tr])
        t, q, s, d = rules_scores([feats[i] for i in np.where(te)[0]], table)
        tot[te], qual[te], sem[te], dec[te] = t, q, s, d
    return tot, qual, sem, dec


# ------------------------------------------------------------------ adapted
def adapted_scores(Xg, yg, Xo):
    """IndEgo-signed per-domain z-score equal mean (m/adapt160 recipe a)."""
    au = np.asarray([float(roc_auc_score(yg, Xg[:, j])) for j in range(Xg.shape[1])])
    sg = np.where(au >= 0.5, 1.0, -1.0)
    zg = (Xg - Xg.mean(0)) / np.clip(Xg.std(0), 1e-9, None)
    zo = (Xo - Xo.mean(0)) / np.clip(Xo.std(0), 1e-9, None)
    return (zg * sg).mean(1), (zo * sg).mean(1)


# ------------------------------------------------------------------ split table
def subset_auroc(y, bad_score, reasons, category):
    """AUROC of {good} vs {bad windows whose reason is in category}."""
    y = np.asarray(y)
    reasons = np.asarray(reasons)
    pos = (y == 0) & np.isin(reasons, list(category))
    neg = (y == 1)
    m = pos | neg
    if pos.sum() < 1:
        return None, int(pos.sum())
    yy = np.where(y[m] == 1, 1, 0)
    return round(auc(yy, np.asarray(bad_score)[m]), 4), int(pos.sum())


# ------------------------------------------------------------------ main
def main():
    t0 = time.perf_counter()
    gold = load_gold()
    ood = load_ood()
    sha = git_sha()

    # rules
    g_tot, g_qual, g_sem, g_dec = gold_loco_rules(gold["feats"], gold["y"], gold["clips"])
    o_tot, o_qual, o_sem, o_dec = rules_scores(ood["feats"], FROZEN_RULES)

    # adapted
    a_g, a_o = adapted_scores(gold["Xz"], gold["y"], ood["Xo"])

    # bad-scores (higher = more likely bad)
    scorers = {
        "rules": {"gold": g_tot.astype(float), "ood": o_tot.astype(float)},
        "frozen": {"gold": -gold["frozen"], "ood": -ood["frozen"]},
        "adapted": {"gold": -a_g, "ood": -a_o},
        "rules_quality": {"gold": g_qual.astype(float), "ood": o_qual.astype(float)},
        "rules_semantic": {"gold": g_sem.astype(float), "ood": o_sem.astype(float)},
    }
    good_scores = {
        "rules": {"gold": -g_tot.astype(float), "ood": -o_tot.astype(float)},
        "frozen": {"gold": gold["frozen"], "ood": ood["frozen"]},
        "adapted": {"gold": a_g, "ood": a_o},
    }

    # ---- N3.01 quality vs semantic ----
    split = {"experiment": "N3.01",
             "title": "Quality-vs-semantic split: which sub-problem transfers?",
             "device": DEV, "git_sha": sha,
             "definitions": {
                 "quality_reasons": sorted(QUALITY),
                 "semantic_reasons": sorted(SEMANTIC),
                 "auroc": "bad-score AUROC: {good} vs {bad windows with reason in category}",
             },
             "domains": {}}
    for dom, dkey, yy, reasons, groups in (
            ("indego_loco", "gold", gold["y"], gold["reasons"], gold["clips"]),
            ("ood160", "ood", ood["y"], ood["reasons"], ood["factory"])):
        rows = {}
        for name, sc in scorers.items():
            bs = sc[dkey]
            all_auc = round(auc(yy, bs), 4)
            qa, nq = subset_auroc(yy, bs, reasons, QUALITY)
            sa, ns = subset_auroc(yy, bs, reasons, SEMANTIC)
            row = {"auroc_all": all_auc,
                   "auroc_quality": qa, "n_quality_bad": nq,
                   "auroc_semantic": sa, "n_semantic_bad": ns,
                   "auroc_all_ci95_grouped": grouped_boot(yy, bs, groups, auc)}
            if qa is not None:
                m = ((yy == 0) & np.isin(reasons, list(QUALITY))) | (yy == 1)
                row["auroc_quality_ci95_grouped"] = grouped_boot(yy[m], bs[m], np.asarray(groups)[m], auc)
            if sa is not None:
                m = ((yy == 0) & np.isin(reasons, list(SEMANTIC))) | (yy == 1)
                row["auroc_semantic_ci95_grouped"] = grouped_boot(yy[m], bs[m], np.asarray(groups)[m], auc)
            rows[name] = row
        # reason counts
        import collections
        rc = collections.Counter(reasons[yy == 0].tolist())
        split["domains"][dom] = {"n": int(len(yy)), "n_good": int((yy == 1).sum()),
                                 "n_bad": int((yy == 0).sum()),
                                 "bad_reason_counts": dict(rc), "scorers": rows}
    # native deterministic decision breakdown (shows what the rules actually DO)
    for dom, yy, reasons, dec in (("indego_loco", gold["y"], gold["reasons"], g_dec),
                                  ("ood160", ood["y"], ood["reasons"], o_dec)):
        d = split["domains"][dom]
        d["rules_native_decisions"] = {}
        for cat, name in ((QUALITY, "quality_bad"), (SEMANTIC, "semantic_bad")):
            m = (yy == 0) & np.isin(reasons, list(cat))
            if m.sum() == 0:
                d["rules_native_decisions"][name] = {"n": 0}
                continue
            dd = np.asarray(dec)[m]
            d["rules_native_decisions"][name] = {
                "n": int(m.sum()),
                "reject": int((dd == "REJECT").sum()),
                "review": int((dd == "REVIEW").sum()),
                "accept": int((dd == "ACCEPT").sum())}
        mg = (yy == 1)
        dd = np.asarray(dec)[mg]
        d["rules_native_decisions"]["good"] = {
            "n": int(mg.sum()), "reject": int((dd == "REJECT").sum()),
            "review": int((dd == "REVIEW").sum()),
            "accept": int((dd == "ACCEPT").sum())}
    json.dump(split, open(os.path.join(OUT, "N3.01.json"), "w"), indent=1, sort_keys=True)
    print("N3.01 done", round(time.perf_counter() - t0, 1), "s")

    # ---- calibration (review proxy) ----
    cal = {}
    for dom, yy, groups in (("gold", gold["y"], gold["clips"]),
                            ("ood", ood["y"], ood["factory"])):
        cal[dom] = {}
        for name in ("rules", "frozen", "adapted"):
            cal[dom][name] = platt_cv_good(yy, good_scores[name][dom], groups)

    # ---- N3.02 routing ----
    g_desc = [gold["desc"][i] for i in gold["ids"]]
    o_desc = [ood["desc"][i] for i in ood["ids"]]

    def route(fn, dom):
        descs = g_desc if dom == "gold" else o_desc
        return np.asarray([fn(d) for d in descs])

    def prof(d):
        return choose_profile(d)["profile"]

    def is_quality_desc(d):
        return (float(d.get("luma_mean", 1)) <= 0.24
                or float(d.get("sharp_norm", 1e9)) <= 2500.0
                or float(d.get("clip_frac", 0)) >= 0.30)

    def aspect(d):
        h = float(d.get("height", 1) or 1)
        return float(d.get("width", 0)) / max(h, 1.0)

    # policies: name -> callable(desc, rules_native_decision) -> scorer name
    policies = {
        "route_fps_ge25": lambda d, dec: "adapted" if float(d.get("fps", 0)) >= 25 else "rules",
        "route_profile_factory": lambda d, dec: "adapted" if prof(d) in ("factory_1080p30",) else "rules",
        "route_aspect_wide": lambda d, dec: "adapted" if aspect(d) >= 1.3 else "rules",
        "route_quality_desc": lambda d, dec: "rules" if is_quality_desc(d) else "adapted",
        "route_rules_review_to_adapted": lambda d, dec: "adapted" if dec == "REVIEW" else "rules",
        "route_aspect_or_quality": lambda d, dec: "rules" if (aspect(d) < 1.3 and is_quality_desc(d)) else "adapted",
    }
    routing = {"experiment": "N3.02",
               "title": "Deterministic regime router: routed system vs single scorers",
               "device": DEV, "git_sha": sha,
               "review_proxy": ("Platt-calibrated P(good) (grouped CV: LOCO gold / "
                                "leave-one-factory-out OOD); REVIEW iff P in [0.4,0.6]. "
                                "AUROC is computed on the raw (monotonic) rank score; "
                                "calibration is only an operating-point measurement proxy."),
               "domains": {}}
    for dom, yy, groups, gkey in (("indego_loco", gold["y"], gold["clips"], "gold"),
                                  ("ood160", ood["y"], ood["factory"], "ood")):
        rows = {}
        raw = {name: rank01(good_scores[name][gkey]) for name in ("rules", "frozen", "adapted")}
        # single scorers
        for name in ("rules", "frozen", "adapted"):
            p = cal[gkey][name]
            rows[name] = {"auroc": round(auc(yy, -raw[name]), 4),
                          "auroc_ci95_grouped": grouped_boot(yy, -raw[name], groups, auc),
                          **review_metrics(yy, p)}
        # routed policies (choose scorer per window; raw rank for AUROC, cal for review)
        decs = g_dec if gkey == "gold" else o_dec
        descs = g_desc if gkey == "gold" else o_desc
        for pname, fn in policies.items():
            chosen = np.asarray([fn(descs[i], decs[i]) for i in range(len(yy))])
            p = np.empty(len(yy))
            rraw = np.empty(len(yy))
            for name in ("rules", "frozen", "adapted"):
                m = chosen == name
                if m.any():
                    p[m] = cal[gkey][name][m]
                    rraw[m] = raw[name][m]
            rows[pname] = {"auroc": round(auc(yy, -rraw), 4),
                           "auroc_ci95_grouped": grouped_boot(yy, -rraw, groups, auc),
                           "route_share": {k: round(float((chosen == k).mean()), 4)
                                           for k in ("rules", "frozen", "adapted")
                                           if (chosen == k).any()},
                           **review_metrics(yy, p)}
        routing["domains"][dom] = {"n": int(len(yy)),
                                   "n_good": int((yy == 1).sum()),
                                   "n_bad": int((yy == 0).sum()),
                                   "scorers": rows}
    # cross-domain summary: mean of per-domain AUROC (equal weight on the two domains)
    names = list(routing["domains"]["indego_loco"]["scorers"].keys())
    routing["summary_mean_domain_auroc"] = {
        n: round(float(np.mean([routing["domains"][d]["scorers"][n]["auroc"]
                                for d in routing["domains"]])), 4)
        for n in names}
    json.dump(routing, open(os.path.join(OUT, "N3.02.json"), "w"), indent=1, sort_keys=True)
    print("N3.02 done", round(time.perf_counter() - t0, 1), "s")

    # ---- N3.03 fusion ----
    fusion = {"experiment": "N3.03",
              "title": "Rank-average fusion: adapted x frozen (x rules) and review proxy",
              "device": DEV, "git_sha": sha,
              "method": "rank-average of per-domain rank01 good-scores (Y3 recipe)",
              "domains": {}}
    for dom, yy, groups, gkey in (("indego_loco", gold["y"], gold["clips"], "gold"),
                                  ("ood160", ood["y"], ood["factory"], "ood")):
        # good-direction ranks
        r_frozen = rank01(good_scores["frozen"][gkey])
        r_adapt = rank01(good_scores["adapted"][gkey])
        r_rules = rank01(good_scores["rules"][gkey])
        # singles use their raw good-score (consistent with N3.02); combos are rank-averages
        combos = {
            "frozen": good_scores["frozen"][gkey],
            "adapted": good_scores["adapted"][gkey],
            "rules": good_scores["rules"][gkey],
            "rank_adapted+frozen": (r_adapt + r_frozen) / 2.0,
            "rank_adapted+frozen+rules": (r_adapt + r_frozen + r_rules) / 3.0,
        }
        rows = {}
        for name, gs in combos.items():
            bs = -np.asarray(gs, float)          # bad-score
            p = platt_cv_good(yy, gs, groups)    # calibrated P(good) for review proxy
            r = {"auroc": round(auc(yy, bs), 4),
                 "auroc_ci95_grouped": grouped_boot(yy, bs, groups, auc),
                 **review_metrics(yy, p)}
            r["review_share_band_25_75"] = round(float(((p >= 0.25) & (p <= 0.75)).mean()), 4)
            rows[name] = r
        # paired gains
        bs_f = -combos["frozen"]
        bs_a = -combos["adapted"]
        bs_x = -combos["rank_adapted+frozen"]
        rows["rank_adapted+frozen"]["gain_vs_frozen_paired"] = paired_grouped_boot(yy, bs_x, bs_f, groups)
        rows["rank_adapted+frozen"]["gain_vs_adapted_paired"] = paired_grouped_boot(yy, bs_x, bs_a, groups)
        fusion["domains"][dom] = {"n": int(len(yy)),
                                  "n_good": int((yy == 1).sum()),
                                  "n_bad": int((yy == 0).sum()),
                                  "combos": rows}
    cnames = list(fusion["domains"]["indego_loco"]["combos"].keys())
    fusion["summary_mean_domain_auroc"] = {
        n: round(float(np.mean([fusion["domains"][d]["combos"][n]["auroc"]
                                for d in fusion["domains"]])), 4)
        for n in cnames}
    # joint point: routed+fused on OOD (fused score, quality-gate to rules) vs OOD error
    # Best fused configuration = rank_adapted+frozen; route: quality-desc -> rules else fused
    yy = ood["y"]
    r_frozen = rank01(good_scores["frozen"]["ood"])
    r_adapt = rank01(good_scores["adapted"]["ood"])
    fused = (r_adapt + r_frozen) / 2.0
    r_rules = rank01(good_scores["rules"]["ood"])
    r_adapt_o = rank01(good_scores["adapted"]["ood"])
    p_fused = platt_cv_good(yy, fused, ood["factory"])
    p_rules = cal["ood"]["rules"]
    p_adapt = cal["ood"]["adapted"]
    o_desc = [ood["desc"][i] for i in ood["ids"]]

    def joint_config(name, chooser, raw_by, p_by):
        raw = np.asarray([raw_by[chooser(o_desc[i])][i] for i in range(len(yy))])
        p = np.asarray([p_by[chooser(o_desc[i])][i] for i in range(len(yy))])
        ch = np.asarray([chooser(o_desc[i]) for i in range(len(yy))])
        return {
            "system": name,
            "ood_auroc": round(auc(yy, -raw), 4),
            "ood_auroc_ci95_grouped": grouped_boot(yy, -raw, ood["factory"], auc),
            "route_share": {k: round(float((ch == k).mean()), 4)
                            for k in sorted(set(ch.tolist()))},
            **review_metrics(yy, p)}

    raw_by = {"rules": r_rules, "fused": fused, "adapted": r_adapt_o}
    p_by = {"rules": p_rules, "fused": p_fused, "adapted": p_adapt}
    fusion["joint_point_ood"] = {
        "adapted_only": joint_config("adapted_only", lambda d: "adapted", raw_by, p_by),
        "quality_gate_rules_else_fused": joint_config(
            "quality-desc -> rules else rank_adapted+frozen",
            lambda d: "rules" if is_quality_desc(d) else "fused", raw_by, p_by),
        "aspect_wide_fused_else_rules": joint_config(
            "aspect>=1.3 -> fused else rules",
            lambda d: "fused" if aspect(d) >= 1.3 else "rules", raw_by, p_by),
    }
    json.dump(fusion, open(os.path.join(OUT, "N3.03.json"), "w"), indent=1, sort_keys=True)
    print("N3.03 done", round(time.perf_counter() - t0, 1), "s")

    print(json.dumps({
        "split": {d: {s: {k: v for k, v in r.items() if k in ("auroc_all", "auroc_quality", "auroc_semantic")}
                       for s, r in split["domains"][d]["scorers"].items()}
                  for d in split["domains"]},
        "routing": {d: {s: {k: v for k, v in r.items() if k in ("auroc", "review_share", "balacc_band_as_accept", "err_decided_only")}
                        for s, r in routing["domains"][d]["scorers"].items()}
                    for d in routing["domains"]},
        "fusion": {d: {c: {k: v for k, v in r.items() if k in ("auroc", "review_share", "balacc_band_as_accept")}
                       for c, r in fusion["domains"][d]["combos"].items()}
                   for d in fusion["domains"]},
        "joint": fusion["joint_point_ood"],
    }, indent=1))


if __name__ == "__main__":
    main()
