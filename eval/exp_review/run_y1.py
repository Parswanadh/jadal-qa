#!/usr/bin/env python3
"""Y1.01 — Review-share reduction for the frozen PoC: abstain-band + calibration sweep.

Goal: cut the E8.01 review share (75.5% of 1,546 Egocentric-10K windows) to <60%
by narrowing the frozen abstain band, with the error cost measured honestly.

Protocol / honesty
------------------
* E8.01 (1,546 Egocentric-10K windows) has NO human labels (label_source 'none'):
  its review% is a *descriptive/workload* number, not accuracy.
* In-domain error = balanced accuracy (BA) on the 150-window IndEgo human gold,
  leave-one-clip-out (LOCO, 7 clips). Calibrators are fit LOCO on the gold only.
* Out-of-domain error = BA on the 60 human-labelled Egocentric-10K windows
  (51 good / 9 bad). This is the PRE-REGISTERED TEST: evaluation only, never fit on.
  n_bad=9 -> CIs are wide; the set is small.
* BA is computed over *decided* windows only (band-outside); windows inside the
  band are REVIEW and cost human time, not a model error.
* Error = 1 - BA. Secondary cost = good-windows-wrongly-rejected rate.

Outputs a single results JSON with n + label source + protocol for every number.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
BUNDLE = "/workspace/jadal_pod_bundle"
E8_JSONL = "/workspace/artifacts/windows_rescored.jsonl"
SEED = 7
GRID = [round(0.02 * i, 2) for i in range(16)]  # 0.00 .. 0.30


# --------------------------------------------------------------------------- IO
def load_e8_scores():
    rows = []
    with open(E8_JSONL) as f:
        for line in f:
            d = json.loads(line)
            if d.get("type") == "header":
                continue
            rows.append(d)
    score = np.array([r["score"] for r in rows], float)
    thr = [(r["versions"]["profile_thresholds"]["accept"],
            r["versions"]["profile_thresholds"]["reject"]) for r in rows]
    verd = [r["verdict"] for r in rows]
    return score, thr, verd


def load_gold():
    d = np.load(os.path.join(BUNDLE, "caches/ensemble/oof_probs.npz"), allow_pickle=True)
    y = d["y"].astype(int)
    clip = d["clip"].astype(str)
    p = d["p_ens"].astype(float)
    return y, clip, p


def load_ood():
    scores = json.load(open(os.path.join(BUNDLE, "caches/ensemble/ood_scores.json")))["scores"]
    labels = json.load(open(os.path.join(BUNDLE, "labels/ood_labels.json")))
    lab = {e["segment_id"]: e["verdict"] for e in labels["history"]}
    ids, y, p = [], [], []
    for s in scores:
        if s["id"] not in lab:
            continue
        ids.append(s["id"])
        y.append(1 if lab[s["id"]] == "good" else 0)
        p.append(float(s["usable_prob"]))
    return np.array(y), np.array(p), ids


# ---------------------------------------------------------------- calibration
def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_temperature(p, y):
    z = _logit(p)

    def nll(logT):
        t = np.exp(logT)
        q = 1.0 / (1.0 + np.exp(-z / t))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    r = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(r.x))


def apply_temperature(p, T):
    z = _logit(p)
    return 1.0 / (1.0 + np.exp(-z / T))


def fit_platt(p, y):
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(_logit(p).reshape(-1, 1), y)
    return lr


def apply_platt(p, lr):
    return lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def fit_isotonic(p, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return iso


def loco_calibrate(p, y, clip):
    """Return dict of LOCO-calibrated OOF probs (temp/platt/isotonic)."""
    out = {"temp": np.zeros_like(p), "platt": np.zeros_like(p), "isotonic": np.zeros_like(p)}
    for c in sorted(set(clip)):
        te = clip == c
        tr = ~te
        T = fit_temperature(p[tr], y[tr])
        out["temp"][te] = apply_temperature(p[te], T)
        out["platt"][te] = apply_platt(p[te], fit_platt(p[tr], y[tr]))
        out["isotonic"][te] = fit_isotonic(p[tr], y[tr]).predict(p[te])
    return out


def global_calibrators(p, y):
    return {
        "temp": fit_temperature(p, y),
        "platt": fit_platt(p, y),
        "isotonic": fit_isotonic(p, y),
    }


def apply_global(p, cal, name):
    if name == "temp":
        return apply_temperature(p, cal["temp"])
    if name == "platt":
        return apply_platt(p, cal["platt"])
    return cal["isotonic"].predict(p)


# -------------------------------------------------------------------- metrics
def balacc(y, p, w):
    """BA over decided windows for symmetric band half-width w around 0.5."""
    dec = (p <= 0.5 - w) | (p >= 0.5 + w)
    if dec.sum() == 0:
        return float("nan"), 0, 0, 0
    pred = (p >= 0.5 + w).astype(int)
    yy, pp = y[dec], pred[dec]
    tp = int(((yy == 1) & (pp == 1)).sum())
    fn = int(((yy == 1) & (pp == 0)).sum())
    tn = int(((yy == 0) & (pp == 0)).sum())
    fp = int(((yy == 0) & (pp == 1)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    ba = 0.5 * (tpr + tnr)
    return ba, int(dec.sum()), int(fp), int(fn)  # fp=good rejected, fn=bad accepted


def review_share(e8, w):
    return float(np.mean((e8 > 0.5 - w) & (e8 < 0.5 + w)))


def sweep(e8, gold_y, gold_p, gold_clip, ood_y, ood_p):
    rows = []
    for w in GRID:
        ba_g, n_g, fp_g, fn_g = balacc(gold_y, gold_p, w)
        ba_o, n_o, fp_o, fn_o = balacc(ood_y, ood_p, w)
        rows.append({
            "w": w,
            "review_share_1546": review_share(e8, w),
            "gold_ba_decided": ba_g, "gold_n_decided": n_g,
            "gold_err": 1 - ba_g if ba_g == ba_g else None,
            "gold_good_wrongly_rejected_rate": (fp_g / (gold_y == 1).sum()) if (gold_y == 1).sum() else None,
            "gold_bad_accepted_rate": (fn_g / (gold_y == 0).sum()) if (gold_y == 0).sum() else None,
            "ood_ba_decided": ba_o, "ood_n_decided": n_o,
            "ood_err": 1 - ba_o if ba_o == ba_o else None,
            "ood_good_wrongly_rejected_rate": (fp_o / (ood_y == 1).sum()) if (ood_y == 1).sum() else None,
            "ood_bad_accepted_rate": (fn_o / (ood_y == 0).sum()) if (ood_y == 0).sum() else None,
        })
    return rows


def boot_ci_gold(y, p, clip, w, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    clips = sorted(set(clip))
    vals = []
    for _ in range(n):
        pick = rng.choice(len(clips), size=len(clips), replace=True)
        idx = np.concatenate([np.where(clip == clips[j])[0] for j in pick])
        ba, nd, _, _ = balacc(y[idx], p[idx], w)
        if ba == ba:
            vals.append(ba)
    vals = np.array(vals)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def boot_ci_ood(y, p, w, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        ba, nd, _, _ = balacc(y[idx], p[idx], w)
        if ba == ba:
            vals.append(ba)
    vals = np.array(vals)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def crossing(rows, target):
    """Widest w on the grid with review_share < target (review is increasing in w)."""
    ok = [r for r in rows if r["review_share_1546"] < target]
    return max(ok, key=lambda r: r["w"]) if ok else None


def main():
    e8, e8_thr, e8_verd = load_e8_scores()
    gy, gclip, gp = load_gold()
    oy, op, oids = load_ood()

    baseline_verd = Counter(e8_verd)
    n = len(e8)

    # sanity: profile-rule reproduction
    match = sum(1 for s, (a, r), v in zip(e8, e8_thr, e8_verd)
                if ("accept" if s >= a else ("reject" if s <= r else "review")) == v)

    raw = sweep(e8, gy, gp, gclip, oy, op)

    # gold LOCO calibration
    gloco = loco_calibrate(gp, gy, gclip)
    gcal = global_calibrators(gp, gy)
    # E8 + OOD use the global (all-150 gold) calibrator, never fit on E8/OOD
    results = {"raw": {"rows": raw}}
    for name in ("temp", "platt", "isotonic"):
        e8c = apply_global(e8, gcal, name)
        oodc = apply_global(op, gcal, name)
        rows = sweep(e8c, gy, gloco[name], gclip, oy, oodc)
        results[name] = {"rows": rows}

        # per-calibration review at the frozen w=0.25 and at raw-matched w
    # BA at w=0 (all decided) sanity
    ba0_g, n0_g, _, _ = balacc(gy, gp, 0.0)
    ba0_gc, _, _, _ = balacc(gy, gloco["platt"], 0.0)

    # baseline E8 verdict distribution
    # reproduction of secondary / primary
    out = {
        "id": "Y1.01",
        "title": "Review-share reduction of the frozen PoC: abstain-band half-width sweep + LOCO calibration on IndEgo gold",
        "owner": "Y1",
        "protocol": {
            "band_rule": "accept if score >= 0.5+w ; reject if score <= 0.5-w ; else review (symmetric abstain band, half-width w)",
            "e8_baseline_rule": "frozen profile factory_1080p30: score>=0.75 accept, <=0.25 reject",
            "review_share_denominator": "n=1546 Egocentric-10K windows from E8.01 (label_source=none, DESCRIPTIVE/workload)",
            "in_domain_error": "balanced accuracy over DECIDED IndEgo gold windows, leave-one-clip-out (7 clips), human gold (single labeller), 100 good/50 bad",
            "ood_error": "balanced accuracy over DECIDED Egocentric-10K labelled sample (51 good / 9 bad); PRE-REGISTERED TEST, evaluation only, never fit on",
            "calibration": "temperature / Platt / isotonic fit LOCO on the gold only; the SAME all-150-gold calibrator is then applied to E8 and OOD (never fit on them)",
            "error": "1 - balanced_accuracy; good_wrongly_rejected = good windows decided reject",
            "seed": SEED,
            "ci": "gold=clip-bootstrap 2000x; OOD=window-bootstrap 2000x (optimistic)",
        },
        "inputs": {
            "e8_results_json": "/workspace/wt/exp-rescore/eval/results/exp_rescore/E8.01.json",
            "e8_jsonl": E8_JSONL,
            "gold_manifest": f"{BUNDLE}/labels/manifest_gold.json",
            "gold_oof": f"{BUNDLE}/caches/ensemble/oof_probs.npz",
            "ood_labels": f"{BUNDLE}/labels/ood_labels.json",
            "ood_scores": f"{BUNDLE}/caches/ensemble/ood_scores.json",
        },
        "n": {"e8_1546": n, "gold_150": int(len(gy)), "ood_60": int(len(oy)),
              "ood_bad": int((oy == 0).sum()), "ood_good": int((oy == 1).sum())},
        "label_source": {"e8_1546": "none", "gold_150": "human gold (single labeller)",
                         "ood_60": "human-labelled Egocentric-10K sample (51 good / 9 bad)"},
        "reproduction": {
            "e8_n": n,
            "e8_verdict_distribution": dict(baseline_verd),
            "e8_verdict_share": {k: v / n for k, v in baseline_verd.items()},
            "profile_rule_matches_rows": match,
            "report_review_share": 0.7549,
            "uniform_w0.25_review_share": review_share(e8, 0.25),
            "note": "uniform symmetric band w=0.25 reproduces the frozen review share within 1 window (the 6 default-profile windows use the default operating point in E8.01)",
        },
        "calibration_gold": {
            "raw_ba_at_w0_full_oof": ba0_g,
            "platt_loco_ba_at_w0": ba0_gc,
            "fitted_temperature_global": gcal["temp"],
            "global_platt_coef_intercept": [float(gcal["platt"].coef_[0][0]), float(gcal["platt"].intercept_[0])],
            "clear_superseded": [],
        },
        "sweep": {},
    }

    # attach CIs at the key crossing points for each method
    methods = [("raw", gp, None, op), ("temp", gloco["temp"], None, None),
               ("platt", gloco["platt"], None, None), ("isotonic", gloco["isotonic"], None, None)]
    # (ood calibrated probs handled below)
    ood_cal = {name: apply_global(op, gcal, name) for name in ("temp", "platt", "isotonic")}
    method_err = {"raw": (gp, op)}
    for name in ("temp", "platt", "isotonic"):
        method_err[name] = (gloco[name], ood_cal[name])

    for name, res in results.items():
        rows = res["rows"]
        gp_m, op_m = method_err[name]
        summary = {"rows": rows}
        for target in (0.60, 0.50, 0.40):
            r = crossing(rows, target)
            if r is None:
                summary[f"first_w_review_lt_{int(target*100)}"] = None
                continue
            w = r["w"]
            ba_g = r["gold_ba_decided"]; ba_o = r["ood_ba_decided"]
            entry = {
                "w": w,
                "review_share_1546": r["review_share_1546"],
                "gold_ba": ba_g, "gold_err": r["gold_err"],
                "gold_ba_ci95_clipbootstrap": boot_ci_gold(gy, gp_m, gclip, w) if name == "raw" else boot_ci_gold(gy, gp_m, gclip, w),
                "ood_ba": ba_o, "ood_err": r["ood_err"],
                "ood_ba_ci95_windowbootstrap": boot_ci_ood(oy, op_m, w),
                "gold_n_decided": r["gold_n_decided"], "ood_n_decided": r["ood_n_decided"],
                "gold_good_wrongly_rejected_rate": r["gold_good_wrongly_rejected_rate"],
                "ood_good_wrongly_rejected_rate": r["ood_good_wrongly_rejected_rate"],
            }
            summary[f"first_w_review_lt_{int(target*100)}"] = entry
        out["sweep"][name] = summary

    # calibration delta at equal-ish operating point: review at frozen error
    out["figures"] = {}
    out["command"] = "cd /workspace/wt/m-review-band && OMP_NUM_THREADS=2 python3 eval/exp_review/run_y1.py"
    out["results_dir"] = "eval/results/exp_review/Y1.01.json"
    out["honesty"] = [
        "E8.01 review share is DESCRIPTIVE on unlabelled windows, not accuracy.",
        "OOD-60 is the pre-registered test, evaluation only; nothing was fit on it.",
        "n_bad=9 -> OOD balanced-accuracy CIs are wide and the set is small; OOD AUROC for this recipe is 0.41-0.53 (near chance), so band choice cannot fix OOD decisions.",
        "IndEgo-only calibration may not transfer OOD; the OOD columns are reported for honesty, not used to pick the band.",
    ]

    os.makedirs(os.path.join(ROOT, "eval/results/exp_review"), exist_ok=True)
    outpath = os.path.join(ROOT, "eval/results/exp_review/Y1.01.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("wrote", outpath)

    # console table
    print(f"\nE8 baseline verdicts: {dict(baseline_verd)}  review={baseline_verd['review']/n:.4f}")
    for name in ("raw", "temp", "platt", "isotonic"):
        print(f"\n=== {name} ===")
        print(f"{'w':>5} {'review%':>8} {'goldBA':>7} {'goldErr':>8} {'n_dec':>6} {'oodBA':>7} {'oodErr':>8} {'n_dec':>6}")
        for r in out["sweep"][name]["rows"]:
            print(f"{r['w']:5.2f} {r['review_share_1546']*100:8.2f} "
                  f"{r['gold_ba_decided']:7.3f} {r['gold_err']:8.3f} {r['gold_n_decided']:6d} "
                  f"{r['ood_ba_decided']:7.3f} {r['ood_err']:8.3f} {r['ood_n_decided']:6d}")
        for target in (0.60, 0.50, 0.40):
            e = out["sweep"][name].get(f"first_w_review_lt_{int(target*100)}")
            if e:
                print(f"  review<{int(target*100)}%: w={e['w']:.2f} review={e['review_share_1546']*100:.1f}% "
                      f"goldBA={e['gold_ba']:.3f}{e['gold_ba_ci95_clipbootstrap']} "
                      f"oodBA={e['ood_ba']:.3f}{e['ood_ba_ci95_windowbootstrap']}")


if __name__ == "__main__":
    main()
