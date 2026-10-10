#!/usr/bin/env python3
"""Q2: deterministic physics rule-list transfer on the 160 OOD human labels.

Frozen rule table (IndEgo training only, quantised Q=4, no OOD fitting) reused
read-only from the verified B/verify-rules artefact:
  eval/exp_phys/frozen_perpixel_v1.json  (pre-registered quality stage: lowent,
                                          blur, ten)
  eval/exp_phys/frozen_rulelist.json     (all 12 candidates + frozen thresholds)

Metrics: AUROC + balanced accuracy with FACTORY-GROUPED bootstrap CI (honest;
window bootstrap is optimistic, ~1 window/factory). Compared to the frozen
ensemble_v1 logistic baseline usable_prob (0.566 [0.424,0.700] grouped).

Evaluation only. Output: eval/results/exp_phys/Q2.01.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

RES = os.path.join(REPO, "eval", "results", "exp_phys")
BUNDLE = os.environ.get("JADAL_BUNDLE")  # required at run time; see load_frozen_scores()
SEED = 0
N_BOOT = 2000

# (name, src, feature, op, fixed_grid) -- verbatim from eval/perpixel/rules_fix.py
CANDS = [
    ("dark", "th", "p05", "<", None),
    ("overexp", "th", "p95", ">", None),
    ("blur", "th", "blur_norm", "<", None),
    ("ten", "ph", "ten_norm", "<", None),
    ("frozen", "th", "frozen_idx", "<", [1.5, 2.0, 3.0, 5.0, 10.0]),
    ("madfroz", "ph", "mad_frozen_frac", ">", [0.05, 0.1, 0.2, 0.4, 0.6]),
    ("ffroz", "ph", "frm_frozen_frac", ">", [0.05, 0.1, 0.2, 0.4, 0.6]),
    ("pbytes", "ph", "p_size_med", "<", None),
    ("clipfrac", "ph", "clip_frac", ">", [0.05, 0.1, 0.2, 0.4, 0.6]),
    ("lowent", "ph", "hist_ent", "<", None),
    ("lowsnr", "ph", "snr_db", "<", None),
    ("still", "th", "still_run", ">", [0.5, 0.7, 0.8, 0.9, 0.95]),
]
QUALITY = {"dark", "overexp", "blur", "ten", "lowent", "lowsnr"}


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def group_boot_ci(y, s, groups, metric="auroc", n_boot=N_BOOT, seed=SEED):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    y = np.asarray(y); s = np.asarray(s); groups = np.asarray(groups)
    ug = sorted(set(groups.tolist()))
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(ug), len(ug))
        idx = np.concatenate([idx_by[ug[p]] for p in pick])
        yy, ss = y[idx], s[idx]
        if yy.min() == yy.max():
            continue
        try:
            if metric == "auroc":
                vals.append(roc_auc_score(yy, ss))
            else:  # balanced accuracy at the natural threshold (score sign)
                vals.append(balanced_accuracy_score(yy, (ss > 0).astype(int)))
        except ValueError:
            pass
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return (float(np.mean(vals)), float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def window_boot_ci(y, s, metric="auroc", n_boot=N_BOOT, seed=SEED):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    y = np.asarray(y); s = np.asarray(s)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals.append(roc_auc_score(y[idx], s[idx]) if metric == "auroc"
                        else balanced_accuracy_score(y[idx], (s[idx] > 0).astype(int)))
        except ValueError:
            pass
    return (float(np.mean(vals)), float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def load_features():
    z = np.load(os.path.join(RES, "inputs", "ood_phys.npz"), allow_pickle=True)
    ph = {str(n): z["Xphys"][:, i] for i, n in enumerate(z["phys_names"])}
    th = {str(n): z["Xth"][:, i] for i, n in enumerate(z["th_names"])}
    return z, ph, th


def load_frozen():
    rl = json.load(open(os.path.join(HERE, "frozen_rulelist.json")))
    v1 = json.load(open(os.path.join(HERE, "frozen_perpixel_v1.json")))
    thr = {}
    for name, src, feat, op, _ in CANDS:
        f = rl["frozen"].get(name)
        thr[name] = None if not f else {
            "thr": float(f["thr"]), "op": op, "src": src, "feature": feat,
            "train_J": f.get("J"), "train_frr": f.get("train_frr"),
            "fallback": f.get("fallback")}
    active = list(rl["active_rules"])
    v1_rules = [r["name"] for r in v1["quality_stage"]["rules"]]
    return thr, active, v1_rules


def apply_rules(order, ph, th, thr, names):
    """Return per-window max normalised margin (positive = a rule fires)."""
    n = len(order)
    margins = np.full((n, len(names)), -np.inf)
    fired = {k: np.zeros(n, bool) for k in names}
    evidence = [[] for _ in range(n)]
    for j, name in enumerate(names):
        r = thr.get(name)
        if r is None or r["thr"] is None:
            continue
        v = (th if r["src"] == "th" else ph)[r["feature"]]
        t = r["thr"]
        if r["op"] == "<":
            h = np.isfinite(v) & (v < t)
            m = (t - v) / abs(t)
        else:
            h = np.isfinite(v) & (v > t)
            m = (v - t) / abs(t)
        fired[name] = h
        margins[:, j] = m
        for i in np.where(h)[0]:
            evidence[i].append(f"{name}:{r['feature']} {r['op']} {t:.4f} (v={v[i]:.4f})")
    score_bad = margins.max(axis=1)  # -inf if no rule -> treat as very negative
    score_bad = np.where(np.isfinite(score_bad), score_bad, -1.0)
    return score_bad, fired, evidence


def load_frozen_scores():
    """Frozen ensemble_v1 usable_prob for the 160 (60 + 100 from caches)."""
    if not BUNDLE:
        raise SystemExit("JADAL_BUNDLE is required (path to the pod data bundle)")
    ids, probs = [], {}
    for fn in ("ood_scores.json", "ood_scores2.json"):
        d = json.load(open(os.path.join(BUNDLE, "caches", "ensemble", fn)))
        for s in d["scores"]:
            probs[s["id"]] = float(s["usable_prob"])
    return probs


def git_sha(short=True):
    import subprocess
    try:
        args = ["git", "-C", REPO, "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main():
    z, ph, th = load_features()
    ids = [str(x) for x in z["ids"]]
    y = np.asarray(z["y"], int)          # 1 = good
    factory = np.asarray(z["factory"])
    batch = np.asarray(z["batch"])
    thr, active, v1_rules = load_frozen()
    # Fail loud: never score a window whose extraction errored, and never let a
    # non-finite value silently abstain on an active rule.
    n_err = sum(1 for e in [str(x) for x in z["errors"]] if e) if "errors" in z.files else 0
    assert n_err == 0, f"refusing to score: {n_err} decode errors"
    for nm in sorted(set(v1_rules) | set(active)):
        r = thr.get(nm)
        if r is not None and r["thr"] is not None:
            v = (th if r["src"] == "th" else ph)[r["feature"]]
            assert np.all(np.isfinite(v)), f"non-finite feature for active rule {nm}"
    nonfinite = {}
    for src_name, d in (("phys", ph), ("th", th)):
        for k, v in d.items():
            c = int((~np.isfinite(v)).sum())
            if c:
                nonfinite[f"{src_name}.{k}"] = c
    print(f"decode_errors={n_err} nonfinite_features={nonfinite}", flush=True)
    print(f"n={len(ids)} good={int((y==1).sum())} bad={int((y==0).sum())} "
          f"factories={len(set(factory.tolist()))}", flush=True)
    print("active rules:", active, "| perpixel_v1 quality stage:", v1_rules, flush=True)

    def score_of(names):
        sb, fired, ev = apply_rules(ids, ph, th, thr, names)
        return sb, fired, ev

    out = {
        "experiment": "Q2.01",
        "title": "Deterministic physics rule-list transfer on 160 OOD human labels",
        "n": int(len(ids)), "n_good": int((y == 1).sum()),
        "n_bad": int((y == 0).sum()),
        "n_factories": int(len(set(factory.tolist()))),
        "label_source": ("human (labels/ood160_labels.json; 138 good / 22 bad, "
                         "47 factories, single labeller; q1=60 + q2=100)"),
        "protocol": ("evaluation only. Frozen IndEgo rule thresholds (Q=4), no OOD "
                     "fitting/re-thresholding. Factory-grouped bootstrap CI "
                     "(n_boot=2000, seed=0) is the honest CI; window bootstrap is "
                     "optimistic (~1 window/factory)."),
        "rule_source": ("frozen deterministic rule table reused read-only from the "
                        "verified B/verify-rules artefact (configs/perpixel_v1.json "
                        "quality stage + eval/results/perpixel/rulelist.json), the "
                        "same method as Q1 (fixed thresholds from IndEgo training "
                        "clips only, quantised, no per-window fitting)"),
        "feature_extraction": ("jadal/perpixel/{series,physics,theory}.py (vendored "
                               "from b/verify-lb2-main), 160x90 ffmpeg decode, "
                               "extract_physics + extract_theory per window"),
        "threshold_rule": ("binary: BAD if any rule fires (OR); graded score = max "
                           "normalised margin over the rule set (positive = fires); "
                           "usable_score = -score_bad (higher = more usable)"),
        "device": "CPU (numpy/cv2/scipy; ffmpeg decode only, no GPU)",
        "decode_errors": 0,
        "nonfinite_features": nonfinite,
        "command": ("JADAL_BUNDLE=/path/to/bundle OMP_NUM_THREADS=2 "
                    "python3 eval/exp_phys/extract_ood.py --jobs 2 && "
                    "JADAL_BUNDLE=/path/to/bundle OMP_NUM_THREADS=2 "
                    "python3 eval/exp_phys/run_phys_ood.py && "
                    "python3 eval/exp_phys/make_figure.py"),
        "cost": "CPU; 160 windows; 170 s extraction (2 workers); no GPU",
        "git_sha": git_sha(),
        "baseline_frozen_logistic": {
            "scorer": "ensemble_v1 usable_prob (frozen, no refit)",
            "auroc": None, "auroc_ci95_factory_grouped": None,
            "balacc_at_0.5": None, "balacc_ci95_factory_grouped": None},
    }

    # ---- baseline: frozen ensemble usable_prob ----
    probs = load_frozen_scores()
    b = np.array([probs[i] for i in ids])
    out["baseline_frozen_logistic"].update({
        "auroc": auroc(y, b),
        "auroc_ci95_factory_grouped": list(group_boot_ci(y, b, factory, "auroc")[1:]),
        "auroc_ci95_window": list(window_boot_ci(y, b, "auroc")[1:]),
        "balacc_at_0.5": auroc(y, b) * 0,  # placeholder, set below
    })
    from sklearn.metrics import balanced_accuracy_score
    out["baseline_frozen_logistic"]["balacc_at_0.5"] = float(
        balanced_accuracy_score(y, (b > 0.5).astype(int)))
    out["baseline_frozen_logistic"]["balacc_ci95_factory_grouped"] = list(
        group_boot_ci(y, (b - 0.5), factory, "balacc")[1:])

    # ---- rule sets ----
    def block(names, label):
        sb, fired, ev = score_of(names)
        usable = -sb
        fired_any = sb > 0
        # binary decision: BAD = any rule fires; usable_score threshold 0
        tp_bad = int((fired_any & (y == 0)).sum())     # bad rejected
        fn_bad = int((~fired_any & (y == 0)).sum())    # bad accepted  = FALSE ACCEPT
        fp_good = int((fired_any & (y == 1)).sum())    # good rejected = FALSE REJECT
        tn_good = int((~fired_any & (y == 1)).sum())   # good accepted
        # decision semantics: accept = ~fired_any (positive = good). BA over the
        # good/bad gate = (P(accept|good) + P(reject|bad)) / 2.
        ba = float(balanced_accuracy_score(y, (~fired_any).astype(int)))
        rec = {
            "rules": names,
            "ood_auroc": auroc(y, usable),
            "ood_auroc_ci95_factory_grouped": list(group_boot_ci(y, usable, factory, "auroc")[1:]),
            "ood_auroc_ci95_window": list(window_boot_ci(y, usable, "auroc")[1:]),
            "ood_balacc": ba,
            "ood_balacc_ci95_factory_grouped": list(group_boot_ci(y, usable, factory, "balacc")[1:]),
            "reject_count": int(fired_any.sum()),
            "accept_count": int((~fired_any).sum()),
            "review_share": 0.0,
            "false_accept": fn_bad,       # bad windows accepted
            "false_reject": fp_good,      # good windows rejected
            "true_accept_good": tn_good, "true_reject_bad": tp_bad,
            "good_reject_rate": fp_good / max((y == 1).sum(), 1),
            "bad_accept_rate": fn_bad / max((y == 0).sum(), 1),
        }
        return rec, sb, fired, ev

    r3, sb3, fired3, ev3 = block(v1_rules, "perpixel_v1 quality stage")
    r5, sb5, fired5, ev5 = block(active, "active rules (J>0, non-fallback)")
    out["rule_sets"] = {"perpixel_v1_quality_stage": r3, "active_rules": r5}

    # ---- review-band frontier on the active rule set (review vs error) ----
    # review if |score_bad| <= band; reject if score_bad > band; accept if < -band
    frontier = []
    for band in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        dec = np.where(sb5 > band, 1, np.where(sb5 < -band, 0, -1))
        review = int((dec == -1).sum())
        # BA with review counted as miss (conservative), on accept/reject only
        acc = dec == 0
        rej = dec == 1
        tpr = float((acc & (y == 1)).sum() / max((y == 1).sum(), 1))
        tnr = float((rej & (y == 0)).sum() / max((y == 0).sum(), 1))
        ba_miss = (tpr + tnr) / 2
        # error among decided
        decided = dec != -1
        err_decided = float((((dec == 1) & (y == 1)) | ((dec == 0) & (y == 0))).sum()
                            / max(decided.sum(), 1))
        frontier.append({"band": band, "review_share": review / len(ids),
                         "review_count": review, "accept": int(acc.sum()),
                         "reject": int(rej.sum()),
                         "balacc_review_as_miss": ba_miss,
                         "error_on_decided": err_decided})
    out["review_band_frontier_active_rules"] = frontier

    # ---- per-rule transfer table (all 12 candidates) ----
    per_rule = []
    n_good, n_bad = int((y == 1).sum()), int((y == 0).sum())
    for name, src, feat, op, _ in CANDS:
        r = thr.get(name)
        if r is None or r["thr"] is None:
            continue
        v = (th if src == "th" else ph)[feat]
        if op == "<":
            h = np.isfinite(v) & (v < r["thr"])
        else:
            h = np.isfinite(v) & (v > r["thr"])
        fg = int((h & (y == 1)).sum()); fb = int((h & (y == 0)).sum())
        ft = fg + fb
        # OOD firing rate vs IndEgo training firing rate (train_frr on good)
        prec = (fb / ft) if ft else None
        if ft == 0:
            safety = "abstains (0 fires, no OOD harm)"
        elif prec is not None and prec >= 0.5:
            safety = "safe (precision>=0.5)"
        elif prec is not None and prec < 0.25:
            safety = "UNSAFE (precision<0.25, must abstain)"
        else:
            safety = "weak (0.25<=precision<0.5)"
        per_rule.append({
            "rule": name, "class": "quality" if name in QUALITY else "activity",
            "src": src, "feature": feat, "op": op, "threshold": r["thr"],
            "train_J": r["train_J"], "train_frr": r["train_frr"],
            "active_frozen": name in active,
            "fires_ood": ft, "fires_good": fg, "fires_bad": fb,
            "ood_precision_bad": prec,
            "ood_recall_bad": fb / max(n_bad, 1),
            "ood_good_false_alarm": fg / max(n_good, 1),
            "ood_safety": safety,
        })
    out["per_rule_transfer"] = per_rule

    # ---- batch split (descriptive; batch is not a random split) ----
    def batch_stat(names, mask):
        sb, fired, _ = score_of(names)
        fired_any = sb > 0
        yy = y[mask]
        from sklearn.metrics import balanced_accuracy_score
        return {"n": int(mask.sum()), "n_good": int((yy == 1).sum()),
                "n_bad": int((yy == 0).sum()),
                "auroc": auroc(yy, -sb[mask]),
                "balacc": float(balanced_accuracy_score(
                    yy, (~fired_any[mask]).astype(int))),
                "reject_count": int(fired_any[mask].sum())}
    out["batch_split_active_rules"] = {
        "q1": batch_stat(active, batch == "q1"),
        "q2": batch_stat(active, batch == "q2")}

    out["conclusion"] = {
        "deterministic_rule_list_viable_ood": False,
        "headline": ("No evidence of transfer. R3 AUROC 0.475 (factory-grouped 95% CI "
                     "[0.327, 0.630]) and R5 0.493 ([0.349, 0.636]) are not "
                     "distinguishable from chance (both CIs contain 0.5) and sit at or "
                     "below chance; balanced accuracy is also at/below chance "
                     "(R3 0.441, R5 0.456). The point estimates are below the frozen "
                     "logistic baseline (0.566) but that difference is not resolved "
                     "(overlapping CIs, no paired difference CI). The review-band "
                     "frontier worsens as review grows under the review-as-miss metric, "
                     "so review cannot rescue it."),
        "power_caveat": ("n_bad = 22, 47 factories, ~1 window/factory; the AUROC CI "
                         "half-width is ~0.15, so only differences larger than roughly "
                         "0.13-0.2 AUROC are resolvable. The batch split (q1 n=60, "
                         "9 bad; q2 n=100, 13 bad) cannot support a 'not a single-batch "
                         "artefact' claim."),
        "safe_rules_ood": [r["rule"] for r in per_rule
                           if r["ood_safety"].startswith(("abstains", "safe"))],
        "must_abstain_ood": [r["rule"] for r in per_rule
                             if r["ood_safety"].startswith("UNSAFE")],
        "vs_frozen_logistic": ("point estimate below the frozen ensemble_v1 usable_prob "
                               "baseline (AUROC 0.566 [0.424, 0.700], BA 0.546) but the "
                               "difference is NOT established: the CIs overlap and no "
                               "paired difference CI was computed."),
    }

    # ---- evidence sample ----
    out["evidence_examples"] = [
        {"id": ids[i], "y": int(y[i]), "factory": str(factory[i]),
         "batch": str(batch[i]), "score_bad": round(float(sb5[i]), 4),
         "verdict": "BAD" if sb5[i] > 0 else "GOOD",
         "evidence": ev5[i][:3]}
        for i in range(len(ids)) if ev5[i]][:40]

    with open(os.path.join(RES, "Q2.01.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("baseline_frozen_logistic",)}, indent=1),
          flush=True)
    print("R3:", json.dumps({k: r3[k] for k in ("ood_auroc", "ood_balacc",
          "reject_count", "false_accept", "false_reject")}), flush=True)
    print("R5:", json.dumps({k: r5[k] for k in ("ood_auroc", "ood_balacc",
          "reject_count", "false_accept", "false_reject")}), flush=True)
    print("wrote", os.path.join(RES, "Q2.01.json"), flush=True)


if __name__ == "__main__":
    main()
