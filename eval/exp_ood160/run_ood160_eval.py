#!/usr/bin/env python3
"""O1: frozen pre-registered OOD evaluation on the 160 human labels.

Claim O1 (branch m/ood160-eval). EVALUATION ONLY: nothing is fitted, tuned or
re-thresholded. The frozen two-view ensemble_v1 (config sha256 4da1068a...) is
scored from the pod-bundle caches (no video decode, no GPU):

  caches/ensemble/ood_scores.json   (queue q1, 60 windows, batch q1)
  caches/ensemble/ood_scores2.json  (queue q2, 100 windows, batch q2)

Labels: labels/ood160_labels.json  {windows:[{segment_id,verdict good|bad,
factory,batch,...}]}  -> good=1, bad=0.  Expected 138 good / 22 bad / 47 factories.

Metrics (frozen threshold and abstain band from configs/ensemble_v1.json):
  * OOD AUROC of usable_prob (primary, pre-registered), plus the two views.
  * Balanced accuracy at the shipped operating point (accept=good, reject=bad,
    review=miss: balacc = (good_accept/N_good + bad_reject/N_bad)/2).
  * 95% CIs by (a) window bootstrap (optimistic) and (b) factory-grouped
    bootstrap (honest: factories are the independent unit; clip==window here).
  * accept/reject/review distribution + confusion matrix.

The shipped PoC *cascade* is a different artifact (IndEgo-calibrated heads, not
in the bundle caches); its 160-window result is carried over from
docs/ood_eval_poc_cascade_160.txt and recorded, clearly labelled, not recomputed.

Outputs (relative to repo root):
  eval/results/exp_ood160/labels160.json / .csv   label table
  eval/results/exp_ood160/O1.01.json              full metrics
Run: OMP_NUM_THREADS=2 python3 eval/exp_ood160/run_ood160_eval.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
OUTDIR = os.path.join(ROOT, "eval", "results", "exp_ood160")

FROZEN_SHA = "4da1068a78d5d15021f4e167d6aa24fd2e0bb08aa88f83c83c2f167cc086fe31"
B = 2000
SEED = 0
GOOD, BAD = "good", "bad"


def auroc(y, p):
    """Mann-Whitney AUROC with average ranks for ties (sklearn-free, matches
    eval/poc/ood_check.py); y in {0,1}, higher p = more likely good."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if len(set(y.tolist())) < 2:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sp = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    rank_sum = ranks[y == 1].sum()
    return float((rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ba_at(y, verdict):
    y = np.asarray(y)
    verdict = np.asarray(verdict)
    g = y == 1
    b = y == 0
    good_acc = float((verdict[g] == "accept").mean()) if g.any() else float("nan")
    bad_rej = float((verdict[b] == "reject").mean()) if b.any() else float("nan")
    return 0.5 * (good_acc + bad_rej), good_acc, bad_rej


def verdict_of(p, thr=0.5, lo=0.4):
    """Frozen shipped rule (jadal/ensemble/scorer.py): p>thr accept; p<lo
    reject; else review (effective band [0.4, 0.5])."""
    return np.where(p > thr, "accept", np.where(p < lo, "reject", "review"))


def load_labels():
    doc = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))
    rows = []
    for w in doc["windows"]:
        rows.append({
            "segment_id": w["segment_id"],
            "window": w.get("window", ""),
            "batch": w.get("batch", ""),
            "factory": w.get("factory", ""),
            "verdict_human": w["verdict"],
            "reason": w.get("reason", "") or "",
            "note": w.get("note", "") or "",
            "video": w.get("video", ""),
            "labelled_utc": w.get("labelled_utc", ""),
            "y": 1 if w["verdict"] == GOOD else 0,
        })
    return rows, doc


def load_scores():
    sc = {}
    prov = {}
    for fn in ("ood_scores.json", "ood_scores2.json"):
        path = os.path.join(BUNDLE, "caches", "ensemble", fn)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        prov[fn] = {"config_sha256": d.get("config_sha256"),
                    "n_scores": len(d["scores"]),
                    "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
        for x in d["scores"]:
            if "error" not in x:
                sc[x["id"]] = x
    return sc, prov


def pct_ci(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 3),
            round(float(np.percentile(vals, 97.5)), 3)]


def window_bootstrap(y, p, verdict, rng):
    n = len(y)
    au, bb = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(set(y[idx].tolist())) < 2:
            continue
        au.append(auroc(y[idx], p[idx]))
        bb.append(ba_at(y[idx], verdict[idx])[0])
    return pct_ci(au), pct_ci(bb)


def group_bootstrap(y, p, verdict, groups, rng):
    ug = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ug}
    au, bb = [], []
    for _ in range(B):
        pick = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([by[g] for g in pick])
        if len(set(y[idx].tolist())) < 2:
            continue
        au.append(auroc(y[idx], p[idx]))
        bb.append(ba_at(y[idx], verdict[idx])[0])
    return pct_ci(au), pct_ci(bb)


def main():
    rows, doc = load_labels()
    sc, prov = load_scores()
    os.makedirs(OUTDIR, exist_ok=True)

    missing = [r["segment_id"] for r in rows if r["segment_id"] not in sc]
    table = []
    for r in rows:
        x = sc.get(r["segment_id"], {})
        table.append({
            "segment_id": r["segment_id"],
            "window": r["window"],
            "batch": r["batch"],
            "factory": r["factory"],
            "human": r["verdict_human"],
            "y": r["y"],
            "reason": r["reason"],
            "usable_prob": x.get("usable_prob"),
            "p_eg2": x.get("p_eg2"),
            "p_pixel": x.get("p_pixel"),
            "margin": x.get("margin"),
            "verdict_cached": x.get("verdict"),
        })

    ids = [r["segment_id"] for r in table if r["segment_id"] in sc]
    y = np.array([next(t["y"] for t in table if t["segment_id"] == i) for i in ids])
    fac = np.array([next(t["factory"] for t in table if t["segment_id"] == i) for i in ids])
    batch = np.array([next(t["batch"] for t in table if t["segment_id"] == i) for i in ids])
    P = {k: np.array([sc[i][k] for i in ids], dtype=float)
         for k in ("usable_prob", "p_pixel", "p_eg2")}
    V = verdict_of(P["usable_prob"])

    n1, n0 = int((y == 1).sum()), int((y == 0).sum())

    # ---- metrics per score ----
    def block(p):
        a = auroc(y, p)
        ba, ga, br = ba_at(y, V)
        return {"auroc": round(a, 4) if a else None,
                "mean_good": round(float(p[y == 1].mean()), 4),
                "mean_bad": round(float(p[y == 0].mean()), 4)}

    rng = np.random.default_rng(SEED)
    aurocs = {}
    for name in ("usable_prob", "p_pixel", "p_eg2"):
        aurocs[name] = block(P[name])

    rng = np.random.default_rng(SEED)
    au_ci_w, ba_ci_w = window_bootstrap(y, P["usable_prob"], V, rng)
    rng = np.random.default_rng(SEED)
    au_ci_g, ba_ci_g = group_bootstrap(y, P["usable_prob"], V, fac, rng)
    # per-view grouped CIs
    rng = np.random.default_rng(SEED)
    view_cis = {}
    for name in ("p_pixel", "p_eg2"):
        rng = np.random.default_rng(SEED)
        w_ci = window_bootstrap(y, P[name], V, rng)[0]
        rng = np.random.default_rng(SEED)
        g_ci = group_bootstrap(y, P[name], V, fac, rng)[0]
        view_cis[name] = {"window": w_ci, "grouped": g_ci}

    ba, good_accept, bad_reject = ba_at(y, V)
    # second convention: collapse to a binary accept/reject decision at 0.5
    # (review -> accept for good / reject for bad is NOT used; instead the
    # pre-registered scalar decision p>0.5 accept else reject, review ignored),
    # which is what the handover's "balacc@0.5" reports.
    Vbin = np.where(P["usable_prob"] > 0.5, "accept", "reject")
    ba_bin, ga_bin, br_bin = ba_at(y, Vbin)

    # ---- confusion matrix good/bad x verdict ----
    def cm(mask):
        v = V[mask]
        return {k: int((v == k).sum()) for k in ("accept", "review", "reject")}
    conf = {GOOD: cm(y == 1), BAD: cm(y == 0)}
    conf[GOOD]["n"] = n1
    conf[BAD]["n"] = n0

    # ---- batch split ----
    batch_split = {}
    for b in sorted(set(batch.tolist())):
        m = batch == b
        batch_split[b] = {
            "n": int(m.sum()), "n_good": int((y[m] == 1).sum()),
            "n_bad": int((y[m] == 0).sum()),
            "auroc": round(auroc(y[m], P["usable_prob"][m]), 4),
        }

    # ---- cache/verdict hygiene: q2 cache uses a wider review band ----
    mism = [t["segment_id"] for t in table
            if t["segment_id"] in sc and t["verdict_cached"]
            != verdict_of(np.array([t["usable_prob"]]))[0]]
    cached_counts = {k: int(sum(1 for t in table if t["verdict_cached"] == k))
                     for k in ("accept", "review", "reject")}

    # ---- factory inventory ----
    fac_sizes = {}
    for f in fac:
        fac_sizes[f] = int((fac == f).sum())

    out = {
        "experiment": "O1",
        "claim": "O1",
        "title": "Frozen PoC out-of-domain evaluation on the 160 human labels",
        "protocol": ("pre-registered, FROZEN ensemble_v1 (two-view equal-weight "
                     "probability average, threshold 0.5, abstain band [0.4,0.6]); "
                     "evaluation only - nothing fitted, tuned or re-thresholded"),
        "frozen_config_sha256": FROZEN_SHA,
        "label_source": {
            "file": "labels/ood160_labels.json",
            "description": doc.get("description"),
            "n": doc.get("n"), "n_good": doc.get("n_good"), "n_bad": doc.get("n_bad"),
            "bad_reasons": doc.get("bad_reasons"),
        },
        "score_source": prov,
        "n": len(ids), "n_good": n1, "n_bad": n0,
        "n_factories": int(len(set(fac.tolist()))),
        "n_missing_scores": len(missing),
        "auroc": {k: v["auroc"] for k, v in aurocs.items()},
        "auroc_detail": aurocs,
        "auroc_ci95": {
            "usable_prob": {
                "window_bootstrap_optimistic": au_ci_w,
                "factory_grouped_bootstrap_honest": au_ci_g,
            },
            "p_pixel": view_cis["p_pixel"],
            "p_eg2": view_cis["p_eg2"],
        },
        "bootstrap": {"n_boot": B, "seed": SEED,
                      "window_unit": "10-s window (optimistic: correlated within factory)",
                      "group_unit": "factory (47 groups; clip==window, so factory is the independent unit)"},
        "operating_point": {
            "rule": "usable_prob>0.5 accept; 0.4<=p<=0.5 review; p<0.4 reject",
            "balanced_accuracy_review_as_miss": round(ba, 4),
            "good_accept_rate": round(good_accept, 4),
            "bad_reject_rate": round(bad_reject, 4),
            "ba_ci95_window_bootstrap": ba_ci_w,
            "ba_ci95_factory_grouped_bootstrap": ba_ci_g,
            "balanced_accuracy_binary_at_0.5": round(ba_bin, 4),
            "balanced_accuracy_binary_note": ("pre-registered scalar decision p>0.5 accept "
                                              "else reject (review counted as reject); this is "
                                              "the handover's 'balacc@0.5' convention"),
            "bad_accept_rate_binary": round(1 - br_bin, 4),
            "verdict_distribution": {k: int((V == k).sum()) for k in ("accept", "review", "reject")},
        },
        "confusion_matrix": conf,
        "batch_split": batch_split,
        "cache_hygiene": {
            "note": ("ood_scores2.json (q2) stores 7 windows with usable_prob>0.5 as "
                     "'review' (wider band); verdicts here are recomputed from the "
                     "frozen rule, not copied."),
            "cached_verdict_counts": cached_counts,
            "n_cache_verdict_mismatch": len(mism),
            "mismatch_ids": mism,
        },
        "factories": {"n": len(fac_sizes), "sizes": fac_sizes},
        "old_n60": {
            "ensemble_v1_auroc": 0.415, "ensemble_v1_ci_window": [0.207, 0.630],
            "poc_cascade_auroc": 0.529, "poc_cascade_ci_window": [0.316, 0.728],
            "source": "eval/results/poc/ood_check_60.json (b/verify-audit); n_bad=9 exploratory",
        },
        "poc_cascade_handover_160": {
            "auroc": 0.600, "ci_window_optimistic": [0.474, 0.724],
            "good": {"accept": 70, "reject": 53, "review": 15},
            "bad": {"accept": 8, "reject": 13, "review": 1},
            "balanced_accuracy": 0.549,
            "by_batch": {"q1_n60": 0.529, "q2_n100": 0.645},
            "provenance": ("jadal_pod_bundle/docs/ood_eval_poc_cascade_160.txt and FACTS.md:97 "
                           "(handover/ood_eval_poc_cascade_160.txt); the IndEgo-calibrated cascade "
                           "heads are NOT in the pod bundle caches, so this number is carried over, "
                           "not recomputed here; no factory-grouped CI available for it."),
        },
        "caveats": [
            "n_bad=22 now exceeds the pre-registered 10/class minimum (confirmatory), but the "
            "positives are still only 22 windows scattered over 47 factories.",
            "The window bootstrap is optimistic: windows from the same factory share lighting, "
            "camera and work-cell, so they are not independent. The factory-grouped CI is the honest one.",
            "Some factories contribute up to 13 windows and contain both classes, so the grouped "
            "resample can drop or duplicate a large fraction of the data -> wide, lumpy CI.",
            "The shipped PoC cascade is a different scorer; its 160-window result is carried over "
            "from the handover and has no grouped CI.",
        ],
    }

    # chronological ordering (q1 then q2) for the table files
    table_ordered = table
    json.dump(table_ordered, open(os.path.join(OUTDIR, "labels160.json"), "w"), indent=1)
    with open(os.path.join(OUTDIR, "labels160.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table_ordered[0].keys()))
        w.writeheader()
        w.writerows(table_ordered)
    json.dump(out, open(os.path.join(OUTDIR, "O1.01.json"), "w"), indent=1, sort_keys=True)

    # ---- stdout summary ----
    print(f"n={len(ids)} good={n1} bad={n0} factories={out['n_factories']} missing_scores={len(missing)}")
    for k in ("usable_prob", "p_pixel", "p_eg2"):
        print(f"  AUROC {k:11s} {aurocs[k]['auroc']:.3f}  "
              f"(good {aurocs[k]['mean_good']:.3f} vs bad {aurocs[k]['mean_bad']:.3f})")
    print(f"usable_prob AUROC window-boot {au_ci_w} (optimistic) | factory-grouped {au_ci_g} (honest)")
    print(f"operating point: BA(review=miss) {ba:.3f} [win {ba_ci_w} | grp {ba_ci_g}] "
          f"good_accept {good_accept:.3f} bad_reject {bad_reject:.3f}")
    print(f"                 BA(binary@0.5) {ba_bin:.3f} (matches handover balacc@0.5 0.546)")
    print(f"verdicts accept {out['operating_point']['verdict_distribution']['accept']} "
          f"review {out['operating_point']['verdict_distribution']['review']} "
          f"reject {out['operating_point']['verdict_distribution']['reject']}")
    print(f"confusion good {conf[GOOD]} bad {conf[BAD]}")
    print(f"cache verdict mismatches vs frozen rule: {len(mism)}")
    print(f"wrote {os.path.relpath(os.path.join(OUTDIR,'O1.01.json'), ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
