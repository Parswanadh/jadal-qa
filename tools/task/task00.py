#!/usr/bin/env python
"""TASK.00(a) -- do NO_ACTIVE_WORK labels predict the shipped 26-feature verdicts?

Zero training, zero new inference.  Reads:
  * the 160 human-labelled Egocentric-10K windows (labels/ood160_labels.json),
  * the frozen old-26 scores + verdicts for those windows
    (interview/dsh_pack/ood160_heads_old26_vs_new22.json, "old26" = original portal).

The shipped verdict is re-derived from the frozen score with the band that
`jadal/poc/cascade.py::decide_window` implements, and the derivation is CHECKED
against the verdict the file itself carries (0 mismatches is the assertion).

Measures, as asked by the brief:
  (i)   accept / review / reject share for NO_ACTIVE_WORK windows vs all others;
  (ii)  per-factory NO_ACTIVE_WORK share vs per-factory reject+review share
        (Spearman + Pearson, n = 47 factories, permutation p);
  (iii) AUROC of the frozen score for NO_ACTIVE_WORK vs good, factory-grouped CI.

CPU only.  Run:
  JADAL_BUNDLE=<bundle> /home/parshu/envs/jadal/bin/python tools/task/task00.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dsh import common  # noqa: E402

ID = "TASK.00"
OUT = os.path.join(common.REPO, "eval", "results", "task", f"{ID}.json")

# the shipped band, from jadal/poc/cascade.py (THRESHOLD=0.6, MARGIN=0.25)
MARGIN = 0.25
THRESHOLD = 0.6


def derive_verdict(p):
    """cascade.decide_window's S1 branch, verbatim in effect.

    |p-0.5| < MARGIN -> review; then p >= THRESHOLD -> accept else reject.
    With MARGIN = 0.25 the review band is (0.25, 0.75), so the S1 branch is only
    reached for p <= 0.25 or p >= 0.75 and THRESHOLD = 0.6 is inert.
    """
    if p is None:
        return None
    if abs(p - 0.5) < MARGIN:
        return "review"
    return "accept" if p >= THRESHOLD else "reject"


def resolve_bundle():
    """ood160_labels.json is NOT in the bundle path the brief exports.

    It lives one level up, in handover/pod_bundle.  Both bundles carry a
    byte-identical manifest_gold.json and features.json (sha256 verified), so
    reading ood160 from the sibling is safe; record which path was used.
    `common._ood160_manifest` (dsh/infra 7865448) now searches the same
    candidates; this helper is kept so the script also runs against the
    pre-fix harness, and it is what the results JSON records.
    """
    b = common.BUNDLE
    cands = [b,
             os.path.join(os.path.dirname(b), "pod_bundle"),
             os.path.join(os.path.dirname(os.path.dirname(b)), "pod_bundle")]
    for c in cands:
        if os.path.exists(os.path.join(c, "labels", "ood160_labels.json")):
            return c
    raise SystemExit(f"ood160_labels.json not found under any of {cands}")


def spearman(x, y):
    """Rank correlation with average ranks for ties (no scipy)."""
    def rank(v):
        v = np.asarray(v, float)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        sv = v[order]
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    bundle = resolve_bundle()
    os.environ["JADAL_BUNDLE"] = bundle
    d = common.load_ood160(bundle)
    ids, y, fac, reason = d["ids"], d["y"], d["factory"], d["reason"]
    p26, v26 = d["p26"], d["v26"]
    n = len(ids)

    # ---- 0. verify the band reproduces the shipped verdicts -----------------
    derived = [derive_verdict(None if not np.isfinite(p) else float(p)) for p in p26]
    mism = [(ids[i], v26[i], derived[i], p26[i])
            for i in range(n) if derived[i] != v26[i]]
    if mism:
        raise SystemExit(f"band does not reproduce the shipped verdicts: {mism[:5]}")
    n_null = int(sum(1 for p in p26 if not np.isfinite(p)))

    is_na = np.asarray([r == "NO_ACTIVE_WORK" for r in reason])
    is_good = y == 1
    is_other_bad = (~is_good) & (~is_na)
    n_na, n_good, n_other = int(is_na.sum()), int(is_good.sum()), int(is_other_bad.sum())

    # ---- (i) verdict shares -------------------------------------------------
    def shares(mask):
        v = np.asarray(derived, dtype=object)[mask]
        m = len(v)
        out = {"n": m}
        for k in ("accept", "review", "reject"):
            c = int((v == k).sum())
            out[k] = c
            out[f"{k}_share"] = round(c / m, 4) if m else float("nan")
        # reject+review == not accepted in the shipped band
        out["reject_or_review"] = out["review"] + out["reject"]
        out["reject_or_review_share"] = round(out["reject_or_review"] / m, 4) if m else float("nan")
        return out

    shares_na = shares(is_na)
    shares_other = shares(~is_na)
    shares_good = shares(is_good)
    shares_all = shares(np.ones(n, bool))
    # breakdown by the three bad reasons, because NO_ACTIVE_WORK is not all the bad
    by_reason = {}
    for r in sorted(set(reason.tolist())):
        if r == "":
            continue
        m = np.asarray([x == r for x in reason])
        by_reason[r] = shares(m)

    # ---- (ii) per-factory shares and their correlation ----------------------
    factories = sorted(set(fac.tolist()))
    rows = []
    for f in factories:
        m = fac == f
        na = int(is_na[m].sum())
        v = np.asarray(derived, dtype=object)[m]
        rr = int(((v == "reject") | (v == "review")).sum())
        rows.append({"factory": f, "n_windows": int(m.sum()), "n_no_active_work": na,
                     "no_active_work_share": round(na / int(m.sum()), 4),
                     "reject_or_review_share": round(rr / int(m.sum()), 4)})
    x = np.asarray([r["no_active_work_share"] for r in rows], float)
    yv = np.asarray([r["reject_or_review_share"] for r in rows], float)
    rho = spearman(x, yv)
    r_pear = float(np.corrcoef(x, yv)[0, 1]) if x.std() and yv.std() else float("nan")
    # permutation p (two-sided) on the Pearson r; 20000 draws, seed 0
    rng = np.random.RandomState(0)
    perm = np.asarray([abs(float(np.corrcoef(x, rng.permutation(yv))[0, 1]))
                       for _ in range(20000)])
    p_perm = float((perm >= abs(r_pear)).mean())
    n_fac_with_na = int((x > 0).sum())
    n_fac_ge5 = int(sum(1 for r in rows if r["n_windows"] >= 5))

    # ---- (iii) AUROC of the frozen score ------------------------------------
    def auroc_block(sel, label="good"):
        yy = y[sel]
        ss = p26[sel]
        g = fac[sel]
        a = common.fast_auc(yy, ss)
        lo, hi, nb = common.group_boot_ci(yy, ss, g, n_boot=2000, seed=0)
        wlo, whi, wnb = common.window_boot_ci(yy, ss, n_boot=2000, seed=0)
        npos, nneg = int((yy == 1).sum()), int((yy == 0).sum())
        return {"label_1": label, "n": int(len(yy)), "n_good": npos, "n_bad": nneg,
                "n_factories": int(len(set(g.tolist()))),
                "auroc": round(float(a), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "ci95_type": "factory-grouped bootstrap (2000 draws, seed 0)",
                "n_boot_valid": nb,
                "ci95_window_optimistic": [round(wlo, 4), round(whi, 4)],
                "ci95_type_window_optimistic": "window bootstrap (2000 draws, seed 0) -- optimistic, factories correlated"}

    sel_na_vs_good = is_na | is_good
    auc_na_vs_good = auroc_block(sel_na_vs_good, "good (1) vs NO_ACTIVE_WORK (0)")
    auc_all_bad = auroc_block(np.ones(n, bool), "good (1) vs all 22 bad (0)")
    auc_na_vs_quiet = auroc_block(is_na | is_good | is_other_bad,
                                  "good (1) vs all 22 bad (0) [same set, sensitivity]")
    # "NO_ACTIVE_WORK vs good" read the other way: 1 = NO_ACTIVE_WORK
    auc_flipped = {
        "label_1": "NO_ACTIVE_WORK (1) vs good (0)",
        "auroc": round(1.0 - auc_na_vs_good["auroc"], 4),
        "n": auc_na_vs_good["n"], "n_good": auc_na_vs_good["n_bad"],
        "n_bad": auc_na_vs_good["n_good"],
        "ci95": [round(1 - auc_na_vs_good["ci95"][1], 4),
                 round(1 - auc_na_vs_good["ci95"][0], 4)],
        "ci95_type": auc_na_vs_good["ci95_type"],
    }

    # how many NO_ACTIVE_WORK windows does the head ACCEPT (false accepts)?
    fa = int((is_na & (np.asarray(derived, dtype=object) == "accept")).sum())
    # score separation
    sep = {"good_median": round(float(np.median(p26[is_good])), 4),
           "no_active_work_median": round(float(np.median(p26[is_na])), 4),
           "other_bad_median": round(float(np.median(p26[is_other_bad])), 4)}

    cfg = {"id": ID, "margin": MARGIN, "threshold": THRESHOLD,
           "band": "review iff |p-0.5|<0.25; else accept iff p>=0.6 (inert) -> accept iff p>=0.75, reject iff p<=0.25",
           "score_file": "interview/dsh_pack/ood160_heads_old26_vs_new22.json [old26]",
           "labels": "labels/ood160_labels.json", "n_boot": 2000, "n_perm": 20000, "seed": 0}

    payload = {
        "id": ID,
        "status": "done",
        "headline": "AUROC of the frozen old26 score for NO_ACTIVE_WORK (0) vs good (1)",
        "auroc": auc_na_vs_good["auroc"],
        "ci95": auc_na_vs_good["ci95"],
        "ci95_type": auc_na_vs_good["ci95_type"],
        "n": auc_na_vs_good["n"],
        "n_bad": auc_na_vs_good["n_bad"],
        "n_good": auc_na_vs_good["n_good"],
        "n_groups": auc_na_vs_good["n_factories"],
        "label_source": d["label_source"],
        "protocol": "zero-training agreement/association study on frozen scores; verdicts re-derived from the frozen score with the jadal/poc/cascade.py band and checked against the shipped verdict field (0/160 mismatches); factory-grouped bootstrap for every CI",
        "dataset": {"n_windows": n, "n_good": n_good, "n_bad": int((y == 0).sum()),
                    "n_no_active_work": n_na, "n_other_bad": n_other,
                    "n_factories": len(factories), "n_null_scores": n_null,
                    "bad_reasons": {k: int(v) for k, v in
                                    zip(*np.unique(reason[y == 0], return_counts=True))}},
        "band_check": {"derived_vs_shipped_mismatches": len(mism), "n_checked": n},
        "verdict_shares": {"no_active_work": shares_na, "all_other_windows": shares_other,
                           "good_only": shares_good, "all_160": shares_all,
                           "by_bad_reason": by_reason},
        "false_accepts_no_active_work": fa,
        "score_separation": sep,
        "auroc_blocks": {"no_active_work_vs_good": auc_na_vs_good,
                         "flipped_no_active_work_as_positive": auc_flipped,
                         "good_vs_all_bad": auc_all_bad,
                         "good_vs_all_bad_same_rows": auc_na_vs_quiet},
        "per_factory": {
            "n_factories": len(rows),
            "n_factories_with_any_no_active_work": n_fac_with_na,
            "n_factories_with_ge5_windows": n_fac_ge5,
            "spearman_rho": round(rho, 4),
            "pearson_r": round(r_pear, 4),
            "permutation_p_two_sided": round(p_perm, 4),
            "n_perm": 20000, "seed": 0,
            "caveat": ("weak test: 47 factory-level points, 41 of 47 factories have <=5 windows, "
                       "only %d have any NO_ACTIVE_WORK at all, and the factory shares are coarse "
                       "ratios of small counts" % n_fac_with_na),
            "rows": rows,
        },
        "caveat": ("Rates here are descriptive agreement on 160 windows a single human labelled; "
                   "the negative class is small (19 NO_ACTIVE_WORK) and 47 factories with 1-13 "
                   "windows each make factory-grouped intervals wide. No accuracy claim is made "
                   "about unlabelled video."),
        "config": cfg,
        "config_sha256": common.config_sha256(cfg),
        "device": common.DEVICE_CPU,
        "cost": "0 GPU; cached scores only; <5 s CPU",
        "command": "JADAL_BUNDLE=<bundle> /home/parshu/envs/jadal/bin/python tools/task/task00.py",
        "git_sha": common.git_sha(),
        "seed": 0,
        "bundle_used_for_ood160_labels": bundle,
    }
    common.save_results(OUT, payload)
    print(json.dumps({k: payload[k] for k in
                      ("headline", "auroc", "ci95", "n", "n_bad", "n_good", "n_groups")},
                     indent=1))
    print("verdict shares NO_ACTIVE_WORK:", shares_na)
    print("verdict shares all other     :", shares_other)
    print("by bad reason:", json.dumps(by_reason, indent=1))
    print("per-factory rho=%.4f r=%.4f p_perm=%.4f (n=%d factories, %d with any NO_ACTIVE_WORK)"
          % (rho, r_pear, p_perm, len(rows), n_fac_with_na))
    print("good_vs_all_bad AUROC", auc_all_bad["auroc"], auc_all_bad["ci95"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
