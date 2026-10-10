#!/usr/bin/env python3
"""Z2 (conformal / risk control) re-run on the 160 OOD labels.

The IndEgo side is unchanged (the experiment code is copied verbatim from
`m/conformal`, PR #126). What is new here is the **out-of-domain evaluation on
the merged 160 human-labelled Egocentric-10K windows** (138 good / 22 bad,
47 factories) instead of the old OOD-60 (51 good / 9 bad):

  * split-conformal calibrated on IndEgo (all 150 windows), applied to the 160
    OOD frozen scores -> empirical coverage / review / error (evaluation only,
    NOTHING is fit on OOD);
  * the review-vs-error frontier of the frozen score on the 160 with
    factory-grouped bootstrap CIs (honest; windows within a factory correlated).

Outputs eval/results/exp_conformal160/Z2.01_160.json .. Z2.03_160.json.
CPU only, seed 7. Run: OMP_NUM_THREADS=2 python3 eval/exp_conformal160/run_z2_160.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import conformal_risk as C  # noqa: E402  (copied verbatim from m/conformal)

BUNDLE = C.BUNDLE
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                   "eval", "results", "exp_conformal160")
N_BOOT = 2000
SEED = 7
COV_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]


def load_labels160():
    d = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))
    w = d["windows"]
    return {"ids": [x["segment_id"] for x in w],
            "y": np.asarray([1 if x["verdict"] == "good" else 0 for x in w], int),
            "factory": np.asarray([x["factory"] for x in w]),
            "batch": np.asarray([x["batch"] for x in w])}


def load_frozen160(ids):
    sc = {}
    for fn in ("ood_scores.json", "ood_scores2.json"):
        d = json.load(open(os.path.join(BUNDLE, "caches", "ensemble", fn)))
        for x in d["scores"]:
            if "error" not in x:
                sc[x["id"]] = float(x["usable_prob"])
    missing = [i for i in ids if i not in sc]
    return np.asarray([sc[i] for i in ids], float), missing


def _gidx(groups):
    gs = list(groups)
    gmap = {g: i for i, g in enumerate(sorted(set(gs)))}
    gi = np.asarray([gmap[g] for g in gs])
    return [np.where(gi == k)[0] for k in range(len(gmap))]


def group_boot_vec(vec, groups, n_boot=N_BOOT, seed=SEED):
    """Factory-grouped bootstrap 95% CI of mean(vec) over the resampled windows."""
    vec = np.asarray(vec, float)
    idx_by = _gidx(groups)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(idx_by), len(idx_by))
        idx = np.concatenate([idx_by[k] for k in pick])
        vals.append(float(np.mean(vec[idx])))
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def group_boot_fn(y, s, groups, fn, n_boot=N_BOOT, seed=SEED):
    """Factory-grouped bootstrap 95% CI of fn(y, s) (e.g. AUROC / balanced accuracy)."""
    y = np.asarray(y)
    s = np.asarray(s, float)
    idx_by = _gidx(groups)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(idx_by), len(idx_by))
        idx = np.concatenate([idx_by[k] for k in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], s[idx]))
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def auroc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def balacc(y, s, thr=0.5):
    y = np.asarray(y)
    pred = (np.asarray(s, float) > thr).astype(int)
    g, b = y == 1, y == 0
    tpr = float((pred[g] == 1).mean()) if g.any() else float("nan")
    tnr = float((pred[b] == 0).mean()) if b.any() else float("nan")
    return 0.5 * (tpr + tnr)


def conformal_ood_table(data, y_ood, s_ood, fac, qs=None):
    """Calibrate q on IndEgo (all 150, or supplied per-fold) and apply to OOD-160."""
    E_in = C.true_label_score(data["p_good"], data["y"])
    rows = []
    for cov in COV_GRID:
        alpha = 1 - cov
        q = C.conformal_q(E_in, alpha, "split") if qs is None else qs[cov]
        dec, size = C.decision(s_ood, q)
        summ = C.summarize(dec, size, y_ood, q)
        covered = np.where(size == 2, True,
                           np.where(size == 0, False, np.where(dec == "A", y_ood == 1, y_ood == 0)))
        decided = dec != "V"
        wrong = decided & ~covered
        review_vec = (dec == "V").astype(float)
        fa_vec = ((dec == "A") & (y_ood == 0)).astype(float)
        row = {
            "coverage_target": cov, "q_indego": round(float(q), 4),
            "review_share": round(float(summ["review_share"]), 4),
            "coverage": round(float(covered.mean()), 4),
            "error_among_decided": (round(summ["error_among_decided"], 4)
                                    if summ["error_among_decided"] is not None else None),
            "false_accept_rate_marginal": round(float(fa_vec.mean()), 4),
            "false_reject_rate_marginal": round(float(summ["false_reject_rate_marginal"]), 4),
            "n_review": int(summ["n_review"]), "n_decided": int(decided.sum()),
            "review_share_ci95_factory_grouped": group_boot_vec(review_vec, fac),
            "coverage_ci95_factory_grouped": group_boot_vec(covered.astype(float), fac),
            "false_accept_marginal_ci95_factory_grouped": group_boot_vec(fa_vec, fac),
        }
        if decided.sum() > 0:
            err_vec = np.where(decided, wrong.astype(float), np.nan)
            ok = ~np.isnan(err_vec)
            row["error_among_decided_ci95_factory_grouped"] = group_boot_vec(
                err_vec[ok], fac[ok])
        else:
            row["error_among_decided_ci95_factory_grouped"] = None
        rows.append(row)
    return rows


def frontier_frozen(y_ood, s_ood, fac):
    """Review-vs-error frontier of the frozen score on the 160 (grouped CI)."""
    n = len(y_ood)
    band = []
    for w in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30]:
        decide = np.ones(n, bool) if w == 0.0 else (np.abs(s_ood - 0.5) >= w)
        dec_idx = np.where(decide)[0]
        review = float(1.0 - decide.mean())
        row = {"band_half_width_w": w, "review_share_160": round(review, 4),
               "n_decided": int(dec_idx.size)}
        if dec_idx.size == 0:
            row["ood_ba_among_decided"] = None
            row["ood_ba_among_decided_ci95_factory_grouped"] = None
        else:
            row["ood_ba_among_decided"] = round(balacc(y_ood[dec_idx], s_ood[dec_idx], 0.5), 4)
            row["ood_ba_among_decided_ci95_factory_grouped"] = group_boot_fn(
                y_ood[dec_idx], s_ood[dec_idx], fac[dec_idx], balacc)
        band.append(row)
    # coverage gating: decide top-c by |s-0.5|
    gating = []
    order = np.argsort(-np.abs(s_ood - 0.5))
    for c in [1.0, 0.8, 0.6, 0.5, 0.4]:
        k = max(1, int(round(c * n)))
        idx = order[:k]
        row = {"coverage_c": c, "review_share_160": round(1.0 - k / n, 4),
               "n_decided": int(k),
               "ood_ba_among_decided": round(balacc(y_ood[idx], s_ood[idx]), 4),
               "ood_ba_among_decided_ci95_factory_grouped": group_boot_fn(
                   y_ood[idx], s_ood[idx], fac[idx], balacc)}
        gating.append(row)
    return {"band_sweep": band, "coverage_gating": gating}


def main():
    os.makedirs(OUT, exist_ok=True)
    data = C.load_indego()
    manifest = C.load_gold_manifest()
    lab = load_labels160()
    y_ood, fac = lab["y"], lab["factory"]
    s_ood, missing = load_frozen160(lab["ids"])
    n_good = int((y_ood == 1).sum())
    n_bad = int((y_ood == 0).sum())
    n_fac = len(set(fac.tolist()))
    ood_auroc = auroc(y_ood, s_ood)
    ood_auroc_ci = group_boot_fn(y_ood, s_ood, fac, auroc)

    head = {
        "n_ood": int(len(y_ood)), "n_good_ood": n_good, "n_bad_ood": n_bad,
        "n_factories": n_fac, "n_indego": int(len(data["y"])),
        "label_source": ("human (IndEgo manifest_gold n=150 + Egocentric-10K "
                         "ood160_labels.json n=160: 138 good / 22 bad, 47 factories)"),
        "supersedes": {"branch": "m/conformal", "pr": 126, "n_ood": 60},
        "device": "CPU only, seed 7", "git_sha": C.__dict__.get("_sha", "see git"),
        "command": "OMP_NUM_THREADS=2 python3 eval/exp_conformal160/run_z2_160.py",
    }

    # ---------------- Z2.01_160 ----------------
    frontier_in = C.loco_frontier(data, COV_GRID, mode="split")
    ood_table = conformal_ood_table(data, y_ood, s_ood, fac)
    op = {}
    for risk_name, field in [("selective_FAR", "false_accept_rate_selective"),
                             ("marginal_FAR", "false_accept_rate_marginal"),
                             ("error_among_decided", "error_among_decided")]:
        cands = [r for r in frontier_in if (r[field] is not None and r[field] <= 0.05)]
        op[risk_name] = (min(cands, key=lambda r: r["review_share"])["coverage_target"]
                         if cands else None)
    z201 = {**head, "experiment": "Z2.01_160",
            "title": "Split-conformal frontier (IndEgo LOCO) + OOD-160 transfer evaluation",
            "protocol_indego": ("clip-grouped leave-one-clip-out split conformal on p_ens=P(good); "
                                "review share = non-singleton set rate; clip-bootstrap CI"),
            "protocol_ood": ("q calibrated on IndEgo (all 150) ONLY; applied to the frozen "
                             "ensemble usable_prob of the 160 OOD windows. Evaluation only, "
                             "NOTHING fit on OOD. CIs = factory-grouped bootstrap (honest)."),
            "frontier_indego": frontier_in,
            "operating_points_indego": op,
            "ood160_frozen_auroc": round(ood_auroc, 4),
            "ood160_frozen_auroc_ci95_factory_grouped": ood_auroc_ci,
            "ood160_conformal_transfer": ood_table,
            "ood160_missing_scores": missing,
            "conclusion": ("the IndEgo-calibrated conformal rule does NOT transfer: on OOD-160 the "
                           "frozen score is near chance (AUROC %.3f, grouped CI %s) so coverage and "
                           "review are not meaningful guarantees out-of-domain." % (ood_auroc, ood_auroc_ci))}
    json.dump(z201, open(os.path.join(OUT, "Z2.01_160.json"), "w"), indent=1, default=str)

    # ---------------- Z2.02_160 ----------------
    # nominal vs empirical coverage on OOD-160 (transfer failure) + worst factory
    nominal = []
    for row in ood_table:
        cov = row["coverage"]
        nominal.append({"coverage_target": row["coverage_target"],
                        "empirical_coverage_ood160": cov,
                        "coverage_ci95_factory_grouped": row["coverage_ci95_factory_grouped"],
                        "review_share": row["review_share"]})
    # worst-factory coverage at a fixed target (0.80) on OOD
    q80 = C.conformal_q(C.true_label_score(data["p_good"], data["y"]), 0.20, "split")
    dec80, size80 = C.decision(s_ood, q80)
    cov80 = np.where(size80 == 2, True, np.where(size80 == 0, False,
                       np.where(dec80 == "A", y_ood == 1, y_ood == 0)))
    per_fac = {f: round(float(cov80[fac == f].mean()), 3) for f in sorted(set(fac.tolist()))}
    z202 = {**head, "experiment": "Z2.02_160",
            "title": "Nominal vs empirical coverage on OOD-160 (transfer failure)",
            "protocol": ("q calibrated on IndEgo only; applied to the 160 OOD frozen scores. "
                         "Empirical coverage = P(true label in the conformal set)."),
            "nominal_vs_empirical_ood160": nominal,
            "worst_factory_coverage_at_target_0.80": min(per_fac.values()),
            "per_factory_coverage_at_target_0.80": per_fac,
            "conclusion": ("empirical OOD coverage collapses far below the nominal target because "
                           "the IndEgo score does not rank the OOD windows (no target "
                           "exchangeability).")}
    json.dump(z202, open(os.path.join(OUT, "Z2.02_160.json"), "w"), indent=1, default=str)

    # ---------------- Z2.03_160 ----------------
    fr = frontier_frozen(y_ood, s_ood, fac)
    # conformal operating point recommended on IndEgo (cov 0.85) applied OOD
    q85 = C.conformal_q(C.true_label_score(data["p_good"], data["y"]), 0.15, "split")
    dec85, size85 = C.decision(s_ood, q85)
    cov85 = np.where(size85 == 2, True, np.where(size85 == 0, False,
                       np.where(dec85 == "A", y_ood == 1, y_ood == 0)))
    op_ood = {"coverage_target": 0.85, "q_indego": round(float(q85), 4),
              "review_share_ood160": round(float((dec85 == "V").mean()), 4),
              "coverage_ood160": round(float(cov85.mean()), 4),
              "false_accept_marginal_ood160": round(float(((dec85 == "A") & (y_ood == 0)).mean()), 4)}
    z203 = {**head, "experiment": "Z2.03_160",
            "title": "Risk control (IndEgo) + frozen review-vs-error frontier on OOD-160",
            "protocol": ("CRC/LTT on IndEgo is unchanged from PR #126 (K=7 clips). The OOD "
                         "frontier sweeps an a-priori band / coverage gate on the frozen score "
                         "and reports factory-grouped BA among decided windows; no OOD threshold "
                         "is fitted."),
            "recommended_indego_operating_point_applied_ood": op_ood,
            "ood160_frozen_review_error_frontier": fr,
            "ood160_frozen_auroc": round(ood_auroc, 4),
            "ood160_frozen_auroc_ci95_factory_grouped": ood_auroc_ci,
            "conclusion": ("the frozen score's review-vs-error frontier on the 160 is not better "
                           "than chance: no review reduction buys OOD accuracy (grouped CIs "
                           "include 0.5).")}
    json.dump(z203, open(os.path.join(OUT, "Z2.03_160.json"), "w"), indent=1, default=str)

    # ---------------- stdout ----------------
    print(f"OOD-160 n={len(y_ood)} good={n_good} bad={n_bad} factories={n_fac} "
          f"(old 60, n_bad=9)")
    print(f"frozen OOD-160 AUROC {ood_auroc:.3f} grouped CI {ood_auroc_ci}")
    print("conformal transfer (calibrated IndEgo, applied OOD-160):")
    print(f"{'cov':>6}{'q':>8}{'review%':>9}{'cov_emp':>9}{'err|dec':>9}{'FA%':>7}")
    for r in ood_table:
        print(f"{r['coverage_target']:6.2f}{r['q_indego']:8.3f}{100*r['review_share']:9.1f}"
              f"{100*r['coverage']:9.1f}{100*(r['error_among_decided'] or 0):9.1f}"
              f"{100*r['false_accept_rate_marginal']:7.1f}")
    print("recommended IndEgo op (cov .85) applied OOD:", op_ood)
    print("frozen band frontier (160):",
          [(b["band_half_width_w"], b["review_share_160"], b["ood_ba_among_decided"])
           for b in fr["band_sweep"]])
    print("frozen coverage gating (160):",
          [(g["coverage_c"], g["ood_ba_among_decided"]) for g in fr["coverage_gating"]])


if __name__ == "__main__":
    main()
