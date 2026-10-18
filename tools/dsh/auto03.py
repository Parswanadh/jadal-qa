#!/usr/bin/env python3
"""AUTO.03 - risk-coverage frontier and a conformal-style bound chosen on
training folds only.

Policies
  P1  s1_forced_0.5          score p26, threshold 0.5
  P2  youden_perfold         score p26, threshold = Youden on the 6 training
                             clips' inner-LOCO probs of that fold
  P3  band_plus_second_view  score = p26 outside |p26-0.5|<0.25, else the EG2
                             second view; threshold 0.5
                             (IndEgo only: the cached EG2 window embeddings
                             cover the 150 IndEgo windows and nothing else)

Coverage sweep: at coverage c the c-fraction of windows with the largest
confidence |score - threshold| is decided automatically and the rest go to a
human; risk = error among the AUTO-DECIDED windows.  Intervals resample clips
(IndEgo) or factories (OOD-160) and re-select inside every resample.

Conformal-style bound: for target error alpha in {5 %, 10 %, 20 %}, the largest
coverage whose finite-sample-corrected calibration risk
(n/(n+1)) * risk_hat(lambda) + 1/(n+1) <= alpha
(Bates et al. CRC selection rule) holds.  lambda is chosen from TRAINING FOLDS
ONLY: in-domain per outer fold from the 6 training clips' inner-LOCO probs, and
for OOD-160 from the IndEgo-150 out-of-fold scores (OOD has no training folds of
its own).  The word used in the report is "conformal-style bound under
exchangeability, assumptions listed" - never "certified".

Run:  JADAL_BUNDLE=<bundle> OMP_NUM_THREADS=2 nice -n 10 \
      /home/parshu/envs/jadal/bin/python tools/dsh/auto03.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auto_util as A  # noqa: E402
import common as C  # noqa: E402

CONFIG = {
    "id": "AUTO.03",
    "policies": ["s1_forced_0.5", "youden_perfold", "band_plus_second_view"],
    "band": "|p26 - 0.5| < 0.25",
    "coverage_grid": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    "risk_definition": "misclassification rate among the AUTO-DECIDED windows (both a false accept and a false reject count)",
    "coverage_definition": "share of windows decided automatically; the rest go to a human",
    "crc_rule": "(n/(n+1))*risk_hat(lambda) + 1/(n+1) <= alpha; primary pick = smallest lambda that passes (crc_published); a monotone-envelope pick and an uncorrected pick are reported alongside",
    "targets_alpha": [0.05, 0.10, 0.20],
    "lambda_grid": "101 quantiles of the calibration confidence",
    "n_boot": 2000, "seed": 0,
}
OUT = os.path.join(C.REPO, "eval", "results", "auto", "AUTO.03.json")

ASSUMPTIONS = [
    "Exchangeability: the calibration windows and the windows the bound is applied to are exchangeable. "
    "In-domain the split is at CLIP level (LOCO), so the honest exchangeable unit is the clip, not the window.",
    "One labeller: every label comes from the same human, so label noise is not modelled and no inter-rater "
    "variance is included in any interval.",
    "The calibration scores are inner-LOCO out-of-fold: no calibration window's own clip trained the model "
    "that scored it.",
    "Intra-clip correlation BREAKS window-level exchangeability. The repo already measured this "
    "(docs/FACTS.md): nominal 2.7 % false-reject became 46 % on held-out clips, intra-clip correlation 0.58, "
    "effective n about 11. That measurement is the repo's, not re-measured in this stream.",
    "OOD-160 is a different distribution (an IndEgo-trained head applied to Egocentric-10K). No target "
    "exchangeability holds there, so an IndEgo-selected lambda carries no guarantee on OOD-160 and the "
    "OOD rows are descriptive.",
    "The CRC rule bounds the expected risk of the SELECTED rule under a monotone risk curve; it is a "
    "statement about the calibration draw, not a per-window certificate, and it is not a licence to skip "
    "the audit.",
    "Risk and coverage are defined as in CONFIG: coverage is the share auto-decided, risk the error among "
    "the auto-decided windows.",
]


def _round(o, nd=4):
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return round(f, nd) if np.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


# ------------------------------------------------------------------ sweep ----
def sweep_stats(y, conf, p_used, thr, c):
    """Auto-decide the c-fraction with the largest confidence among `sel`."""
    n = len(y)
    k = max(1, int(round(c * n)))
    sel = np.argsort(-conf, kind="mergesort")[:k]
    ys = y[sel]
    dec = (p_used[sel] >= thr[sel]).astype(int)
    good, bad = ys == 1, ys == 0
    if good.sum() == 0 or bad.sum() == 0:
        return None
    return {
        "n_decided": int(k), "coverage": k / n,
        "bal_acc": float(C.balanced_accuracy(ys, dec)),
        "false_accept_rate": float((dec[bad] == 1).mean()),
        "false_reject_rate": float((dec[good] == 0).mean()),
        "error_rate": float((dec != ys).mean()),
        "accept_share": float((dec == 1).mean()),
    }


def frontier(y, groups, conf, p_used, thr, cov_grid, n_boot, seed):
    rows = []
    for c in cov_grid:
        point = sweep_stats(y, conf, p_used, thr, c)
        if point is None:
            rows.append({"coverage_target": c, "available": False,
                         "why": "one class missing at this coverage"})
            continue
        row = {"coverage_target": c, "available": True, **_round(point)}
        for name in ("bal_acc", "false_accept_rate", "false_reject_rate", "error_rate", "accept_share"):
            r = A.group_boot_stat(
                y, groups,
                lambda d, idx, nm=name, cc=c: (lambda s: None if s is None else s[nm])(
                    sweep_stats(y[idx], conf[idx], p_used[idx], thr[idx], cc)),
                n_boot=n_boot, seed=seed)
            row[name + "_ci95"] = [round(r["lo"], 4), round(r["hi"], 4)]
            row["n_boot_used"] = r["n_boot"]
        rows.append(row)
    return rows


# --------------------------------------------------------------- conformal ----
def crc_picks(conf_cal, p_used_cal, thr_cal, y_cal, alpha):
    """Three lambda picks from CALIBRATION data only, all on a 101-point
    confidence-quantile grid (no test label, no held-out window enters here).

    crc_published          smallest lambda with (n/(n+1))*risk_hat + 1/(n+1) <= alpha
                           -- the published CRC/LTT selection rule; largest
                           calibration coverage that passes the corrected bound.
    crc_monotone_envelope  smallest lambda such that every larger lambda also
                           passes (conservative variant; reported because a
                           finite grid can dip below alpha by luck).
    naive_uncorrected      smallest lambda with risk_hat <= alpha (no
                           finite-sample correction) -- shown for contrast only.
    """
    n = len(y_cal)
    qs = np.linspace(0.0, 1.0, 101)
    lams = np.quantile(conf_cal, qs)
    rows = []
    for q, lam in zip(qs, lams):
        m = conf_cal >= lam
        k = int(m.sum())
        if k == 0:
            continue
        risk = float(np.mean((p_used_cal[m] >= thr_cal[m]).astype(int) != y_cal[m]))
        rows.append({"q": float(q), "lambda": float(lam), "k": k, "coverage_cal": k / n,
                     "risk_hat_cal": risk,
                     "risk_bound": (n / (n + 1.0)) * risk + 1.0 / (n + 1.0)})
    out = {}
    cand = [r for r in rows if r["risk_bound"] <= alpha]
    out["crc_published"] = (min(cand, key=lambda r: r["lambda"]) if cand else None)
    env, ok = None, True
    for r in sorted(rows, key=lambda r: -r["lambda"]):
        if not ok:
            break
        if r["risk_bound"] <= alpha:
            env = r
        else:
            ok = False
    out["crc_monotone_envelope"] = env
    cand2 = [r for r in rows if r["risk_hat_cal"] <= alpha]
    out["naive_uncorrected"] = (min(cand2, key=lambda r: r["lambda"]) if cand2 else None)
    return out


def apply_lambda(conf, p_used, thr, y, lam):
    """Auto-decide the windows whose confidence clears lambda; risk among them."""
    m = conf >= lam
    k = int(m.sum())
    if k == 0:
        return {"n_decided": 0, "coverage": 0.0, "risk": float("nan")}
    dec = (p_used[m] >= thr[m]).astype(int)
    return {"n_decided": k, "coverage": k / len(y), "risk": float(np.mean(dec != y[m]))}


def pooled_ci(per_fold, n_boot=2000, seed=0):
    """Clip bootstrap over ALL folds (a fold where no lambda could be selected
    counts as coverage 0, which is what deployment of this rule would give)."""
    rng = np.random.RandomState(seed)
    if not per_fold:
        return {"coverage_ci95_clip": None, "risk_ci95_clip": None}
    cov, risk = [], []
    for _ in range(n_boot):
        pick = rng.randint(0, len(per_fold), len(per_fold))
        n_dec = sum(per_fold[i]["n_decided"] for i in pick)
        n_tot = sum(per_fold[i]["n"] for i in pick)
        e = sum(per_fold[i]["errors"] for i in pick)
        if n_tot == 0:
            continue
        cov.append(n_dec / n_tot)
        if n_dec:
            risk.append(e / n_dec)
    out = {"coverage_ci95_clip": [round(float(np.percentile(cov, 2.5)), 4),
                                  round(float(np.percentile(cov, 97.5)), 4)],
           "n_boot_coverage": len(cov)}
    out["risk_ci95_clip"] = ([round(float(np.percentile(risk, 2.5)), 4),
                              round(float(np.percentile(risk, 97.5)), 4)] if risk else None)
    out["n_boot_risk"] = len(risk)
    return out


def main():
    n_boot, seed = CONFIG["n_boot"], CONFIG["seed"]
    g = C.load_gold()
    X, y, clip = g["X"], g["y"], g["clip"]
    p26 = C.loco_proba(X, y, clip, C=0.3)

    z = np.load(os.path.join(C.BUNDLE, "caches", "frames10k", "segments.npz"), allow_pickle=True)
    F, cnt = z["F"], np.asarray(z["counts"], int)
    Xe = np.stack([np.asarray(F[i][:int(cnt[i])], np.float32).mean(0) for i in range(len(y))]).astype(np.float64)
    p_eg2 = C.loco_proba(Xe, y, clip, C=0.3)
    band = np.abs(p26 - 0.5) < 0.25
    p_used_p3 = np.where(band, p_eg2, p26)

    # per-fold Youden threshold on the training clips (training folds only)
    thr_youden, folds_youden = A.loco_fold_thresholds(X, y, clip, kind="youden", C=0.3)

    policies = {
        "s1_forced_0.5": dict(p_used=p26, thr=np.full(len(y), 0.5)),
        "youden_perfold": dict(p_used=p26, thr=thr_youden),
        "band_plus_second_view": dict(p_used=p_used_p3, thr=np.full(len(y), 0.5)),
    }
    conf = {k: np.abs(v["p_used"] - v["thr"]) for k, v in policies.items()}

    # ---------------- IndEgo frontier + conformal bound ----------------
    indomain_frontier = {}
    for name, v in policies.items():
        indomain_frontier[name] = frontier(y, clip, conf[name], v["p_used"], v["thr"],
                                           CONFIG["coverage_grid"], n_boot, seed)

    # inner-LOCO scores of the TRAINING clips of every outer fold (for lambda)
    inner = {}
    for gg in sorted(set(clip)):
        tr = clip != gg
        inner[gg] = {
            "tr": tr, "y_tr": y[tr], "clip_tr": clip[tr],
            "p26": C.loco_proba(X[tr], y[tr], clip[tr], C=0.3),
            "p_eg2": C.loco_proba(Xe[tr], y[tr], clip[tr], C=0.3),
        }
        inner[gg]["t_youden"], _ = C.youden_threshold(y[tr], inner[gg]["p26"])

    conformal = {"indomain": {}, "ood160_no_training_folds": {}, "ood160_factory_split": {}}
    VARIANTS = ("crc_published", "crc_monotone_envelope", "naive_uncorrected")
    for alpha in CONFIG["targets_alpha"]:
        key = f"{alpha:.2f}"
        folds = []
        for gg in sorted(set(clip)):
            d = inner[gg]
            band_in = np.abs(d["p26"] - 0.5) < 0.25
            te = clip == gg
            band_te = np.abs(p26[te] - 0.5) < 0.25
            per_policy = {
                "s1_forced_0.5": (
                    np.abs(d["p26"] - 0.5), d["p26"], np.full(len(d["y_tr"]), 0.5),
                    conf["s1_forced_0.5"][te], p26[te], np.full(int(te.sum()), 0.5)),
                "youden_perfold": (
                    np.abs(d["p26"] - d["t_youden"]), d["p26"], np.full(len(d["y_tr"]), d["t_youden"]),
                    np.abs(p26[te] - d["t_youden"]), p26[te], np.full(int(te.sum()), d["t_youden"])),
                "band_plus_second_view": (
                    np.abs(np.where(band_in, d["p_eg2"], d["p26"]) - 0.5),
                    np.where(band_in, d["p_eg2"], d["p26"]), np.full(len(d["y_tr"]), 0.5),
                    np.abs(np.where(band_te, p_eg2[te], p26[te]) - 0.5),
                    np.where(band_te, p_eg2[te], p26[te]), np.full(int(te.sum()), 0.5)),
            }
            for pol in CONFIG["policies"]:
                cc, pu, tt, ct, put, ttt = per_policy[pol]
                picks = crc_picks(cc, pu, tt, d["y_tr"], alpha)
                for var, pick in picks.items():
                    if pick is None:
                        folds.append({"policy": pol, "variant": var, "clip": str(gg),
                                      "feasible": False, "n": int(te.sum()),
                                      "n_decided": 0, "errors": 0})
                        continue
                    ap = apply_lambda(ct, put, ttt, y[te], pick["lambda"])
                    ap["errors"] = int(round(ap["risk"] * ap["n_decided"])) if ap["n_decided"] else 0
                    folds.append({"policy": pol, "variant": var, "clip": str(gg), "feasible": True,
                                  "n": int(te.sum()), "n_decided": ap["n_decided"],
                                  "coverage_test": ap["coverage"], "risk_test": ap["risk"],
                                  "errors": ap["errors"], "lambda": pick["lambda"],
                                  "coverage_cal": pick["coverage_cal"], "risk_hat_cal": pick["risk_hat_cal"],
                                  "risk_bound": pick["risk_bound"], "q": pick["q"]})
        for pol in CONFIG["policies"]:
            for var in VARIANTS:
                allf = [f for f in folds if f["policy"] == pol and f["variant"] == var]
                pf = [f for f in allf if f["feasible"]]
                slot = conformal["indomain"].setdefault(key, {}).setdefault(pol, {})
                if not pf:
                    slot[var] = {"feasible": False, "n_folds": len(allf), "n_folds_feasible": 0,
                                 "why": "no training fold could select a lambda at this target"}
                    continue
                n_dec = sum(f["n_decided"] for f in pf)
                n_tot = sum(f["n"] for f in pf)
                n_all = sum(f["n"] for f in allf)
                errs = sum(f["errors"] for f in pf)
                slot[var] = {
                    "feasible": True, "n_folds": len(allf), "n_folds_feasible": len(pf),
                    "coverage_achieved_all_folds": n_dec / n_all,
                    "coverage_achieved_feasible_folds": n_dec / n_tot,
                    "risk_achieved_pooled": errs / n_dec if n_dec else None,
                    "n_decided": n_dec, "n_windows": n_all,
                    **pooled_ci(allf, n_boot=n_boot, seed=seed),
                    "per_fold": allf,
                }

    # lambda from all 150 IndEgo OOF windows, applied to OOD-160 (P1 and P2 only)
    thr_indego_pooled, _ = C.youden_threshold(y, p26)
    per_policy_all = {
        "s1_forced_0.5": (np.abs(p26 - 0.5), p26, np.full(len(y), 0.5)),
        "youden_perfold": (np.abs(p26 - thr_indego_pooled), p26, np.full(len(y), thr_indego_pooled)),
    }
    o = C.load_ood160()
    oy, ofac, op = o["y"], o["factory"], o["p26"]
    for alpha in CONFIG["targets_alpha"]:
        key = f"{alpha:.2f}"
        conformal["ood160_no_training_folds"].setdefault(key, {})
        for pol, (cc, pu, tt) in per_policy_all.items():
            tfix = thr_indego_pooled if pol == "youden_perfold" else 0.5
            picks = crc_picks(cc, pu, tt, y, alpha)
            for var, pick in picks.items():
                if pick is None:
                    conformal["ood160_no_training_folds"][key].setdefault(pol, {})[var] = {
                        "feasible": False, "why": "IndEgo-150 OOF cannot select a lambda at this target"}
                    continue
                m = np.abs(op - tfix) >= pick["lambda"]
                ap = apply_lambda(np.abs(op - tfix), op, np.full(len(oy), tfix), oy, pick["lambda"])
                r = A.group_boot_stat(oy, ofac, lambda d, idx, mm=m: float(mm[idx].mean()),
                                      n_boot=n_boot, seed=seed)
                rr = A.group_boot_stat(
                    oy, ofac,
                    lambda d, idx, mm=m, pp=op, yy=oy, tf=tfix:
                        float(np.mean(((pp[idx][mm[idx]] >= tf).astype(int)) != yy[idx][mm[idx]]))
                        if mm[idx].sum() else float("nan"),
                    n_boot=n_boot, seed=seed)
                conformal["ood160_no_training_folds"][key].setdefault(pol, {})[var] = {
                    "feasible": True, "lambda_from_indego": pick["lambda"],
                    "coverage_cal_indego": pick["coverage_cal"],
                    "risk_hat_cal_indego": pick["risk_hat_cal"],
                    "risk_bound_indego": pick["risk_bound"],
                    "coverage_achieved_ood160": ap["coverage"],
                    "coverage_achieved_ood160_ci95_factory": [round(r["lo"], 4), round(r["hi"], 4)],
                    "risk_achieved_ood160": ap["risk"],
                    "risk_achieved_ood160_ci95_factory": [round(rr["lo"], 4), round(rr["hi"], 4)],
                    "n_decided_ood160": ap["n_decided"],
                    "warning": ("no exchangeability between IndEgo and OOD-160: descriptive only, "
                                "not a bound on OOD performance"),
                }

    # exchangeable-ish construction inside OOD-160: factory-level split
    facs = sorted(set(ofac))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(facs))
    half = len(facs) // 2
    fa = set(np.asarray(facs, dtype=object)[perm[:half]])
    fb = set(np.asarray(facs, dtype=object)[perm[half:]])
    for alpha in CONFIG["targets_alpha"]:
        key = f"{alpha:.2f}"
        conformal["ood160_factory_split"].setdefault(key, {})
        for name, cal_f, ev_f in (("A_cal_B_eval", fa, fb), ("B_cal_A_eval", fb, fa)):
            cal = np.array([f in cal_f for f in ofac])
            ev = ~cal
            for pol, tfix in (("s1_forced_0.5", 0.5), ("youden_perfold", thr_indego_pooled)):
                picks = crc_picks(np.abs(op[cal] - tfix), op[cal],
                                  np.full(int(cal.sum()), tfix), oy[cal], alpha)
                for var, pick in picks.items():
                    slot = f"{name}:{pol}:{var}"
                    if pick is None:
                        conformal["ood160_factory_split"][key][slot] = {
                            "feasible": False, "n_cal": int(cal.sum()),
                            "n_cal_bad": int((oy[cal] == 0).sum()),
                            "why": "no lambda on the calibration factories passes at this target"}
                        continue
                    m = np.abs(op[ev] - tfix) >= pick["lambda"]
                    ap = apply_lambda(np.abs(op[ev] - tfix), op[ev], np.full(int(ev.sum()), tfix),
                                      oy[ev], pick["lambda"])
                    r = A.group_boot_stat(oy[ev], ofac[ev], lambda d, idx, mm=m: float(mm[idx].mean()),
                                          n_boot=n_boot, seed=seed)
                    conformal["ood160_factory_split"][key][slot] = {
                        "feasible": True, "n_cal": int(cal.sum()), "n_cal_bad": int((oy[cal] == 0).sum()),
                        "n_eval": int(ev.sum()), "n_eval_bad": int((oy[ev] == 0).sum()),
                        "lambda": pick["lambda"], "coverage_cal": pick["coverage_cal"],
                        "risk_hat_cal": pick["risk_hat_cal"], "risk_bound": pick["risk_bound"],
                        "coverage_achieved_eval": ap["coverage"],
                        "coverage_achieved_eval_ci95_factory": [round(r["lo"], 4), round(r["hi"], 4)],
                        "risk_achieved_eval": ap["risk"], "n_decided_eval": ap["n_decided"],
                    }

    # ---------------- OOD frontier (P3 unavailable: no second view for OOD) ----
    ood_frontier = {}
    for pol, tfix in (("s1_forced_0.5", 0.5), ("youden_perfold", thr_indego_pooled)):
        ood_frontier[pol] = frontier(oy, ofac, np.abs(op - tfix), op, np.full(len(oy), tfix),
                                     CONFIG["coverage_grid"], n_boot, seed)
    ood_frontier["band_plus_second_view"] = {
        "available": False,
        "why": ("the cached EmbeddingGemma-2 window embeddings cover the 150 IndEgo windows only; "
                "the OOD-160 windows have no cached second-view embedding and this stream did not run "
                "a model to make one"),
    }

    payload = {
        "id": "AUTO.03", "status": "done",
        "n": int(len(y)), "n_bad": int((y == 0).sum()), "n_good": int((y == 1).sum()),
        "n_ood": int(len(oy)), "n_bad_ood": int((oy == 0).sum()),
        "n_groups": len(set(clip)), "groups": "7 IndEgo clips / 47 OOD factories",
        "label_source": g["label_source"] + " | " + o["label_source"],
        "protocol": ("coverage sweep over a confidence ranking with clip-grouped (IndEgo) / factory-grouped "
                     "(OOD-160) bootstrap that re-selects inside every resample; conformal-style lambda "
                     "chosen on training folds only (per outer fold for IndEgo, IndEgo-150 for OOD)"),
        "ci95": [indomain_frontier["s1_forced_0.5"][0]["bal_acc_ci95"][0],
                 indomain_frontier["s1_forced_0.5"][0]["bal_acc_ci95"][1]],
        "ci95_type": A.CI_TYPE_CLIP + " on the 100 %-coverage balanced accuracy of P1",
        "device": C.DEVICE_CPU,
        "cost": ("0 GPU; cached scores and embeddings; ~25 s CPU (7 outer x 2 views + 42 inner x 2 views "
                 "logistic fits, sweep + CRC bootstraps)"),
        "command": ("JADAL_BUNDLE=<bundle> OMP_NUM_THREADS=2 nice -n 10 "
                    "/home/parshu/envs/jadal/bin/python tools/dsh/auto03.py"),
        "git_sha": C.git_sha(), "seed": seed,
        "config_sha256": C.config_sha256(CONFIG), "config": CONFIG,
        "bound_wording": ("conformal-style bound under exchangeability, assumptions listed"),
        "assumptions": ASSUMPTIONS,
        "exchangeability_prior_art": {
            "source": "docs/FACTS.md (repo, on main)",
            "measured": "nominal 2.7 % false-reject became 46 % on held-out clips; intra-clip correlation 0.58; effective n about 11",
            "status": "quoted from the repo's own record, not re-measured in this stream",
        },
        "youden_threshold_indego_pooled": round(thr_indego_pooled, 4),
        "indomain_frontier": indomain_frontier,
        "ood160_frontier": ood_frontier,
        "conformal_style_bound": conformal,
        "ood_caveat": A.decision_note_ood(),
        "notes": [
            "Risk at coverage c is measured on the auto-decided windows only; the reviewed windows are not counted as errors.",
            "Confidence for P2 uses each window's own fold threshold, so the confidence is comparable across folds.",
            "The frontier bootstrap re-selects the top-c fraction inside every resample, so the interval covers the selection as well as the metric.",
            "All three policies are FORCED-decision policies (accept iff score >= threshold); the shipped "
            "cascade instead accepts outside the band at p >= 0.75. The forced variant is used so that the "
            "100 %-coverage numbers are comparable across policies and with AUTO.00.",
        ],
    }
    C.save_results(OUT, payload)

    def brief(fr):
        return [{k: r.get(k) for k in ("coverage_target", "n_decided", "bal_acc", "false_accept_rate",
                                       "false_reject_rate", "error_rate")} for r in fr if r.get("available")]
    print(json.dumps({
        "indomain": {k: brief(v) for k, v in indomain_frontier.items()},
        "ood": {k: (brief(v) if isinstance(v, list) else v) for k, v in ood_frontier.items()},
        "conformal_indomain": {a: {p: {kk: vv for kk, vv in d.items() if kk != "per_fold"}
                                   for p, d in pols.items()}
                               for a, pols in conformal["indomain"].items()},
        "conformal_ood_transfer": conformal["ood160_no_training_folds"],
        "conformal_ood_factory_split": {a: {k: {kk: vv for kk, vv in v.items()}
                                            for k, v in pols.items()}
                                        for a, pols in conformal["ood160_factory_split"].items()},
    }, indent=1, default=str))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
