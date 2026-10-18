#!/usr/bin/env python3
"""AUTO.00 - forced-decision baseline: the full-coverage error with the review
band removed.

The shipped funnel sends |p - 0.5| < 0.25 to a human.  This experiment replaces
that band with a forced decision and measures the error at 100 % coverage:

  in-domain  IndEgo-150, 26 cached per-pixel features, leave-one-clip-out.
             (a) accept iff p >= 0.5
             (b) Youden threshold chosen on the TRAINING clips only, per fold
                 (inner LOCO inside the 6 training clips; the held-out clip's
                 labels never enter the threshold)
  OOD-160    the FROZEN shipped-head scores (load_ood160()["p26"]), i.e. the
             original portal's 26-number head on the 480p window clips.
             Factory-grouped CIs; the window bootstrap is shown marked
             optimistic.
  round-3    read-only.  No per-window score file exists for the 26-number
             single head without re-running a model, so the single-head arm is
             SKIPPED and stated as such.  The C1 PR's committed per-window
             scores of the frozen two-view shipped ensemble are reported as a
             clearly labelled DIFFERENT-MODEL arm (nothing was scored here).

Every OOD number carries the caveat string (n=160, n_bad=22, 19/22 bad are
NO_ACTIVE_WORK) and the reason-stratified AUROC for NO_ACTIVE_WORK only.

Run:  JADAL_BUNDLE=<bundle> OMP_NUM_THREADS=2 nice -n 10 \
      /home/parshu/envs/jadal/bin/python tools/dsh/auto00.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auto_util as A  # noqa: E402
import common as C  # noqa: E402

CONFIG = {
    "id": "AUTO.00",
    "head": "StandardScaler+LogisticRegression(C=0.3,class_weight=balanced)",
    "features": "26 per-pixel, caches/gpu/features.json",
    "in_domain_protocol": "leave-one-clip-out over 7 IndEgo clips",
    "policy_a": "accept iff p >= 0.5 (forced, coverage 1.0)",
    "policy_b": "per-fold Youden threshold from the 6 training clips' inner-LOCO probs",
    "band": "|p-0.5| < 0.25 (the shipped review band, reported as a share only)",
    "ood_scores": "load_ood160()['p26'] frozen shipped-head scores (original portal)",
    "n_boot": 2000, "seed": 0,
}
OUT = os.path.join(C.REPO, "eval", "results", "auto", "AUTO.00.json")
C1_INPUTS = os.path.join(C.REPO, "eval", "results", "auto", "inputs")
C1_COMMIT = "origin/m/confirm"  # read-only source of the consumed round-3 scores


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


def read_c1_round3():
    """Read the C1 PR's committed per-window 108-window scores.  NOTHING is
    re-scored: this only parses two files out of the branch (labels + scores)."""
    try:
        raw_scores = subprocess.run(
            ["git", "show", f"{C1_COMMIT}:eval/results/exp_confirm/inputs/confirm108_ensemble.json"],
            capture_output=True, text=True, cwd=C.REPO, timeout=60, check=True).stdout
        raw_windows = subprocess.run(
            ["git", "show", f"{C1_COMMIT}:eval/results/exp_confirm/inputs/windows108.json"],
            capture_output=True, text=True, cwd=C.REPO, timeout=60, check=True).stdout
    except Exception as e:  # pragma: no cover - environment dependent
        return {"available": False, "why": f"{type(e).__name__}: {e}"}
    scores = {r["id"]: r for r in json.loads(raw_scores)["scores"]}
    windows = json.loads(raw_windows)
    ids, y, fac, reason, p_ens, p_eg2, p_px = [], [], [], [], [], [], []
    for w in windows:
        r = scores.get(w["id"])
        if r is None:
            continue
        ids.append(w["id"])
        y.append(1 if w["verdict"] == "good" else 0)
        fac.append(w["factory"])
        reason.append(w.get("reason") or "")
        p_ens.append(r["usable_prob"])
        p_eg2.append(r["p_eg2"])
        p_px.append(r["p_pixel"])
    return {"available": True, "ids": ids, "y": np.asarray(y), "factory": np.asarray(fac),
            "reason": np.asarray(reason, dtype=object), "usable_prob": np.asarray(p_ens),
            "p_eg2": np.asarray(p_eg2), "p_pixel": np.asarray(p_px),
            "labels_sha256_source": "git show %s:eval/results/exp_confirm/inputs/windows108.json" % C1_COMMIT}


def ci_row(y, groups, pred, n_boot, seed):
    return A.rates_ci(y, groups, pred, n_boot=n_boot, seed=seed)


def main():
    n_boot, seed = CONFIG["n_boot"], CONFIG["seed"]
    g = C.load_gold()
    X, y, clip = g["X"], g["y"], g["clip"]
    p = C.loco_proba(X, y, clip, C=0.3)                     # LOCO out-of-fold

    # ---------------- (a) forced 0.5 ----------------
    pred_a = (p >= 0.5).astype(int)
    rates_a = A.policy_rates(y, pred_a)
    ci_a = ci_row(y, clip, pred_a, n_boot, seed)

    # ---------------- (b) per-fold Youden on training clips only ----------------
    thr_b, folds_b = A.loco_fold_thresholds(X, y, clip, kind="youden", C=0.3)
    pred_b = (p >= thr_b).astype(int)
    rates_b = A.policy_rates(y, pred_b)
    ci_b = ci_row(y, clip, pred_b, n_boot, seed)

    # contrast only: pooled in-sample threshold (optimistic, uses the eval labels)
    t_oracle, _ = C.youden_threshold(y, p)
    pred_c = (p >= t_oracle).astype(int)
    rates_c = A.policy_rates(y, pred_c)

    # shipped band, as a share only (it is not a forced-decision policy)
    band = np.abs(p - 0.5) < 0.25
    thr_b_share = float(((p >= 0.5).astype(int)[band] == 1).mean())

    # ---------------- OOD-160: frozen shipped-head scores ----------------
    o = C.load_ood160()
    oy, ofac, oreason, op = o["y"], o["factory"], o["reason"], o["p26"]
    assert not np.isnan(op).any(), "frozen OOD-160 scores contain NaN"
    ood_auroc = C.fast_auc(oy, op)
    ood_ci_fac = C.group_boot_ci(oy, op, ofac, n_boot=n_boot, seed=seed)
    ood_ci_win = C.window_boot_ci(oy, op, n_boot=n_boot, seed=seed)
    # reason-stratified: NO_ACTIVE_WORK only (good windows + NAW bad windows)
    sem = oy == 1
    naw = (oy == 0) & (oreason == "NO_ACTIVE_WORK")
    sub = sem | naw
    auroc_naw = C.fast_auc(oy[sub], op[sub])
    ci_naw_fac = C.group_boot_ci(oy[sub], op[sub], ofac[sub], n_boot=n_boot, seed=seed)
    ci_naw_win = C.window_boot_ci(oy[sub], op[sub], n_boot=n_boot, seed=seed)
    per_reason = {}
    for r in sorted(set(oreason[oy == 0])):
        m = (oy == 1) | ((oy == 0) & (oreason == r))
        c = C.group_boot_ci(oy[m], op[m], ofac[m], n_boot=n_boot, seed=seed)
        per_reason[str(r)] = {"n_bad": int(((oy == 0) & (oreason == r)).sum()),
                             "auroc_vs_good": round(C.fast_auc(oy[m], op[m]), 4),
                             "ci95_factory_grouped": [round(c[0], 3), round(c[1], 3)]}

    pred_ood_05 = (op >= 0.5).astype(int)
    rates_ood_05 = A.policy_rates(oy, pred_ood_05)
    ci_ood_05 = ci_row(oy, ofac, pred_ood_05, n_boot, seed)

    t_indego, _ = C.youden_threshold(y, p)          # IndEgo = training data w.r.t. OOD
    pred_ood_ti = (op >= t_indego).astype(int)
    rates_ood_ti = A.policy_rates(oy, pred_ood_ti)
    ci_ood_ti = ci_row(oy, ofac, pred_ood_ti, n_boot, seed)

    t_ood_oracle, _ = C.youden_threshold(oy, op)    # in-sample: NOT a policy
    pred_ood_to = (op >= t_ood_oracle).astype(int)
    rates_ood_to = A.policy_rates(oy, pred_ood_to)

    ood_band_share = float((np.abs(op - 0.5) < 0.25).mean())

    # ---------------- round-3 108: read-only ----------------
    r3 = read_c1_round3()
    round3_block = {
        "shipped_26_number_single_head": {
            "status": "SKIPPED",
            "why": ("no per-window score file exists for the 26-number single head on the "
                    "108 round-3 windows; the C1 PR scored the frozen TWO-VIEW ensemble, and "
                    "producing a 26-head score would mean re-running a model on a consumed "
                    "hold-out. Not scored here."),
        },
        "frozen_ensemble_readonly_different_model": None,
    }
    if r3.get("available"):
        ry, rf = r3["y"], r3["factory"]
        rows = {}
        for name, s in (("usable_prob_frozen_two_view_ensemble", r3["usable_prob"]),
                        ("p_eg2_only", r3["p_eg2"]), ("p_pixel_only", r3["p_pixel"])):
            pr = (s >= 0.5).astype(int)
            c = C.group_boot_ci(ry, s, rf, n_boot=n_boot, seed=seed)
            cw = C.window_boot_ci(ry, s, n_boot=n_boot, seed=seed)
            rows[name] = {
                "auroc": round(C.fast_auc(ry, s), 4),
                "ci95_factory_grouped": [round(c[0], 3), round(c[1], 3)],
                "ci95_window_optimistic": [round(cw[0], 3), round(cw[1], 3)],
                "forced_at_0.5": _round(A.policy_rates(ry, pr)),
                "rates_ci95_factory_grouped": ci_row(ry, rf, pr, n_boot, seed),
            }
        round3_block["frozen_ensemble_readonly_different_model"] = {
            "model": ("frozen two-view ensemble_v1 (jadal.ensemble.scorer), the shipped cascade's "
                      "score - NOT the 26-number single head measured in the in-domain arm"),
            "source": "read from the C1 PR's committed inputs; NO model was run in this stream",
            "n": int(len(ry)), "n_good": int((ry == 1).sum()), "n_bad": int((ry == 0).sum()),
            "n_factories": int(len(set(rf))),
            "label_source": ("human, single labeller (round-3 confirmatory hold-out, 108 windows); "
                             "CONSUMED - already scored twice (C1 PR #162, physics v2 PR #163)"),
            "rows": rows,
        }

    # ---------------- results ----------------
    payload = {
        "id": "AUTO.00", "status": "done",
        "n": int(len(y)), "n_bad": int((y == 0).sum()), "n_good": int((y == 1).sum()),
        "n_groups": len(set(clip)), "groups": "7 IndEgo clips / 47 OOD factories",
        "label_source": (g["label_source"] + " | OOD-160: " + o["label_source"] +
                         " | round-3 108 (read-only, different model): human, single labeller, CONSUMED"),
        "protocol": ("(in-domain) LOCO over 7 clips; policy (a) fixed 0.5, policy (b) Youden chosen on the "
                     "6 training clips' inner-LOCO probs per fold; (OOD-160) frozen p26 scores, factory-grouped "
                     "CIs, window bootstrap marked optimistic; (round-3) read-only, single-head arm skipped"),
        "ci95": [ci_a["bal_acc_ci95"][0], ci_a["bal_acc_ci95"][1]],
        "ci95_type": A.CI_TYPE_CLIP + " on balanced accuracy of policy (a)",
        "device": C.DEVICE_CPU,
        "cost": "0 GPU; cached features and frozen scores; ~15 s CPU (7 outer + 42 inner logistic fits, 2000-draw bootstraps)",
        "command": ("JADAL_BUNDLE=<bundle> OMP_NUM_THREADS=2 nice -n 10 "
                    "/home/parshu/envs/jadal/bin/python tools/dsh/auto00.py"),
        "git_sha": C.git_sha(), "seed": seed,
        "config_sha256": C.config_sha256(CONFIG), "config": CONFIG,
        "indomain": {
            "auroc": round(C.fast_auc(y, p), 4),
            "auroc_ci95_clip": [round(x, 3) for x in C.group_boot_ci(y, p, clip, n_boot=n_boot, seed=seed)[:2]],
            "policy_a_forced_0.5": _round(rates_a),
            "policy_a_ci95_clip": ci_a,
            "policy_b_youden_per_fold": _round(rates_b),
            "policy_b_ci95_clip": ci_b,
            "policy_b_folds": _round(folds_b),
            "policy_b_threshold_summary": {
                "min": round(float(np.min(thr_b)), 4), "median": round(float(np.median(thr_b)), 4),
                "max": round(float(np.max(thr_b)), 4),
                "note": "one threshold per held-out clip, chosen on the other 6 clips only",
            },
            "optimistic_in_sample_youden_contrast": dict(
                _round(rates_c), threshold=round(t_oracle, 4),
                note="threshold chosen on ALL 150 OOF labels -> uses the evaluation labels; contrast only, not a policy"),
            "shipped_band_share": round(float(band.mean()), 4),
            "shipped_band_accept_within_band": round(thr_b_share, 4),
        },
        "ood160": {
            "caveat": A.decision_note_ood(),
            "n": int(len(oy)), "n_bad": int((oy == 0).sum()), "n_good": int((oy == 1).sum()),
            "n_factories": int(len(set(ofac))),
            "score": "load_ood160()['p26'] (FROZEN shipped 26-number head, original portal)",
            "auroc": round(ood_auroc, 4),
            "auroc_ci95_factory_grouped": [round(ood_ci_fac[0], 3), round(ood_ci_fac[1], 3)],
            "auroc_ci95_window_optimistic": [round(ood_ci_win[0], 3), round(ood_ci_win[1], 3)],
            "auroc_naw_only": {
                "note": "stratified: 138 good + the 19 NO_ACTIVE_WORK bad windows only (n=157)",
                "n": int(sub.sum()), "n_bad": int(naw.sum()),
                "auroc": round(auroc_naw, 4),
                "ci95_factory_grouped": [round(ci_naw_fac[0], 3), round(ci_naw_fac[1], 3)],
                "ci95_window_optimistic": [round(ci_naw_win[0], 3), round(ci_naw_win[1], 3)],
            },
            "per_reason_auroc_vs_good": per_reason,
            "policy_a_forced_0.5": _round(rates_ood_05),
            "policy_a_ci95_factory": ci_ood_05,
            "policy_b_indego_youden_transferred": dict(
                _round(rates_ood_ti), threshold_from_indego=round(t_indego, 4),
                note="threshold chosen on IndEgo-150 OOF (training data w.r.t. OOD), applied to frozen p26"),
            "policy_b_ci95_factory": ci_ood_ti,
            "optimistic_in_sample_ood_youden_contrast": dict(
                _round(rates_ood_to), threshold=round(t_ood_oracle, 4),
                note="threshold chosen on the OOD-160 labels themselves -> in-sample; contrast only"),
            "shipped_band_share": round(ood_band_share, 4),
            "accept_share_at_0.5": round(float((op >= 0.5).mean()), 4),
        },
        "round3_108": round3_block,
        "notes": [
            "False accept = a BAD window accepted; false reject = a GOOD window rejected.",
            "Policy (b) is nested: the inner LOCO runs inside the 6 training clips of every outer fold.",
            "The frozen p26 scores are the shipped head's scores, so this is the shipped head at full coverage.",
        ],
    }
    C.save_results(OUT, payload)
    print(json.dumps({
        "in_domain_a": {"bal_acc": rates_a["bal_acc"], "ci": ci_a["bal_acc_ci95"],
                        "FAR": rates_a["false_accept_rate"], "FRR": rates_a["false_reject_rate"],
                        "accept_share": rates_a["accept_share"]},
        "in_domain_b": {"bal_acc": rates_b["bal_acc"], "ci": ci_b["bal_acc_ci95"],
                        "FAR": rates_b["false_accept_rate"], "FRR": rates_b["false_reject_rate"],
                        "accept_share": rates_b["accept_share"]},
        "ood_auroc": round(ood_auroc, 4), "ood_auroc_ci_factory": [round(ood_ci_fac[0], 3), round(ood_ci_fac[1], 3)],
        "ood_auroc_ci_window_OPTIMISTIC": [round(ood_ci_win[0], 3), round(ood_ci_win[1], 3)],
        "ood_auroc_naw": round(auroc_naw, 4), "ood_naw_ci_factory": [round(ci_naw_fac[0], 3), round(ci_naw_fac[1], 3)],
        "ood_a": {"bal_acc": rates_ood_05["bal_acc"], "ci": ci_ood_05["bal_acc_ci95"],
                  "FAR": rates_ood_05["false_accept_rate"], "FRR": rates_ood_05["false_reject_rate"],
                  "accept_share": rates_ood_05["accept_share"]},
        "ood_b_transferred": {"threshold": round(t_indego, 4), "bal_acc": rates_ood_ti["bal_acc"],
                              "ci": ci_ood_ti["bal_acc_ci95"], "FAR": rates_ood_ti["false_accept_rate"],
                              "FRR": rates_ood_ti["false_reject_rate"]},
        "round3_status": ("SKIPPED for the 26-number single head; read-only different-model arm present: "
                          + str(round3_block["frozen_ensemble_readonly_different_model"] is not None)),
    }, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
