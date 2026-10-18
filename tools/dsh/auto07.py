#!/usr/bin/env python3
"""AUTO.07 - audit policy: audit rate vs detected-error rate vs human hours per
10,000 hours of video.

Recommended deployment form (from the brief): auto-decide everything, send a
random audit sample plus ALL low-confidence windows to humans, track error and
verdict mix per site, alarm on drift.

Everything in the output table is ARITHMETIC over measured shares, and every
row is labelled `arithmetic`.  The three measured inputs are, per dataset:
  * the shipped band share  r  = P(|p - 0.5| < 0.25)              (measured)
  * the forced-decision error rate INSIDE the band   e_band       (measured)
  * the forced-decision error rate OUTSIDE the band  e_out        (measured)
One parameter is NOT measured and is exposed as such: the human time to review
one 10 s window, T_review.  No measured review time exists anywhere in the
bundle or the repo, so the table is given for T_review in {20, 30, 60} s and
labelled ASSUMED.

Run:  JADAL_BUNDLE=<bundle> OMP_NUM_THREADS=2 nice -n 10 \
      /home/parshu/envs/jadal/bin/python tools/dsh/auto07.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auto_util as A  # noqa: E402
import common as C  # noqa: E402

WINDOW_S = 10.0
WINDOWS_PER_VIDEO_HOUR = 3600.0 / WINDOW_S          # arithmetic from the 10 s window
VIDEO_HOURS = 10000.0
W = WINDOWS_PER_VIDEO_HOUR * VIDEO_HOURS            # windows in 10,000 video hours
AUDIT_RATES = [0.02, 0.05, 0.10]
T_REVIEW_ASSUMED_S = [20.0, 30.0, 60.0]
SITE_WINDOWS_PER_DAY = 10000                        # a site's daily volume, arithmetic scenario
CONFIG = {
    "id": "AUTO.07",
    "window_s": WINDOW_S,
    "windows_per_video_hour_arithmetic": WINDOWS_PER_VIDEO_HOUR,
    "video_hours": VIDEO_HOURS,
    "audit_rates": AUDIT_RATES,
    "t_review_seconds_ASSUMED": T_REVIEW_ASSUMED_S,
    "site_windows_per_day_arithmetic": SITE_WINDOWS_PER_DAY,
    "band": "|p-0.5| < 0.25 (the shipped low-confidence review band)",
    "policy": ("auto-decide everything with the forced 0.5 rule; send every band window AND a uniform "
               "random audit sample to a human; track error and verdict mix per site; alarm when the "
               "site's audit-based error upper bound exceeds tolerance"),
    "n_boot": 2000, "seed": 0,
}
OUT = os.path.join(C.REPO, "eval", "results", "auto", "AUTO.07.json")


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


def measured_inputs(y, p, groups, name):
    """The three measured quantities per dataset, with a grouped CI on the band share."""
    band = np.abs(p - 0.5) < 0.25
    pred = (p >= 0.5).astype(int)
    out = {
        "dataset": name, "n": int(len(y)), "n_bad": int((y == 0).sum()),
        "band_share_r": float(band.mean()),
        "band_share_ci95_grouped": None,
        "error_rate_inside_band_forced0.5": float((pred[band] != y[band]).mean()) if band.sum() else None,
        "error_rate_outside_band_forced0.5": float((pred[~band] != y[~band]).mean()) if (~band).sum() else None,
        "n_band": int(band.sum()), "n_outside": int((~band).sum()),
        "bal_acc_full_coverage": float(C.balanced_accuracy(y, pred)),
        "false_accept_rate_full_coverage": float((pred[y == 0] == 1).mean()),
        "false_reject_rate_full_coverage": float((pred[y == 1] == 0).mean()),
        "measurement_note": "band share and the two error rates are measured on this dataset's labels",
    }
    r = A.group_boot_stat(y, groups, lambda d, idx: float((np.abs(p[idx] - 0.5) < 0.25).mean()),
                          n_boot=CONFIG["n_boot"], seed=CONFIG["seed"], min_classes=False)
    out["band_share_ci95_grouped"] = [round(r["lo"], 4), round(r["hi"], 4)]
    return out


def audit_table(m, a, t_s):
    """Closed-form audit arithmetic for one measured input row."""
    r = m["band_share_r"]
    e_band = m["error_rate_inside_band_forced0.5"]
    e_out = m["error_rate_outside_band_forced0.5"]
    n_band = r * W
    n_out = (1 - r) * W
    err_band = e_band * n_band              # caught by the human band review
    err_out = e_out * n_out                 # escaped auto-decisions; only the audit sees them
    err_detected = err_band + a * err_out
    err_total = err_band + err_out
    human_windows = (r + a) * W
    return {
        "label": "arithmetic",
        "audit_rate": a,
        "t_review_seconds_ASSUMED": t_s,
        "review_share_measured_r": round(r, 4),
        "human_visited_share_arithmetic": round(r + a, 4),
        "human_windows_per_10k_video_hours": int(round(human_windows)),
        "human_hours_per_10k_video_hours": round(human_windows * t_s / 3600.0, 1),
        # two explicit units; W/N_10K converts the 10,000-video-hour totals to a
        # per-10,000-window basis (3.6 M windows / 10 000 = 360 windows per unit)
        "escaped_errors_per_10k_windows": int(round(err_out / (W / 10000.0))),
        "detected_escaped_errors_per_10k_windows": int(round(a * err_out / (W / 10000.0))),
        "residual_undetected_errors_per_10k_windows": int(round((1 - a) * err_out / (W / 10000.0))),
        "escaped_errors_per_10k_video_hours": int(round(err_out / VIDEO_HOURS)),
        "detected_escaped_errors_per_10k_video_hours": int(round(a * err_out / VIDEO_HOURS)),
        "detected_error_rate_over_all_errors": round(err_detected / err_total, 4) if err_total else None,
        "expected_share_of_escaped_errors_caught": a,
        "p_at_least_one_escaped_error_in_a_10000_window_site_audit":
            round(1.0 - (1.0 - e_out) ** (a * SITE_WINDOWS_PER_DAY), 6),
        "expected_escaped_errors_detected_in_a_10000_window_site_audit":
            round(a * SITE_WINDOWS_PER_DAY * e_out, 2),
        "note": ("arithmetic, not measurement: shares r/e_band/e_out are measured, T_review is an "
                 "assumed parameter, and the 10 s window gives 360 windows per video hour"),
    }


def main():
    g = C.load_gold()
    p = C.loco_proba(g["X"], g["y"], g["clip"], C=0.3)
    mi = measured_inputs(g["y"], p, g["clip"], "IndEgo-150 (LOCO)")
    o = C.load_ood160()
    mo = measured_inputs(o["y"], o["p26"], o["factory"], "OOD-160 (frozen shipped head)")

    rows = []
    for m in (mi, mo):
        for a in AUDIT_RATES:
            for t_s in T_REVIEW_ASSUMED_S:
                rows.append({"dataset": m["dataset"], **audit_table(m, a, t_s)})

    # drift-alarm arithmetic: how many audited windows a site needs
    drift = {}
    for m in (mi, mo):
        e_out = m["error_rate_outside_band_forced0.5"]
        per = {}
        for a in AUDIT_RATES:
            n_audit = a * SITE_WINDOWS_PER_DAY
            per[f"{a:.2f}"] = {
                "audited_windows_per_site_day": int(round(n_audit)),
                "expected_detected_errors": round(n_audit * e_out, 2),
                "p_at_least_one_detected": round(1 - (1 - e_out) ** n_audit, 6),
                "n_audit_needed_for_95pct_chance_of_one_detection":
                    int(np.ceil(np.log(0.05) / np.log(1 - e_out))) if 0 < e_out < 1 else None,
                "n_audit_needed_to_estimate_site_error_within_5pct_absolute_95pct":
                    int(np.ceil(1.96 ** 2 * e_out * (1 - e_out) / 0.05 ** 2)),
            }
        drift[m["dataset"]] = {"error_rate_outside_band_measured": round(e_out, 4),
                               "site_windows_per_day_ASSUMED_arithmetic": SITE_WINDOWS_PER_DAY,
                               "per_audit_rate": per}

    decision_rule = {
        "rule": ("recommend fully automated only with the measured 100 %-coverage error and its "
                 "interval; if the OOD-160 balanced-accuracy lower bound is below 0.70 the "
                 "recommendation must be tiered automation with audit"),
        "ood160_bal_acc_lower_bound": None,   # filled from AUTO.00's recorded interval
        "recommendation": None,
    }
    try:
        a00 = json.load(open(os.path.join(C.REPO, "eval", "results", "auto", "AUTO.00.json")))
        lo = a00["ood160"]["policy_a_ci95_factory"]["bal_acc_ci95"][0]
        decision_rule["ood160_bal_acc_lower_bound"] = lo
        decision_rule["recommendation"] = ("tiered automation with audit" if lo < 0.70
                                           else "fully automated is defensible")
        decision_rule["source"] = "eval/results/auto/AUTO.00.json policy_a_ci95_factory"
    except Exception as e:  # pragma: no cover
        decision_rule["source"] = f"unavailable: {type(e).__name__}: {e}"

    payload = {
        "id": "AUTO.07", "status": "done",
        "n": int(len(g["y"])), "n_bad": int((g["y"] == 0).sum()),
        "n_ood": int(len(o["y"])), "n_bad_ood": int(o["n_bad"]),
        "n_groups": 7, "groups": "7 IndEgo clips / 47 OOD factories",
        "label_source": g["label_source"] + " | " + o["label_source"],
        "protocol": ("no new measurement: the band share and the in-band / out-of-band error rates are "
                     "computed from the AUTO.00 scores and labels; every table row is closed-form "
                     "arithmetic over those shares with T_review exposed as an assumed parameter"),
        "ci95": mi["band_share_ci95_grouped"],
        "ci95_type": A.CI_TYPE_CLIP + " on the IndEgo band share (the only sampling interval in this file)",
        "device": C.DEVICE_CPU,
        "cost": "0 GPU; ~3 s CPU (one LOCO fit, no bootstrap beyond the band share)",
        "command": ("JADAL_BUNDLE=<bundle> OMP_NUM_THREADS=2 nice -n 10 "
                    "/home/parshu/envs/jadal/bin/python tools/dsh/auto07.py"),
        "git_sha": C.git_sha(), "seed": CONFIG["seed"],
        "config_sha256": C.config_sha256(CONFIG), "config": CONFIG,
        "measured_inputs": {"indego": mi, "ood160": mo,
                            "ood_caveat": A.decision_note_ood()},
        "assumed_parameters": {
            "t_review_seconds": {
                "value": T_REVIEW_ASSUMED_S,
                "status": "ASSUMED, NOT MEASURED",
                "why": ("no measured human review time per window exists in the bundle or the repo; the "
                        "hours columns scale linearly in this parameter and are meaningless without it"),
            },
            "site_windows_per_day": {
                "value": SITE_WINDOWS_PER_DAY,
                "status": "ASSUMED scenario for the drift-alarm table only",
            },
            "windows_per_video_hour": {
                "value": WINDOWS_PER_VIDEO_HOUR,
                "status": "arithmetic from the 10 s non-overlapping window definition",
            },
        },
        "audit_table": rows,
        "drift_alarm_arithmetic": drift,
        "decision_rule_check": decision_rule,
        "notes": [
            "Every row of audit_table carries label='arithmetic'; no row is a measurement.",
            "detected_error_rate_over_all_errors is arithmetic: band errors are caught by the human "
            "review of the band, escaped errors only by the uniform audit sample.",
            "The audit is modelled as a uniform random sample; a real audit that oversamples hard "
            "sites would change the detected count, not the ordering of the rows.",
            "The brief's 'accept share moved from 19 % to 57 % when frame sampling changed' is quoted "
            "in the brief and was NOT re-measured in this stream; it is the reason the per-site drift "
            "tracking is specified, not a number this file relies on.",
        ],
    }
    C.save_results(OUT, payload)
    print("measured: IndEgo band=%.3f (CI %s) e_band=%.3f e_out=%.3f | OOD band=%.3f e_band=%.3f e_out=%.3f" % (
        mi["band_share_r"], mi["band_share_ci95_grouped"], mi["error_rate_inside_band_forced0.5"],
        mi["error_rate_outside_band_forced0.5"], mo["band_share_r"],
        mo["error_rate_inside_band_forced0.5"], mo["error_rate_outside_band_forced0.5"]))
    for r in rows:
        if abs(r["t_review_seconds_ASSUMED"] - 30.0) < 1e-9:
            print("%-34s a=%.2f  human_hours/10k_video_h=%.1f  detected_err_rate=%.3f  escaped/10k=%d  caught/10k=%d" % (
                r["dataset"], r["audit_rate"], r["human_hours_per_10k_video_hours"],
                r["detected_error_rate_over_all_errors"] or float("nan"),
                r["escaped_errors_per_10k_windows"], r["detected_escaped_errors_per_10k_windows"]))
    print("decision rule:", json.dumps(decision_rule))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
