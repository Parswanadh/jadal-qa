#!/usr/bin/env python3
"""DSH-PHYS: score the FROZEN v2 rule table ONCE on the 108-window confirmatory hold-out.

Discipline: this script loads `configs/phys_rules_v2.json` and never modifies it.
After a successful run it writes `HOLDOUT_SCORED.lock`; a second run refuses unless
`--i-know-this-breaks-the-protocol` is passed. The hold-out labels were never used to
choose the rule list, thresholds, integer weights, `t_reject` or the transfer mode.

Two modes are reported because the hold-out is Egocentric-10K footage (same family as
the OOD-160 set), while the table's absolute thresholds come from IndEgo:
  * absolute   -- thresholds as frozen (what an in-domain deployment would run);
  * requantile -- label-free per-domain rank normalisation of the hold-out's OWN
                  unlabelled values, rule list/directions/weights/t_reject frozen.

Output: eval/results/exp_phys_dsh/HOLDOUT108_final.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from eval.exp_phys_dsh.run_v2_lead import (  # noqa: E402
    LEVELS, auroc_bad, ba_at, boot_ci, load_npz, q_int, rule_matrix)

OUTDIR = os.path.join(REPO, "eval", "results", "exp_phys_dsh")
CONFIG = os.path.join(REPO, "configs", "phys_rules_v2.json")
LOCK = os.path.join(OUTDIR, "HOLDOUT_SCORED.lock")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-know-this-breaks-the-protocol", action="store_true",
                    dest="force")
    args = ap.parse_args()
    if os.path.exists(LOCK) and not args.force:
        print(f"REFUSING: {LOCK} exists - the hold-out was already scored once.")
        return 2
    t0 = time.time()
    cfg = json.load(open(CONFIG))
    rules, tau = cfg["rules"], int(cfg["t_reject"])
    ho = load_npz("holdout108")
    # the hold-out npz carries no `clips` key (windows come from many factories);
    # bootstrap by FACTORY, which is the honest clustering here.
    raw = np.load(os.path.join(REPO, "eval", "results", "exp_phys_dsh", "inputs",
                               "holdout108_full.npz"), allow_pickle=True)
    ho["clips"] = [json.loads(str(m)).get("factory", "") for m in raw["meta"]]
    y, clips = ho["y"], ho["clips"]

    thr_abs = {r["rule"] + "|" + r["feature"]: [int(t) for t in r["thresholds_q"]]
               for r in rules}
    Xr = np.array(ho["X"], float, copy=True)
    order = np.argsort(np.argsort(Xr, axis=0), axis=0).astype(float)
    finite = np.isfinite(Xr)
    cnt = finite.sum(axis=0)
    Xr = np.where(finite, (order + 0.5) / np.maximum(cnt, 1), np.nan)
    thr_rank = {r["rule"] + "|" + r["feature"]:
                [q_int(1.0 - l) if r["side"] == "high" else q_int(l) for l in LEVELS]
                for r in rules}

    out = {"experiment": "HOLDOUT108_final",
           "config": os.path.relpath(CONFIG, REPO),
           "config_sha256": __import__("hashlib").sha256(
               open(CONFIG, "rb").read()).hexdigest(),
           "device": "CPU (RunPod A40 host)",
           "protocol": ("confirmatory hold-out, 108 human-labelled Egocentric-10K "
                        "windows (92 good / 16 bad), scorched ONCE with the frozen "
                        "table; window-bootstrap 95% CI (optimistic)"),
           "label_source": "human (labels/round3/ood_labels3_108_with_factory.json)",
           "scored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "command": "python3 eval/exp_phys_dsh/score_holdout.py",
           "t_reject_frozen": tau, "t_accept_frozen": int(cfg.get("t_accept", 0)),
           "n": int(len(y)), "n_good": int((y == 1).sum()), "n_bad": int((y == 0).sum()),
           "modes": {}}
    M_abs = rule_matrix(ho["X"], ho["names"], rules, thr_abs)
    M_rank = rule_matrix(Xr, ho["names"], rules, thr_rank)
    w = np.array([r.get("weight", 1) for r in rules], int)
    for tag, M in (("absolute", M_abs), ("requantile", M_rank)):
        s = M @ w
        ba, tpr, tnr = ba_at(y, s, tau)
        out["modes"][tag] = {
            "auroc": round(auroc_bad(y, s), 4),
            "auroc_ci95_window_bootstrap": [round(v, 4) for v in boot_ci(y, s, clips)],
            "balanced_accuracy": round(ba, 4),
            "reject_rate_all": round(float((s >= tau).mean()), 4),
            "reject_rate_bad": round(tpr, 4),
            "false_reject_good": round(1 - tnr, 4),
            "score_stats": {"min": int(s.min()), "max": int(s.max()),
                            "mean": round(float(s.mean()), 3)}}
        print(f"holdout {tag:11s} AUROC={out['modes'][tag]['auroc']:.4f} "
              f"CI={out['modes'][tag]['auroc_ci95_window_bootstrap']} "
              f"BA={out['modes'][tag]['balanced_accuracy']:.3f} "
              f"reject={out['modes'][tag]['reject_rate_all']:.3f}", flush=True)

    out["honesty"] = ("scored once after the freeze; the hold-out never entered any "
                      "threshold, weight, tau or mode decision. n_bad=16 is small, so "
                      "the CI is wide and this is a single point estimate.")
    out["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "HOLDOUT108_final.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    with open(LOCK, "w") as f:
        f.write(json.dumps({"scored_utc": out["scored_utc"],
                            "config_sha256": out["config_sha256"]}) + "\n")
    print("wrote HOLDOUT108_final.json and the lock")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
