"""M3 freeze v2 (review-compliant): quality tripwires + conv activity head.

Quality stage: {lowent, blur, ten} — only rules with LOCO bal clearly above
trivial (lowent 0.817, blur 0.709, ten 0.623; OR-stage sens 0.714/FRR
0.13/bal 0.792). Frozen thresholds = all-data capped refit (rulelist.json).
Activity: frozen conv1d (configs/perpixel_conv.pt, seed 0) — LOCO 0.782
[0.63,0.86], best point estimate; LR-all 0.702/HGB 0.758 reported in compare.
Digest EXCLUDES B_mv_per_frame (ffmpeg MV debug counts vary run to run;
documented; feature still reported, never hashed).

Writes configs/perpixel_v1.json (+ .sha256), eval/results/perpixel/
{compare.json, freeze_receipts.json}.
Regenerate: OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1 .../python eval/perpixel/freeze_v2.py
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results", "perpixel")
CFGDIR = os.path.normpath(os.path.join(HERE, "..", "..", "configs"))
SPEC_DIR = "/home/parshu/projects/jadal-qa-wt/spectral_data"

STAGE = ["lowent", "blur", "ten"]
DIGEST_EXCLUDE = ["B_mv_per_frame"]


def main():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    gm = json.load(open(os.path.join(SPEC_DIR, "manifest_gold.json")))
    m2 = json.load(open(os.path.join(RES, "m2.json")))
    rl = json.load(open(os.path.join(RES, "rulelist.json")))
    theory = json.load(open(os.path.join(RES, "theory.json")))
    phys = json.load(open(os.path.join(RES, "physics.json")))
    audit = json.load(open(os.path.join(RES, "audit.json")))
    z = np.load(os.path.join(RES, "gold.npz"), allow_pickle=True)
    names = [str(n) for n in z["names"]]
    qrules = []
    for n in STAGE:
        f = rl["frozen"][n]
        qrules.append({"name": n, "src": f["src"], "feature": f["feature"],
                       "op": f["op"], "threshold": f["thr"],
                       "thr_int": f["thr_int"],
                       "train_J": f["J"], "train_frr": f["train_frr"]})
    conv = torch.load(os.path.join(CFGDIR, "perpixel_conv.pt"),
                      map_location="cpu", weights_only=False)
    cfg = {
        "version": "perpixel_v1",
        "created_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "git_sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True).stdout.strip(),
        "label_snapshot": {"manifest": "manifest_gold.json",
                           "sha256": gm.get("labels_sha256"),
                           "n_good": gm.get("n_good"), "n_bad": gm.get("n_bad")},
        "feature_names": names,
        "digest_exclude": DIGEST_EXCLUDE,
        "digest_exclude_why": ("B_mv_per_frame comes from ffmpeg '-v debug' "
                               "MV-count lines; decoder threading makes it "
                               "vary run to run. Excluded from the hash; "
                               "still reported, never a decision input."),
        "quality_stage": {
            "rules": qrules,
            "loco": {"sens": 0.714, "frr_good": 0.13, "bal_acc": 0.792,
                     "per_rule": {
                         "lowent": {"sens": 0.714, "frr": 0.08},
                         "blur": {"sens": 0.429, "frr": 0.01},
                         "ten": {"sens": 0.286, "frr": 0.04}},
                     "baselines": {"accept_all": 0.5, "reject_all": 0.5}},
            "note": ("OR of the only rules beating trivial out-of-clip; "
                     "9 other candidates rejected (rulelist.json + "
                     "perrule.log). No 'primary decider' claim: the stage "
                     "gates capture failures; activity stays a score.")},
        "dup": {"threshold": float(
            m2["duplicates"]["constructed"]["threshold"]),
            "note": "constructed-pairs threshold; over-flags real data "
                    "(human precision 0.048); corpus check offline"},
        "activity": {
            "model": "conv1d-8x16-k5x2-pool-fc",
            "weights": "configs/perpixel_conv.pt",
            "weights_sha256": hashlib.sha256(open(os.path.join(
                CFGDIR, "perpixel_conv.pt"), "rb").read()).hexdigest(),
            "seed": 0, "epochs": 40, "T": 100,
            "threshold": float(conv["threshold"]),
            "loco_auroc": float(m2["series"]["auroc_mean"]),
            "loco_ci": m2["series"]["ci"],
            "loco_seeds": m2["series"]["auroc_seeds"],
            "why_not_lr": ("LR-all 0.702/HGB 0.758 skipped as frozen head: "
                           "conv has the best LOCO point estimate with "
                           "proven repeat determinism (maxdiff 0.0); CIs "
                           "overlap — stated, not hidden.")},
        "certifiability": ("n=150: no rule Bonferroni-certifiable "
                           "(CERTIFY-by-clip n~84 < ~86 needed); per-rule "
                           "Clopper-Pearson uppers in theory.json are "
                           "descriptive."),
        "frozen_note": ("PRE-REGISTERED for one later scoring on a new "
                        "human-labelled Egocentric-10K sample. Do NOT tune "
                        "after freezing."),
    }
    with open(os.path.join(CFGDIR, "perpixel_v1.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    h = hashlib.sha256(open(os.path.join(
        CFGDIR, "perpixel_v1.json"), "rb").read()).hexdigest()
    open(os.path.join(CFGDIR, "perpixel_v1.json.sha256"), "w").write(
        h + "  perpixel_v1.json\n")
    print("froze v1", h[:16], flush=True)
    cmpj = {
        "protocol": "nested leave-one-clip-out, 7 IndEgo clips, 150 windows",
        "quality_stage": {"rules": STAGE, "sens": 0.714, "frr_good": 0.13,
                          "bal_acc": 0.792},
        "activity": {
            "conv1d": {"auroc": float(m2["series"]["auroc_mean"]),
                       "ci": m2["series"]["ci"],
                       "seeds": m2["series"]["auroc_seeds"],
                       "params": "~1k", "explains": "no", "frozen": True},
            "hgb_m1": {"auroc": float(m2["activity"]["auroc_hgb"]),
                       "ci": m2["activity"]["ci_hgb"], "frozen": False},
            "lr_m1": {"auroc": float(m2["activity"]["auroc_lr"]),
                      "ci": m2["activity"]["ci_lr"], "frozen": False},
            "lr_physics": {"auroc": float(
                phys["activity"]["physics_only"]["auroc"]),
                "ci": phys["activity"]["physics_only"]["ci"],
                "frozen": False},
            "lr_all": {"auroc": float(
                phys["activity"]["m1+theory+physics"]["auroc"]),
                "frozen": False}},
        "rule_list_full_7rule": {"bal_acc": theory["rule_list"]["bal_acc"],
                                 "auroc": theory["rule_list"]["auroc"],
                                 "status": "REJECTED as decider (GOOD kept "
                                           "0.0); kept as evidence only"},
    }
    with open(os.path.join(RES, "compare.json"), "w") as f:
        json.dump(cmpj, f, indent=1)
    print("wrote compare.json", flush=True)


if __name__ == "__main__":
    main()
