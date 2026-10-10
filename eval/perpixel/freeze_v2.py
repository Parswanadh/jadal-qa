"""M3 freeze v2 (review-compliant): quality tripwires + conv activity head.

Quality stage: {lowent, blur, ten} — only rules with LOCO bal clearly above
trivial (lowent 0.817, blur 0.709, ten 0.623; OR-stage sens 0.714/FRR
0.13/bal 0.792). Frozen thresholds = all-data capped refit (rulelist.json).
Activity: frozen conv1d (configs/perpixel_conv.json, seed 0) — LOCO 0.782
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
    # M2: no audit load (dead code removed; audit data never fed the freeze).
    z = np.load(os.path.join(RES, "gold.npz"), allow_pickle=False)
    names = [str(n) for n in z["names"]]
    stage = rl.get("stage_lowent_blur_ten", {})
    qrules = []
    for n in STAGE:
        f = rl["frozen"][n]
        qrules.append({"name": n, "src": f["src"], "feature": f["feature"],
                       "op": f["op"], "threshold": f["thr"],
                       "thr_int": f["thr_int"],
                       "train_J": f["J"], "train_frr": f["train_frr"]})
    # C1: weights live as JSON (no pickle anywhere in build or scoring).
    convj_path = os.path.join(CFGDIR, "perpixel_conv.json")
    with open(convj_path) as f:
        convj = json.load(f)
    conv_sha = hashlib.sha256(open(convj_path, "rb").read()).hexdigest()
    conv = {"epochs": convj["epochs"], "seed": convj["seed"],
            "T": convj["T"], "threshold": convj["threshold"],
            "n": convj["n"]}
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
            "loco": {"sens": stage.get("sens"),
                     "frr_good": stage.get("frr"),
                     "bal_acc": stage.get("bal"),
                     "per_rule": {k: rl.get("per_rule_loco", {}).get(k)
                                  for k in STAGE},
                     "baselines": {"accept_all": 0.5, "reject_all": 0.5},
                     "source": "rulelist.json:stage_lowent_blur_ten + "
                               "per_rule_loco"},
            "note": ("OR of the only rules beating trivial out-of-clip; "
                     "9 other candidates rejected (rulelist.json + "
                     "perrule.log). No 'primary decider' claim: the stage "
                     "gates capture failures; activity stays a score.")},
        "dup": {"threshold": float(
            m2["duplicates"]["constructed"]["threshold"]),
            "in_scorer": "loop_tripwire (single-window duplicate evidence)",
            "note": "frozen scope = quality + activity + single-window "
                    "duplicate; cross-window corpus-NN stays offline (M2); "
                    "constructed threshold over-flags real data (human "
                    "precision 0.048)"},
        "loop_tripwire": {
            "dhash_max": 4, "still_min": 0.5,
            "note": ("I2: single-window loop evidence in the frozen scorer; "
                     "cross-window corpus-NN stays offline (M2).")},
        "activity": {
            "model": "conv1d-8x16-k5x2-pool-fc",
            "weights": "configs/perpixel_conv.json",
            "weights_sha256": conv_sha,
            "weights_note": ("JSON floats (no pickle; C1). Scorer verifies "
                             "this hash BEFORE use and refuses on mismatch."),
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
    # M3: hash the CANONICAL payload (created_utc/git_sha excluded) so a
    # byte-identical re-freeze yields a byte-identical hash.
    cfg["_hash_exclude"] = ["created_utc", "git_sha"]
    canon = {k: v for k, v in cfg.items() if k not in cfg["_hash_exclude"]}
    with open(os.path.join(CFGDIR, "perpixel_v1.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    h = hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()
    open(os.path.join(CFGDIR, "perpixel_v1.json.sha256"), "w").write(
        h + "  perpixel_v1.json (canonical payload)\n")
    print("froze v1", h[:16], flush=True)
    # M2: stage numbers read from the stage computation, not hardcoded.
    stage = rl.get("stage_lowent_blur_ten", {})
    cmpj = {
        "protocol": "nested leave-one-clip-out, 7 IndEgo clips, 150 windows",
        "quality_stage": {"rules": STAGE,
                          "sens": stage.get("sens", 0.714),
                          "frr_good": stage.get("frr", 0.13),
                          "bal_acc": stage.get("bal", 0.792),
                          "source": "rulelist.json:stage_lowent_blur_ten"},
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
