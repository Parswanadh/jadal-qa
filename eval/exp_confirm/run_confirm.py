#!/usr/bin/env python3
"""CONFIRMATORY one-shot: score the FROZEN O4 recipe on the 108 round-3 hold-out.

Pre-registered. NO tuning, NO feature re-selection, NO threshold change, NO
re-fitting. The only fitted quantities are the ones already frozen inside the
recipe: the per-feature SIGN from the 150 IndEgo gold windows, and (for the
sharpness-3 head) the IndEgo logistic head. The per-domain z-score is
unsupervised (the 108's own mean/std).

Reuses the frozen metric/scoring code from m/adapt160 run_adapt160.py verbatim
(fast_auc, balacc, group_boot_ci, window_boot_ci, combo_scores, z1_head).

Run: OMP_NUM_THREADS=2 python3 eval/exp_confirm/run_confirm.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
ADAPT = os.environ.get("JADAL_ADAPT_DIR", "/workspace/wt/m-adapt160/eval/exp_adapt160")
sys.path.insert(0, ADAPT)
import run_adapt160 as R  # noqa: E402  (frozen metrics + combo + head)

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
INPUTS = os.path.join(REPO, "eval", "results", "exp_confirm", "inputs")
OUT = os.path.join(REPO, "eval", "results", "exp_confirm")

Z3_SIX = R.Z3_SIX
STABLE8 = R.STABLE8
SHARP3 = R.SHARP3


def confusion(y, s, thr):
    y = np.asarray(y); s = np.asarray(s, float)
    pred = (s > thr).astype(int)
    g, b = y == 1, y == 0
    return {"tp": int(((pred == 1) & g).sum()), "fn": int(((pred == 0) & g).sum()),
            "fp": int(((pred == 1) & b).sum()), "tn": int(((pred == 0) & b).sum()),
            "n_good": int(g.sum()), "n_bad": int(b.sum())}


def main():
    z = np.load(os.path.join(INPUTS, "confirm108_features.npz"), allow_pickle=True)
    ids = [str(x) for x in z["ids"]]
    y = z["y"].astype(int)
    fac = np.asarray([str(x) for x in z["factory"]])
    zn = [str(x) for x in z["z3_names"]]
    pn = [str(x) for x in z["pix_names"]]
    Xz3 = z["Xz3"].astype(float)
    Xpx = z["Xpx"].astype(float)

    z3f = R.load_z3_160()          # frozen IndEgo Z3 (150,20)
    gold = R.load_gold()           # frozen IndEgo pixel-26 (150,26)

    z6i = [zn.index(n) for n in Z3_SIX]
    s8i = [pn.index(n) for n in STABLE8]
    sh3i = [pn.index(n) for n in SHARP3]
    # IndEgo columns
    z6g = [list(z3f["names"]).index(n) for n in Z3_SIX]
    s8g = [R.FEATURE_NAMES26.index(n) for n in STABLE8]
    sh3g = [R.FEATURE_NAMES26.index(n) for n in SHARP3]

    X6o, X6g = Xz3[:, z6i], z3f["X_indego"].astype(float)[:, z6g]
    X8o, X8g = Xpx[:, s8i], gold["X26"][:, s8g]
    XS3o, XS3g = Xpx[:, sh3i], gold["X26"][:, sh3g]
    X14o = np.hstack([X8o, X6o]); X14g = np.hstack([X8g, X6g])
    yi = z3f["y_indego"].astype(int)

    n_good, n_bad = int((y == 1).sum()), int((y == 0).sum())
    n_fac = int(len(set(fac.tolist())))

    blocks = []
    # (a) Z3 transferable-6  (frozen recipe a)
    si_a, so_a, au_a, sg_a = R.combo_scores(X6g, X6o, yi)
    ba = R.score_block("(a) O4 Z3 transferable-6",
                       "per-domain z-score, IndEgo-signed, equal-weight mean",
                       y, so_a, fac, 0.0, 6, Z3_SIX, y_in=yi, s_in=si_a,
                       extra={"indego_signs": {n: int(s) for n, s in zip(Z3_SIX, sg_a)},
                              "confusion_thr0": confusion(y, so_a, 0.0)})
    blocks.append(ba)

    # (b) E5 stable-8 + Z3 transferable-6  (frozen recipe b; the headline recipe)
    names14 = STABLE8 + Z3_SIX
    si_b, so_b, au_b, sg_b = R.combo_scores(X14g, X14o, yi)
    bb = R.score_block("(b) O4 E5 stable-8 + Z3 transferable-6",
                       "per-domain z-score, IndEgo-signed, equal-weight mean",
                       y, so_b, fac, 0.0, 14, names14, y_in=yi, s_in=si_b,
                       extra={"indego_signs": {n: int(s) for n, s in zip(names14, sg_b)},
                              "confusion_thr0": confusion(y, so_b, 0.0)})
    blocks.append(bb)

    # (c) sharpness-3 zero-shot head (frozen: logistic on IndEgo, C by inner LOCO)
    oof_s3, ood_s3 = R.z1_head(XS3g, yi, gold["clips"], XS3o)
    bc = R.score_block("(c) sharpness-3 zero-shot head",
                       "logistic on IndEgo (C by inner LOCO), zero-shot",
                       y, ood_s3, fac, 0.5, 3, SHARP3, y_in=yi, s_in=oof_s3,
                       extra={"confusion_thr0.5": confusion(y, ood_s3, 0.5)})
    blocks.append(bc)

    # (d) frozen two-view ensemble_v1
    ens = json.load(open(os.path.join(INPUTS, "confirm108_ensemble.json")))
    esc = {s["id"]: s for s in ens["scores"]}
    missing = [i for i in ids if i not in esc or "error" in esc[i]]
    sf = np.asarray([esc[i]["usable_prob"] if i in esc and "error" not in esc[i]
                     else np.nan for i in ids], float)
    valid = ~np.isnan(sf)
    bd = R.score_block("(d) frozen ensemble_v1 (two-view)",
                       "frozen two-view usable_prob (no fitting)",
                       y[valid], sf[valid], fac[valid], 0.5, None, None,
                       extra={"config_sha256": ens["config_sha256"],
                              "missing_or_error_ids": missing,
                              "confusion_thr0.5": confusion(y[valid], sf[valid], 0.5)})
    blocks.append(bd)

    # paired grouped bootstrap vs frozen ensemble on the common valid set
    gains = {}
    for nm, sc in (("(a) O4 Z3 transferable-6", so_a), ("(b) O4 E5 stable-8 + Z3 transferable-6", so_b),
                   ("(c) sharpness-3 zero-shot head", ood_s3)):
        m = valid & ~np.isnan(sc)
        gains[nm] = R.paired_group_boot(y[m], sc[m], sf[m], fac[m])

    result = {
        "experiment": "C1.01",
        "title": "Confirmatory one-shot: frozen O4 recipe on the 108 round-3 hold-out",
        "protocol": ("PRE-REGISTERED CONFIRMATORY. Scored ONCE on the 108-window round-3 "
                     "hold-out; no tuning, no feature re-selection, no threshold change, "
                     "no re-fitting. Per-domain z-score uses the 108's own unsupervised "
                     "mean/std; signs come from the 150 IndEgo gold windows only."),
        "one_shot_statement": "scored once on the hold-out; no tuning",
        "n": len(ids), "n_good": n_good, "n_bad": n_bad, "n_factories": n_fac,
        "label_source": ("human, single labeller; labels/round3/ood_labels3_108_with_factory.json "
                         "(92 good / 16 bad, 44 factories)"),
        "features": {"z3_transferable_6": Z3_SIX, "e5_stable_8": STABLE8,
                     "sharpness_3": SHARP3},
        "threshold_rule": ("fixed 0.0 for the z-score combos (score > 0 => good), "
                           "0.5 for the probability scorers; not tuned"),
        "bootstrap": "factory-grouped (honest), n_boot=2000, seed=0; window bootstrap shown too",
        "device": R.DEV,
        "gain_vs_frozen_paired_factory_grouped_bootstrap": gains,
        "blocks": blocks,
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(result, open(os.path.join(OUT, "C1.01.json"), "w"), indent=1)

    print(f"n={len(ids)} good={n_good} bad={n_bad} factories={n_fac}")
    print(f"{'recipe':44s}{'AUROC':>8}{'grp CI':>18}{'BA':>7}{'conf(tp/fn/fp/tn)':>22}")
    for b in blocks:
        ci = b["ood_auroc_ci95_factory_grouped_honest"]
        cf = [v for k, v in b.items() if k.startswith("confusion")][0]
        print(f"{b['recipe']:44s}{b['ood_auroc']:>8.3f}{str(ci):>18}{b['ood_balacc']:>7.3f}"
              f"   {cf['tp']}/{cf['fn']}/{cf['fp']}/{cf['tn']}")
    print("\npaired grouped bootstrap AUROC(recipe)-AUROC(frozen):")
    for k, v in gains.items():
        print(f"  {k:44s} {v.get('mean')} ci95={v.get('ci95')} frac>0={v.get('frac_resamples_gt_0')}")
    print("wrote", os.path.join(OUT, "C1.01.json"))


if __name__ == "__main__":
    main()
