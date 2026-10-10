#!/usr/bin/env python3
"""O4 review point: adapted score vs the frozen review policy (band / gate).

The published review policy (Y1/Y2) is defined on the frozen *probability*
(`usable_prob`): a symmetric abstain band half-width w=0.16 around 0.5, and a
decision-theoretic gate that decides the top c=45% of windows by margin
|p - 0.5| (reviews the rest). The adapted Z3 score is an *uncalibrated signed
z-score* (boundary 0), so the same rules are applied with the boundary at 0 and
w in standardized-score units — clearly a different quantity, reported as such.

We compare, on the SAME 100 sampled unlabelled E8 windows, the review% of the
frozen score vs the adapted score under both rules; and on the labelled 160 we
report the OOD balanced accuracy (review = miss) at those operating points.

Appends a `review_point` block to eval/results/exp_adapt160/O4.01.json.
Run (after extract_e8_sample.py): OMP_NUM_THREADS=2 python3 eval/exp_adapt160/review_point.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import run_adapt160 as R  # noqa: E402

OUT = os.path.join(REPO, "eval", "results", "exp_adapt160")
INPUTS = os.path.join(OUT, "inputs")
# Pod path from the environment only (no tracked default, review finding 6).
E8_JSONL = os.environ.get("JADAL_E8_JSONL")
W_BAND = 0.16
C_GATE = 0.45


def load_frozen_e8():
    if not E8_JSONL:
        raise SystemExit(
            "JADAL_E8_JSONL is not set and no pod path is hard-coded; export it, e.g. "
            "export JADAL_E8_JSONL=/path/windows_rescored.jsonl")
    sc = {}
    for line in open(E8_JSONL):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "header":
            continue
        sc[d["window_id"]] = float(d["score"])
    return sc


def review_mask_band(s, boundary, w):
    return np.abs(np.asarray(s, float) - boundary) <= w


def review_mask_gate(s, boundary, c):
    """Review the (1-c) fraction with the smallest |s-boundary| (decide top c%)."""
    m = np.abs(np.asarray(s, float) - boundary)
    k = int(round((1.0 - c) * len(m)))
    order = np.argsort(m, kind="mergesort")
    review = np.zeros(len(m), bool)
    review[order[:k]] = True
    return review


def ba_review_miss(y, s, boundary, review):
    y = np.asarray(y)
    s = np.asarray(s, float)
    review = np.asarray(review, bool)
    g = (y == 1)
    b = (y == 0)
    good_accept = float(((~review) & (s > boundary))[g].mean()) if g.any() else float("nan")
    bad_reject = float(((~review) & (s < boundary))[b].mean()) if b.any() else float("nan")
    return 0.5 * (good_accept + bad_reject), good_accept, bad_reject


def main():
    # ---- OOD-160 adapted scores ----
    lab = R.load_labels160()
    z3 = R.load_z3_160()
    pix = R.load_pixel160()
    gold = R.load_gold()
    y, fac = lab["y"], lab["factory"]
    ids = lab["ids"]
    Xz3 = z3["X"]
    Z3i = z3["X_indego"]
    yi = z3["y_indego"]
    X26o = R.align(pix["ids"], pix["X10"], ids)
    z6 = [z3["names"].index(n) for n in R.Z3_SIX]
    s8 = [R.FEATURE_NAMES26.index(n) for n in R.STABLE8]
    X6o, X6g = Xz3[:, z6], Z3i[:, z6]
    X14o = np.hstack([X26o[:, s8], X6o])
    X14g = np.hstack([gold["X26"][:, s8], X6g])
    _, so_a, _, _ = R.combo_scores(X6g, X6o, yi)
    _, so_b, _, _ = R.combo_scores(X14g, X14o, gold["y"])
    frozen, _ = R.load_frozen160(ids)

    # ---- E8 sample ----
    zp = np.load(os.path.join(INPUTS, "e8_sample.npz"), allow_pickle=True)
    e8ids = [str(x) for x in zp["ids"]]
    e8names = [str(x) for x in zp["names"]]
    Xe6 = zp["X"][:, [e8names.index(n) for n in R.Z3_SIX]]
    Xe26 = zp["X26"]
    Xe14 = np.hstack([Xe26[:, s8], Xe6])
    # per-domain z-score using the SAMPLE's own mean/std; signs from IndEgo
    se_a = R.combo_scores(X6g, Xe6, yi)[1]
    se_b = R.combo_scores(X14g, Xe14, gold["y"])[1]
    frozen_e8 = load_frozen_e8()
    frozen_s = np.asarray([frozen_e8[i] for i in e8ids], float)
    ok = ~np.isnan(se_a) & ~np.isnan(se_b) & ~np.isnan(frozen_s)
    se_a, se_b, frozen_s = se_a[ok], se_b[ok], frozen_s[ok]
    n_e8 = int(ok.sum())

    def row(name, s_e8, s_ood, boundary):
        rb = review_mask_band(s_e8, boundary, W_BAND)
        rg = review_mask_gate(s_e8, boundary, C_GATE)
        ob = review_mask_band(s_ood, boundary, W_BAND)
        og = review_mask_gate(s_ood, boundary, C_GATE)
        ba_ob = ba_review_miss(y, s_ood, boundary, ob)
        ba_og = ba_review_miss(y, s_ood, boundary, og)
        return {
            "scorer": name,
            "e8_sample_n": n_e8,
            "review_pct_band_w0.16": round(float(rb.mean()) * 100, 2),
            "review_pct_gate_c0.45": round(float(rg.mean()) * 100, 2),
            "ood160_review_pct_band_w0.16": round(float(ob.mean()) * 100, 2),
            "ood160_review_pct_gate_c0.45": round(float(og.mean()) * 100, 2),
            "ood160_ba_review_miss_band": round(ba_ob[0], 4),
            "ood160_ba_review_miss_gate": round(ba_og[0], 4),
            "ood160_auroc": round(R.fast_auc(y, s_ood), 4),
            "ood160_good_accept_band": round(ba_ob[1], 4),
            "ood160_bad_reject_band": round(ba_ob[2], 4),
        }

    rows = [
        row("frozen ensemble_v1 (probability, boundary 0.5)", frozen_s, frozen, 0.5),
        row("(a) Z3 transferable-6 z-score (boundary 0)", se_a, so_a, 0.0),
        row("(b) stable-8+6 z-score (boundary 0)", se_b, so_b, 0.0),
    ]

    block = {
        "note": ("The published band w=0.16 / gate c=0.45 are defined on the frozen "
                 "PROBABILITY usable_prob (boundary 0.5). The adapted score is an "
                 "uncalibrated signed z-score (boundary 0); the same rule is applied with "
                 "w in standardized-score units, so its review% is a different quantity. "
                 "E8 windows are UNLABELLED: review% is descriptive workload, not accuracy. "
                 "OOD-160 BA counts review as a miss (good_accept + bad_reject)/2."),
        "band_w": W_BAND, "gate_c": C_GATE,
        "e8_sampling": {"n_sampled": int(zp["n_total"]) if "n_total" in zp else None,
                        "n_used": n_e8, "seed": int(zp["sample_seed"]) if "sample_seed" in zp else 0,
                        "source": "artifacts/windows_rescored.jsonl (1,546 unlabelled E8 windows)"},
        "rows": rows,
    }

    o401 = json.load(open(os.path.join(OUT, "O4.01.json")))
    o401["review_point"] = block
    json.dump(o401, open(os.path.join(OUT, "O4.01.json"), "w"), indent=1)

    print(f"E8 sample n={n_e8} (unlabelled) | OOD-160 n=160 (22 bad)")
    print(f"{'scorer':52s}{'E8 band%':>10}{'E8 gate%':>10}{'OOD band BA':>13}{'OOD gate BA':>13}")
    for r in rows:
        print(f"{r['scorer']:52s}{r['review_pct_band_w0.16']:>10.2f}"
              f"{r['review_pct_gate_c0.45']:>10.2f}"
              f"{r['ood160_ba_review_miss_band']:>13.3f}{r['ood160_ba_review_miss_gate']:>13.3f}")
    print("wrote review_point -> O4.01.json")


if __name__ == "__main__":
    main()
