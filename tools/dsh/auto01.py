#!/usr/bin/env python3
"""AUTO.01 - second-view resolver for the windows the shipped band sends to review.

Second view (no GPU, no downloads): the cached EmbeddingGemma-2 window
embeddings at caches/frames10k/segments.npz - 150 IndEgo windows x 21 frames x
768-d, ~2 fps.  Representation: the mean of the 21 frame embeddings per window,
then the SAME StandardScaler + LogisticRegression(C=0.3, balanced) recipe and
the same leave-one-clip-out out-of-fold scheme as the shipped 26-feature head.

Evaluated CONDITIONAL on the band (|p26 - 0.5| < 0.25):
  * AUROC inside the band, clip-grouped CI (window bootstrap marked optimistic);
  * the 100 %-coverage balanced accuracy of "S1 outside the band, second view
    inside it", compared with AUTO.00's forced 0.5 rule on the same windows,
    PAIRED clip bootstrap.  A difference counts only if the paired interval
    excludes zero;
  * the same for a rank-average of the 26-feature score and the EG2 score
    inside the band (the Y3-style combination, checked here rather than quoted).

Run:  JADAL_BUNDLE=<bundle> OMP_NUM_THREADS=2 nice -n 10 \
      /home/parshu/envs/jadal/bin/python tools/dsh/auto01.py
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
    "id": "AUTO.01",
    "second_view": "caches/frames10k/segments.npz F (150,21,768) float16, mean-pooled over the 21 frames",
    "recipe": "StandardScaler+LogisticRegression(C=0.3,class_weight=balanced), leave-one-clip-out",
    "band": "|p26 - 0.5| < 0.25 (the shipped review band)",
    "policies": [
        "hybrid: S1 (26 features) outside the band, EG2 inside the band, accept iff p >= 0.5",
        "band+rankavg: S1 outside the band, rank-average of the two OOF scores inside the band",
    ],
    "comparison": "paired clip bootstrap of balanced accuracy vs AUTO.00 forced 0.5",
    "n_boot": 2000, "seed": 0,
}
OUT = os.path.join(C.REPO, "eval", "results", "auto", "AUTO.01.json")


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


def main():
    n_boot, seed = CONFIG["n_boot"], CONFIG["seed"]
    g = C.load_gold()
    X, y, clip = g["X"], g["y"], g["clip"]
    p26 = C.loco_proba(X, y, clip, C=0.3)

    # ---- second view: cached EG2 window embeddings ----
    npz_path = os.path.join(C.BUNDLE, "caches", "frames10k", "segments.npz")
    z = np.load(npz_path, allow_pickle=True)
    F = z["F"]
    ids_eg2 = [str(i) for i in z["ids"]]
    align_ok = (ids_eg2 == list(g["ids"]))
    if not align_ok:  # align by id, never by position, if the cache is reordered
        pos = {k: i for i, k in enumerate(ids_eg2)}
        F = F[[pos[k] for k in g["ids"]]]
    X_eg2 = np.full((len(y), F.shape[2]), np.nan)
    counts = np.asarray(z["counts"], int)
    for i in range(len(y)):
        n_valid = int(counts[i])
        row = np.asarray(F[i][:n_valid], np.float32)
        assert not np.isnan(row).any(), f"window {i} has NaN inside its {n_valid} valid frames"
        X_eg2[i] = row.mean(axis=0)
    assert not np.isnan(X_eg2).any(), "mean-pooled EG2 representation contains NaN"
    assert X_eg2.shape == (len(y), F.shape[2]), X_eg2.shape
    n_frames_used = {"20_frames": int((counts == 20).sum()), "21_frames": int((counts == 21).sum()),
                     "note": "the 21st frame slot is NaN padding for the shorter windows and is excluded"}

    p_eg2 = C.loco_proba(X_eg2, y, clip, C=0.3)

    band = np.abs(p26 - 0.5) < 0.25
    nb = int(band.sum())

    # ---- conditional-on-band metrics ----
    auroc_band_26 = C.fast_auc(y[band], p26[band])
    auroc_band_eg2 = C.fast_auc(y[band], p_eg2[band])
    ci_band_26 = C.group_boot_ci(y[band], p26[band], clip[band], n_boot=n_boot, seed=seed)
    ci_band_eg2 = C.group_boot_ci(y[band], p_eg2[band], clip[band], n_boot=n_boot, seed=seed)
    wci_band_eg2 = C.window_boot_ci(y[band], p_eg2[band], n_boot=n_boot, seed=seed)
    paired_band_auroc = C.paired_group_boot_diff(y[band], p_eg2[band], p26[band],
                                                 clip[band], n_boot=n_boot, seed=seed)

    # ---- policies at 100 % coverage ----
    pred_forced = (p26 >= 0.5).astype(int)                       # AUTO.00 policy (a)
    pred_hybrid = pred_forced.copy()
    pred_hybrid[band] = (p_eg2[band] >= 0.5).astype(int)

    # rank-average inside the band: ranks are computed over the band's OOF
    # scores only, and the band is defined by the S1 score alone, so no label
    # and no held-out information enters the combination.
    r_avg = np.full(len(y), np.nan)
    r_avg[band] = A.rank_average(p26[band], p_eg2[band])
    pred_rankavg = pred_forced.copy()
    pred_rankavg[band] = (r_avg[band] >= 0.5).astype(int)

    # the band's own forced-0.5 sub-policy, for the conditional comparison
    pred_forced_band = (p26[band] >= 0.5).astype(int)

    # EQUAL-COVERAGE CONTROLS.  A rank-average decided at 0.5 accepts the top
    # half of the band by construction, so a win could be the operating point
    # rather than the combination.  Control = the same 50/50 split using ONE
    # view's ranks (i.e. accept the higher-ranked half of the band).
    r26_band = A.rank01(p26[band])
    reg2_band = A.rank01(p_eg2[band])
    pred_rank_p26 = pred_forced.copy()
    pred_rank_p26[band] = (r26_band >= 0.5).astype(int)
    pred_rank_eg2 = pred_forced.copy()
    pred_rank_eg2[band] = (reg2_band >= 0.5).astype(int)

    rates = {
        "forced_0.5": _round(A.policy_rates(y, pred_forced)),
        "hybrid_eg2": _round(A.policy_rates(y, pred_hybrid)),
        "band_rankavg": _round(A.policy_rates(y, pred_rankavg)),
        "band_only_forced_0.5": _round(A.policy_rates(y[band], pred_forced_band)),
        "band_only_eg2": _round(A.policy_rates(y[band], (p_eg2[band] >= 0.5).astype(int))),
        "band_only_rankavg": _round(A.policy_rates(y[band], (r_avg[band] >= 0.5).astype(int))),
        "equal_coverage_control_rank_p26_only": _round(A.policy_rates(y, pred_rank_p26)),
        "equal_coverage_control_rank_eg2_only": _round(A.policy_rates(y, pred_rank_eg2)),
    }
    cis = {
        "forced_0.5": A.rates_ci(y, clip, pred_forced, n_boot, seed),
        "hybrid_eg2": A.rates_ci(y, clip, pred_hybrid, n_boot, seed),
        "band_rankavg": A.rates_ci(y, clip, pred_rankavg, n_boot, seed),
    }

    # ---- paired comparisons (the only basis for calling a difference real) ----
    def ba_of(pred):
        return lambda yy, idx: float(C.balanced_accuracy(yy, pred[idx]))

    def err_of(pred):
        return lambda yy, idx: float(np.mean(pred[idx] != yy))

    # auto_util.paired_group_boot_diff takes STATISTIC FUNCTIONS of (y_sub, idx)
    # and resamples the same groups for both, which is the paired interval.
    paired = {
        "hybrid_vs_forced0.5_bal_acc": A.paired_group_boot_diff(
            y, clip, ba_of(pred_hybrid), ba_of(pred_forced), n_boot=n_boot, seed=seed),
        "hybrid_vs_forced0.5_error": A.paired_group_boot_diff(
            y, clip, err_of(pred_forced), err_of(pred_hybrid), n_boot=n_boot, seed=seed),
        "rankavg_vs_forced0.5_bal_acc": A.paired_group_boot_diff(
            y, clip, ba_of(pred_rankavg), ba_of(pred_forced), n_boot=n_boot, seed=seed),
        "rankavg_vs_hybrid_bal_acc": A.paired_group_boot_diff(
            y, clip, ba_of(pred_rankavg), ba_of(pred_hybrid), n_boot=n_boot, seed=seed),
        "band_only_eg2_vs_band_only_forced0.5_bal_acc": A.paired_group_boot_diff(
            y[band], clip[band],
            lambda yy, idx: float(C.balanced_accuracy(yy, (p_eg2[band] >= 0.5).astype(int)[idx])),
            lambda yy, idx: float(C.balanced_accuracy(yy, pred_forced_band[idx])),
            n_boot=n_boot, seed=seed),
        # the decisive controls: does the rank-AVERAGE beat each single view
        # decided at the SAME (top-half) coverage inside the band?
        "rankavg_vs_equal_coverage_p26_only": A.paired_group_boot_diff(
            y, clip, ba_of(pred_rankavg), ba_of(pred_rank_p26), n_boot=n_boot, seed=seed),
        "rankavg_vs_equal_coverage_eg2_only": A.paired_group_boot_diff(
            y, clip, ba_of(pred_rankavg), ba_of(pred_rank_eg2), n_boot=n_boot, seed=seed),
    }

    # ---- cross-check against the repo's frozen nested-LOCO EG2 view ----
    frozen = None
    try:
        zp = np.load(os.path.join(C.BUNDLE, "caches", "ensemble", "oof_probs.npz"),
                     allow_pickle=True)
        p_eg2_frozen = np.asarray(zp["p_eg2"], float)
        frozen = {
            "source": "caches/ensemble/oof_probs.npz p_eg2 (frozen ensemble_v1 nested-LOCO EG2 view)",
            "note": "the repo's own frozen artifact; shown as a cross-check, NOT this stream's result",
            "n": int(len(p_eg2_frozen)),
            "pearson_r_vs_this_view": round(float(np.corrcoef(p_eg2, p_eg2_frozen)[0, 1]), 4),
            "auroc_frozen_overall": round(C.fast_auc(y, p_eg2_frozen), 4),
            "auroc_this_view_overall": round(C.fast_auc(y, p_eg2), 4),
            "auroc_frozen_in_band": round(C.fast_auc(y[band], p_eg2_frozen[band]), 4),
        }
    except Exception as e:  # pragma: no cover
        frozen = {"available": False, "why": f"{type(e).__name__}: {e}"}

    payload = {
        "id": "AUTO.01", "status": "done",
        "n": int(len(y)), "n_bad": int((y == 0).sum()), "n_good": int((y == 1).sum()),
        "n_band": nb, "band_share": round(nb / len(y), 4),
        "n_groups": len(set(clip)),
        "label_source": g["label_source"],
        "protocol": ("LOCO over 7 clips for both views; conditional-on-band metrics use the band windows only; "
                     "policies are compared at 100 % coverage by PAIRED clip bootstrap (2000 draws, seed 0)"),
        "ci95": [cis["hybrid_eg2"]["bal_acc_ci95"][0], cis["hybrid_eg2"]["bal_acc_ci95"][1]],
        "ci95_type": A.CI_TYPE_CLIP + " on balanced accuracy of the EG2 hybrid at 100 % coverage",
        "device": C.DEVICE_CPU,
        "cost": ("0 GPU, no downloads; cached embeddings (3.5 MB npz); ~8 s CPU (7 outer + 7 outer logistic fits, "
                 "768-d features, 2000-draw bootstraps)"),
        "command": ("JADAL_BUNDLE=<bundle> OMP_NUM_THREADS=2 nice -n 10 "
                    "/home/parshu/envs/jadal/bin/python tools/dsh/auto01.py"),
        "git_sha": C.git_sha(), "seed": seed,
        "config_sha256": C.config_sha256(CONFIG), "config": CONFIG,
        "second_view": {
            "source": npz_path, "ids_aligned_by": "id (positional match verified)" if align_ok else "id",
            "window_representation": "mean over the cached frame embeddings (768-d); the 21st slot is NaN padding on 145/150 windows and is excluded",
            "n_frames_used": n_frames_used,
            "auroc_overall": round(C.fast_auc(y, p_eg2), 4),
            "auroc_overall_ci95_clip": [round(x, 3) for x in C.group_boot_ci(y, p_eg2, clip, n_boot=n_boot, seed=seed)[:2]],
            "frozen_cross_check": frozen,
        },
        "conditional_on_band": {
            "n_band": nb, "n_band_bad": int((y[band] == 0).sum()), "n_band_good": int((y[band] == 1).sum()),
            "auroc_s1_26_in_band": round(auroc_band_26, 4),
            "auroc_s1_26_in_band_ci95_clip": [round(x, 3) for x in ci_band_26[:2]],
            "auroc_eg2_in_band": round(auroc_band_eg2, 4),
            "auroc_eg2_in_band_ci95_clip": [round(x, 3) for x in ci_band_eg2[:2]],
            "auroc_eg2_in_band_ci95_window_optimistic": [round(x, 3) for x in wci_band_eg2[:2]],
            "paired_eg2_minus_s1_in_band": paired_band_auroc,
        },
        "policies_100pct_coverage": rates,
        "policies_ci95_clip": cis,
        "paired_comparisons_clip_bootstrap": paired,
        "verdicts": {
            "hybrid_improves_over_forced0.5": bool(paired["hybrid_vs_forced0.5_bal_acc"]["excludes_zero"]),
            "rankavg_improves_over_forced0.5": bool(paired["rankavg_vs_forced0.5_bal_acc"]["excludes_zero"]),
            "rankavg_differs_from_hybrid": bool(paired["rankavg_vs_hybrid_bal_acc"]["excludes_zero"]),
            "rankavg_beats_equal_coverage_p26_only": bool(
                paired["rankavg_vs_equal_coverage_p26_only"]["excludes_zero"]),
            "rankavg_beats_equal_coverage_eg2_only": bool(
                paired["rankavg_vs_equal_coverage_eg2_only"]["excludes_zero"]),
            "rule": "a difference is stated only when the paired clip-bootstrap interval excludes zero",
        },
        "notes": [
            "The band is defined by the S1 score alone, so the rank transform inside the band uses no labels.",
            "Outside the band both policies are identical (S1 at 0.5), so the paired difference is driven by the band windows.",
            "n_band is small; the conditional intervals are correspondingly wide.",
            "The shipped cascade decides OUTSIDE the band with accept iff p >= 0.75 and reject iff p <= 0.25; "
            "this experiment models the forced variant (accept iff p >= 0.5) so that the comparison with "
            "AUTO.00 is on identical windows. The hybrid is therefore a strengthened funnel that decides the "
            "band instead of reviewing it, not the shipped cascade with one stage swapped.",
            "The rank-average is decided at rank 0.5, which accepts the top half of the band by "
            "construction; the equal-coverage controls (single view's ranks at 0.5) separate a real "
            "combination win from a mere operating-point change.",
        ],
    }
    C.save_results(OUT, payload)
    print(json.dumps({
        "n_band": nb, "band_share": round(nb / len(y), 4),
        "auroc_in_band": {"s1_26": round(auroc_band_26, 4), "eg2": round(auroc_band_eg2, 4),
                          "eg2_ci_clip": [round(x, 3) for x in ci_band_eg2[:2]],
                          "eg2_ci_window_OPTIMISTIC": [round(x, 3) for x in wci_band_eg2[:2]],
                          "paired_eg2_minus_s1": paired_band_auroc},
        "rates_100pct": {k: {kk: v[kk] for kk in ("bal_acc", "false_accept_rate", "false_reject_rate", "accept_share")}
                         for k, v in rates.items()},
        "paired": paired,
        "cross_check_frozen_eg2": frozen,
    }, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
