"""B1-verifier: INDEPENDENT re-check of the LB2 PoC-integration numbers (PR #92).

Written from scratch by the independent verifier. It does NOT import and does
NOT re-run eval/poc_integrate/run_integration.py. It re-implements the whole
scoring path (feature loading, EG2 prefix pooling, LOCO logistic fits, AUROC,
clip bootstrap, verdicts) from the READ-ONLY pod-bundle caches and compares the
outcome against the b/poc-integrate claim.

Risk under test (highest value first): the run used C_eg2=0.003 while the
PUBLISHED reference for the same EG2 2 fps view (ensemble_v1.json ->
views.eg2.full_inner_oof_auroc_by_C) is C_eg2=0.01 -> AUROC 0.848
(C=0.003 -> 0.8514). Pixel view published C=0.3 -> 0.810. So the identical
protocol is re-run at BOTH C_eg2 values and the paired clip-bootstrap
difference (C=0.01 minus C=0.003) is reported.

Output: eval/results/pod_verify/lb2_recheck.json
Run:    OMP_NUM_THREADS=2 python eval/poc_integrate/lb2_independent_recheck.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "pod_verify", "lb2_recheck.json")

SEED = 20261010          # verifier's own bootstrap seed (not the author's seed 7)
BOOT = 2000
C_PX = 0.3               # published pixel C (ensemble_v1 views.pixel.C)
C_EG2 = {"claimed_0.003": 0.003, "published_0.01": 0.01}


# ---------------------------------------------------------------- metrics ---
def auroc_mw(y, p):
    """Own Mann-Whitney AUROC (average ranks for ties), cross-checked vs sklearn."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = ranks[y == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def clip_bootstrap(y, p, clip, n_boot=BOOT, seed=SEED):
    """Percentile 95% CI resampling CLIPS with replacement; returns (ci, used, draws)."""
    y = np.asarray(y)
    clip = np.asarray(clip)
    clips = sorted(set(clip.tolist()))
    idx_by_clip = [np.where(clip == c)[0] for c in clips]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(clips), size=len(clips))
        idx = np.concatenate([idx_by_clip[j] for j in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(auroc_mw(y[idx], np.asarray(p)[idx]))
    vals = np.asarray(vals)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], int(len(vals)), vals


def loco_oof(X, y, clip, C):
    """LOCO: refit scaler+logistic on the other 6 clips for each held-out clip."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    p = np.zeros(len(y), dtype=np.float64)
    for c in sorted(set(np.asarray(clip).tolist())):
        te = np.asarray(clip) == c
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=7)
        clf.fit(sc.transform(X[tr]), y[tr])
        p[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def verdict(p):
    p = float(p)
    return "accept" if p > 0.5 else ("reject" if p < 0.4 else "uncertain")


# ------------------------------------------------------------------ input ---
def main():
    man = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json"), encoding="utf-8"))
    items = man["items"]
    assert len(items) == 150, len(items)
    ids = [it["id"] for it in items]
    y = np.array([int(it["label"]) for it in items], dtype=int)     # label 1 = good
    clip = np.array([it["clip"] for it in items])
    ens = json.load(open(os.path.join(BUNDLE, "caches", "ensemble", "ensemble_v1.json"), encoding="utf-8"))
    px_names = list(ens["views"]["pixel"]["feature_names"])
    assert abs(float(ens["views"]["pixel"]["C"]) - C_PX) < 1e-12
    assert abs(float(ens["views"]["eg2"]["C"]) - 0.003) < 1e-12

    win = json.load(open(os.path.join(BUNDLE, "caches", "gpu", "features.json"), encoding="utf-8"))["windows"]
    Xpx4 = np.array([[win[i]["prefix"][1][f] for f in px_names] for i in ids], dtype=np.float64)
    XpxF = np.array([[win[i]["prefix"][4][f] for f in px_names] for i in ids], dtype=np.float64)

    z = np.load(os.path.join(BUNDLE, "caches", "frames10k", "segments.npz"), allow_pickle=True)
    seg_ids = [str(s) for s in z["ids"]]
    assert sorted(seg_ids) == sorted(ids), "segment ids mismatch"
    order = [seg_ids.index(i) for i in ids]
    F = np.asarray(z["F"][order], dtype=np.float32)
    counts = np.asarray(z["counts"][order])

    def eg2_prefix(m):
        out = np.empty((len(ids), F.shape[2]), dtype=np.float32)
        for i in range(len(ids)):
            k = max(1, min(m, int(counts[i])))
            v = F[i, :k, :].mean(axis=0)
            nrm = float(np.linalg.norm(v))
            out[i] = v / nrm if nrm > 0 else v
        return out

    E4 = eg2_prefix(int(math.ceil(4.0 / 0.5)))    # 8 frames -> 4 s
    EF = eg2_prefix(21)                            # 21 frames -> full window

    # ------------------------------------------------------------ scoring ---
    p_px4 = loco_oof(Xpx4, y, clip, C_PX)
    p_pxF = loco_oof(XpxF, y, clip, C_PX)
    p_e4 = {k: loco_oof(E4, y, clip, c) for k, c in C_EG2.items()}
    p_eF = {k: loco_oof(EF, y, clip, c) for k, c in C_EG2.items()}
    p_f4 = {k: 0.5 * p_px4 + 0.5 * p_e4[k] for k in C_EG2}
    p_fF = {k: 0.5 * p_pxF + 0.5 * p_eF[k] for k in C_EG2}

    # AUROC agreement check: own MW vs sklearn (must be identical to <1e-12)
    checks = {}
    for nm, p in (("pixel26_4s", p_px4), ("eg2_4s_C0.003", p_e4["claimed_0.003"]),
                  ("fused_4s_C0.003", p_f4["claimed_0.003"])):
        a, b = auroc_mw(y, p), float(roc_auc_score(y, p))
        checks[nm] = {"auroc_mw": round(a, 10), "auroc_sklearn": round(b, 10),
                      "abs_diff": round(abs(a - b), 12)}
    assert all(v["abs_diff"] < 1e-9 for v in checks.values()), checks

    metrics = {}
    for nm, p in (("pixel26_4s", p_px4), ("pixel26_full", p_pxF),
                  ("eg2_2fps_4s_C0.003", p_e4["claimed_0.003"]),
                  ("eg2_2fps_4s_C0.01", p_e4["published_0.01"]),
                  ("eg2_2fps_full_C0.003", p_eF["claimed_0.003"]),
                  ("eg2_2fps_full_C0.01", p_eF["published_0.01"]),
                  ("fused_4s_C_eg2_0.003", p_f4["claimed_0.003"]),
                  ("fused_4s_C_eg2_0.01", p_f4["published_0.01"]),
                  ("fused_full_C_eg2_0.003", p_fF["claimed_0.003"]),
                  ("fused_full_C_eg2_0.01", p_fF["published_0.01"])):
        ci, used, _ = clip_bootstrap(y, p, clip)
        metrics[nm] = {"auroc": round(auroc_mw(y, p), 4), "auroc_ci95": [round(ci[0], 4), round(ci[1], 4)],
                       "bootstrap_reps_used": used, "n": int(len(y)),
                       "protocol": "LOCO 7 clips; clip-bootstrap 2000 resamples seed 20261010"}

    # paired bootstrap of the C sensitivity: fused(0.01) - fused(0.003)
    clips = sorted(set(clip.tolist()))
    idx_by_clip = [np.where(clip == c)[0] for c in clips]
    rng = np.random.default_rng(SEED)
    diffs_f4, diffs_e4 = [], []
    for _ in range(BOOT):
        pick = rng.integers(0, len(clips), size=len(clips))
        idx = np.concatenate([idx_by_clip[j] for j in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs_f4.append(auroc_mw(y[idx], p_f4["published_0.01"][idx]) - auroc_mw(y[idx], p_f4["claimed_0.003"][idx]))
        diffs_e4.append(auroc_mw(y[idx], p_e4["published_0.01"][idx]) - auroc_mw(y[idx], p_e4["claimed_0.003"][idx]))
    diffs_f4, diffs_e4 = np.asarray(diffs_f4), np.asarray(diffs_e4)
    c_sensitivity = {
        "fused_4s_delta_auroc_published_minus_claimed": round(metrics["fused_4s_C_eg2_0.01"]["auroc"]
                                                              - metrics["fused_4s_C_eg2_0.003"]["auroc"], 4),
        "fused_4s_delta_ci95": [round(float(np.percentile(diffs_f4, 2.5)), 4),
                                round(float(np.percentile(diffs_f4, 97.5)), 4)],
        "fused_4s_P_delta_lt_0": round(float(np.mean(diffs_f4 < 0)), 4),
        "eg2_4s_delta_auroc_published_minus_claimed": round(metrics["eg2_2fps_4s_C0.01"]["auroc"]
                                                            - metrics["eg2_2fps_4s_C0.003"]["auroc"], 4),
        "eg2_4s_delta_ci95": [round(float(np.percentile(diffs_e4, 2.5)), 4),
                              round(float(np.percentile(diffs_e4, 97.5)), 4)],
    }

    # ------------------------------------------------------------ verdicts ---
    # every-2nd grid: range(0, n, 2) inside each clip, manifest order
    sampled = np.zeros(len(y), dtype=bool)
    for c in clips:
        idxs = np.where(clip == c)[0]
        for j in range(0, len(idxs), 2):
            sampled[idxs[j]] = True
    assert int(sampled.sum()) == 77, int(sampled.sum())

    verdicts = {}
    for k in C_EG2:
        v_all = [verdict(p) for p in p_f4[k]]
        v_smp = [verdict(p) for p, s in zip(p_f4[k], sampled) if s]
        verdicts[k] = {
            "all_150": {t: v_all.count(t) for t in ("accept", "reject", "uncertain")},
            "every_2nd_grid_77": {t: v_smp.count(t) for t in ("accept", "reject", "uncertain")},
        }

    # clip verdicts (descriptive, both C)
    clip_rows = []
    for k in C_EG2:
        for c in clips:
            idxs = np.where(clip == c)[0]
            bad = [verdict(p_f4[k][i]) == "reject" for i in idxs if sampled[i]]
            frac = float(np.mean(bad)) if bad else float("nan")
            truth = float(np.mean(y[idxs] == 0))
            clip_rows.append({"C_eg2": C_EG2[k], "clip": c, "n_windows": int(len(idxs)),
                              "n_sampled": int(sampled[idxs].sum()),
                              "sampled_bad_fraction": round(frac, 4),
                              "clip_verdict": "reject" if frac >= 0.3 else "accept",
                              "truth_clip_verdict": "reject" if truth >= 0.3 else "accept"})
    for k in C_EG2:
        rows = [r for r in clip_rows if r["C_eg2"] == C_EG2[k]]
        verdicts[k]["clip_verdict_agreement"] = f"{sum(1 for r in rows if r['clip_verdict'] == r['truth_clip_verdict'])}/7"

    claim = {"fused_4s_auroc": 0.874, "fused_4s_ci95": [0.742, 0.942],
             "eg2_2fps_4s": 0.860, "eg2_2fps_4s_ci95": [0.715, 0.944],
             "pixel26_4s": 0.797, "pixel26_4s_ci95": [0.658, 0.860],
             "fused_full": 0.849, "verdicts": {"accept": 108, "reject": 27, "uncertain": 15},
             "sampled": 77}

    def close(a, b, tol=0.0006):
        return abs(a - b) <= tol

    reproduced = (close(metrics["fused_4s_C_eg2_0.003"]["auroc"], claim["fused_4s_auroc"])
                  and close(metrics["eg2_2fps_4s_C0.003"]["auroc"], claim["eg2_2fps_4s"])
                  and close(metrics["pixel26_4s"]["auroc"], claim["pixel26_4s"])
                  and close(metrics["fused_full_C_eg2_0.003"]["auroc"], claim["fused_full"])
                  and verdicts["claimed_0.003"]["all_150"] == claim["verdicts"]
                  and int(sampled.sum()) == claim["sampled"])

    out = {
        "verifier_branch": "b/verify-lb2",
        "verifier_git_sha": os.popen("git rev-parse HEAD").read().strip(),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "claim_under_test": "PR #92 (b/poc-integrate): " + json.dumps(claim, sort_keys=True),
        "independence": ("own script eval/poc_integrate/lb2_independent_recheck.py; does not import or re-run "
                         "eval/poc_integrate/run_integration.py; own Mann-Whitney AUROC cross-checked against "
                         "sklearn (<1e-9) and own bootstrap seed 20261010"),
        "n": 150,
        "n_clips": len(clips),
        "label_source": "human gold (single labeller, 100 good / 50 bad), labels/manifest_gold.json",
        "labels_sha256": man.get("labels_sha256"),
        "protocol": ("LOCO over the 7 IndEgo clips; StandardScaler+LogisticRegression refit per held-out clip; "
                     "eg2_2fps = L2-normalised mean of first 8 frames (4 s, 2 fps) or 21 frames (full); "
                     "pixel26 = features.json prefix[1] (4 s) / prefix[4] (full); fused = equal-weight probability "
                     "average; verdicts >0.5 accept, <0.4 reject else uncertain; clip-bootstrap 2000, seed 20261010"),
        "inputs": {"features": os.path.join(BUNDLE, "caches/gpu/features.json"),
                   "segments": os.path.join(BUNDLE, "caches/frames10k/segments.npz"),
                   "ensemble": os.path.join(BUNDLE, "caches/ensemble/ensemble_v1.json"),
                   "manifest": os.path.join(BUNDLE, "labels/manifest_gold.json")},
        "auroc_impl_check": checks,
        "C_used": {"pixel26": C_PX, "eg2_claimed": 0.003, "eg2_published": 0.01},
        "metrics": metrics,
        "c_sensitivity": c_sensitivity,
        "verdict_counts": verdicts,
        "grid": {"rule": "range(0, n_windows, 2) per clip, manifest order",
                 "n_sampled": int(sampled.sum()), "of": int(len(y))},
        "clip_verdicts": clip_rows,
        "reproduced_claim_branch": bool(reproduced),
        "published_reference": {
            "source": "caches/ensemble/ensemble_v1.json -> views.<view>.full_inner_oof_auroc_by_C",
            "eg2_full_inner_oof_by_C": ens["views"]["eg2"]["full_inner_oof_auroc_by_C"],
            "pixel_full_inner_oof_by_C": ens["views"]["pixel"]["full_inner_oof_auroc_by_C"],
            "note": ("published EG2 number 0.848 is at C=0.01 (0.8514 at C=0.003); published pixel 0.810 at C=0.3. "
                     "The 4 s prefix LOCO values above are re-derived at both C_eg2 values under the identical "
                     "protocol."),
        },
        "conclusion": ("reproduced" if reproduced else "NOT reproduced") + "; C_eg2 sensitivity "
                      + ("immaterial (delta = %.4f, CI %s)" % (
                          c_sensitivity["fused_4s_delta_auroc_published_minus_claimed"],
                          c_sensitivity["fused_4s_delta_ci95"])
                         if abs(c_sensitivity["fused_4s_delta_auroc_published_minus_claimed"]) <= 0.005
                         else "MATERIAL"),
        "preexisting_artifact_note": ("an untracked eval/results/pod_verify/lb2_recheck.json (sha256 "
                                      "6998f8ee1737826adc682d347a23bf8ff5ffe97d1dc59b5a0677e95758d657c3) and an "
                                      "untracked verify_lb2_recheck.py were present in this worktree before this "
                                      "run; they were NOT read for the computation and were overwritten by this "
                                      "file"),
        "regenerate": "OMP_NUM_THREADS=2 python eval/poc_integrate/lb2_independent_recheck.py",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("wrote", OUT)
    print("fused4 C0.003", metrics["fused_4s_C_eg2_0.003"]["auroc"], "| C0.01", metrics["fused_4s_C_eg2_0.01"]["auroc"])
    print("eg2_4s C0.003", metrics["eg2_2fps_4s_C0.003"]["auroc"], "| C0.01", metrics["eg2_2fps_4s_C0.01"]["auroc"])
    print("pixel4", metrics["pixel26_4s"]["auroc"], "| fusefull", metrics["fused_full_C_eg2_0.003"]["auroc"])
    print("verdicts", verdicts["claimed_0.003"], "sampled", int(sampled.sum()), "reproduced", reproduced)
    return 0


if __name__ == "__main__":
    sys.exit(main())
