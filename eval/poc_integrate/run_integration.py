"""LB2 -- apply the adopted PoC policy to the cached scorer path.

CPU-only, deterministic, no GPU, no network.  Reads:

- ``labels/manifest_gold.json``            150 human-labelled IndEgo windows (READ-ONLY)
- ``caches/gpu/features.json``             26 per-pixel features, full + 2/4/6/8/10 s prefixes
- ``caches/frames10k/segments.npz``        EmbeddingGemma-2 frame embeddings (2 fps), 150 windows
- ``caches/ensemble/ensemble_v1.json``     frozen view hyperparameters (C, fusion, thresholds)
- ``caches/ensemble/ood_scores.json``      60 OOD-labelled windows, frozen full-window scores
- ``labels/ood_labels.json``               60 human OOD labels (READ-ONLY)

Writes (under ``eval/results/poc_integrated/``):

- ``window_verdicts.jsonl``  one line per window (150 IndEgo gold + 60 OOD, OOD descriptive only)
- ``summary.json``           top-level n / label_source / protocol / descriptive_only + metrics

Protocol (IndEgo): leave-one-clip-out (LOCO) over the 7 clips.  For every
held-out clip the two logistic views are re-fitted on the other 6 clips at the
*observed* prefix (4 s), so no window is ever scored by a model that saw its
own clip.  Views are fused by equal-weight probability average and thresholded
with the frozen verdict rule.  95% CIs are clip-bootstrap (2000 resamples,
seed 7).  All numbers are DESCRIPTIVE: the E3.05 selection is a working policy
on IndEgo, not validated, and the OOD rows are not an accuracy claim.

Regenerate:
  OMP_NUM_THREADS=2 python eval/poc_integrate/run_integration.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from jadal.poc_stream import policy  # noqa: E402

SEED = 7
BOOT = 2000
DEFAULT_OUT = os.path.join(REPO_ROOT, "eval", "results", "poc_integrated")

POLICY_LABEL = "every-2-window grid + 4s early exit + EG2-2fps + pixel26-logistic (frozen-v1)"
GOLD_PROTOCOL = (
    "LOCO over the 7 IndEgo clips (clip held out entirely; heads re-fitted on the other 6 "
    "clips at the observed prefix); clip-bootstrap 95% CI (2000 resamples, seed 7); "
    "frozen ensemble_v1 hyperparameters/fusion/thresholds; equal-weight probability average"
)
OOD_PROTOCOL = (
    "frozen full-window ensemble scores read from caches/ensemble/ood_scores.json "
    "(no cached features/embeddings for these 60 windows in the pod bundle, so the "
    "adopted 4 s prefix policy could NOT be re-applied to them)"
)


def _sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def load_gold(bundle):
    with open(os.path.join(bundle, "labels", "manifest_gold.json"), "r", encoding="utf-8") as fh:
        man = json.load(fh)
    items = man["items"]
    if len(items) != 150:
        raise ValueError(f"expected 150 gold items, got {len(items)}")
    return man, items


def load_features(bundle):
    with open(os.path.join(bundle, "caches", "gpu", "features.json"), "r", encoding="utf-8") as fh:
        feats = json.load(fh)
    return feats["windows"]


def load_segments(bundle):
    z = np.load(os.path.join(bundle, "caches", "frames10k", "segments.npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}


def build_matrices(items, windows, seg, ens):
    """Return feature matrices aligned to the manifest order."""
    ids = [it["id"] for it in items]
    missing = [i for i in ids if i not in windows]
    if missing:
        raise ValueError(f"features.json missing gold windows: {missing[:5]}")
    px_names = list(ens["views"]["pixel"]["feature_names"])
    pi = policy.pixel_prefix_index()
    Xpx4 = np.array([[windows[i]["prefix"][pi][f] for f in px_names] for i in ids], dtype=np.float64)
    Xpxfull = np.array([[windows[i]["prefix"][4][f] for f in px_names] for i in ids], dtype=np.float64)

    seg_ids = [str(s) for s in seg["ids"]]
    if sorted(seg_ids) != sorted(ids):
        raise ValueError("segments.npz ids do not match the 150 gold windows")
    order = [seg_ids.index(i) for i in ids]
    F = seg["F"][order]
    counts = seg["counts"][order]
    E4 = policy.eg2_prefix_mean(F, counts)  # 4 s -> first 8 frames @2 fps
    Efull = policy.eg2_prefix_mean(F, counts, exit_s=10.5)
    return Xpx4, Xpxfull, E4, Efull


def loco_oof(X, y, clip, C):
    """Leave-one-clip-out out-of-fold probabilities (deterministic)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    clip = np.asarray(clip)
    p = np.zeros(len(y), dtype=np.float64)
    for c in sorted(set(clip.tolist())):
        te = clip == c
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=SEED)
        clf.fit(sc.transform(X[tr]), y[tr])
        p[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def auroc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def clip_bootstrap_ci(y, p, clip, n_boot=BOOT, seed=SEED):
    """Percentile 95% CI resampling clips with replacement (pooled windows)."""
    y = np.asarray(y)
    p = np.asarray(p)
    clip = np.asarray(clip)
    clips = sorted(set(clip.tolist()))
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(clips), size=len(clips), replace=True)
        idx = np.concatenate([np.where(clip == clips[j])[0] for j in pick])
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        vals.append(auroc(yy, p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)], len(vals)


def balanced_acc(y, p, thr=0.5):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, (np.asarray(p) > thr).astype(int)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=policy.DEFAULT_BUNDLE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--boot", type=int, default=BOOT)
    args = ap.parse_args()
    if not args.bundle:
        ap.error("no pod bundle: set JADAL_BUNDLE or pass --bundle")

    os.makedirs(args.out, exist_ok=True)
    ens = policy.load_frozen_ensemble(args.bundle)
    man, items = load_gold(args.bundle)
    windows = load_features(args.bundle)
    seg = load_segments(args.bundle)
    Xpx4, Xpxfull, E4, Efull = build_matrices(items, windows, seg, ens)

    y = np.array([it["label"] for it in items], dtype=int)
    clip = np.array([it["clip"] for it in items])
    C_px = float(ens["views"]["pixel"]["C"])
    C_e2 = float(ens["views"]["eg2"]["C"])

    p_px4 = loco_oof(Xpx4, y, clip, C_px)
    p_e24 = loco_oof(E4, y, clip, C_e2)
    p_pxfull = loco_oof(Xpxfull, y, clip, C_px)
    p_e2full = loco_oof(Efull, y, clip, C_e2)
    p_fuse4 = policy.fuse_probabilities(p_px4, p_e24)
    p_fusefull = policy.fuse_probabilities(p_pxfull, p_e2full)

    metrics = {}
    for name, p in (("pixel26_4s", p_px4), ("eg2_2fps_4s", p_e24), ("fused_4s", p_fuse4),
                    ("pixel26_full", p_pxfull), ("eg2_2fps_full", p_e2full), ("fused_full", p_fusefull)):
        ci, nb = clip_bootstrap_ci(y, p, clip, n_boot=args.boot)
        metrics[name] = {
            "auroc": round(auroc(y, p), 4),
            "auroc_ci95": [round(ci[0], 4), round(ci[1], 4)],
            "balanced_accuracy_at_0.5": round(balanced_acc(y, p), 4),
            "n": int(len(y)),
            "label_source": "human gold (single labeller), manifest_gold.json",
            "protocol": GOLD_PROTOCOL,
            "bootstrap_reps_used": nb,
        }

    # ---- per-window verdicts + every-2nd sampling grid (per clip, manifest order) ----
    verdicts4 = [policy.window_verdict(v) for v in p_fuse4]
    order_pos = {}
    for i, it in enumerate(items):
        order_pos.setdefault(it["clip"], []).append(i)
    sampled = np.zeros(len(items), dtype=bool)
    for c, idxs in order_pos.items():
        for j in policy.sampled_indices(len(idxs)):
            sampled[idxs[j]] = True

    clip_rows = []
    for c, idxs in order_pos.items():
        bad = [verdicts4[i] == "reject" for i in idxs if sampled[i]]
        frac = float(np.mean(bad)) if bad else float("nan")
        truth_frac = float(np.mean([y[i] == 0 for i in idxs]))  # label 0 = bad in the gold manifest
        clip_rows.append({
            "clip": c,
            "n_windows": len(idxs),
            "n_sampled": int(sum(sampled[i] for i in idxs)),
            "sampled_bad_fraction": round(frac, 4) if bad else None,
            "clip_verdict": policy.clip_verdict(bad),
            "truth_bad_fraction": round(truth_frac, 4),
            "truth_clip_verdict": "reject" if truth_frac >= policy.ADOPTED["clip_bad_fraction_threshold"] else "accept",
        })
    clip_verdict_hits = sum(1 for r in clip_rows if r["clip_verdict"] == r["truth_clip_verdict"])

    # ---- OOD rows: frozen full-window scores, descriptive only ----
    with open(os.path.join(args.bundle, "labels", "ood_labels.json"), "r", encoding="utf-8") as fh:
        ood_labels = json.load(fh)
    latest = ood_labels["latest"]
    with open(os.path.join(args.bundle, "caches", "ensemble", "ood_scores.json"), "r", encoding="utf-8") as fh:
        ood_scores = json.load(fh)
    frozen_by_id = {s["id"]: s for s in ood_scores["scores"]}
    ood_rows = []
    ood_counts = {"accept": 0, "reject": 0, "uncertain": 0}
    for wid, lab in latest.items():
        s = frozen_by_id.get(wid)
        if s is None:
            continue
        v = {"accept": "accept", "reject": "reject", "review": "uncertain"}.get(s["verdict"], "uncertain")
        ood_counts[v] += 1
        ood_rows.append({
            "window_id": wid, "clip": None, "label_domain": "Egocentric-10K-OOD-human",
            "human_label": lab.get("verdict"), "human_reason": lab.get("reason"),
            "chosen_views": ["eg2_2fps", "pixel26"], "observed_seconds": 10.0,
            "score": s["usable_prob"], "p_pixel": s.get("p_pixel"), "p_eg2": s.get("p_eg2"),
            "verdict": v, "policy": "frozen ensemble_v1 full-window score (adopted 4 s prefix NOT applicable: no cached features)",
            "score_protocol": OOD_PROTOCOL, "sampled": None, "descriptive_only": True,
            "claim": "none - descriptive only, not an accuracy claim",
        })

    # ---- write window_verdicts.jsonl ----
    out_jsonl = os.path.join(args.out, "window_verdicts.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for i, it in enumerate(items):
            rec = {
                "window_id": it["id"], "clip": it["clip"], "label_domain": "IndEgo-gold",
                "human_label": int(y[i]), "human_reason": it.get("reason"),
                "chosen_views": list(policy.ADOPTED["views"]),
                "observed_seconds": policy.sampled_seconds_per_window(),
                "score": round(float(p_fuse4[i]), 6),
                "p_pixel": round(float(p_px4[i]), 6), "p_eg2": round(float(p_e24[i]), 6),
                "verdict": verdicts4[i], "policy": POLICY_LABEL,
                "score_protocol": "LOCO-OOF (the window's clip was held out of both view fits)",
                "sampled": bool(sampled[i]),
                "window_index_in_clip": order_pos[it["clip"]].index(i),
                "descriptive_only": True,
            }
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        for rec in ood_rows:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    counts = {v: verdicts4.count(v) for v in ("accept", "reject", "uncertain")}
    summary = {
        "descriptive_only": True,
        "claim": "none - descriptive policy integration on IndEgo gold; OOD rows are not an accuracy claim",
        "n": int(len(items)),
        "label_source": "human gold (single labeller, 100 good / 50 bad), manifest_gold.json",
        "labels_sha256": man.get("labels_sha256"),
        "protocol": GOLD_PROTOCOL,
        "policy": policy.policy_fingerprint(),
        "policy_label": POLICY_LABEL,
        "device": "CPU only (torch.cuda not used; cached features)",
        "deterministic": True,
        "frozen_ensemble_sha256_verified": policy.FROZEN_PREREG_SHA256,
        "frozen_ensemble_path": policy.FROZEN_PREREG_REL,
        "independent_verification": {
            "source": "LB4 independent verification, PR #98",
            "result": "REPRODUCED: fused 4 s AUROC 0.8736 at C_eg2=0.003 vs 0.8734 at the published C_eg2=0.01; paired clip-bootstrap delta -0.0002, CI [-0.0133, +0.0095]; C-insensitive (+/-0.0002)",
        },
        "framing": {
            "do_not_read_as_improvement": "fused_4s AUROC is NOT comparable to the published 0.848: that is a full-window inner-OOF number (the same table gives 0.8514 at C=0.003), not a like-for-like 4 s-LOCO comparator",
            "c_sensitivity": "C-insensitive (+/-0.0002) per independent verification PR #98",
        },
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _sh(["git", "rev-parse", "HEAD"]),
        "metrics": metrics,
        "verdict_counts": counts,
        "verdict_counts_note": "counts over all 150 gold windows under the 4 s LOCO policy; 'sampled' in the JSONL marks the every-2nd grid",
        "n_sampled_windows": int(sampled.sum()),
        "sampled_fraction": round(float(sampled.mean()), 4),
        "clip_verdicts": clip_rows,
        "clip_verdict_agreement": f"{clip_verdict_hits}/{len(clip_rows)}",
        "clip_verdict_note": "descriptive only; E3.01/E3.05 quote 7/7 on label-oracle clip error, this is the model-based count",
        "cost": {
            "every_2_speedup": policy.cost_ratio("every_2_speedup"),
            "every_2_4s_speedup": policy.cost_ratio("every_2_4s_speedup"),
            "every_2_4s_total_frac": policy.ADOPTED["cost"]["every_2_4s_total_frac"],
            "source": "E3.04 (arithmetic on E3.01/E3.03 fractions; decode-bound upper bound)",
        },
        "ood_descriptive": {
            "n": len(ood_rows),
            "n_bad": int(sum(1 for r in ood_rows if r["human_label"] == "bad")),
            "label_source": "human OOD labels, ood_labels.json (READ-ONLY)",
            "protocol": OOD_PROTOCOL,
            "verdict_counts": ood_counts,
            "claim": "none - descriptive only; no OOD/accuracy claim",
        },
        "limitations": [
            "E3.05 status is 'working policy on IndEgo, not validated' (7 clips, sparse gold windows)",
            "OOD 4 s prefix could not be scored: no cached features/embeddings for the 60 OOD windows",
            "clip-bootstrap CIs assume clips are exchangeable; 7 clips only, pallet2 has 6 windows",
            "balanced accuracy at 0.5 is descriptive; no certified error bounds at n=150",
        ],
        "regenerate": "OMP_NUM_THREADS=2 python eval/poc_integrate/run_integration.py",
    }
    out_sum = os.path.join(args.out, "summary.json")
    with open(out_sum, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
        fh.write("\n")

    print(f"wrote {out_jsonl} ({len(items)} gold + {len(ood_rows)} OOD rows)")
    print(f"wrote {out_sum}")
    print("fused 4s AUROC", metrics["fused_4s"]["auroc"], metrics["fused_4s"]["auroc_ci95"],
          "| pixel", metrics["pixel26_4s"]["auroc"], "| eg2", metrics["eg2_2fps_4s"]["auroc"])
    print("verdicts", counts, "| sampled", int(sampled.sum()), "| clip agreement", summary["clip_verdict_agreement"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
