"""Independent recompute of the frames10k EG2 frame-head -> human-window transfer.

Reads only:
  segments.npz  (150 human windows x <=21 EG2 frame embeddings, 768-d, fp16)
  frame_heads.used.json      (8k lean heads, C=1)
  frame_heads.used_10k.json  (10k lean heads, C=1)
and re-implements the protocol described in eval/frames10k/analyze_transfer.py
(which is NOT imported). Outputs JSON with AUROCs and both the published
clip-bootstrap and a proper cluster bootstrap.

CPU only, OMP_NUM_THREADS=2.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "_artifacts")
C_GRID = (0.01, 0.1, 1.0, 10.0)


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def head_prob(Fn, h):
    Xs = (Fn - np.asarray(h["mean"])) / np.asarray(h["scale"])
    return sigmoid(Xs @ np.asarray(h["coef"]) + h["intercept"])


def pool_feats(P):
    feats = []
    for row in P:
        v = row[~np.isnan(row)]
        if len(v) == 0:
            feats.append([0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
        else:
            feats.append([v.mean(), v.max(), np.percentile(v, 25), np.percentile(v, 75),
                          (v > 0.5).mean(), v.var()])
    return np.asarray(feats)


def loco_oof(X, y, clips):
    """Outer leave-one-clip-out; inner LOCO on train clips picks C (same rules)."""
    clips = np.asarray(clips)
    oof = np.zeros(len(y))
    for held in sorted(set(clips)):
        te = np.where(clips == held)[0]
        tr = np.where(clips != held)[0]
        tclips = clips[tr]
        best_c, best = 1.0, -1.0
        for c in C_GRID:
            aucs = []
            for h2 in sorted(set(tclips)):
                itr = tr[tclips != h2]
                ite = tr[tclips == h2]
                if len(np.unique(y[ite])) < 2 or len(np.unique(y[itr])) < 2:
                    continue
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(C=c, max_iter=3000,
                                                       class_weight="balanced"))
                clf.fit(X[itr], y[itr])
                try:
                    aucs.append(roc_auc_score(y[ite], clf.predict_proba(X[ite])[:, 1]))
                except ValueError:
                    pass
            m = float(np.mean(aucs)) if aucs else 0.5
            if m > best:
                best, best_c = m, c
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=best_c, max_iter=3000,
                                               class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def clip_boot_published(y, s, clips, n_boot=2000, seed=0):
    """Exactly the published loop: resample the 7 clips, union their windows."""
    rng = np.random.RandomState(seed)
    uclips = np.asarray(sorted(set(clips)))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uclips, size=len(uclips), replace=True)
        idx = np.where(np.isin(clips, pick))[0]
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals.append(roc_auc_score(y[idx], s[idx]))
        except ValueError:
            pass
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def clip_boot_proper(y, s, clips, n_boot=2000, seed=0):
    """Cluster bootstrap: resample clips and duplicate all their windows."""
    rng = np.random.RandomState(seed)
    uclips = np.asarray(sorted(set(clips)))
    idx_by_clip = {c: np.where(clips == c)[0] for c in uclips}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uclips, size=len(uclips), replace=True)
        idx = np.concatenate([idx_by_clip[c] for c in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    z = np.load(os.path.join(D, "segments.npz"), allow_pickle=True)
    F = z["F"].astype(np.float64)
    clips = np.asarray([str(x) for x in z["clips"]])
    y = z["labels"].astype(int)
    counts = z["counts"]
    Fn = np.nan_to_num(F, nan=0.0)
    mask = ~np.isnan(F[:, :, 0])
    print(f"n={len(y)} good={int(y.sum())} bad={int((1-y).sum())} T={F.shape[1]} "
          f"counts min/max={counts.min()}/{counts.max()} mean={counts.mean():.3f}")

    seg_mean = np.array([Fn[i, :counts[i]].mean(axis=0) for i in range(len(y))])
    seg_mean = seg_mean / np.maximum(np.linalg.norm(seg_mean, axis=1, keepdims=True), 1e-12)

    out = {"n": int(len(y)), "n_good": int(y.sum()), "n_bad": int((1 - y).sum()),
           "n_clips": int(len(set(clips))), "T": int(F.shape[1]),
           "frames_mean_per_window": float(counts.mean()),
           "runs": {}}

    for tag, hfile in (("10k_heads", "frame_heads.used_10k.json"),
                       ("8k_heads", "frame_heads.used.json")):
        heads = json.load(open(os.path.join(D, hfile)))["heads"]
        Pw = head_prob(Fn, heads["active_labor"])
        Ph = head_prob(Fn, heads["hands_ok"])
        Pw = np.where(mask, Pw, np.nan)
        Ph = np.where(mask, Ph, np.nan)
        Pw_0 = np.where(mask, head_prob(Fn, heads["active_labor"]), np.nan)
        Fp = np.concatenate([pool_feats(Pw), pool_feats(Ph)], axis=1)
        arms = {"pretrained_pooled": Fp,
                "human_only_embed": seg_mean,
                "combo": np.concatenate([Fp, seg_mean], axis=1)}
        res = {"head_file": hfile, "arms": {}}
        # single-feature reference AUROCs (no fitting)
        res["single_feature_ref"] = {
            "active_labor_pool_mean": float(roc_auc_score(y, np.nan_to_num(Pw, nan=0.5)[:, :].mean(axis=1))),
            "active_labor_per_frame_mean_of_scores": float(roc_auc_score(y, np.nanmean(Pw, axis=1))),
            "hands_ok_per_frame_mean_of_scores": float(roc_auc_score(y, np.nanmean(Ph, axis=1))),
        }
        for name, X in arms.items():
            oof = loco_oof(X, y, clips)
            auc = float(roc_auc_score(y, oof))
            m, lo, hi = clip_boot_published(y, oof, clips)
            plo, phi = clip_boot_proper(y, oof, clips)
            res["arms"][name] = {
                "dim": int(X.shape[1]),
                "oof_auroc": auc,
                "balacc@0.5": float(balanced_accuracy_score(y, oof > 0.5)),
                "acc@0.5": float(accuracy_score(y, oof > 0.5)),
                "clip_boot_published_mean": m,
                "clip_boot_published_ci95": [lo, hi],
                "cluster_boot_proper_ci95": [plo, phi],
            }
            print(f"[{tag}] {name}: AUROC={auc:.4f} pubCI=[{lo:.3f},{hi:.3f}] "
                  f"properCI=[{plo:.3f},{phi:.3f}]")
        out["runs"][tag] = res

    OUT = os.environ.get("OUT", os.path.join(os.getcwd(), "recompute_frame_heads.json"))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
