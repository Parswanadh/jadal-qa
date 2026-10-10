"""KEY TEST: pooled frame-head outputs -> HUMAN usable/bad label (CPU).

Compares, under leave-one-clip-out CV (7 IndEgo clips; segments of a clip
never in train+test):
  (i)   pretrained: logistic on 12 pooled frame-head feats (mean/max/p25/p75/
        frac>0.5/var x {p_work, p_hands}), heads trained on 10k AI labels
  (ii)  human-only: logistic on 768-d segment-mean EG2 embedding (LOCO)
  (iii) combo: 12 pooled + 768-d embedding
Metrics: AUROC on LOCO out-of-fold scores + clip-bootstrap 95% CI, balanced
accuracy, accuracy. Per-bad-reason one-vs-rest AUROC (usable vs each reason).
C picked by inner LOCO on train clips only. class_weight=balanced.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[2] / "eval" / "results" / "frames10k"
SEG = RES / "human_segments" / "segments.npz"


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def pool_feats(P):
    """P: (n_seg, T) with nan pad -> (n,6): mean,max,p25,p75,frac>0.5,var."""
    feats = []
    for row in P:
        v = row[~np.isnan(row)]
        if len(v) == 0:
            feats.append([0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
            continue
        feats.append([v.mean(), v.max(), np.percentile(v, 25), np.percentile(v, 75),
                      (v > 0.5).mean(), v.var()])
    return np.asarray(feats)


def clip_boot_ci(y, s, clips, n_boot=2000, seed=0):
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(seed)
    uclips = np.asarray(sorted(set(clips)))
    vals = []
    y = np.asarray(y)
    s = np.asarray(s)
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


def loco_oof(X, y, clips):
    """Leave-one-clip-out OOF scores; inner LOCO on train clips picks C."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clips = np.asarray(clips)
    uclips = sorted(set(clips))
    oof = np.zeros(len(y))
    for held in uclips:
        te = np.where(clips == held)[0]
        tr = np.where(clips != held)[0]
        tclips = clips[tr]
        best_c, best = 1.0, -1.0
        from sklearn.metrics import roc_auc_score

        for c in (0.01, 0.1, 1.0, 10.0):
            aucs = []
            for h2 in sorted(set(tclips)):
                itr = tr[tclips != h2]
                ite = tr[tclips == h2]
                if len(np.unique(y[ite])) < 2 or len(np.unique(y[itr])) < 2:
                    continue
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))
                clf.fit(X[itr], y[itr])
                try:
                    aucs.append(roc_auc_score(y[ite], clf.predict_proba(X[ite])[:, 1]))
                except ValueError:
                    pass
            m = float(np.mean(aucs)) if aucs else 0.5
            if m > best:
                best, best_c = m, c
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def main() -> None:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    z = np.load(SEG, allow_pickle=True)
    F = z["F"].astype(np.float64)  # (150, T, 768), nan-padded
    ids = [str(x) for x in z["ids"]]
    clips = np.asarray([str(x) for x in z["clips"]])
    y = z["labels"].astype(int)
    reasons = np.asarray([str(x) for x in z["reasons"]])
    counts = z["counts"]
    print(f"segments={len(y)} usable={int(y.sum())} bad={int((1-y).sum())} "
          f"clips={sorted(set(clips))} T={F.shape[1]} mean_frames={counts.mean():.1f}", flush=True)

    # frame-head probabilities per frame
    heads = json.load(open(RES / "egocentric_10k" / "frame_heads.json"))["heads"]
    Fn = np.nan_to_num(F, nan=0.0)
    P = {}
    for name in ("active_labor", "hands_ok"):
        h = heads[name]
        Xs = (Fn - np.asarray(h["mean"])) / np.asarray(h["scale"])
        P[name] = sigmoid(Xs @ np.asarray(h["coef"]) + h["intercept"])
    mask = ~np.isnan(F[:, :, 0])
    Pw = np.where(mask, P["active_labor"], np.nan)
    Ph = np.where(mask, P["hands_ok"], np.nan)

    Fp = np.concatenate([pool_feats(Pw), pool_feats(Ph)], axis=1)  # (150,12)
    seg_mean = np.array([Fn[i, :counts[i]].mean(axis=0) for i in range(len(y))])
    nrm = np.linalg.norm(seg_mean, axis=1, keepdims=True)
    seg_mean = seg_mean / np.maximum(nrm, 1e-12)

    arms = {"pretrained_pooled": Fp,
            "human_only_embed": seg_mean,
            "combo": np.concatenate([Fp, seg_mean], axis=1)}
    rep = {"n": len(y), "n_good": int(y.sum()), "n_bad": int((1 - y).sum()),
           "clips": sorted(set(clips.tolist())),
           "label_source": "human (single labeller; manifest_gold snapshot)",
           "protocol": "leave-one-clip-out CV; C in {0.01,0.1,1,10} via inner LOCO on train clips; class_weight=balanced",
           "frame_heads_source": "AI labels (gemini-2.5-flash) on 8000 factory frames (lean C=1 heads)"}
    for name, X in arms.items():
        oof = loco_oof(X, y, clips)
        auc = roc_auc_score(y, oof)
        m, lo, hi = clip_boot_ci(y, oof, clips)
        ba = balanced_accuracy_score(y, oof > 0.5)
        acc = accuracy_score(y, oof > 0.5)
        rep[name] = {"oof_auroc": float(auc), "clip_boot_mean": m, "clip_boot_ci95": [lo, hi],
                     "balacc": float(ba), "acc": float(acc), "dim": int(X.shape[1])}
        print(f"{name}: AUROC={auc:.3f} (clip-boot {m:.3f} [{lo:.3f},{hi:.3f}]) "
              f"balacc={ba:.3f} acc={acc:.3f}", flush=True)
        # per-bad-reason: usable (1) vs each reason
        per = {}
        for r in sorted(set(reasons[y == 0].tolist())):
            sel = (y == 1) | (reasons == r)
            try:
                a = roc_auc_score(y[sel], oof[sel])
            except ValueError:
                a = float("nan")
            per[r] = {"auroc": float(a), "n_bad": int(((reasons == r)).sum())}
        rep[name]["per_reason"] = per
        print(f"   per-reason: " + ", ".join(f"{k}={v['auroc']:.3f}(n={v['n_bad']})" for k, v in per.items()), flush=True)
    (RES / "human_segments" / "transfer.json").write_text(json.dumps(rep, indent=1))
    print("wrote transfer.json")


if __name__ == "__main__":
    main()
