"""M2: frame heads -> every window frame -> pooled window features -> LOCO head.

Label honesty: frame heads use AI labels only; window head uses HUMAN gold only.
Frame-head config is selected by inner CV on frame data (never on windows).
LOCO: leave-one-clip-out over the 7 gold clips; clip-bootstrap CI for AUROC.
Usage: python run_m2.py --device cpu --epochs 30
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

WIN = Path("/home/parshu/projects/jadal-qa-wt/frames-spectral/eval/results/frames_spectral/human_windows_raw.npz")
OUT = Path(__file__).resolve().parents[1] / "results" / "gpu_heads"
OUT.mkdir(parents=True, exist_ok=True)


def setup(seed=0):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_final_head(Xf, yf, cfgs, epochs, device, seed):
    """Inner 2-fold selection on frame data only; refit winner on all frames."""
    from jadal.gpuheads.data import standardize_apply
    from jadal.gpuheads.train import train_groups
    from jadal.gpuheads.metrics import auroc as auc
    by_h: dict[int, list] = {}
    for c in cfgs:
        by_h.setdefault(c.hidden, []).append(c)
    order = sorted(by_h)
    bounds, o = {}, 0
    for h in order:
        bounds[h] = (o, o + len(by_h[h]))
        o += bounds[h][1] - bounds[h][0]
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    scores = np.zeros(len(cfgs))
    for itri, ivi in skf.split(Xf, yf):
        mu = Xf[itri].mean(0)
        sd = Xf[itri].std(0); sd[sd < 1e-8] = 1.0
        A = ((Xf[itri] - mu) / sd).astype(np.float32)
        V = ((Xf[ivi] - mu) / sd).astype(np.float32)
        mods, _, _ = train_groups(torch.from_numpy(A), torch.from_numpy(yf[itri].astype(np.float32)),
                                  by_h, Xf.shape[1], epochs, 2048, device, seed)
        with torch.no_grad():
            Xv = torch.from_numpy(V).to(device)
            for h, m in mods.items():
                a, b = bounds[h]
                lv = m(Xv).detach().cpu().numpy()
                for k in range(b - a):
                    try:
                        scores[a + k] += auc(yf[ivi], sigmoid(lv[k])) / 2.0
                    except Exception:
                        pass
        del mods
        if device == "cuda":
            torch.cuda.empty_cache()
    best = int(np.argmax(scores))
    cfg = cfgs[best]
    mu = Xf.mean(0)
    sd = Xf.std(0); sd[sd < 1e-8] = 1.0
    A = ((Xf - mu) / sd).astype(np.float32)
    mods, _, dt = train_groups(torch.from_numpy(A), torch.from_numpy(yf.astype(np.float32)),
                               {cfg.hidden: [cfg]}, Xf.shape[1], epochs * 2, 4096, device, seed)
    return cfg, mods[cfg.hidden], mu, sd, float(scores[best]), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", choices=["hands", "work"], default="hands")
    args = ap.parse_args()
    setup(args.seed)
    from jadal.gpuheads.config import build_grid
    from jadal.gpuheads.data import load_spectral, pool_window
    from jadal.gpuheads.metrics import auroc as auc, balanced_acc as ba, f1 as f1m, bootstrap_ci_auroc

    spec = load_spectral()
    yf = ((spec["hand"] >= 1) if args.target == "hands" else (spec["work"] == 1)).astype(np.int64)
    Xf = spec["X"].astype(np.float32)
    cfgs = build_grid(hiddens=[0, 64, 256], seeds=[0, 1])
    t0 = time.time()
    cfg, mod, mu, sd, inner_auc, dt = train_final_head(Xf, yf, cfgs, args.epochs, args.device, args.seed)
    print(f"frame-head: hidden={cfg.hidden} wd={cfg.wd} cw={cfg.class_weight} inner_auc={inner_auc:.4f} train_s={dt:.1f}", flush=True)

    d = np.load(str(WIN), allow_pickle=True)  # trusted local file; reasons is object array
    Xw = d["X"].astype(np.float32)  # (150, T, 53)
    ids = d["ids"].astype(str)
    clips = d["clips"].astype(str)
    yl = d["labels"].astype(np.int64)
    reasons = d["reasons"].astype(str) if "reasons" in d.files else np.array([""] * len(yl))
    T = Xw.shape[1]
    t1 = time.time()
    with torch.no_grad():
        mods = mod.to(args.device)
        Tn = (Xw.reshape(-1, Xw.shape[2]) - mu) / sd
        logits = mods(torch.from_numpy(Tn.astype(np.float32)).to(args.device)).detach().cpu().numpy()[0]
    t_score = time.time() - t1
    S = sigmoid(logits).reshape(len(yl), T)  # (150, T) frame usable-ish scores
    P = np.array([pool_window(s) for s in S])  # (150, 9)
    print(f"scored {len(yl)} windows x {T} frames in {t_score:.2f}s ({t_score/len(yl)*1000:.1f} ms/window)", flush=True)

    # LOCO over clips
    uclips = sorted(set(clips.tolist()))
    pred = np.zeros(len(yl))
    for c in uclips:
        te = np.where(clips == c)[0]
        tr = np.where(clips != c)[0]
        clf = LogisticRegression(max_iter=2000)
        clf.fit(P[tr], yl[tr])
        pred[te] = clf.predict_proba(P[te])[:, 1]
    ci_lo, ci_hi = bootstrap_ci_auroc(yl, pred, n_boot=1000, seed=args.seed)
    # clip-bootstrap CI: resample clips
    rng = np.random.default_rng(args.seed)
    cvals = []
    for _ in range(1000):
        samp = rng.choice(uclips, len(uclips), replace=True)
        idx = np.where(np.isin(clips, samp))[0]
        if len(np.unique(yl[idx])) < 2:
            continue
        try:
            from sklearn.metrics import roc_auc_score
            cvals.append(roc_auc_score(yl[idx], pred[idx]))
        except Exception:
            pass
    c_lo, c_hi = ((float(np.quantile(cvals, 0.025)), float(np.quantile(cvals, 0.975))) if cvals else (float("nan"), float("nan")))
    res = {"n_windows": int(len(yl)), "n_clips": len(uclips),
           "n_usable": int(yl.sum()), "n_bad": int((1 - yl).sum()),
           "label_source": "human gold (150 windows)",
           "protocol": "LOCO over 7 clips; frame head selected on AI frame labels only",
           "frame_head": {"hidden": cfg.hidden, "wd": cfg.wd, "lr": cfg.lr,
                          "seed": cfg.seed, "class_weight": cfg.class_weight,
                          "frame_target": args.target, "inner_auc": inner_auc},
           "auroc": auc(yl, pred), "auroc_ci95_segboot": [ci_lo, ci_hi],
           "auroc_ci95_clipboot": [float(c_lo), float(c_hi)],
           "bal_acc": ba(yl, pred), "f1": f1m(yl, pred),
           "per_clip": [], "per_reason_recall": {}, "wall_s": time.time() - t0,
           "ms_per_window": t_score / len(yl) * 1000}
    for c in uclips:
        i = np.where(clips == c)[0]
        try:
            a = auc(yl[i], pred[i]) if len(np.unique(yl[i])) == 2 else None
        except Exception:
            a = None
        res["per_clip"].append({"clip": c, "n": int(len(i)), "n_usable": int(yl[i].sum()),
                                "auroc": a, "bal_acc": ba(yl[i], pred[i])})
    # per-reason recall at 0.5 (bad windows with pred<0.5 = correctly rejected)
    for r in sorted(set(reasons.tolist())):
        if not r:
            continue
        i = np.where(reasons == r)[0]
        res["per_reason_recall"][r] = {"n": int(len(i)), "recall_bad": float((pred[i] < 0.5).mean())}
    # per-reason heads: reason-vs-usable LOCO AUROC where computable
    res["per_reason_heads"] = {}
    for r in ["HANDS_NOT_VISIBLE", "NO_ACTIVE_WORK", "TOO_DARK", "DUPLICATE"]:
        i = np.where((reasons == r) | (yl == 1))[0]
        yy = (yl[i] == 1).astype(int)  # 1=usable, 0=this-reason
        if yy.sum() == 0 or (1 - yy).sum() < 2:
            res["per_reason_heads"][r] = {"n_reason": int((1 - yy).sum()), "note": "too few for LOCO"}
            continue
        pr = np.zeros(len(i))
        ok = True
        for c in sorted(set(clips[i].tolist())):
            te = np.where(clips[i] == c)[0]
            tr = np.where(clips[i] != c)[0]
            if len(np.unique(yy[tr])) < 2:
                ok = False
                break
            clf = LogisticRegression(max_iter=2000)
            clf.fit(P[i][tr], yy[tr])
            pr[te] = clf.predict_proba(P[i][te])[:, 1]
        try:
            a = auc(yy, pr) if ok else None
        except Exception:
            a = None
        res["per_reason_heads"][r] = {"n_reason": int((1 - yy).sum()), "n_usable": int(yy.sum()),
                                      "auroc_reason_vs_usable": a}
    fn = OUT / f"m2_windows_{args.target}.json"
    json.dump(res, open(fn, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("auroc", "auroc_ci95_clipboot", "bal_acc", "f1",
                                          "per_reason_recall", "per_reason_heads")}, indent=1), flush=True)
    print(f"wrote {fn}", flush=True)


if __name__ == "__main__":
    main()
