"""M1: frame-level nested 5-fold stratified CV over batched heads.

Deterministic: torch.manual_seed, deterministic algorithms, CUBLAS config
expected in env, fixed fold seeds, fixed batch order, float32.
Nested selection happens INSIDE training folds only.
Usage: python run_m1.py --feature both --target hands --epochs 30 --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

OUT = Path(__file__).resolve().parents[2] / "results" / "gpu_heads"
OUT.mkdir(parents=True, exist_ok=True)


def setup_determinism(seed=0):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def get_xy(feature, target):
    from jadal.gpuheads.data import (load_frame_eg2, load_spectral, align,
                                      standardize_fit, standardize_apply)
    spec = load_spectral()
    eg2 = load_frame_eg2()
    if target == "hands":
        y_spec = (spec["hand"] >= 1).astype(np.int64)
    else:
        y_spec = (spec["work"] == 1).astype(np.int64)
    if feature == "spec":
        return spec["X"].astype(np.float32), y_spec, ["spec53"]
    assert eg2 is not None, "no EG2 shards yet"
    if target == "hands":
        y_eg = (eg2["hand"] >= 1).astype(np.int64)
    else:
        y_eg = (eg2["work"] == 1).astype(np.int64)
    ia, ib = align(eg2["ids"], spec["ids"])
    assert len(ia) > 0, "id overlap is empty"
    # sanity: AI labels agree on overlap?
    agree_h = float(((eg2["hand"][ia] >= 1) == (spec["hand"][ib] >= 1)).mean())
    if feature == "eg2":
        return eg2["emb"][ia], y_eg[ia], ["eg2-768"]
    Xb = np.concatenate([eg2["emb"][ia], spec["X"][ib]], axis=1)
    return Xb.astype(np.float32), y_eg[ia], ["eg2-768+spec53"]


def nested_cv(X, y, cfgs, epochs, device, seed, n_outer=5):
    from jadal.gpuheads.data import standardize_fit, standardize_apply
    from jadal.gpuheads.train import train_groups
    from jadal.gpuheads.metrics import auroc as auc, balanced_acc as ba, f1 as f1m
    from jadal.gpuheads.config import HeadConfig
    torch.manual_seed(seed)
    skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    # group flat configs by hidden
    by_h: dict[int, list] = {}
    for c in cfgs:
        by_h.setdefault(c.hidden, []).append(c)
    order_h = sorted(by_h)
    bounds = {}
    o = 0
    for h in order_h:
        bounds[h] = (o, o + len(by_h[h]))
        o += bounds[h][1] - bounds[h][0]
    outer_pred, outer_true, chosen = [], [], []
    t_train = 0.0
    n_heads_trained = 0
    for fold, (tri, tei) in enumerate(skf.split(X, y)):
        # inner 2-fold selection inside outer-train only
        inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed + fold)
        scores = np.zeros(len(cfgs))
        for itri, ivi in inner.split(X[tri], y[tri]):
            mu = X[tri][itri].mean(axis=0)
            sd = X[tri][itri].std(axis=0); sd[sd < 1e-8] = 1.0
            A = (X[tri][itri] - mu) / sd
            V = (X[tri][ivi] - mu) / sd
            Xa = torch.from_numpy(A.astype(np.float32))
            ya = torch.from_numpy(y[tri][itri].astype(np.float32))
            mods, _, dt = train_groups(Xa, ya, by_h, X.shape[1], epochs, 2048, device, seed + fold)
            t_train += dt
            n_heads_trained += len(cfgs)
            with torch.no_grad():
                Xv = torch.from_numpy(V.astype(np.float32)).to(device)
                for h, m in mods.items():
                    a, b = bounds[h]
                    lv = m(Xv).detach().cpu().numpy()  # (K,Nv)
                    for k in range(b - a):
                        try:
                            s = 1 / (1 + np.exp(-lv[k]))
                            scores[a + k] += auc(y[tri][ivi], s) / 2.0
                        except Exception:
                            scores[a + k] += 0.0
            del mods
            if device == "cuda":
                torch.cuda.empty_cache()
        best = int(np.argmax(scores))
        # refit best config on full outer-train
        mu = X[tri].mean(axis=0); sd = X[tri].std(axis=0); sd[sd < 1e-8] = 1.0
        A = (X[tri] - mu) / sd; T = (X[tei] - mu) / sd
        cfg = cfgs[best]
        mods, _, dt = train_groups(torch.from_numpy(A.astype(np.float32)),
                                   torch.from_numpy(y[tri].astype(np.float32)),
                                   {cfg.hidden: [cfg]}, X.shape[1], epochs * 2, 4096, device, seed)
        t_train += dt
        n_heads_trained += 1
        with torch.no_grad():
            Xt = torch.from_numpy(T.astype(np.float32)).to(device)
            lv = mods[cfg.hidden](Xt).detach().cpu().numpy()[0]
            s = 1 / (1 + np.exp(-lv))
        outer_pred.append(s)
        outer_true.append(y[tei])
        chosen.append({"fold": fold, "hidden": cfg.hidden, "wd": cfg.wd,
                       "lr": cfg.lr, "seed": cfg.seed, "class_weight": cfg.class_weight,
                       "inner_auc": float(scores[best])})
        del mods
        if device == "cuda":
            torch.cuda.empty_cache()
    P = np.concatenate(outer_pred)
    Y = np.concatenate(outer_true)
    res = {"auroc": auc(Y, P), "bal_acc": ba(Y, P), "f1": f1m(Y, P),
           "n": int(len(Y)), "n_pos": int(Y.sum()), "chosen": chosen,
           "t_train_s": t_train, "n_heads_trained": n_heads_trained,
           "heads_per_s": (n_heads_trained / t_train) if t_train > 0 else 0.0}
    return res, Y, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", choices=["eg2", "spec", "both"], default="both")
    ap.add_argument("--target", choices=["hands", "work"], default="hands")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hidden", type=int, default=256,
                    help="cap hidden grid (1024 excluded if 256 for speed)")
    args = ap.parse_args()
    setup_determinism(args.seed)
    from jadal.gpuheads.config import build_grid
    hiddens = [0, 64, 256] + ([1024] if args.max_hidden >= 1024 else [])
    cfgs = build_grid(hiddens=hiddens)
    print(f"configs={len(cfgs)} feature={args.feature} target={args.target} device={args.device}", flush=True)
    X, y, fnames = get_xy(args.feature, args.target)
    print(f"N={len(y)} D={X.shape[1]} pos_rate={y.mean():.3f} files={fnames}", flush=True)
    if args.device == "cuda":
        assert torch.cuda.is_available()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res, Y, P = nested_cv(X, y, cfgs, args.epochs, args.device, args.seed)
    wall = time.time() - t0
    from jadal.gpuheads.metrics import bootstrap_ci_auroc
    lo, hi = bootstrap_ci_auroc(Y, P)
    res.update({"feature": args.feature, "target": args.target, "label_source": "AI (EG2 worker)",
                "protocol": "5-fold stratified CV, nested selection inside train folds",
                "auroc_ci95": [lo, hi], "wall_s": wall,
                "vram_peak_GiB": (torch.cuda.max_memory_allocated() / 2**30) if args.device == "cuda" else 0.0,
                "epochs": args.epochs, "seed": args.seed})
    # single-logistic baseline (same outer folds, sklearn, train-only scaling)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    Pb = np.zeros_like(P)
    for tri, tei in skf.split(X, y):
        mu = X[tri].mean(axis=0); sd = X[tri].std(axis=0); sd[sd < 1e-8] = 1.0
        clf = LogisticRegression(max_iter=1000)
        clf.fit((X[tri] - mu) / sd, y[tri])
        Pb[tei] = clf.predict_proba((X[tei] - mu) / sd)[:, 1]
    from jadal.gpuheads.metrics import auroc as auc
    res["single_logistic_auroc"] = auc(Y, Pb)
    res["delta_many_minus_single"] = res["auroc"] - res["single_logistic_auroc"]
    fn = OUT / f"m1_{args.feature}_{args.target}.json"
    json.dump(res, open(fn, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("auroc", "auroc_ci95", "bal_acc", "f1", "single_logistic_auroc",
                                          "delta_many_minus_single", "n", "n_pos", "wall_s",
                                          "vram_peak_GiB", "heads_per_s")}, indent=1), flush=True)
    print(f"wrote {fn}", flush=True)


if __name__ == "__main__":
    main()
