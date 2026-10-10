"""Train the rhythm ensemble under one outer leave-one-clip-out fold.

Protocol (see docs/reports/rhythm.md and ALGO_SPEC.md):
  outer : hold out ONE clip; train on the other 6.
  inner : inside those 6, leave one out at a time to score each ensemble member,
          then keep the top-M members (selection never touches the held-out clip).
  output: per-window logits from the selected members' mean, for the held-out clip.

Everything is deterministic: fixed seeds, deterministic algorithms where
available, no sampling at inference.

Usage
  python eval/rhythm/train.py --fold 0 --K 64 --top-m 32 --out eval/results/rhythm/fold0.json
  python eval/rhythm/train.py --all --K 64                      # all 7 folds

Ablation flags: --G, --no-ego, --no-skin, --no-aug, --K 1, --no-synth,
--tag <name> (writes results under a tagged filename so runs never overwrite).

WHAT IS NOT DONE:
  * No early stopping on a separate validation loss; members are selected by
    inner-LOCO AUROC and trained for a fixed number of epochs.
  * No learning-rate search; the schedule is fixed by the config.
  * BatchNorm running statistics are computed on augmented batches, so the
    inference path sees slightly different statistics than training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from jadal.rhythm import features as F  # noqa: E402
from jadal.rhythm.model import RhythmEnsemble, count_params  # noqa: E402

CACHE = Path("/home/parshu/projects/jadal-qa-wt/custom_algo_data/cache/main")
GOLD = Path("/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json")
REASON_TO_AUX = {"NO_ACTIVE_WORK": 0, "HANDS_NOT_VISIBLE": 1}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_gold(G: int = 6, tag: str = ""):
    """Load cached tensors for all gold windows -> dict of arrays."""
    cache = CACHE.parent / (tag or "main")
    items = json.load(open(GOLD))["items"]
    Xs, gs, y, clips, reasons, ids = [], [], [], [], [], []
    for it in items:
        c = F.load_cache(cache, it["id"], G)
        if c is None:
            continue
        Xs.append(c["X"])
        gs.append(c["g"])
        y.append(int(it["label"]))
        clips.append(it["clip"])
        reasons.append(it.get("reason", ""))
        ids.append(it["id"])
    return {"X": Xs, "g": gs, "y": np.array(y, np.int64), "clip": np.array(clips),
            "reason": np.array(reasons), "id": np.array(ids)}


def pad_to(Xs, gs, T: int):
    """Stack variable-length windows into (n,T,G,G,C) and (n,T,8), truncating/padding."""
    n = len(Xs)
    G = Xs[0].shape[1]
    C = Xs[0].shape[3]
    X = np.zeros((n, T, G, G, C), np.float32)
    g = np.zeros((n, T, 8), np.float32)
    for i, (a, b) in enumerate(zip(Xs, gs)):
        t = min(len(a), T)
        X[i, :t] = a[:t]
        g[i, :t] = b[:t]
        if t < T:  # repeat the last frame rather than leaving zeros
            X[i, t:] = a[t - 1]
            g[i, t:] = b[t - 1]
    return X, g


def synth_negatives(data, T: int, kinds=("synth_blur", "synth_dark", "synth_frozen")):
    """Tensor-level label-0 derivatives of the GOOD windows (true by construction).

    These are NOT pixel-exact re-renderings of degraded video; they are the same
    approximations used in eval/rhythm/extract.py. They add quality negatives so
    the model does not learn "any dark window is fine".
    """
    Xs, gs, y, clips, reasons, ids = [], [], [], [], [], []
    for i in range(len(data["X"])):
        if data["y"][i] != 1:
            continue
        for k in kinds:
            X = data["X"][i].copy()
            g = data["g"][i].copy()
            if k == "synth_blur":
                for c in (0, 1, 4, 5):
                    ch = X[:, :, :, c]
                    sm = ch.copy()
                    sm[:, 1:, :] = 0.5 * (ch[:, 1:, :] + ch[:, :-1, :])
                    sm[:, :, 1:] = 0.5 * (sm[:, :, 1:] + sm[:, :, :-1])
                    X[:, :, :, c] = sm
                g[:, 1] *= 0.15
                g[:, 7] *= 0.25
            elif k == "synth_dark":
                X[:, :, :, 0] *= 0.2
                X[:, :, :, 1] *= 0.2
                X[:, :, :, 3] *= 0.2
                g[:, 0] *= 0.25
                g[:, 1] *= 0.15
                g[:, 5] = np.clip(g[:, 5] + 0.55, 0, 1)
                g[:, 7] *= 0.3
            elif k == "synth_frozen":
                X[:] = X[:1]
                g[:] = g[:1]
            Xs.append(X)
            gs.append(g)
            y.append(0)
            clips.append(data["clip"][i])  # stays in the same clip -> same fold
            reasons.append("SYNTHETIC")
            ids.append(f"{data['id'][i]}_{k}")
    return {"X": Xs, "g": gs, "y": np.array(y, np.int64), "clip": np.array(clips),
            "reason": np.array(reasons), "id": np.array(ids)}


def subset(data, mask):
    return {"X": [data["X"][i] for i in np.where(mask)[0]],
            "g": [data["g"][i] for i in np.where(mask)[0]],
            "y": data["y"][mask], "clip": data["clip"][mask],
            "reason": data["reason"][mask], "id": data["id"][mask]}


# --------------------------------------------------------------------------- #
# augmentation (all on GPU, label-preserving)
# --------------------------------------------------------------------------- #
def augment(X: torch.Tensor, g: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """X (K,B,T,G,G,C), g (K,B,T,8) -> augmented copies.

    Each member gets its OWN independent augmentation: the K dimension is a real
    batch dimension, so this is where the ensemble gets its diversity for free.

    Index arithmetic note: a (K,B,T) integer index tensor selecting into a
    flattened (K*B*T, ...) array must be offset by k*(B*T) + b*T, NOT (k*B+b)*T.
    """
    K, B, T, G, _, C = X.shape
    dev = X.device
    kb = (torch.arange(K, device=dev)[:, None, None] * (B * T) +
          torch.arange(B, device=dev)[None, :, None] * T)

    # temporal crop 60-100% then resample back to T (nearest within the crop)
    if cfg.get("crop", True):
        t_keep = torch.clamp((torch.empty(K, B, device=dev).uniform_(0.6, 1.0) * T).long(),
                             min=max(2, T // 3))
        idx = (torch.rand(K, B, T, device=dev) * t_keep[:, :, None].float()).long().clamp(max=T - 1)
        flat = (idx + kb).reshape(-1)
        X = X.reshape(K * B * T, G, G, C)[flat].reshape(K, B, T, G, G, C)
        g = g.reshape(K * B * T, 8)[flat].reshape(K, B, T, 8)

    # time reversal
    if cfg.get("reverse", True):
        flip = (torch.rand(K, B, device=dev) < 0.5)
        X = torch.where(flip[:, :, None, None, None, None], X.flip(2), X)
        g = torch.where(flip[:, :, None, None], g.flip(2), g)

    # speed jitter: linear resample of the time axis by a random factor
    if cfg.get("speed", True):
        sp = torch.empty(K, B, device=dev).uniform_(0.8, 1.25)
        src = (torch.arange(T, device=dev).float()[None, None, :] * sp[:, :, None]).clamp(0, T - 1)
        lo = src.floor().long()
        hi = (lo + 1).clamp(max=T - 1)
        w = (src - lo.float())                      # (K,B,T)
        idx_lo = (lo + kb).reshape(-1)
        idx_hi = (hi + kb).reshape(-1)
        wf = w.reshape(-1, 1)
        Xf = X.reshape(K * B * T, G * G * C)
        X = (Xf[idx_lo] * (1 - wf) + Xf[idx_hi] * wf).reshape(K, B, T, G, G, C)
        gf = g.reshape(K * B * T, 8)
        g = (gf[idx_lo] * (1 - wf) + gf[idx_hi] * wf).reshape(K, B, T, 8)

    # horizontal flip of the grid (framing is arbitrary)
    if cfg.get("hflip", True):
        flip = (torch.rand(K, B, device=dev) < 0.5)
        X = torch.where(flip[:, :, None, None, None, None], X.flip(3), X)

    # random cell dropout: mask whole grid cells
    if cfg.get("cell_drop", True):
        keep = (torch.rand(K, B, G, G, device=dev) > 0.15).float()
        X = X * keep[:, :, None, :, :, None]

    # additive noise on the residual channels
    if cfg.get("noise", True):
        X = X + torch.randn_like(X) * 0.01

    # synthetic camera shake injected into the residual channels
    if cfg.get("shake", True):
        sh = torch.randn(K, B, T, 1, 1, 2, device=dev).abs() * 0.05
        X = X.clone()
        X[..., 0:1] = (X[..., 0:1] + sh[..., 0:1]).clamp(0, 1)
        X[..., 4:5] = (X[..., 4:5] + sh[..., 1:2]).clamp(0, 1)
    return X, g


def mixup(X, g, y, alpha: float, rng: torch.Generator | None = None):
    """Within-batch mixup. Labels are mixed with the same lambda."""
    K, B = X.shape[0], X.shape[1]
    lam = np.random.beta(alpha, alpha, size=(K, B)).astype(np.float32)
    lam_t = torch.as_tensor(lam, device=X.device)
    perm = torch.randperm(B, device=X.device)
    Xm = X * lam_t[:, :, None, None, None, None] + X[:, perm] * (1 - lam_t[:, :, None, None, None, None])
    gm = g * lam_t[:, :, None, None] + g[:, perm] * (1 - lam_t[:, :, None, None])
    ym = y * lam_t + y[:, perm] * (1 - lam_t)
    return Xm, gm, ym


# --------------------------------------------------------------------------- #
# training one ensemble
# --------------------------------------------------------------------------- #
def train_ensemble(Xtr, gtr, ytr, aux_tr, cfg, seed: int = 0):
    """Train K members in parallel. Returns (model, per-member inner scores)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = cfg["K"]
    model = RhythmEnsemble(K=K, grid_ch=cfg.get("grid_ch", 6), glob_ch=8,
                           chans=tuple(cfg.get("chans", (8, 12, 16))),
                           glob_chans=tuple(cfg.get("glob_chans", (8, 8))),
                           kernel=cfg.get("kernel", 5),
                           dilations=tuple(cfg.get("dilations", (1, 2, 4))),
                           head_hidden=cfg.get("head_hidden", 32),
                           dropout=cfg.get("dropout", 0.1)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-3),
                            weight_decay=cfg.get("wd", 1e-2))
    epochs = cfg.get("epochs", 60)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = len(ytr)
    Xtr_t = torch.as_tensor(Xtr, device=dev)
    gtr_t = torch.as_tensor(gtr, device=dev)
    ytr_t = torch.as_tensor(ytr, device=dev, dtype=torch.float32)
    aux_t = torch.as_tensor(aux_tr, device=dev, dtype=torch.float32)
    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=dev)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    ls = cfg.get("label_smooth", 0.05)
    aug_cfg = cfg.get("aug", {})

    bs = min(cfg.get("batch", 16), n)
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            b = len(idx)
            # replicate the batch across members so each member augments its own copy
            Xb = Xtr_t[idx].unsqueeze(0).repeat(K, *([1] * Xtr_t.dim())).contiguous()
            gb = gtr_t[idx].unsqueeze(0).repeat(K, *([1] * gtr_t.dim())).contiguous()
            yb = ytr_t[idx][None, :].expand(K, b)
            ab = aux_t[idx][None, :].expand(K, b, -1)
            if aug_cfg:
                Xb, gb = augment(Xb, gb, aug_cfg)
            if cfg.get("mixup", 0.0) > 0:
                Xb, gb, yb = mixup(Xb, gb, yb, cfg["mixup"])
            yb = yb * (1 - ls) + 0.5 * ls
            logit, aux_out = model(Xb, gb)
            loss = bce(logit, yb)
            if ab.shape[-1] and cfg.get("aux_weight", 0.0) > 0:
                m = ~torch.isnan(ab)
                if m.any():
                    loss = loss + cfg["aux_weight"] * torch.nn.functional.binary_cross_entropy_with_logits(
                        aux_out[m], ab[m])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, X, g, batch: int = 32):
    """Returns (K, n) logits."""
    dev = next(model.parameters()).device
    K = model.K
    n = len(X)
    out = np.zeros((K, n), np.float32)
    for s in range(0, n, batch):
        idx = slice(s, min(s + batch, n))
        b = len(X[idx])
        Xb = torch.as_tensor(X[idx], device=dev).unsqueeze(0).repeat(K, *([1] * X[idx].ndim)).contiguous()
        gb = torch.as_tensor(g[idx], device=dev).unsqueeze(0).repeat(K, *([1] * g[idx].ndim)).contiguous()
        lo, _ = model(Xb, gb)
        out[:, idx] = lo.cpu().numpy()
    return out


def auroc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    npos = int((y == 1).sum())
    nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None, help="outer clip index (0-6)")
    ap.add_argument("--folds", default=None, help="comma-separated outer clip indices, e.g. 0,3,6")
    ap.add_argument("--all", action="store_true", help="run all 7 outer folds")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--top-m", type=int, default=None, help="members kept (default K//2)")
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-ego", action="store_true")
    ap.add_argument("--no-aug", action="store_true", help="ablation: disable all augmentation")
    ap.add_argument("--no-synth", action="store_true")
    ap.add_argument("--single", action="store_true",
                    help="ablation: K=1 (one member). Keeps augmentation unless --no-aug is also given.")
    args = ap.parse_args()

    # `--single` only changes the member count; augmentation is controlled solely
    # by `--no-aug`. The two are separate axes of the ablation matrix.
    cfg = {
        "K": 1 if args.single else args.K,
        "epochs": args.epochs,
        "lr": 3e-3, "wd": 1e-2, "dropout": 0.1, "label_smooth": 0.05,
        "mixup": 0.0 if (args.no_aug or args.single) else 0.2,
        "aux_weight": 0.3, "batch": args.batch,
        "aug": {} if args.no_aug else
               {"crop": True, "reverse": True, "speed": True, "hflip": True,
                "cell_drop": True, "noise": True, "shake": True},
    }
    # mixup needs more than one member to have anything to mix across in the
    # per-member sense; keep it only for real ensembles.
    if cfg["K"] == 1:
        cfg["mixup"] = 0.0
    tag = args.tag or ("main" if not args.no_ego else "noego")

    data = load_gold(G=args.G, tag=tag)
    if not data["y"].size:
        print("ERROR: no cached windows found. Run eval/rhythm/extract.py first.", file=sys.stderr)
        return 2
    if not args.no_synth:
        syn = synth_negatives(data, T=cfg.get("T", 100))
        for k in ("X", "g"):
            data[k] = data[k] + syn[k]
        data["y"] = np.concatenate([data["y"], syn["y"]])
        data["clip"] = np.concatenate([data["clip"], syn["clip"]])
        data["reason"] = np.concatenate([data["reason"], syn["reason"]])
        data["id"] = np.concatenate([data["id"], syn["id"]])

    clips = sorted(set(data["clip"].tolist()))
    if args.folds:
        folds = [int(v) for v in args.folds.split(",") if v.strip() != ""]
    else:
        folds = list(range(len(clips))) if args.all else [args.fold if args.fold is not None else 0]
    top_m = args.top_m or max(1, cfg["K"] // 2)
    T = int(np.median([a.shape[0] for a in data["X"]]))
    T = min(T, 100)

    results = []
    for fo in folds:
        held = clips[fo]
        t0 = time.time()
        tr_mask = data["clip"] != held
        te_mask = data["clip"] == held
        Xtr, gtr = pad_to([data["X"][i] for i in np.where(tr_mask)[0]],
                          [data["g"][i] for i in np.where(tr_mask)[0]], T)
        Xte, gte = pad_to([data["X"][i] for i in np.where(te_mask)[0]],
                          [data["g"][i] for i in np.where(te_mask)[0]], T)
        ytr = data["y"][tr_mask]
        yte = data["y"][te_mask]
        aux_tr = np.full((len(ytr), 2), np.nan, np.float32)
        for i, r in enumerate(data["reason"][tr_mask]):
            if r in REASON_TO_AUX:
                aux_tr[i, REASON_TO_AUX[r]] = 1.0

        model = train_ensemble(Xtr, gtr, ytr, aux_tr, cfg, seed=1000 + fo)
        ptr_all = predict(model, Xtr, gtr)     # (K, n_train)
        pte_all = predict(model, Xte, gte)     # (K, n_test)

        # member selection by inner-LOCO over the TRAINING clips only
        inner_clips = sorted(set(data["clip"][tr_mask].tolist()))
        inner_scores = np.zeros(cfg["K"], np.float32)
        for ic in inner_clips:
            va = data["clip"][tr_mask] == ic
            if len(np.unique(ytr[va])) < 2:
                continue
            s = auroc(ytr[va], ptr_all[:, va].mean(0))
            if not math.isnan(s):
                inner_scores += s / max(len(inner_clips), 1)
        keep = np.argsort(-inner_scores)[:top_m]
        pte = pte_all[keep].mean(0)

        results.append({
            "fold": fo, "held_out_clip": held, "n_test": int(te_mask.sum()),
            "n_bad_test": int((yte == 0).sum()),
            "auroc": auroc(yte, pte),
            "members_kept": int(len(keep)),
            "inner_scores_mean": float(np.nanmean(inner_scores)),
            "seconds": round(time.time() - t0, 1),
            "ids": data["id"][te_mask].tolist(),
            "y": yte.tolist(),
            "reason": data["reason"][te_mask].tolist(),
            "pred": [round(float(v), 4) for v in pte],
        })
        print(f"fold {fo} held={held:11s} n={int(te_mask.sum()):3d} "
              f"bad={int((yte==0).sum()):2d} AUROC={results[-1]['auroc']:.3f} "
              f"({results[-1]['seconds']}s)", flush=True)

    out = Path(args.out) if args.out else Path(f"eval/results/rhythm/fold_{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": cfg, "G": args.G, "top_m": top_m, "T": T,
        "params": count_params(RhythmEnsemble(K=cfg["K"])),
        "label_source": "human gold (150 windows, 7 clips, one labeller) + synthetic-by-construction negatives",
        "protocol": "leave-one-clip-out; members selected by inner-LOCO on training clips only",
        "command": " ".join(sys.argv),
        "folds": results,
        "mean_auroc": float(np.nanmean([r["auroc"] for r in results])),
    }
    json.dump(payload, open(out, "w"), indent=1)
    print(f"wrote {out}  mean fold AUROC {payload['mean_auroc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
