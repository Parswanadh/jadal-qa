"""M3: determinism — 5 repeated trainings, sha256 of weights + window outputs.

Claims bit-identical selected-head weights and identical per-window outputs
ON THIS GPU. Nothing is claimed across devices/drivers.
Usage: python run_m3.py --device cuda --feature spec --target hands --epochs 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parents[1] / "results" / "gpu_heads"
OUT.mkdir(parents=True, exist_ok=True)

WIN = Path("/home/parshu/projects/jadal-qa-wt/frames-spectral/eval/results/frames_spectral/human_windows_raw.npz")


def setup(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--feature", choices=["spec", "both"], default="spec")
    ap.add_argument("--target", choices=["hands", "work"], default="hands")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    from eval.gpu_heads.run_m1 import get_xy
    from jadal.gpuheads.config import HeadConfig
    from jadal.gpuheads.train import train_groups
    from jadal.gpuheads.data import pool_window

    X, y, _ = get_xy(args.feature, args.target)
    cfg = HeadConfig(hidden=256, wd=1e-3, lr=0.05, seed=args.seed, class_weight=True)
    hashes, wouts = [], []
    d = np.load(str(WIN), allow_pickle=True)
    Xw = d["X"].astype(np.float32)
    for r in range(args.repeats):
        setup(args.seed)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        mu = X.mean(0)
        sd = X.std(0); sd[sd < 1e-8] = 1.0
        A = ((X - mu) / sd).astype(np.float32)
        mods, _, _ = train_groups(torch.from_numpy(A), torch.from_numpy(y.astype(np.float32)),
                                  {cfg.hidden: [cfg]}, X.shape[1], args.epochs, 2048,
                                  args.device, args.seed)
        m = mods[cfg.hidden]
        blob = b"".join(p.detach().cpu().numpy().tobytes() for p in m.parameters())
        hashes.append(hashlib.sha256(blob).hexdigest())
        with torch.no_grad():
            Tn = ((Xw.reshape(-1, Xw.shape[2]) - mu) / sd).astype(np.float32)
            lv = m(torch.from_numpy(Tn).to(args.device)).detach().cpu().numpy()[0]
        s = (1 / (1 + np.exp(-lv))).reshape(len(Xw), Xw.shape[1])
        wouts.append(np.array([pool_window(x) for x in s]))
    same_w = all(h == hashes[0] for h in hashes)
    same_o = all(np.array_equal(w, wouts[0]) for w in wouts)
    res = {"repeats": args.repeats, "feature": args.feature, "target": args.target,
           "device": args.device, "sha256": hashes,
           "bit_identical_weights": same_w, "identical_window_outputs": same_o,
           "not_claimed": "no claim across devices, drivers, or library versions",
           "vram_peak_GiB": (torch.cuda.max_memory_allocated() / 2**30) if args.device == "cuda" else 0.0}
    fn = OUT / f"m3_determinism_{args.feature}_{args.target}.json"
    json.dump(res, open(fn, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
