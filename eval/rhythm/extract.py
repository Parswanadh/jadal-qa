"""Extract the R representation for gold (+synthetic) windows into the cache.

Usage (sharded, one process per shard):
  python eval/rhythm/extract.py --kind gold --shard 0 --nshards 4 --G 6
  python eval/rhythm/extract.py --kind synth --G 6

Writes <cache>/<id>_G<G>.npz  (float16 X, g) and a manifest with digests.
Everything is deterministic: re-running overwrites with byte-identical tensors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from jadal.rhythm import features as F  # noqa: E402

GOLD = Path("/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json")
CACHE = Path("/home/parshu/projects/jadal-qa-wt/custom_algo_data/cache")


def load_gold():
    return json.load(open(GOLD))["items"]


def synth_from_tensor(X: np.ndarray, g: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic label-0 derivatives of a good window's tensor.

    Tensor-level approximations of the synthetic degradations used elsewhere in
    this project (blur / dark / frozen). They are TRUE BY CONSTRUCTION as
    negatives, but they are not pixel-exact re-renderings of the video.
    """
    X = X.copy()
    g = g.copy()
    if kind == "synth_blur":
        # blur removes spatial detail -> smooth the grid, cut the high-frequency
        # global channel, keep motion (a blurred clip still moves)
        for c in (0, 1, 4, 5):
            ch = X[:, :, :, c]
            sm = ch.copy()
            sm[:, 1:, :] = 0.5 * (ch[:, 1:, :] + ch[:, :-1, :])
            sm[:, :, 1:] = 0.5 * (sm[:, :, 1:] + sm[:, :, :-1])
            X[:, :, :, c] = sm
        g[:, 1] *= 0.15   # lap_var collapses
        g[:, 7] *= 0.25   # dct_hf collapses
    elif kind == "synth_dark":
        X[:, :, :, 0] *= 0.2
        X[:, :, :, 1] *= 0.2
        X[:, :, :, 3] *= 0.2
        g[:, 0] *= 0.25   # luma_mean collapses
        g[:, 1] *= 0.15   # lap_var falls with brightness (known artefact)
        g[:, 5] = np.clip(g[:, 5] + 0.55, 0, 1)  # under_exposed_frac rises
        g[:, 7] *= 0.3
    elif kind == "synth_frozen":
        X[:] = X[:1]      # first frame repeated for the whole window
        g[:] = g[:1]
    else:
        raise ValueError(kind)
    return X, g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["gold", "synth"], default="gold")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--no-ego", action="store_true", help="ablation: no ego-motion compensation")
    ap.add_argument("--no-flow", action="store_true", help="ablation: no optical-flow channels")
    ap.add_argument("--no-skin", action="store_true", help="ablation: zero the skin chromaticity channel")
    ap.add_argument("--tag", default="", help="suffix for the cache subdir (ablations)")
    args = ap.parse_args()

    cache_dir = args.cache / (args.tag or "main")
    cache_dir.mkdir(parents=True, exist_ok=True)

    items = load_gold()
    if args.kind == "gold":
        todo = [it for i, it in enumerate(items) if i % args.nshards == args.shard]
    else:
        # synthetic derivatives are built from GOOD windows only
        good = [it for it in items if it["label"] == 1]
        kinds = ["synth_blur", "synth_dark", "synth_frozen"]
        todo = []
        for i, it in enumerate(good):
            if i % args.nshards != args.shard:
                continue
            for k in kinds:
                todo.append({**it, "id": f"{it['id']}_{k}", "kind": k, "base": it["id"]})

    manifest = []
    t_start = time.time()
    for n, it in enumerate(todo):
        t0 = time.time()
        out = F.extract_window(it["path"], it["start_s"], it["end_s"], G=args.G,
                               ego_compensate=not args.no_ego, use_flow=not args.no_flow)
        X, g = out["X"], out["g"]
        if args.no_skin:
            X[:, :, :, 2] = 0.0   # ablation: skin_frac channel removed
        if it.get("kind"):
            X, g = synth_from_tensor(X, g, it["kind"])
        dig = F.tensor_digest(X, g)
        F.save_cache(cache_dir, it["id"], {"X": X, "g": g, "fps": out["fps"], "T": out["T"]}, args.G)
        manifest.append({
            "id": it["id"], "clip": it["clip"], "label": int(it["label"]),
            "reason": it.get("reason", ""), "kind": it.get("kind", "real"),
            "base": it.get("base"), "T": out["T"], "fps": out["fps"],
            "digest": dig, "seconds": round(time.time() - t0, 3),
        })
        if (n + 1) % 10 == 0:
            print(f"  [{args.kind} s{args.shard}] {n+1}/{len(todo)} "
                  f"({time.time()-t_start:.0f}s)", flush=True)

    mpath = cache_dir / f"manifest_{args.kind}_shard{args.shard}.json"
    json.dump({"kind": args.kind, "shard": args.shard, "nshards": args.nshards,
               "G": args.G, "ego_compensate": not args.no_ego, "use_flow": not args.no_flow,
               "tag": args.tag, "n": len(manifest), "items": manifest},
              open(mpath, "w"), indent=1)
    print(f"[{args.kind} s{args.shard}] done {len(manifest)} windows in "
          f"{time.time()-t_start:.0f}s -> {mpath}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
