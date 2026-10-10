"""Per-config wall-cost micro-bench for claim X3.

The main run encodes all 12 configs in ONE batch per window, so per-config
embed ms cannot be attributed.  Here each config is built and encoded
SEPARATELY over a window sample (same batch size for every config) so we can
report honest ms/win for build (CPU, scales with canvas px) and EG2 embed (GPU).

Writes eval/results/exp_mosaic_res/cfg_timing.json.
Regen: bash tools/pod/gpu_run.sh python3 eval/exp_mosaic_res/bench_cfg.py --sample 48
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from jadal.mosaic import build as B  # noqa: E402
from eval.exp_mosaic_res.run_res import (mosaic_configs, load_windows,  # noqa: E402
                                         decode_window, native_fps, Embedder)

RESULTS = os.path.join(ROOT, "eval", "results", "exp_mosaic_res")
WINDOW_S = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=48)
    args = ap.parse_args()
    wins = load_windows()
    step = max(1, len(wins) // args.sample)
    wins = wins[::step][:args.sample]
    cfgs = mosaic_configs()
    print(f"timing sample {len(wins)} windows, {len(cfgs)} configs", flush=True)

    fps = {}
    decoded = []
    t0 = time.perf_counter()
    for w in wins:
        if w["path"] not in fps:
            fps[w["path"]] = native_fps(w["path"])
        nf = int(round(WINDOW_S * fps[w["path"]]))
        decoded.append((w, decode_window(w["path"], w["start_s"], nf)))
    decode_ms = (time.perf_counter() - t0) * 1000.0
    print(f"decode {decode_ms/len(wins):.0f} ms/win", flush=True)

    emb = Embedder()
    out = {"n_windows": len(wins), "decode_ms_per_win": decode_ms / len(wins),
           "configs": {}}
    for (tag, r, c, tp, ts, sh) in cfgs:
        t0 = time.perf_counter()
        imgs = [B.build_window(w["id"], w["source"], fr, fps[w["path"]],
                               n_tiles=r * c, tile_px=tp, timestamps=ts,
                               shuffle_seed=(1 if sh else None))[0]
                for (w, fr) in decoded]
        b_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        emb.encode(imgs)
        e_ms = (time.perf_counter() - t0) * 1000.0
        canvas = (c * tp + (c + 1) * 2) * (r * tp + (r + 1) * 2)
        out["configs"][tag] = {"build_ms_per_win": round(b_ms / len(wins), 1),
                               "embed_ms_per_win": round(e_ms / len(wins), 1),
                               "canvas_px": canvas, "frames": r * c,
                               "tile_px": tp}
        print(f"  {tag:18s} build {b_ms/len(wins):6.1f} embed {e_ms/len(wins):6.1f} "
              f"ms/win canvas {canvas}", flush=True)
    json.dump(out, open(os.path.join(RESULTS, "cfg_timing.json"), "w"), indent=1)
    print("wrote", os.path.join(RESULTS, "cfg_timing.json"))


if __name__ == "__main__":
    main()
