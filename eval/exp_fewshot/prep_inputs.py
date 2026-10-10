#!/usr/bin/env python3
"""Build the 160-window per-pixel feature input (10 fps-effective view).

Batch-1 (60 windows, 30 fps source transcodes): reuse the frozen
``ood_pixel_selfdec.npz`` ``X_10fps`` column (every 3rd frame of the 160x90
ffmpeg decode, fps=10 -> 100 frames / 10 s).

Batch-2 (100 windows, native 10 fps transcodes): decode with the SAME frozen
recipe (``jadal.ensemble.scorer.decode_pixel_window`` 160x90 rgb24 ffmpeg) and
compute ``jadal.gpu.features.cpu_window_features(luma, rgb, fps=10.0)`` on all
100 frames (also 100 frames / 10 s). This makes the batch-2 window identical in
regime to batch-1's 10 fps-effective column.

Output (git-ignored, ``*.npz``):
  eval/results/exp_fewshot/inputs/ood160_pixel.npz
    {ids (160,), y (160,), factory (160,), batch (160,), names (26,),
     X10 (160,26)}

Run:  OMP_NUM_THREADS=2 python3 eval/exp_fewshot/prep_inputs.py [--jobs 2]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from eval.exp_fewshot import o2lib as L  # noqa: E402

Q1 = os.path.join(L.BUNDLE, "queues", "label_queue_random.json")
Q2 = os.path.join(L.BUNDLE, "queues", "label_queue_random2.json")
# frozen batch-1 pixel features live on the exp-domain-adapt branch result tree;
# accept an override so this is reproducible if that tree moves.
DEFAULT_SELF_DEC = os.environ.get(
    "JADAL_OOD_SELFDEC",
    "/workspace/wt/exp-domain-adapt/eval/results/exp_domain/inputs/ood_pixel_selfdec.npz")


def _one(seg):
    from jadal.ensemble.scorer import decode_pixel_window
    from jadal.gpu.features import cpu_window_features
    try:
        luma, rgb, fps = decode_pixel_window(seg["path"], seg["start_s"], seg["end_s"], 160, 90)
        f = cpu_window_features(luma, rgb, fps=10.0)
        return seg["segment_id"], [float(f[n]) for n in L.FEATURE_NAMES], int(len(luma)), float(fps)
    except Exception as e:  # keep going; report errors
        return seg["segment_id"], None, 0, str(e)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--self-dec", default=DEFAULT_SELF_DEC)
    args = ap.parse_args()

    labels = L.load_labels160()
    order = {sid: i for i, sid in enumerate(labels["ids"])}
    X10 = np.full((labels["n"], len(L.FEATURE_NAMES)), np.nan)

    # --- batch 1: reuse frozen X_10fps ---
    sd = np.load(args.self_dec, allow_pickle=True)
    sidx = {str(i): k for k, i in enumerate(sd["ids"])}
    n1 = 0
    for sid in labels["ids"]:
        if sid in sidx and labels["batch"][order[sid]] == "q1":
            X10[order[sid]] = sd["X_10fps"][sidx[sid]]
            n1 += 1
    print(f"batch-1 reused from {args.self_dec}: {n1}/60", flush=True)

    # --- batch 2: frozen decode recipe ---
    segs = [s for s in json.load(open(Q2))["segments"] if s["segment_id"] in order]
    print(f"batch-2 decoding {len(segs)} windows with the frozen 160x90/10fps recipe", flush=True)
    with mp.Pool(min(args.jobs, len(segs))) as pool:
        rows = pool.map(_one, segs)
    n2, errs = 0, []
    for sid, feats, _t, _fps in rows:
        if feats is None:
            errs.append(sid); continue
        X10[order[sid]] = feats
        n2 += 1
    print(f"batch-2 decoded ok: {n2}/{len(segs)}; errors={errs[:5]}", flush=True)

    ok = ~np.isnan(X10).any(axis=1)
    os.makedirs(L.INPUTS, exist_ok=True)
    np.savez_compressed(os.path.join(L.INPUTS, "ood160_pixel.npz"),
                        ids=np.asarray(labels["ids"])[ok],
                        y=labels["y"][ok], factory=labels["factory"][ok],
                        batch=labels["batch"][ok],
                        names=np.asarray(L.FEATURE_NAMES), X10=X10[ok])
    print(f"wrote {os.path.join(L.INPUTS, 'ood160_pixel.npz')}: {int(ok.sum())}/160 windows, "
          f"good={int(labels['y'][ok].sum())} bad={int((1-labels['y'][ok]).sum())}", flush=True)


if __name__ == "__main__":
    main()
