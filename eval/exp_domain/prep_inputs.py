"""E5 input prep (CPU): OOD pixel features + OOD EG2 window means.

Outputs (committed, small):
  eval/results/exp_domain/inputs/ood_pixel_selfdec.npz
      {ids (60,), y (60,), factory (60,), names (26,), X_native (60,26), X_10fps (60,26)}
      Decoded from the label_windows transcodes with the frozen pixel recipe
      (ffmpeg 160x90 rgb24 -> jadal.gpu.features.cpu_window_features).
      X_native = native fps (~30 for the transcodes); X_10fps = idx=arange(0,T,3),
      fps_eff=10 (matches IndEgo's ~10 fps so E5.03 can separate fps from domain).
  eval/results/exp_domain/inputs/ood_eg2_means.npz
      {ids (60,), y (60,), factory (60,), means (60,768) float32 L2-normalised}
      Derived from the frozen EG2 frame embeddings of the 60 OOD windows
      (source: commit aece6d7 `eval/results/ensemble/ood_frames.npz`, reproduced
      from the frozen 0.5 s sampling; provided via --frames).

Run:
  OMP_NUM_THREADS=1 python3 eval/exp_domain/prep_inputs.py \
      --frames eval/results/exp_domain/inputs/ood_eg2_frames.npz
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
from eval.exp_domain import e5lib as L  # noqa: E402

QUEUE = os.path.join(L.BUNDLE, "queues", "label_queue_random.json")


def _one(seg):
    from jadal.ensemble.scorer import decode_pixel_window
    from jadal.gpu.features import cpu_window_features
    try:
        luma, rgb, fps = decode_pixel_window(seg["path"], seg["start_s"], seg["end_s"], 160, 90)
        f_native = cpu_window_features(luma, rgb, fps=fps)
        idx = np.arange(0, len(luma), 3)
        f10 = cpu_window_features(luma[idx].copy(), rgb[idx].copy(), fps=10.0)
        return seg["segment_id"], [float(f_native[n]) for n in L.FEATURE_NAMES], \
            [float(f10[n]) for n in L.FEATURE_NAMES], float(fps), int(len(luma))
    except Exception as e:  # keep going; report errors
        return seg["segment_id"], None, None, 0.0, 0, str(e)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=os.path.join(L.INPUTS, "ood_eg2_frames.npz"),
                    help="ood_frames.npz with F (60,T,768) EG2 frame embeddings")
    ap.add_argument("--fetch-git", action="store_true",
                    help="fetch the frozen EG2 frames from origin/feat/ensemble-views if absent")
    ap.add_argument("--skip-pixel", action="store_true",
                    help="reuse an existing ood_pixel_selfdec.npz (only redo the EG2 means)")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(L.INPUTS, exist_ok=True)

    ref = L.load_ood()
    order = {sid: i for i, sid in enumerate(ref["ids"])}
    segs = [s for s in json.load(open(QUEUE))["segments"] if s["segment_id"] in order]

    if args.skip_pixel:
        print("skip-pixel: keeping existing ood_pixel_selfdec.npz", flush=True)
    else:
        with mp.Pool(min(args.jobs, len(segs))) as pool:
            rows = pool.map(_one, segs)
        ok = np.array([r[1] is not None for r in rows])
        Xn = np.array([r[1] for r in rows if r[1] is not None], dtype=np.float64)
        X10 = np.array([r[2] for r in rows if r[2] is not None], dtype=np.float64)
        okids = [r[0] for r in rows if r[1] is not None]
        y = np.array([ref["y"][order[i]] for i in okids], dtype=int)
        fac = np.array([ref["factory"][order[i]] for i in okids])
        np.savez_compressed(os.path.join(L.INPUTS, "ood_pixel_selfdec.npz"),
                            ids=np.asarray(okids), y=y, factory=fac,
                            names=np.asarray(L.FEATURE_NAMES), X_native=Xn, X_10fps=X10)
        print(f"ood_pixel_selfdec: {Xn.shape}, ok={int(ok.sum())}/{len(rows)}, "
              f"errors={[r for r in rows if r[1] is None]}", flush=True)

    # EG2 window means
    if not os.path.exists(args.frames) and args.fetch_git:
        import subprocess
        os.makedirs(os.path.dirname(args.frames) or ".", exist_ok=True)
        with open(args.frames, "wb") as f:
            subprocess.run(["git", "show",
                            "origin/feat/ensemble-views:eval/results/ensemble/ood_frames.npz"],
                           stdout=f, check=True, cwd=L.REPO)
        print(f"fetched frozen EG2 frames from origin/feat/ensemble-views -> {args.frames}", flush=True)
    if not os.path.exists(args.frames):
        print(f"WARNING: {args.frames} missing -> skipping ood_eg2_means.npz "
              f"(pass --frames or --fetch-git; source eval/results/ensemble/ood_frames.npz)")
        return
    z = np.load(args.frames, allow_pickle=True)
    F = z["F"].astype(np.float64)
    zids = [str(x) for x in z["ids"]]
    means = np.empty((len(zids), F.shape[2]), dtype=np.float32)
    for i, row in enumerate(F):
        v = row[~np.isnan(row)]
        m = v.mean(axis=0)
        means[i] = (m / max(np.linalg.norm(m), 1e-12)).astype(np.float32)
    morder = {sid: i for i, sid in enumerate(zids)}
    y_by_id = {sid: int(v) for sid, v in zip(ref["ids"], ref["y"])}
    fac_by_id = {sid: f for sid, f in zip(ref["ids"], ref["factory"])}
    pairs = [(sid, morder[sid]) for sid in ref["ids"] if sid in morder]
    np.savez_compressed(os.path.join(L.INPUTS, "ood_eg2_means.npz"),
                        ids=np.asarray([sid for sid, _ in pairs]),
                        y=np.asarray([y_by_id[sid] for sid, _ in pairs], int),
                        factory=np.asarray([fac_by_id[sid] for sid, _ in pairs]),
                        means=np.asarray([means[j] for _, j in pairs], dtype=np.float32))
    print(f"ood_eg2_means: {means.shape} -> {len(pairs)} matched", flush=True)


if __name__ == "__main__":
    main()
