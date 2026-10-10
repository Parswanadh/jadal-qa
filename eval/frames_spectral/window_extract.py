"""Decode EVERY frame of each of the 150 human windows and extract per-frame
spectral features (same extractor as the 10k frames: resize to 480-px width).

Source: /home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json
(READ-ONLY; human labels, 150 windows, 7 clips).
Two variants cached: raw and gaussian-denoised (3x3, spatial-denoise ablation).
Bilateral/NLM cost measured on a 5-frame sample only (too slow, excluded w/
reason — cost recorded in the JSON).

Usage: python -m eval.frames_spectral.window_extract [--procs 8]
Out: eval/results/frames_spectral/human_windows_raw.npz (list of T_i x d),
     human_windows_gauss.npz, plus meta json (ids, clip, label, reason, fps).
<= 8 procs. Deterministic order.
"""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

from eval.frames_spectral.frame_features import FEATURES, extract_frame_bgr

GOLD = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"
OUT = "eval/results/frames_spectral"


def decode_window(path, start_s, end_s):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    frames = []
    while True:
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        ok, fr = cap.read()
        if not ok or pos >= end_s - 1e-9:
            break
        if pos >= start_s - 1e-9:
            frames.append(fr)
    cap.release()
    return frames, float(fps)


def _work(args):
    i, path, start_s, end_s, denoise = args
    frames, fps = decode_window(path, start_s, end_s)
    if denoise == "gauss":
        frames = [cv2.GaussianBlur(f, (3, 3), 0) for f in frames]
    X = np.stack([extract_frame_bgr(f) for f in frames]).astype(np.float32) \
        if frames else np.zeros((0, len(FEATURES)), np.float32)
    return i, X, fps, len(frames)


def extract_variant(items, denoise, procs):
    args = [(i, it["path"], it["start_s"], it["end_s"], denoise)
            for i, it in enumerate(items)]
    Xs = [None] * len(items)
    fpss = np.zeros(len(items))
    counts = np.zeros(len(items), dtype=int)
    with Pool(min(procs, 8)) as pool:
        for i, X, fps, c in pool.imap_unordered(_work, args, chunksize=4):
            Xs[i] = X
            fpss[i] = fps
            counts[i] = c
    return Xs, fpss, counts


def main():
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    t0 = time.time()
    with open(GOLD) as f:
        man = json.load(f)
    items = man["items"]
    print(f"gold snapshot={man['snapshot_utc']} n={len(items)} "
          f"good={man['n_good']} bad={man['n_bad']}", flush=True)
    # denoise cost sample (bilateral + NLM excluded from full run if slow)
    import time as _t
    cap = cv2.VideoCapture(items[0]["path"])
    ok, fr = cap.read()
    cap.release()
    cost = {}
    for name, fn in (("gauss", lambda f: cv2.GaussianBlur(f, (3, 3), 0)),
                     ("bilateral", lambda f: cv2.bilateralFilter(f, 5, 50, 50)),
                     ("nlm", lambda f: cv2.fastNlMeansDenoisingColored(f, None, 5, 5, 7, 21))):
        a = _t.time()
        for _ in range(3):
            fn(fr)
        cost[name] = (_t.time() - a) / 3 * 1000.0
    print("denoise ms/frame sample (480px): " +
          ", ".join(f"{k}={v:.0f}" for k, v in cost.items()), flush=True)
    meta = {"snapshot_utc": man["snapshot_utc"], "labels_sha256": man["labels_sha256"],
            "denoise_cost_ms": cost,
            "denoise_note": "bilateral/NLM excluded from full ablation: "
                            f"{cost['bilateral']:.0f}/{cost['nlm']:.0f} ms/frame would dominate "
                            "the 25ms extractor; recorded here as cost evidence."}
    for denoise in ("raw", "gauss"):
        Xs, fpss, counts = extract_variant(items, denoise, procs)
        maxT = max(x.shape[0] for x in Xs)
        d = len(FEATURES)
        pad = np.zeros((len(items), maxT, d), np.float32)
        for i, x in enumerate(Xs):
            pad[i, :x.shape[0]] = x
        np.savez_compressed(
            f"{OUT}/human_windows_{denoise}.npz", X=pad,
            lengths=counts.astype(np.int32), fps=fpss.astype(np.float32),
            features=np.asarray(FEATURES),
            ids=np.asarray([it["id"] for it in items]),
            clips=np.asarray([it["clip"] for it in items]),
            labels=np.asarray([it["label"] for it in items], dtype=np.int8),
            reasons=np.asarray([it.get("reason", "") for it in items]))
        print(f"{denoise}: windows={len(items)} maxT={maxT} "
              f"total_frames={counts.sum()} median_frames={np.median(counts):.0f} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
    meta["elapsed_s"] = time.time() - t0
    with open(f"{OUT}/human_windows_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
