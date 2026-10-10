"""E1.03: OOD fps sweep on the ORIGINAL 1080p/30fps sources {30,15,10,5,3,2,1}.

Decode each OOD window's ORIGINAL source clip (queues/label_queue_random*.json
field `source_path` + `source_start_s`, 1080p HEVC 30fps) at 160x90, 300 frames,
subsample by the index rule idx=arange(0,T,step), compute the same 26
cpu_window_features with the TRUE effective fps=30/step.

Two queues:
  --queue 1 : the 60 human-labelled windows (labels/ood_labels.json, 51/9) -> ood_shard_*
  --queue 2 : the 100 windows of labels/ood_labels2.json if present, else
              UNLABELLED (descriptive score stability only)             -> ood2_shard_*

Decode is CPU (cv2, no NVDEC): recorded per window. All paths from $JADAL_BUNDLE.

Regen (per shard, OMP_NUM_THREADS=1):
  python3 eval/exp_fps/run_ood.py --queue 1 --shard 0/2
  python3 eval/exp_fps/run_ood.py --queue 2 --shard 0/4
then: python3 eval/exp_fps/merge_ood.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fps_lib import (OOD_LABELS, OOD_LABELS2, QUEUE1, QUEUE2, RESULTS,
                     decode_window, git_sha, gpu_features, subsample_idx)

OOD_STEPS = {30: 1, 15: 2, 10: 3, 5: 6, 3: 10, 2: 15, 1: 30}
NATIVE_FPS = 30.0


def probe(path, cache):
    """width/height/r_frame_rate/nb_frames of the original source (cached)."""
    if path in cache:
        return cache[path]
    import subprocess
    info = {"width": None, "height": None, "r_frame_rate": None, "nb_frames": None}
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries",
                            "stream=width,height,r_frame_rate,nb_frames",
                            "-of", "json", path],
                           capture_output=True, text=True, timeout=120)
        s = json.loads(r.stdout)["streams"][0]
        info = {"width": s.get("width"), "height": s.get("height"),
                "r_frame_rate": s.get("r_frame_rate"),
                "nb_frames": int(s["nb_frames"]) if s.get("nb_frames") else None}
    except Exception as e:  # pragma: no cover - probe is best effort
        info["probe_error"] = str(e)
    cache[path] = info
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=int, choices=[1, 2], default=1)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default=None, help="override output path")
    a = ap.parse_args()
    si, sn = [int(x) for x in a.shard.split("/")]
    qpath = QUEUE1 if a.queue == 1 else QUEUE2
    lpath = OOD_LABELS if a.queue == 1 else OOD_LABELS2
    segs = json.load(open(qpath))["segments"]
    lab = json.load(open(lpath))["latest"] if os.path.exists(lpath) else None
    mine = [s for k, s in enumerate(segs) if k % sn == si]
    gf = gpu_features()
    names = list(gf.FEATURE_NAMES)
    assert len(names) == 26, names
    probe_cache = {}
    out = {"names": names, "queue": a.queue, "queue_path": qpath,
           "label_path": lpath if lab is not None else None,
           "n_queue": len(segs), "shard": a.shard,
           "decode": "CPU cv2.VideoCapture POS_MSEC seek, 160x90 analysis res",
           "git_sha": git_sha(), "windows": []}
    for s in mine:
        sid = s["segment_id"]
        v = lab.get(sid) if lab is not None else None
        rgb, luma, dms = decode_window(s["source_path"], s["source_start_s"],
                                       n_frames=300)
        T = len(luma)
        rec = {"segment_id": sid, "factory": s.get("factory"),
               "worker": s.get("worker"),
               "y": (1 if v["verdict"] == "good" else 0) if v else None,
               "reason": v.get("reason") if v else None,
               "n_decoded": T, "decode_ms": round(dms, 1),
               "decode_ms_per_frame": round(dms / max(T, 1), 3),
               "source": probe(s["source_path"], probe_cache),
               "fps": {}}
        for fps, step in OOD_STEPS.items():
            idx = subsample_idx(T, step)
            feff = NATIVE_FPS / step
            t0 = time.perf_counter()
            d = gf.cpu_window_features(luma[idx].copy(), rgb[idx].copy(),
                                       fps=float(feff))
            fms = (time.perf_counter() - t0) * 1000.0
            rec["fps"][str(fps)] = {"feats": [d[n] for n in names],
                                    "n_frames": len(idx),
                                    "feat_ms": round(fms, 1),
                                    "feat_ms_per_frame": round(fms / max(len(idx), 1), 3)}
        out["windows"].append(rec)
        print(f"q{a.queue} {sid} y={rec['y']} T={T} dec={dms:.0f}ms "
              f"({rec['decode_ms_per_frame']:.2f}ms/frame)", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    prefix = "ood_shard" if a.queue == 1 else "ood2_shard"
    p = a.out or os.path.join(RESULTS, f"{prefix}_{si}of{sn}.json")
    json.dump(out, open(p, "w"))
    print(f"wrote {p} n={len(out['windows'])}")


if __name__ == "__main__":
    main()
