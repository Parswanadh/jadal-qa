"""CPU 26-feature extraction for the 60 OOD windows (no GPU lock needed).

Same recipe as the frozen pixel view (ffmpeg CPU 160x90 rgb24 ->
jadal.gpu.features:cpu_window_features). Writes
eval/results/ensemble/ood_pixel.npz {ids, X (60,26), names}.
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUEUE = "/home/parshu/projects/jadal-qa-wt/scale-data/eval/results/scale/label_queue_random.json"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "ensemble", "ood_pixel.npz")


def main():
    import numpy as np
    from jadal.ensemble.scorer import load_config, pixel_features_for_window
    cfg, _ = load_config()
    names = cfg["views"]["pixel"]["feature_names"]
    segs = json.load(open(QUEUE))["segments"]
    X, ids, errs = [], [], 0
    for k, s in enumerate(segs):
        try:
            feats, _ = pixel_features_for_window(s["path"], s["start_s"], s["end_s"], cfg)
            X.append([float(feats[n]) for n in names])
            ids.append(s["segment_id"])
        except Exception as e:
            errs += 1
            print(f"ERR {s['segment_id']}: {str(e)[:150]}", flush=True)
        if (k + 1) % 10 == 0:
            print(f"{k+1}/{len(segs)}", flush=True)
    X = np.array(X)
    np.savez(OUT, ids=np.asarray(ids), X=X, names=np.asarray(names))
    print(f"wrote {OUT} X={X.shape} errs={errs}")


if __name__ == "__main__":
    main()
