"""Extract the 3 frozen rule features (hist_ent, ten_norm, blur_norm) for OOD windows.

Uses Q1's real modules on branch b/ood-main (jadal.perpixel.{series,physics,theory}).
Run from the b/ood-main worktree. Output: JSON list of {window_id, features, meta}.
One-shot: the result is frozen into eval/exp_phys/input_windows.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
from jadal.perpixel import physics as P  # noqa: E402
from jadal.perpixel import series as S  # noqa: E402
from jadal.perpixel import theory as T  # noqa: E402

B = os.environ.get("JADAL_BUNDLE")
if not B:
    raise SystemExit("JADAL_BUNDLE is required (path to the pod data bundle)")
LABELS = os.path.join(B, "labels", "ood160_labels.json")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/ood_feats.json"

d = json.load(open(LABELS))
windows = d["windows"][:N]
out = []
t0 = time.time()
for i, w in enumerate(windows):
    path = os.path.join(B, w["video"])
    start, end = 0.0, 10.0
    fr, fps = S.decode_160x90(path, start, end)
    ph = P.extract_physics(fr, fps, path, start, end)
    th = T.extract_theory(fr, fps)
    out.append({
        "window_id": w["segment_id"],
        "split": "ood",
        "label": 1 if w["verdict"] == "good" else 0,
        "label_reason": w.get("reason") or None,
        "factory": w.get("factory"),
        "n_frames": int(fr.shape[0]),
        "fps": round(float(fps), 4),
        "features": {
            "hist_ent": float(ph["hist_ent"]),
            "ten_norm": float(ph["ten_norm"]),
            "blur_norm": float(th["blur_norm"]),
        },
    })
    print(f"[{i+1}/{N}] {w['segment_id']} {w['verdict']} "
          f"hist_ent={ph['hist_ent']:.4f} ten_norm={ph['ten_norm']:.4f} "
          f"blur_norm={th['blur_norm']:.4f}", flush=True)

json.dump({"source": "ood160_labels.json + local label_windows mp4s",
           "extractor": "b/ood-main jadal.perpixel.{series,physics,theory}",
           "n": len(out), "windows": out}, open(OUT, "w"), indent=1)
print(f"wrote {OUT} n={len(out)} wall={time.time()-t0:.1f}s")
