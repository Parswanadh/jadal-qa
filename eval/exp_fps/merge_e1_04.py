"""Merge select shards -> LOCO per strategy -> eval/results/exp_fps/e1_04.json

Only SPATIAL_FEATURES (dt-independent, 18 of 26) are scored, because the
selection arms are not uniformly spaced. The keyframe arm is reported as
INVALID (no AUROC) if any window had fewer than 20 I-frame positions.

Regen: PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/exp_fps/merge_e1_04.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fps_lib import SPATIAL_FEATURES, loco_proba, clip_bootstrap_ci, bal_acc
from sklearn.metrics import roc_auc_score

D = os.path.join(os.path.dirname(__file__), "..", "results", "exp_fps")
shards = [json.load(open(os.path.join(D, f"shard_select_{i}of2.json")))
          for i in range(2)]
order = np.argsort([x for s in shards for x in s["ids"]], kind="stable")
clips = np.array([x for s in shards for x in s["clips"]])[order]
y = np.array([x for s in shards for x in s["y"]])[order]
names = shards[0]["names"]
cols = [names.index(n) for n in SPATIAL_FEATURES]
tags = list(shards[0]["feats"].keys())
feats = {t: np.array([x for s in shards for x in s["feats"][t]])[order]
         for t in tags}
fms = {t: np.array([x for s in shards for x in s["feat_ms"][t]])[order]
       for t in shards[0]["feat_ms"]}
dec = np.array([x for s in shards for x in s["decode_ms"]])[order]
kf_nif = np.array([x for s in shards for x in s["keyframe_n_iframes"]])[order]
kf_fb = np.array([x for s in shards for x in s["keyframe_fallback"]])[order]

rows = []
for t in ["uniform20", "motion20", "random20", "keyframe"]:
    if t not in feats:
        continue
    if t == "keyframe" and kf_fb.any():
        rows.append({"strategy": t, "frames": 20, "status": "INVALID",
                     "auroc": None, "ci95": None, "bal_acc": None,
                     "keyframe_fallback_windows": int(kf_fb.sum()),
                     "n_windows": int(len(y)),
                     "iframes_per_window_min": int(kf_nif.min()),
                     "iframes_per_window_median": float(np.median(kf_nif)),
                     "iframes_per_window_max": int(kf_nif.max()),
                     "reason": "fewer than 20 distinct I-frame positions on the "
                               "10 fps grid in these windows; the arm would not "
                               "be a 20-frame keyframe selection, so no result is "
                               "reported"})
        continue
    X = feats[t][:, cols]
    p = loco_proba(X, y, clips, C=0.3)
    lo, hi = clip_bootstrap_ci(y, p, clips)
    rows.append({"strategy": t, "frames": 20, "status": "valid",
                 "auroc": round(float(roc_auc_score(y, p)), 4),
                 "ci95": [round(lo, 4), round(hi, 4)],
                 "bal_acc": round(bal_acc(y, p), 4),
                 "feat_ms_mean": round(float(fms[t].mean()), 1)})
res = {"experiment": "E1.04", "n": int(len(y)),
       "label_source": "human (IndEgo gold)",
       "protocol": "LOCO logistic C=0.3 balanced; CI=clip-bootstrap B=1000 seed0",
       "features": f"{len(SPATIAL_FEATURES)} spatial (dt-independent) features only",
       "budget": "20 frames/window (~2 fps); motion=top-19 inter-frame-diff + frame0; "
                 "random=seed0; keyframe=I-frame positions from ffprobe (valid only if >=20 per window)",
       "decode_ms_per_window": round(float(dec.mean()), 1),
       "command": "PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python "
                  "eval/exp_fps/run_fps.py --mode select --shard {0,1}/2; then merge_e1_04.py",
       "rows": rows}
json.dump(res, open(os.path.join(D, "e1_04.json"), "w"), indent=1)
for r in rows:
    if r["status"] != "valid":
        print(f"{r['strategy']:>10} INVALID keyframe_fallback "
              f"{r['keyframe_fallback_windows']}/{r['n_windows']}")
        continue
    print(f"{r['strategy']:>10} AUROC {r['auroc']:.4f} "
          f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] bal {r['bal_acc']:.3f}")
