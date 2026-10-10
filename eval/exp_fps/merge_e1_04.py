"""Merge select shards -> LOCO per strategy -> eval/results/exp_fps/e1_04.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fps_lib import loco_proba, clip_bootstrap_ci, bal_acc
from sklearn.metrics import roc_auc_score

D = os.path.join(os.path.dirname(__file__), "..", "results", "exp_fps")
shards = [json.load(open(os.path.join(D, f"shard_select_{i}of2.json")))
          for i in range(2)]
order = np.argsort([x for s in shards for x in s["ids"]], kind="stable")
clips = np.array([x for s in shards for x in s["clips"]])[order]
y = np.array([x for s in shards for x in s["y"]])[order]
tags = list(shards[0]["feats"].keys())
feats = {t: np.nan_to_num(np.array([x for s in shards for x in s["feats"][t]])[order])
         for t in tags}
fms = {t: np.array([x for s in shards for x in s["feat_ms"][t]])[order]
       for t in shards[0]["feat_ms"]}
dec = np.array([x for s in shards for x in s["decode_ms"]])[order]

rows = []
for t in ["uniform20", "motion20", "random20", "keyframe"]:
    if t not in feats:
        continue
    X = feats[t]
    if np.isnan(X).all() or (X.var(axis=0) == 0).all():
        continue
    p = loco_proba(X, y, clips, C=0.3)
    lo, hi = clip_bootstrap_ci(y, p, clips)
    rows.append({"strategy": t, "frames": 20,
                 "auroc": round(float(roc_auc_score(y, p)), 4),
                 "ci95": [round(lo, 4), round(hi, 4)],
                 "bal_acc": round(bal_acc(y, p), 4),
                 "feat_ms_mean": round(float(fms[t].mean()), 1)})
res = {"experiment": "E1.04", "n": 150,
       "label_source": "human (IndEgo gold)",
       "protocol": "LOCO logistic C=0.3; CI=clip-bootstrap B=1000 seed0",
       "budget": "20 frames/window (~2fps); motion=top-20 inter-frame-diff + frame0 anchor; random=seed0; keyframe=I-frame timestamps via ffprobe padded with uniform to 20",
       "decode_ms_mean": round(float(dec.mean()), 1),
       "command": "eval/exp_fps/run_fps.py --mode select --shard {0,1}/2",
       "rows": rows}
json.dump(res, open(os.path.join(D, "e1_04.json"), "w"), indent=1)
for r in rows:
    print(f"{r['strategy']:>10} AUROC {r['auroc']:.4f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] bal {r['bal_acc']:.3f}")
