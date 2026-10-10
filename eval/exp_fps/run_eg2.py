"""E1.02: EG2 window-embedding fps subsample from cached segments.npz (no GPU).

F (150,21,768) fp16 at 0.5s spacing ~= 2fps. Subsample rows:
 2fps: all rows (counts-aware, recipe = ensemble_views_exploratory.py)
 1fps: rows ::2 ; 0.5fps: rows ::4
LOCO C=0.01 + clip-bootstrap CI. Regen: python3 eval/exp_fps/run_eg2.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fps_lib import RESULTS, SEGMENTS_NPZ, git_sha, loco_proba, clip_bootstrap_ci, bal_acc

from sklearn.metrics import roc_auc_score

D = RESULTS
z = np.load(SEGMENTS_NPZ, allow_pickle=True)
ids = [str(i) for i in z["ids"]]
clips = np.array([str(c) for c in z["clips"]])
y = np.array(z["labels"]).astype(int)
E = z["F"].astype(np.float32)
cnt = np.array(z["counts"]).astype(int)


def window_mean(step):
    Xs = []
    for k in range(len(ids)):
        n = max(1, int(cnt[k]))
        rows = E[k, :n:step]
        if len(rows) == 0:
            rows = E[k, :1]
        v = rows.mean(axis=0)
        Xs.append(v)
    X = np.stack(Xs)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


rows = []
probs = {}
for tag, step in [("2fps", 1), ("1fps", 2), ("0.5fps", 4)]:
    X = window_mean(step)
    nfr = float(np.mean([max(1, int(c)) // step + (1 if step > 1 and int(c) % step else 0)
                         for c in cnt]))
    p = loco_proba(X, y, clips, C=0.01)
    probs[tag] = p
    lo, hi = clip_bootstrap_ci(y, p, clips)
    rows.append({"fps": tag,
                 "frames_per_window": round(nfr, 1),
                 "auroc": round(float(roc_auc_score(y, p)), 4),
                 "ci95": [round(lo, 4), round(hi, 4)],
                 "bal_acc": round(bal_acc(y, p), 4)})
full = rows[0]["auroc"]
for r in rows:
    r["delta_vs_2fps"] = round(r["auroc"] - full, 4)

# two-view probability average with pixel view at matching fps (from e1_01 shards)
sh = [json.load(open(os.path.join(D, f"shard_sweep_{i}of2.json"))) for i in range(2)]
order = np.argsort([x for s in sh for x in s["ids"]], kind="stable")
sids = [x for s in sh for x in s["ids"]]
sids = [sids[i] for i in order]
assert sids == ids, "id order mismatch between EG2 cache and pixel shards"
px = {t: np.nan_to_num(np.array([x for s in sh for x in s["feats"][t]])[order])
      for t in sh[0]["feats"]}
pix_map = {"2fps": "2", "1fps": "1"}  # pixel sweep has no 0.5fps; pair available
ens_rows = []
for tag in ["2fps", "1fps"]:
    pg = loco_proba(px[pix_map[tag]], y, clips, C=0.3)
    ensp = (probs[tag] + pg) / 2
    lo, hi = clip_bootstrap_ci(y, ensp, clips)
    ens_rows.append({"fps": tag, "auroc": round(float(roc_auc_score(y, ensp)), 4),
                     "ci95": [round(lo, 4), round(hi, 4)],
                     "bal_acc": round(bal_acc(y, ensp), 4)})

res = {"experiment": "E1.02", "n": 150, "n_good": int(y.sum()),
       "n_bad": int((1 - y).sum()),
       "label_source": "human (IndEgo gold)",
       "protocol": "LOCO, L2-normalised mean embedding + StandardScaler + logistic C=0.01; CI=clip-bootstrap B=1000 seed0",
       "subsample_rule": "cached 0.5s-spacing rows; 2fps=all rows counts-aware, 1fps=rows ::2, 0.5fps=rows ::4; NO re-embed",
       "command": "python3 eval/exp_fps/run_eg2.py",
       "device": "A40 (pod host); CPU eval on cached embeddings",
       "git_sha": git_sha(),
       "eg2_rows": rows, "two_view_prob_avg": ens_rows,
       "note_5_10fps": "pending GPU-lock re-embed (E1.02b)"}
json.dump(res, open(os.path.join(D, "e1_02.json"), "w"), indent=1)
for r in rows:
    print(f"EG2 {r['fps']:>6} nfr~{r['frames_per_window']:<5} AUROC {r['auroc']:.4f} "
          f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] bal {r['bal_acc']:.3f} d={r['delta_vs_2fps']:+.4f}")
for r in ens_rows:
    print(f"ENS {r['fps']:>6} AUROC {r['auroc']:.4f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] bal {r['bal_acc']:.3f}")
