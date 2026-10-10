"""Merge OOD shards -> frozen-model AUROC, per-feature AUROC, rank stability."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fps_lib import loco_proba  # noqa (kept for protocol parity)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

D = os.path.join(os.path.dirname(__file__), "..", "results", "exp_fps")
shards = [json.load(open(os.path.join(D, f"ood_shard_{i}of2.json")))
          for i in range(2)]
wins = [w for s in shards for w in s["windows"]]
wins.sort(key=lambda w: w["segment_id"])
names = shards[0]["names"]
FPS = ["30", "15", "10", "5", "3", "2", "1"]
y = np.array([w["y"] for w in wins])
assert len(wins) == 60, len(wins)
print(f"n={len(wins)} n_bad={int((1-y).sum())}")

# frozen model: train on ALL IndEgo pixel features at native 10fps
sh = [json.load(open(os.path.join(D, f"shard_sweep_{i}of2.json"))) for i in range(2)]
Xg = np.nan_to_num(np.array([x for s in sh for x in s["feats"]["10"]]))
sc = StandardScaler().fit(Xg)
yg = np.array([x for s in sh for x in s["y"]])
frz = LogisticRegression(max_iter=4000, C=0.3, class_weight="balanced").fit(
    sc.transform(Xg), yg)

rng = np.random.default_rng(0)


def win_boot_ci(p, B=2000):
    outs = []
    n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        outs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(outs, 2.5)), float(np.percentile(outs, 97.5))


scores = {}
frozen_rows = []
for f in FPS:
    X = np.nan_to_num(np.array([w["fps"][f]["feats"] for w in wins]))
    p = frz.predict_proba(sc.transform(X))[:, 1]
    scores[f] = p
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    lo, hi = win_boot_ci(p)
    frozen_rows.append({"fps": int(f),
                        "frames_per_window": wins[0]["fps"][f]["n_frames"],
                        "frozen_auroc": round(float(auc), 4),
                        "ci95_window_boot": [round(lo, 4), round(hi, 4)],
                        "feat_ms_mean": round(float(np.mean(
                            [w["fps"][f]["feat_ms"] for w in wins])), 1)})

# per-feature direct AUROC vs fps (signed)
pf = {}
for f in FPS:
    X = np.nan_to_num(np.array([w["fps"][f]["feats"] for w in wins]))
    d = {}
    for j, n in enumerate(names):
        try:
            a = roc_auc_score(y, X[:, j])
        except ValueError:
            a = float("nan")
        d[n] = round(float(a), 4)
    pf[f] = d

# label-free stability: spearman of frozen scores between fps settings
stab = {}
for i, a in enumerate(FPS):
    for b in FPS[i + 1:]:
        r, _ = spearmanr(scores[a], scores[b])
        stab[f"{a}vs{b}"] = round(float(r), 4)

dec_mean = float(np.mean([w["decode_ms"] for w in wins]))
res = {"experiment": "E1.03", "n": 60, "n_good": int(y.sum()),
       "n_bad": int((1 - y).sum()),
       "label_source": "human ood_labels/labels.json (51 good/9 bad)",
       "protocol": "frozen IndEgo logistic (all-150, 10fps features) applied to OOD; CI=window-bootstrap B=2000 (optimistic: ~1 window/factory); per-feature AUROC direct on OOD labels",
       "decode": "original 1080p30 HEVC source_path @160x90 analysis res; idx=arange(0,T,step) steps {30:1,15:2,10:3,5:6,3:10,2:15,1:30}; true fps=30/step",
       "decode_ms_mean_300f": round(dec_mean, 1),
       "command": "eval/exp_fps/run_ood.py --shard {0,1}/2, merge_ood.py",
       "note_labels2": "labels2.json absent at run time; OOD n=60 only",
       "frozen_rows": frozen_rows, "per_feature_signed_auroc": pf,
       "score_rank_stability_spearman": stab}
json.dump(res, open(os.path.join(D, "e1_03.json"), "w"), indent=1)
for r in frozen_rows:
    print(f"fps={r['fps']:>3} nfr={r['frames_per_window']:>3} frozen AUROC {r['frozen_auroc']:.4f} "
          f"[{r['ci95_window_boot'][0]:.3f},{r['ci95_window_boot'][1]:.3f}] feat {r['feat_ms_mean']:.0f}ms")
print("stability:", json.dumps(stab))
