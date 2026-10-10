"""E1.03 merge: OOD 1080p30-source fps sweep.

Labelled part (queue 1, 60 windows, 51 good / 9 bad, human):
  (a) FROZEN IndEgo logistic (fit on all 150 IndEgo windows at native 10fps,
      C=0.3 balanced) applied to OOD -> AUROC per fps + window-bootstrap CI95
      (optimistic: ~1 window per factory)
  (b) per-feature signed AUROC per fps (no transfer needed)
  (c) label-free score rank-correlation (spearman) across fps
Unlabelled part (queue 2, 100 windows): descriptive score stability only,
  NO accuracy is claimed.

-> eval/results/exp_fps/E1.03.json
Regen: run_ood.py --queue {1,2} shards, then this file.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fps_lib import RESULTS, git_sha
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

D = RESULTS
FPS = ["30", "15", "10", "5", "3", "2", "1"]
NATIVE = 30.0


def load_shards(prefix, n):
    out = []
    for i in range(n):
        p = os.path.join(D, f"{prefix}_{i}of{n}.json")
        if os.path.exists(p):
            out.append(json.load(open(p)))
    return out


def win_boot_ci(y, p, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    outs, n = [], len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        outs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(outs, 2.5)), float(np.percentile(outs, 97.5))


shards = load_shards("ood_shard", 2)
wins = [w for s in shards for w in s["windows"]]
wins.sort(key=lambda w: w["segment_id"])
names = shards[0]["names"]
y = np.array([w["y"] for w in wins])
assert len(wins) == 60 and len(y) == 60, len(wins)
n_good, n_bad = int(y.sum()), int((1 - y).sum())
print(f"labelled n={len(wins)} n_good={n_good} n_bad={n_bad}")

# --- frozen IndEgo model: all 150 IndEgo windows, native 10fps features
ind = load_shards("shard_sweep", 2)
assert len(ind) == 2, "IndEgo sweep shards missing (run run_fps.py --mode sweep)"
Xg = np.nan_to_num(np.array([x for s in ind for x in s["feats"]["10"]]))
yg = np.array([x for s in ind for x in s["y"]])
sc = StandardScaler().fit(Xg)
frz = LogisticRegression(max_iter=4000, C=0.3,
                         class_weight="balanced").fit(sc.transform(Xg), yg)

scores, frozen_rows, pf = {}, [], {}
for f in FPS:
    X = np.nan_to_num(np.array([w["fps"][f]["feats"] for w in wins]))
    p = frz.predict_proba(sc.transform(X))[:, 1]
    scores[f] = p
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    lo, hi = win_boot_ci(y, p)
    nfr = float(np.mean([w["fps"][f]["n_frames"] for w in wins]))
    fms = float(np.mean([w["fps"][f]["feat_ms"] for w in wins]))
    frozen_rows.append({"fps": int(f), "fps_eff": NATIVE / (30 / int(f)),
                        "frames_per_window": wins[0]["fps"][f]["n_frames"],
                        "n_frames_mean": round(nfr, 1),
                        "frozen_auroc": round(auc, 4),
                        "ci95_window_boot": [round(lo, 4), round(hi, 4)],
                        "feat_ms_mean": round(fms, 1),
                        "feat_ms_per_frame": round(fms / max(nfr, 1), 3)})
    d = {}
    for j, nm in enumerate(names):
        try:
            d[nm] = round(float(roc_auc_score(y, X[:, j])), 4)
        except ValueError:
            d[nm] = float("nan")
    pf[f] = d

stab = {}
for i, a in enumerate(FPS):
    for b in FPS[i + 1:]:
        r, _ = spearmanr(scores[a], scores[b])
        stab[f"{a}vs{b}"] = round(float(r), 4)

# per-feature stability across fps: does the direct-AUROC ranking hold?
feat_stab = {}
for nm in names:
    v = [pf[f][nm] for f in FPS]
    feat_stab[nm] = {"min": round(float(np.nanmin(v)), 4),
                     "max": round(float(np.nanmax(v)), 4),
                     "range": round(float(np.nanmax(v) - np.nanmin(v)), 4),
                     "auroc_30": pf["30"][nm], "auroc_1": pf["1"][nm]}

dec_ms = float(np.mean([w["decode_ms"] for w in wins]))
dec_mpf = float(np.mean([w["decode_ms_per_frame"] for w in wins]))
src = wins[0].get("source", {})

# --- queue 2 (unlabelled) descriptive score stability
q2 = load_shards("ood2_shard", 4)
q2_block = {"present": bool(q2),
            "note": "UNLABELLED (labels/ood_labels2.json absent at run time): "
                    "descriptive score stability only, NO accuracy claimed"}
if q2:
    w2 = [w for s in q2 for w in s["windows"]]
    w2.sort(key=lambda w: w["segment_id"])
    q2_block["n"] = len(w2)
    q2_block["n_with_labels"] = int(sum(1 for w in w2 if w.get("y") is not None))
    s2 = {}
    for f in FPS:
        X = np.nan_to_num(np.array([w["fps"][f]["feats"] for w in w2]))
        s2[f] = frz.predict_proba(sc.transform(X))[:, 1]
    st2 = {}
    for i, a in enumerate(FPS):
        for b in FPS[i + 1:]:
            r, _ = spearmanr(s2[a], s2[b])
            st2[f"{a}vs{b}"] = round(float(r), 4)
    q2_block["frozen_score_spearman_across_fps"] = st2
    q2_block["frozen_score_mean_by_fps"] = {f: round(float(s2[f].mean()), 4) for f in FPS}
    q2_block["frozen_score_std_by_fps"] = {f: round(float(s2[f].std()), 4) for f in FPS}
    q2_block["feat_ms_mean_by_fps"] = {f: round(float(np.mean([w["fps"][f]["feat_ms"] for w in w2])), 1) for f in FPS}
    q2_block["decode_ms_mean_300f"] = round(float(np.mean([w["decode_ms"] for w in w2])), 1)
    q2_block["decode_ms_per_frame"] = round(float(np.mean([w["decode_ms_per_frame"] for w in w2])), 3)

res = {"experiment": "E1.03", "n": len(wins), "n_good": n_good, "n_bad": n_bad,
       "label_source": "human (labels/ood_labels.json, 51 good / 9 bad, one labeller)",
       "protocol": "out-of-domain: frozen IndEgo logistic (all 150 IndEgo windows, native 10fps, C=0.3 balanced) applied to the 60 labelled Egocentric-10K windows; CI95 = window bootstrap B=2000 seed0 (OPTIMISTIC: ~1 window per factory); per-feature AUROC computed directly against the OOD labels (no transfer)",
       "ci95": {r["fps"]: r["ci95_window_boot"] for r in frozen_rows},
       "ci95_note": "ci95[fps] = window-bootstrap CI95 of the frozen-model AUROC at that fps (optimistic: ~1 window per factory)",
       "cost": {"frames_per_window_by_fps": {r["fps"]: r["n_frames_mean"] for r in frozen_rows},
                "feat_ms_per_window_by_fps": {r["fps"]: r["feat_ms_mean"] for r in frozen_rows},
                "feat_ms_per_frame_by_fps": {r["fps"]: r["feat_ms_per_frame"] for r in frozen_rows},
                "decode_ms_per_window": round(dec_ms, 1),
                "decode_ms_per_frame": round(dec_mpf, 3),
                "device_note": "decode=CPU cv2 160x90, features=CPU float64; the whole 300-frame 1080p window is decoded regardless of the target fps, so decode does not shrink with fps",
                "clean_bench": "see eval/results/exp_fps/pareto.json"},
       "regime": "out-of-domain Egocentric-10K 1080p 30fps HEVC, 60 windows / 47 factories",
       "decode": "ORIGINAL source_path (1080p HEVC 30fps) via cv2 POS_MSEC seek -> 160x90 analysis res, 300 frames, CPU decode; idx=arange(0,T,step), steps {30:1,15:2,10:3,5:6,3:10,2:15,1:30}; TRUE effective fps=30/step passed to cpu_window_features",
       "source_sample": src,
       "decode_ms_mean_300f": round(dec_ms, 1),
       "decode_ms_per_frame": round(dec_mpf, 3),
       "device": "A40 (pod host); decode=CPU cv2 160x90, features=CPU float64",
       "command": "OMP_NUM_THREADS=1 eval/exp_fps/run_ood.py --queue 1 --shard {0,1}/2 && OMP_NUM_THREADS=1 eval/exp_fps/run_fps.py --mode sweep --shard {0,1}/2 && python3 eval/exp_fps/merge_ood.py",
       "git_sha": git_sha(),
       "warning": "n_bad=9 -> exploratory only; CI by window is optimistic",
       "frozen_rows": frozen_rows,
       "per_feature_signed_auroc": pf,
       "per_feature_stability_across_fps": feat_stab,
       "score_rank_stability_spearman": stab,
       "queue2_unlabelled": q2_block}
json.dump(res, open(os.path.join(D, "E1.03.json"), "w"), indent=1)
for r in frozen_rows:
    print(f"fps={r['fps']:>3} nfr={r['frames_per_window']:>3} frozen AUROC "
          f"{r['frozen_auroc']:.4f} [{r['ci95_window_boot'][0]:.3f},"
          f"{r['ci95_window_boot'][1]:.3f}] feat {r['feat_ms_mean']:.0f}ms")
print("stability:", json.dumps(stab))
best = sorted(feat_stab.items(), key=lambda kv: kv[1]["range"])[:5]
print("most fps-stable features:", json.dumps({k: v for k, v in best}))
if q2:
    print("queue2 stability:", json.dumps(q2_block["frozen_score_spearman_across_fps"]))
