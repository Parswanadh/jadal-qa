"""Merge select shards -> LOCO per strategy + overlap/rank vs uniform
-> eval/results/exp_fps/e1_04.json

Regen: OMP_NUM_THREADS=1 python3 eval/exp_fps/run_fps.py --mode select --shard {0,1}/2
       python3 eval/exp_fps/merge_e1_04.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fps_lib import RESULTS, git_sha, loco_proba, clip_bootstrap_ci, bal_acc, jaccard
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

D = RESULTS
STRATEGIES = ["uniform20", "motion20", "random20", "keyframe"]
shards = [json.load(open(os.path.join(D, f"shard_select_{i}of2.json")))
          for i in range(2)]
order = np.argsort([x for s in shards for x in s["ids"]], kind="stable")
ids = [x for s in shards for x in s["ids"]]
ids = [ids[i] for i in order]
clips = np.array([x for s in shards for x in s["clips"]])[order]
y = np.array([x for s in shards for x in s["y"]])[order]
tags = list(shards[0]["feats"].keys())
feats = {t: np.nan_to_num(np.array([x for s in shards for x in s["feats"][t]])[order])
         for t in tags}
fms = {t: np.array([x for s in shards for x in s["feat_ms"][t]])[order]
       for t in shards[0]["feat_ms"]}
dec = np.array([x for s in shards for x in s["decode_ms"]])[order]
nf = np.array([x for s in shards for x in s["nframes"]])[order]
keep = {t: [x for s in shards for x in s["keep_idx"][t]][0:] for t in tags}
keep = {t: [keep[t][i] for i in order] for t in tags}
meta = {t: [x for s in shards for x in s["meta"][t]] for t in tags}
meta = {t: [meta[t][i] for i in order] for t in tags}

probs, rows = {}, []
for t in STRATEGIES:
    if t not in feats:
        continue
    X = feats[t]
    p = loco_proba(X, y, clips, C=0.3)
    probs[t] = p
    lo, hi = clip_bootstrap_ci(y, p, clips)
    rows.append({"strategy": t, "frames": 20,
                 "auroc": round(float(roc_auc_score(y, p)), 4),
                 "ci95": [round(lo, 4), round(hi, 4)],
                 "bal_acc": round(bal_acc(y, p), 4),
                 "n_frames_mean": round(float(np.mean([m["n_frames"]
                                                       for m in meta[t]])), 1),
                 "feat_ms_mean": round(float(fms[t].mean()), 1),
                 "feat_ms_per_frame": round(float(fms[t].mean() /
                                                   max(np.mean([m["n_frames"] for m in meta[t]]), 1)), 3)})

# per-window overlap (Jaccard of kept frame indices) vs uniform20
uni = keep["uniform20"]
ov = {}
for t in STRATEGIES:
    if t not in keep:
        continue
    js, inter = [], []
    for a, b in zip(keep[t], uni):
        js.append(jaccard(a, b))
        inter.append(len(set(int(x) for x in a) & set(int(x) for x in b)))
    ov[t] = {"mean_jaccard_vs_uniform20": round(float(np.mean(js)), 4),
             "mean_frames_shared": round(float(np.mean(inter)), 2),
             "n_windows_identical_to_uniform": int(sum(1 for a, b in zip(keep[t], uni) if set(a) == set(b)))}
# label-free score rank-correlation vs uniform20
rk = {}
for t in STRATEGIES:
    if t in probs:
        r, _ = spearmanr(probs[t], probs["uniform20"])
        rk[t] = round(float(r), 4)

kf = [m for m in meta.get("keyframe", []) if m.get("n_iframes") is not None]
kf_info = {"mean_iframes_in_window": round(float(np.mean([m["n_iframes"] for m in kf])), 2) if kf else None,
           "n_windows_iframes_lt_budget": int(sum(1 for m in kf if m["n_iframes"] < 20)),
           "n_windows": len(kf)}

base = next((r["auroc"] for r in rows if r["strategy"] == "uniform20"), None)
for r in rows:
    r["delta_vs_uniform20"] = round(r["auroc"] - base, 4) if base is not None else None

res = {"experiment": "E1.04", "n": len(y), "n_good": int(y.sum()),
       "n_bad": int((1 - y).sum()),
       "label_source": "human (IndEgo gold manifest_gold.json)",
       "protocol": "leave-one-clip-out over 7 clips, StandardScaler+LogisticRegression C=0.3 balanced; CI=clip-bootstrap B=1000 seed0",
       "ci95": next((r["ci95"] for r in rows if r["strategy"] == "uniform20"), None),
       "ci95_note": "ci95 = uniform20 clip-bootstrap CI95 (the reference arm); per-arm CI95 in rows[]",
       "budget": "20 frames/window (~2fps at 10fps native); uniform=arange(0,T,5); motion=top-19 mean|inter-frame diff| + frame0 anchor; random=seed0; keyframe=I-frame pts_time via ffprobe mapped to 10fps grid, padded with uniform when <20 I-frames",
       "cost": {"frames_per_window": 20,
                "decode_ms_per_window": round(float(dec.mean()), 1),
                "decode_ms_per_frame": round(float(dec.mean() / max(nf.mean(), 1)), 3),
                "feat_ms_per_window_by_arm": {t: round(float(fms[t].mean()), 1) for t in fms},
                "note": "decode is the full 100-frame native window in every arm (selection happens after decode), so selection cannot reduce decode cost"},
       "decode_ms_mean": round(float(dec.mean()), 1),
       "decode_ms_per_frame": round(float(dec.mean() / max(nf.mean(), 1)), 3),
       "device": "A40 (pod host); decode=CPU cv2 160x90, features=CPU float64",
       "command": "OMP_NUM_THREADS=1 eval/exp_fps/run_fps.py --mode select --shard {0,1}/2 && python3 eval/exp_fps/merge_e1_04.py",
       "git_sha": git_sha(),
       "rows": rows, "overlap_vs_uniform20": ov,
       "score_spearman_vs_uniform20": rk, "keyframe_info": kf_info}

# C1 (PR #88 review): a keyframe arm that falls back to uniform for EVERY window
# is not a keyframe selection -> mark it INVALID and withhold its AUROC/CI so it
# can never be read as a result.
if kf_info.get("n_windows") and \
        kf_info["n_windows_iframes_lt_budget"] == kf_info["n_windows"]:
    reason = ("every window has fewer than the 20-frame budget of I-frames "
              f"(mean {kf_info['mean_iframes_in_window']:.2f} I-frames/window; "
              f"{kf_info['n_windows_iframes_lt_budget']}/{kf_info['n_windows']} windows), "
              "so the arm is padded back to uniform and is not a keyframe selection; "
              "its AUROC/CI are withheld")
    res["invalid_arms"] = {"keyframe": {"reason": reason, "diagnostic": {
        "mean_iframes_in_window": kf_info["mean_iframes_in_window"],
        "n_windows_iframes_lt_budget": kf_info["n_windows_iframes_lt_budget"],
        "n_windows": kf_info["n_windows"],
        "overlap_vs_uniform20": ov.get("keyframe")}}}
    for r in rows:
        if r["strategy"] == "keyframe":
            r["invalid"] = True
            r["invalid_reason"] = reason
            r["withheld"] = {"auroc": r.pop("auroc"), "ci95": r.pop("ci95"),
                             "bal_acc": r.pop("bal_acc"),
                             "delta_vs_uniform20": r.pop("delta_vs_uniform20")}

json.dump(res, open(os.path.join(D, "E1.04.json"), "w"), indent=1)
for r in rows:
    if "auroc" not in r:
        print(f"{r['strategy']:>10} INVALID (AUROC withheld)")
        continue
    print(f"{r['strategy']:>10} AUROC {r['auroc']:.4f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] "
          f"bal {r['bal_acc']:.3f} d={r['delta_vs_uniform20']:+.4f} "
          f"nfr {r['n_frames_mean']} feat {r['feat_ms_mean']:.0f}ms")
print("overlap:", json.dumps(ov))
print("spearman vs uniform20:", json.dumps(rk))
print("keyframe:", json.dumps(kf_info))
