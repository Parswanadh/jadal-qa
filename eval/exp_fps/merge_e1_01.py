"""Merge sweep shards -> LOCO eval per fps -> eval/results/exp_fps/e1_01.json

Headline = all 26 features (as originally run). Also reported: the 18
dt-independent SPATIAL_FEATURES only, which are the fair comparison across fps.
Regen: PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/exp_fps/merge_e1_01.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fps_lib import (FPS_STEPS, SPATIAL_FEATURES, loco_proba,
                     clip_bootstrap_ci, bal_acc)

D = os.path.join(os.path.dirname(__file__), "..", "results", "exp_fps")
shards = [json.load(open(os.path.join(D, f"shard_sweep_{i}of2.json")))
          for i in range(2)]
order = np.argsort([x for s in shards for x in s["ids"]], kind="stable")
ids = [x for s in shards for x in s["ids"]]
ids = [ids[i] for i in order]
clips = np.array([x for s in shards for x in s["clips"]])[order]
y = np.array([x for s in shards for x in s["y"]])[order]
names = shards[0]["names"]
feats = {}
for tag in shards[0]["feats"]:
    F = np.array([x for s in shards for x in s["feats"][tag]])[order]
    feats[tag] = F
dec_ms = np.array([x for s in shards for x in s["decode_ms"]])[order]
feat_ms = {t: np.array([x for s in shards for x in s["feat_ms"][t]])[order]
           for t in shards[0]["feat_ms"]}
nf = np.array([x for s in shards for x in s["nframes"]])[order]

from sklearn.metrics import roc_auc_score

fps_of = {10: 10.0, 5: 5.0, 3.33: 10.0 / 3, 2: 2.0, 1: 1.0}
nframes_of = {t: len(np.arange(0, 100, s)) for t, s in FPS_STEPS.items()}
sp_cols = [names.index(n) for n in SPATIAL_FEATURES]
rows = []
pf_auc = {}
for tag in [10, 5, 3.33, 2, 1]:
    X = np.nan_to_num(feats[str(tag)])
    p = loco_proba(X, y, clips, C=0.3)
    lo, hi = clip_bootstrap_ci(y, p, clips)
    auc = roc_auc_score(y, p)
    Xs = X[:, sp_cols]
    ps = loco_proba(Xs, y, clips, C=0.3)
    slo, shi = clip_bootstrap_ci(y, ps, clips)
    rows.append({"fps": tag, "fps_eff": fps_of[tag],
                 "frames_per_window": nframes_of[tag],
                 "mean_decoded_frames": round(float(nf.mean()), 1),
                 "n_features": len(names),
                 "auroc": round(float(auc), 4),
                 "ci95": [round(lo, 4), round(hi, 4)],
                 "bal_acc": round(bal_acc(y, p), 4),
                 "n_features_spatial": len(SPATIAL_FEATURES),
                 "auroc_spatial": round(float(roc_auc_score(y, ps)), 4),
                 "ci95_spatial": [round(slo, 4), round(shi, 4)],
                 "bal_acc_spatial": round(bal_acc(y, ps), 4),
                 "feat_ms_mean": round(float(feat_ms[str(tag)].mean()), 1)})
    pf = {}
    for j, n in enumerate(names):
        try:
            a = roc_auc_score(y, X[:, j])
        except ValueError:
            a = float("nan")
        pf[n] = round(float(a), 4)
    pf_auc[str(tag)] = pf

full = rows[0]["auroc"]
full_sp = rows[0]["auroc_spatial"]
for r in rows:
    r["delta_vs_full"] = round(r["auroc"] - full, 4)
    r["delta_vs_full_spatial"] = round(r["auroc_spatial"] - full_sp, 4)

res = {
    "experiment": "E1.01", "n": 150, "n_good": int(y.sum()),
    "n_bad": int((1 - y).sum()),
    "label_source": "human (IndEgo gold manifest_gold.json)",
    "protocol": "leave-one-clip-out, StandardScaler+LogisticRegression C=0.3 balanced; CI=clip-bootstrap B=1000 seed0",
    "subsample_rule": "decode once native ~10fps via POS_MSEC seek + consecutive reads at 160x90; idx=arange(0,T,step), step {10:1,5:2,3.33:3,2:5,1:10}; cpu_window_features with TRUE effective fps (10/step)",
    "command": "PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python eval/exp_fps/run_fps.py --mode sweep --shard {0,1}/2; then merge_e1_01.py",
    "decode_ms_per_window": round(float(dec_ms.mean()), 1),
    "claim_status": "descriptive only: no equivalence margin was pre-stated and no "
                    "paired test was run; the 95% CIs are ~0.2 wide at n=150 (50 bad), "
                    "so differences between fps settings are not resolvable at this n",
    "rows": rows, "per_feature_signed_auroc": pf_auc,
    "fps_comparability": {
        "NOT_comparable_across_fps": [
            "temp_b0..b3 (band edges in Hz shift with df=fps/T; b2/b3 empty below Nyquist)",
            "flicker_idx (b3/total)",
            "motion_energy (mean |frame diff| grows with inter-frame dt)",
            "rppg_pulse (code uses hardcoded FPS_DEFAULT=30 regardless of fps arg; T<16 returns 0: constant at 1fps)",
            "frozen_frac (energy-threshold based; definition shifts with band layout)"],
        "comparable_endpoint_spatial": [
            "global_shift (first vs middle frame displacement)",
            "zncc_self (first vs last frame)",
            "radial_slope, hf_ratio, dct_energy (middle frame only)",
            "luma_mean/p05/p50/p95, dark_frac, bright_frac, pca_resid, top1_share, illum_r/g/b (pooled; sampling noise only)",
            "skin_frac (subsampled frames; sampling noise only)",
            "dup_hash_mean (middle frame only)"]},
}
json.dump(res, open(os.path.join(D, "e1_01.json"), "w"), indent=1)
for r in rows:
    print(f"fps={r['fps']:>5} nfr={r['frames_per_window']:>3} AUROC {r['auroc']:.4f} "
          f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] d={r['delta_vs_full']:+.4f} | "
          f"spatial AUROC {r['auroc_spatial']:.4f} d={r['delta_vs_full_spatial']:+.4f} "
          f"feat {r['feat_ms_mean']:.0f}ms")
