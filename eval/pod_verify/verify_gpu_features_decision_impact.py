"""LEAD B / LB1 — does the GPU feature path change the DECISION, or only the numbers?

verify_gpu_kernel_parity.py showed that the GPU kernels agree with the verified CPU reference on
22 of 26 features to <1e-6, but four move materially (rppg_pulse max |delta| 0.384 / 66% relative,
radial_slope 0.039, dup_hash_mean 0.025, pca_top1_share 5.9e-4).

That raises the only question that matters for the throughput story: if the fast GPU path is used
instead of the CPU path, does the headline in-domain number move?

This runs ALL 150 gold windows through the GPU kernels and scores them with the SAME LOCO protocol
used for the published 0.810, then compares:

  * per-feature agreement, CPU vs GPU, over all 150 windows
  * LOCO AUROC from GPU features vs the published 0.810 (and the CPU features recomputed here)
  * the paired clip-bootstrap delta, so the comparison is honest rather than eyeballed

CPU features are NOT recomputed here (that takes ~70 s of decode either way; the cache is already
verified to machine epsilon against a fresh derivation — see feature_reproduction_all150.json); the
cached values are used as the CPU reference.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.metrics import roc_auc_score                    # noqa: E402
from sklearn.preprocessing import StandardScaler             # noqa: E402

from eval.gpu.decode import decode_window                    # noqa: E402
from jadal.gpu.features import FEATURE_NAMES, gpu_window_features_batch  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
OUT = os.path.join(ROOT, "eval", "results", "pod_verify", "gpu_features_decision_impact.json")
W, H, NBOOT, SEED = 160, 90, 2000, 0


def pod_path(p):
    if os.path.exists(p):
        return p
    for a, b in (("/home/parshu/projects/awing/data/", "/workspace/jadal_pod_bundle/data/"),
                 ("/home/parshu/projects/jadal-qa-wt/spectral_data/", "/workspace/jadal_pod_bundle/labels/"),
                 ("/home/parshu/projects/jadal-qa-wt/", "/workspace/jadal_pod_bundle/")):
        if p.startswith(a) and os.path.exists(b + p[len(a):]):
            return b + p[len(a):]
    return p


def luma_of(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def loco(X, y, clip, C=0.3):
    p = np.zeros(len(y))
    for c in sorted(set(clip)):
        tr, te = clip != c, clip == c
        sc = StandardScaler().fit(X[tr])
        p[te] = LogisticRegression(max_iter=3000, C=C, class_weight="balanced") \
            .fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]
    return p


def clip_bootstrap_delta(y, p_gpu, p_cpu, clip, n=NBOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    uc = np.array(sorted(set(clip)))
    d = []
    for _ in range(n):
        pick = rng.choice(uc, size=len(uc), replace=True)
        idx = np.concatenate([np.where(clip == c)[0] for c in pick])
        if len(set(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], p_gpu[idx]) - roc_auc_score(y[idx], p_cpu[idx]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 150")
    args = ap.parse_args()

    import torch
    assert torch.cuda.is_available(), "no GPU"
    torch.cuda.set_device(0)

    cache = json.load(open(os.path.join(BUNDLE, "caches", "gpu", "features.json")))["windows"]
    items = [i for i in json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))["items"]]
    if args.limit:
        items = items[: args.limit]
    items = [i for i in items if i["id"] in cache]

    rows, t0 = [], time.time()
    dec_s = 0.0
    for s in range(0, len(items), args.batch):
        chunk = items[s:s + args.batch]
        lums, rgbs, fps_l = [], [], []
        for it in chunk:
            ta = time.time()
            rgb, fps = decode_window(pod_path(it["path"]), it.get("start_s", 0.0),
                                     round(it["end_s"] - it["start_s"], 2), W, H, "rgb24")
            dec_s += time.time() - ta
            lums.append(luma_of(rgb).astype(np.float32))
            rgbs.append(rgb.astype(np.float32))
            fps_l.append(min(fps, 30.0))
        T = min(a.shape[0] for a in lums)
        bl = torch.from_numpy(np.stack([a[:T] for a in lums])).cuda().float()
        br = torch.from_numpy(np.stack([a[:T] for a in rgbs])).cuda().float()
        out = gpu_window_features_batch(bl, br, fps=fps_l[0]).detach().float().cpu().numpy()
        for j, it in enumerate(chunk):
            cpu = cache[it["id"]]["full"]
            g = {n: float(out[j][k]) for k, n in enumerate(FEATURE_NAMES)}
            d = {n: abs(g[n] - float(cpu[n])) for n in g if n in cpu}
            rows.append({"id": it["id"], "clip": it["clip"], "label": it["label"],
                         "gpu": g, "per_feature_abs_diff": d,
                         "max_abs_diff": max(d.values()), "worst": max(d, key=d.get)})
        if (s // args.batch) % 8 == 0:
            print(f"  {len(rows)}/{len(items)} windows  ({time.time()-t0:.0f}s)", flush=True)

    names = [n for n in FEATURE_NAMES if n in rows[0]["gpu"]]
    y = np.array([r["label"] for r in rows])
    clip = np.array([r["clip"] for r in rows])
    Xgpu = np.array([[r["gpu"][n] for n in names] for r in rows], float)
    Xcpu = np.array([[float(cache[r["id"]]["full"][n]) for n in names] for r in rows], float)
    Xgpu, Xcpu = np.nan_to_num(Xgpu), np.nan_to_num(Xcpu)

    pg, pc = loco(Xgpu, y, clip), loco(Xcpu, y, clip)
    ag, ac = float(roc_auc_score(y, pg)), float(roc_auc_score(y, pc))
    lo, hi, mean = clip_bootstrap_delta(y, pg, pc, clip)
    per_feat = {n: float(np.max([r["per_feature_abs_diff"][n] for r in rows])) for n in names}

    out = {
        "experiment": "LB1-gpu-features-decision-impact",
        "verified_by": "lead-b",
        "question": ("the GPU feature path moves 4 of 26 features; does it move the verdict-level number?"),
        "n": len(rows), "n_bad": int((y == 0).sum()), "n_clips": len(set(clip)),
        "label_source": "human gold IndEgo (labels/manifest_gold.json READ-ONLY)",
        "protocol": "LOCO logistic C=0.3 class_weight=balanced on all 26 features; GPU features from "
                    "gpu_window_features_batch (fp32 CUDA), CPU features from the verified cache; "
                    "clip-bootstrap 95% CI on the paired AUROC difference (2000, seed 0)",
        "device": "NVIDIA A40, torch " + torch.__version__,
        "timing_s": {"decode": round(dec_s, 1), "total": round(time.time() - t0, 1)},
        "auroc_gpu_features": round(ag, 4),
        "auroc_cpu_features": round(ac, 4),
        "published_reference": 0.810,
        "paired_delta_gpu_minus_cpu": {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]},
        "per_feature_max_abs_diff": per_feat,
        "features_moving_more_than_1e-3": {k: v for k, v in per_feat.items() if v > 1e-3},
        "windows": [{"id": r["id"], "clip": r["clip"], "max_abs_diff": r["max_abs_diff"],
                     "worst": r["worst"]} for r in rows],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nGPU features LOCO AUROC {ag:.4f} | CPU features {ac:.4f} | published 0.810")
    print(f"paired delta (gpu-cpu) = {mean:+.4f} [{lo:+.4f},{hi:+.4f}]")
    print("features moving >1e-3:", {k: round(v, 4) for k, v in out["features_moving_more_than_1e-3"].items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
