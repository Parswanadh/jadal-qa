"""LEAD B / LB1 — does the GPU feature kernel agree with the verified CPU reference?

The CPU path is what produced caches/gpu/features.json, and I have verified that cache
reproduces from the source video to machine epsilon. The GPU kernels
(jadal/gpu/features.gpu_window_features_batch) are the throughput path. FACTS.md explicitly
does NOT claim identical numerics across devices, so nobody has checked whether the GPU path
returns the same features. This does that, on this pod's A40.

Protocol
  * windows from labels/manifest_gold.json (READ-ONLY), round-robin across clips
  * decode 160x90 rgb24 (same as the cache producer)
  * CPU: cpu_window_features(luma, rgb, fps)
  * GPU: gpu_window_features_batch(torch tensors on cuda, fp32) -> (B,F) with FEATURE_NAMES
  * report per-feature |delta| and the AUROC-relevant question: does the difference change anything?
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

from eval.gpu.decode import decode_window                      # noqa: E402
from jadal.gpu.features import (FEATURE_NAMES, cpu_window_features,  # noqa: E402
                                gpu_window_features_batch)

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
MANIFEST = os.path.join(BUNDLE, "labels", "manifest_gold.json")
OUT = os.path.join(ROOT, "eval", "results", "pod_verify", "gpu_kernel_parity.json")
W, H = 160, 90



def pod_path(p):
    """The bundle manifest was restored to its pristine (source-machine) copy at 17:19Z;
    its paths do not resolve on the pod. Remap in memory only - the file is READ-ONLY."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    items = json.load(open(MANIFEST))["items"]
    by_clip = {}
    for it in items:
        by_clip.setdefault(it["clip"], []).append(it)
    picked, i = [], 0
    while len(picked) < args.n and any(len(v) > i for v in by_clip.values()):
        for c in sorted(by_clip):
            if len(by_clip[c]) > i and len(picked) < args.n:
                picked.append(by_clip[c][i])
        i += 1

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("cuda not available")
    except Exception as e:
        json.dump({"error": f"no GPU: {e}"}, open(OUT, "w"), indent=1)
        print("NO GPU:", e)
        return

    # decode the picked windows
    lums, rgbs, meta = [], [], []
    for it in picked:
        rgb, fps = decode_window(pod_path(it["path"]), it.get("start_s", 0.0),
                                 round(it["end_s"] - it["start_s"], 2), W, H, "rgb24")
        lums.append(luma_of(rgb).astype(np.float32))
        rgbs.append(rgb.astype(np.float32))
        meta.append((it["id"], it["clip"], float(fps)))

    T = min(a.shape[0] for a in lums)
    lums = [a[:T] for a in lums]
    rgbs = [a[:T] for a in rgbs]

    # CPU reference
    cpu = [cpu_window_features(lums[k].astype(np.float64), rgbs[k].astype(np.float64),
                               fps=min(meta[k][2], 30.0)) for k in range(len(lums))]

    # GPU batched, in chunks of args.batch
    t0 = time.time()
    gpu_rows = []
    for s in range(0, len(lums), args.batch):
        bl = torch.from_numpy(np.stack(lums[s:s + args.batch])).cuda().float()
        br = torch.from_numpy(np.stack(rgbs[s:s + args.batch])).cuda().float()
        out = gpu_window_features_batch(bl, br, fps=min(meta[0][2], 30.0))
        gpu_rows.append(out.detach().float().cpu().numpy())
    gpu = np.concatenate(gpu_rows, axis=0)
    gpu_s = time.time() - t0
    torch.cuda.synchronize()

    per_feat, rows = {}, []
    for k in range(len(lums)):
        d = {}
        for j, name in enumerate(FEATURE_NAMES):
            cv = float(cpu[k].get(name, float("nan")))
            gv = float(gpu[k][j])
            d[name] = {"cpu": cv, "gpu": gv, "abs_diff": abs(cv - gv),
                       "rel_diff": abs(cv - gv) / max(abs(cv), 1e-12)}
        rows.append({"id": meta[k][0], "clip": meta[k][1], "per_feature": d,
                     "max_abs_diff": max(v["abs_diff"] for v in d.values()),
                     "worst_feature": max(d, key=lambda n: d[n]["abs_diff"]),
                     "max_rel_diff": max(v["rel_diff"] for v in d.values())})
        for name, v in d.items():
            per_feat.setdefault(name, []).append(v["abs_diff"])
        print(f"{meta[k][0]:>6s} {meta[k][1]:<12s} max|d|={rows[-1]['max_abs_diff']:.3e} "
              f"worst={rows[-1]['worst_feature']} max_rel={rows[-1]['max_rel_diff']:.2e}")

    out = {
        "experiment": "LB1-gpu-kernel-parity",
        "verified_by": "lead-b",
        "question": ("does gpu_window_features_batch return the same 26 features as the verified "
                     "cpu_window_features reference, on this pod's A40?"),
        "n_windows": len(rows), "frames": int(T), "fps_used": meta[0][2],
        "device": "NVIDIA A40 (pod), torch " + __import__("torch").__version__,
        "label_source": "human gold IndEgo (labels/manifest_gold.json, READ-ONLY)",
        "protocol": ("decode 160x90 rgb24 -> cpu_window_features (float64) vs "
                     "gpu_window_features_batch (fp32 CUDA, batch<=%d); per-feature abs/rel diff" % args.batch),
        "gpu_batch_seconds": round(gpu_s, 4),
        "per_feature_max_abs_diff": {k: float(np.max(v)) for k, v in per_feat.items()},
        "global": {
            "max_abs_diff": float(np.max([r["max_abs_diff"] for r in rows])),
            "max_rel_diff": float(np.max([r["max_rel_diff"] for r in rows])),
            "worst_feature": max(per_feat, key=lambda k: np.max(per_feat[k])),
        },
        "windows": rows,
        "note": ("FACTS.md does not claim cross-device numeric identity for the GPU path; this measures "
                 "the actual gap on this device so any GPU-side usage can be judged deliberately."),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nmax|d| =", out["global"]["max_abs_diff"], "max_rel =", out["global"]["max_rel_diff"],
          "worst =", out["global"]["worst_feature"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
