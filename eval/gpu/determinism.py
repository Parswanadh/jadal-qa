"""M5 determinism: 10 repeated runs of the same window give bit-identical
quantised outputs (what you hash). CPU part runs without lock; GPU part
needs --gpu (run under flock).

Writes eval/results/gpu/determinism.json
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "gpu", "determinism.json")
GOLD = "/workspace/jadal_pod_bundle/labels/manifest_gold.json"


def canon(feats, nd=4):
    q = {k: (feats[k] if isinstance(feats[k], str) else round(float(feats[k]), nd))
         for k in sorted(feats)}
    return json.dumps(q, sort_keys=True)


def digest(feats):
    return hashlib.sha256(canon(feats).encode()).hexdigest()


def main(use_gpu=False):
    sys.path.insert(0, os.path.join(HERE, "..", ".."))
    from eval.gpu.decode import decode_window
    from jadal.gpu.features import cpu_window_features
    gold = json.load(open(GOLD))["items"]
    g = next(it for it in gold if it["id"] == "g001")
    rgb, fps = decode_window(g["path"], g["start_s"],
                             round(g["end_s"] - g["start_s"], 2), 160, 90)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    res = {"window": "g001", "frames": int(rgb.shape[0]),
           "quantisation": "round(4) of each feature; fp_hash exact"}
    cpu_digests = [digest(cpu_window_features(luma, rgb, fps=min(fps, 30.0)))
                   for _ in range(10)]
    res["cpu_10x_identical"] = len(set(cpu_digests)) == 1
    res["cpu_digest"] = cpu_digests[0]
    if use_gpu:
        import torch
        from jadal.gpu.features import gpu_window_features_batch, FEATURE_NAMES
        assert torch.cuda.is_available()
        torch.cuda.set_device(0)
        lb = torch.tensor(luma.astype("float32")).unsqueeze(0).cuda().half()
        rb = torch.tensor(rgb.astype("float32")).unsqueeze(0).cuda().half()
        gd = []
        for _ in range(10):
            with torch.no_grad():
                t = gpu_window_features_batch(lb, rb, fps=10.0).float().cpu().numpy()[0]
            gd.append(digest(dict(zip(FEATURE_NAMES, map(float, t)))))
        res["gpu_10x_identical"] = len(set(gd)) == 1
        res["gpu_digest"] = gd[0]
        res["gpu_peak_vram_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    res["note"] = ("GPU numerics are NOT claimed identical across devices or "
                   "vs CPU (fp16/fp32 vs float64; tolerances in tests/). "
                   "The edge (Pi) path keeps the CPU reference implementation. "
                   "Only the quantised digest is claimed bit-identical across repeats.")
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main(use_gpu="--gpu" in sys.argv)
