"""RES.04: out-of-domain resolution sensitivity for the FROZEN pixel-26 view.

Uses all 60 human-labelled random Egocentric-10K windows (51 good / 9 bad, 48 clips).
For each window the FROZEN pixel-26 head (logistic C=0.3 balanced, trained on the
150 IndEgo gold windows) is scored at three effective source resolutions, all
decoded through the canonical 160x90 target (ALGO_SPEC):
  native1080p : NVDEC decode 1080p HEVC -> scale_cuda 160:90
  480p        : NVDEC decode -> scale_cuda 854:480 -> scale_cuda 160:90   (one pass)
  360p        : NVDEC decode -> scale_cuda 640:360 -> scale_cuda 160:90   (one pass)
Frames subsampled to ~10 fps (every 3rd of the 30 fps stream) to match the gold
pipeline.  Reports AUROC and delta vs native with a window-bootstrap CI (n small).

Downscale is done at decode time (resolution reduction, no re-encode) so the run
fits the timebox; this isolates the resolution effect from codec artifacts.
The EG2 window-embedding view is NOT recomputed (needs a GPU frame embedder).

GPU command -> MUST be launched via:  bash tools/pod/gpu_run.sh python3 eval/exp_res/run_ood_res.py
Output: eval/results/exp_res/res04_ood_pixel.json
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from jadal.gpu.features import cpu_window_features   # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
GOLD = os.path.join(BUNDLE, "labels", "manifest_gold.json")
CACHED = os.path.join(BUNDLE, "caches", "gpu", "features.json")
QUEUE = os.path.join(BUNDLE, "queues", "label_queue_random.json")
OODLAB = os.path.join(BUNDLE, "labels", "ood_labels.json")
OUT = os.path.join(ROOT, "eval", "results", "exp_res")
DW, DH = 160, 90
PRE = {"native1080p": "", "480p": "scale_cuda=854:480,", "360p": "scale_cuda=640:360,"}


def nvdec_scaled(path, start_s, dur_s, pre, w=DW, h=DH):
    vf = f"{pre}scale_cuda={w}:{h}:format=nv12,hwdownload,format=nv12,format=rgb24"
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
           "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
           "-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}", "-i", path,
           "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        # fall back to CPU decode
        vf2 = vf.replace("scale_cuda=", "scale=").replace(":format=nv12", "").replace(",hwdownload", "").replace(",format=nv12", "")
        cmd2 = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin", "-ss", f"{start_s:.3f}",
                "-t", f"{dur_s:.3f}", "-i", path, "-vf", vf2, "-f", "rawvideo",
                "-pix_fmt", "rgb24", "pipe:1"]
        p = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-200:].decode("utf-8", "replace"))
    px = w * h * 3
    n = len(p.stdout) // px
    if n == 0:
        raise RuntimeError("no frames")
    a = np.frombuffer(p.stdout[:n * px], dtype=np.uint8).astype(np.float64) / 255.0
    return a.reshape(n, h, w, 3)


def luma_of(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def src_feats(args):
    seg, labels = args
    sid = seg["segment_id"]
    src, ss = seg["source_path"], seg["source_start_s"]
    dur = round(seg["end_s"] - seg["start_s"], 2)
    out = {"id": sid, "clip": seg["clip"], "label": labels[sid]}
    for tag, pre in PRE.items():
        try:
            rgb = nvdec_scaled(src, ss, dur, pre)[::3]   # ~10 fps
            out[f"full_{tag}"] = cpu_window_features(luma_of(rgb), rgb, fps=10.0)
        except Exception as e:
            return sid, {"error": f"{tag}: {e}"}
    return sid, out


def frozen_head():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    F = json.load(open(CACHED))["windows"]
    M = {i["id"]: i for i in json.load(open(GOLD))["items"]}
    ids = [k for k in M if k in F and F[k].get("full")]
    names = sorted(F[ids[0]]["full"].keys())
    X = np.nan_to_num(np.array([[F[k]["full"][n] for n in names] for k in ids], float))
    y = np.array([M[k]["label"] for k in ids])
    sc = StandardScaler().fit(X)
    m = LogisticRegression(max_iter=4000, C=0.3, class_weight="balanced").fit(sc.transform(X), y)
    return m, sc, names


def main():
    jobs = int(os.environ.get("RES_JOBS", "4"))
    os.makedirs(OUT, exist_ok=True)
    segs = json.load(open(QUEUE))["segments"]
    labels = {k: (1 if v["verdict"] == "good" else 0)
              for k, v in json.load(open(OODLAB))["latest"].items()}
    m, sc, names = frozen_head()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = dict(ex.map(src_feats, [(s, labels) for s in segs]))
    errs = [k for k, v in rows.items() if "error" in v]
    ok = [k for k, v in rows.items() if "error" not in v]
    y = np.array([rows[k]["label"] for k in ok])
    clips = np.array([rows[k]["clip"] for k in ok])
    from sklearn.metrics import roc_auc_score

    def score(tag):
        X = np.nan_to_num(np.array([[rows[k][tag][n] for n in names] for k in ok], float))
        return m.predict_proba(sc.transform(X))[:, 1]

    conds = {"native1080p": "full_native1080p", "480p": "full_480p", "360p": "full_360p"}
    preds = {c: score(t) for c, t in conds.items()}
    rng = np.random.default_rng(0)
    res = {}
    for c, p in preds.items():
        auc = float(roc_auc_score(y, p))
        bs = []
        for _ in range(2000):
            idx = rng.integers(0, len(y), len(y))
            if len(set(y[idx])) < 2:
                continue
            bs.append(roc_auc_score(y[idx], p[idx]))
        lo, hi = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (float("nan"),) * 2
        res[c] = {"auroc": round(auc, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    for c in ["480p", "360p"]:
        d = []
        for _ in range(2000):
            idx = rng.integers(0, len(y), len(y))
            if len(set(y[idx])) < 2:
                continue
            d.append(roc_auc_score(y[idx], preds[c][idx]) - roc_auc_score(y[idx], preds["native1080p"][idx]))
        res[c]["delta_vs_native"] = round(res[c]["auroc"] - res["native1080p"]["auroc"], 4)
        res[c]["delta_ci95"] = [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
    out = {
        "experiment": "RES.04", "combo": "ood_resolution_pixel26",
        "n": len(ok), "n_good": int(y.sum()), "n_bad": int((1 - y).sum()),
        "n_clips": len(set(clips)), "n_errors": len(errs), "errors": errs[:10],
        "label_source": "human (Egocentric-10K OOD, 51 good/9 bad)",
        "protocol": "frozen pixel-26 head (logistic C=0.3 balanced, trained on 150 IndEgo gold); NVDEC 1080p HEVC window -> scale_cuda to 160x90 (native) or via 854x480 / 640x360 (one-pass downscale, no re-encode); ~10 fps; AUROC + window-bootstrap CI B=2000 seed0",
        "conditions": res,
        "note_eg2": "EG2 window-embedding view NOT recomputed (needs a GPU frame embedder on downscaled clips); pixel-26 only.",
        "note_scale": "downscale done at decode time; isolates resolution effect from codec artifacts",
        "command": "bash tools/pod/gpu_run.sh python3 eval/exp_res/run_ood_res.py",
    }
    json.dump(out, open(os.path.join(OUT, "res04_ood_pixel.json"), "w"), indent=1)
    json.dump(rows, open(os.path.join(OUT, "_feats_res04_ood.json"), "w"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
