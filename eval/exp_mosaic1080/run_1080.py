"""H2 true-1080p mosaic test (E2.1080): tile-resolution ladder + equal-canvas duel.

1080p arm: REAL 1080p Ego10K HEVC frames (30 fps), NVDEC cuvid decode with a
select filter (only wanted frames cross the pipe); 480p arm: the labelled
10 s transcodes (batch1 30 fps H264, batch2 10 fps H264). Per-arm fps is
probed; tile times are cell-centred (same times, nearest index per arm).

  ladder (1080p src) : 3x3 t{224,168,112}, 4x4 t{168,112}
  duel (equal canvas): 1080p 3x3/t224 (680^2=462400px) vs 480p 4x4/t168 (682^2=465124px)
  source ablation    : 480p 3x3/t224 vs 1080p 3x3/t224
  control            : mean of the SAME frames' EG2 embeddings (N=9, N=16, 1080p src)
All mosaics: X3 builder (top-right timestamps, JSON sidecars), row-major order.

Eval: stratified 5-fold CV logistic C=0.01 on L2-normalised EG2 embeddings;
out-of-fold AUROC + factory-bootstrap CI (B=1000, seed0). OOD n_bad=22 ->
wide CI; optimistic read, NOT transfer proof.

Regen: bash tools/pod/gpu_run.sh python3 eval/exp_mosaic1080/run_1080.py [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from jadal.mosaic import build as B  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
EG2_DIR = os.environ.get(
    "JADAL_EG2_DIR",
    "/workspace/.cache/huggingface/hub/models--google--embeddinggemma-2/"
    "snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b")
RESULTS = os.path.join(ROOT, "eval", "results", "exp_mosaic1080")
MOSAIC_DIR = os.environ.get("JADAL_MOSAIC1080_DIR", "/workspace/artifacts/mosaics1080")
WINDOW_S = 10.0
BATCH = 16

MOSAIC_CFGS = [
    ("h1080_3x3_t224", 3, 3, 224, "1080p"),
    ("h1080_3x3_t168", 3, 3, 168, "1080p"),
    ("h1080_3x3_t112", 3, 3, 112, "1080p"),
    ("h1080_4x4_t168", 4, 4, 168, "1080p"),
    ("h1080_4x4_t112", 4, 4, 112, "1080p"),
    ("h480_4x4_t168", 4, 4, 168, "480p"),
    ("h480_3x3_t224", 3, 3, 224, "480p"),
]
MEAN_NS = [9, 16]
ARMS = {"1080p": (1920, 1080, "hevc_cuvid"), "480p": (854, 480, "h264_cuvid")}


def canvas_px(r, c, tp):
    return (c * tp + (c + 1) * B.GUTTER) * (r * tp + (r + 1) * B.GUTTER)


def load_windows():
    lab = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))
    q1 = json.load(open(os.path.join(BUNDLE, "queues", "label_queue_random.json")))
    q2 = json.load(open(os.path.join(BUNDLE, "queues", "label_queue_random2.json")))
    seg = {s["segment_id"]: s for s in q1["segments"]}
    seg.update({s["segment_id"]: s for s in q2["segments"]})
    wins = []
    for w in lab["windows"]:
        s = seg[w["segment_id"]]
        wins.append({"id": w["segment_id"], "y": 1 if w["verdict"] == "good" else 0,
                     "factory": w["factory"], "reason": w.get("reason", ""),
                     "src1080": s["source_path"], "src_start": float(s["source_start_s"]),
                     "path480": os.path.join(BUNDLE, w["video"])})
    return wins


_FPS_CACHE = {}


def probe_fps(path):
    if path in _FPS_CACHE:
        return _FPS_CACHE[path]
    o = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate", "-of",
         "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True).stdout.strip()
    num, den = o.split("/")
    fps = float(num) / float(den)
    _FPS_CACHE[path] = fps
    return fps


def arm_plan(path, start_s, ns):
    """Return dict with fps, n_frames, and per-N wanted native indices."""
    fps = probe_fps(path)
    nf = int(round(WINDOW_S * fps))
    return {"fps": fps, "nf": nf,
            "want": {n: B.frame_indices(B.tile_times(n, WINDOW_S), fps, nf) for n in ns},
            "start": start_s}


def decode_want(path, start_s, want_sorted, w, h, codec, order):
    """Decode 10 s window; select-filter keeps only wanted indices (in order)."""
    sel = "+".join(f"eq(n\\,{k})" for k in want_sorted)
    last_err = ""
    for att in order:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if att == "nvdec":
            cmd += ["-c:v", codec]
        cmd += ["-ss", f"{start_s:.3f}", "-i", path, "-t", f"{WINDOW_S}",
                "-vf", f"select='{sel}'", "-vsync", "0",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            frames = []
            fb = w * h * 3
            for _ in want_sorted:
                buf = p.stdout.read(fb)
                if not buf or len(buf) < fb:
                    break
                frames.append(np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy())
            try:
                p.stdout.close()
            except Exception:
                pass
            p.wait()
            if len(frames) >= len(want_sorted) - 1:
                return dict(zip(want_sorted, frames)), att
            last_err = f"short: {len(frames)}/{len(want_sorted)}"
        except Exception as e:
            last_err = str(e)
    return None, last_err


class Embedder:
    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.manual_seed(1234)
        np.random.seed(1234)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        assert torch.cuda.is_available(), "EG2 requires CUDA"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.model = SentenceTransformer(str(EG2_DIR),
                                         model_kwargs={"torch_dtype": dtype},
                                         trust_remote_code=True)
        self.model.eval()
        self.torch = torch

    def encode(self, pils, batch=BATCH):
        if not pils:
            return np.zeros((0, 768), dtype=np.float32)
        with self.torch.inference_mode():
            e = self.model.encode(
                pils, batch_size=batch, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
                processing_kwargs={"image": {"max_soft_tokens": 140}})
        return np.asarray(e, dtype=np.float32)


def _l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def cv_proba(X, y, seed=0, k=5, C=0.01):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
        m.fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def factory_boot_ci(y, p, groups, Bn=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    names = sorted(set(groups))
    out = []
    for _ in range(Bn):
        gs = rng.choice(names, len(names))
        idx = np.concatenate([np.where(groups == g)[0] for g in gs])
        if len(set(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    return [round(float(np.percentile(out, 2.5)), 4),
            round(float(np.percentile(out, 97.5)), 4)]


def evaluate(X, y, groups):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    p = cv_proba(_l2(X), y)
    return {"n": int(len(y)), "n_good": int(y.sum()), "n_bad": int((1 - y).sum()),
            "auroc": round(float(roc_auc_score(y, p)), 4),
            "ci95": factory_boot_ci(y, p, groups),
            "bal_acc": round(float(balanced_accuracy_score(y, (p > 0.5).astype(int))), 4)}


def build_one(wid, src, frames, times, sidx, grid, tile_px, fps_arm):
    img, sc = B.build_mosaic(frames, times, sidx, grid, tile_px,
                             window_id=wid, source=src, fps_used=fps_arm)
    return img, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--decode", default="nvdec", choices=["nvdec", "cpu"])
    ap.add_argument("--skip-mean", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(MOSAIC_DIR, exist_ok=True)

    wins = load_windows()
    if args.limit:
        bad = [w for w in wins if w["y"] == 0]
        good = [w for w in wins if w["y"] == 1]
        ng = max(0, args.limit - len(bad))
        good = good[::max(1, len(good) // max(1, ng))][:ng]
        wins = bad + good
    print(f"windows={len(wins)} (bad={sum(1 for w in wins if w['y']==0)})", flush=True)
    order = ["nvdec", "cpu"] if args.decode == "nvdec" else ["cpu"]

    from PIL import Image
    t_all = time.perf_counter()
    build_ms = {c[0]: 0.0 for c in MOSAIC_CFGS}
    cost = {"decode_ms": 0.0, "build_ms": 0.0, "mosaic_embed_ms": 0.0,
            "frame_embed_ms": 0.0}
    dec_used, kept, meta = {}, [], {}
    cache10, cache48 = {}, {}  # wid -> (frames dict, plan)

    for wi, w in enumerate(wins):
        t0 = time.perf_counter()
        p10 = arm_plan(w["src1080"], w["src_start"], MEAN_NS)
        p48 = arm_plan(w["path480"], 0.0, MEAN_NS)
        w10, h10, c10 = ARMS["1080p"]
        w48, h48, c48 = ARMS["480p"]
        want10 = sorted(set(p10["want"][9]) | set(p10["want"][16]))
        want48 = sorted(set(p48["want"][9]) | set(p48["want"][16]))
        f10, u10 = decode_want(w["src1080"], w["src_start"], want10, w10, h10, c10, order)
        f48, u48 = decode_want(w["path480"], 0.0, want48, w48, h48, c48, order)
        cost["decode_ms"] += (time.perf_counter() - t0) * 1000.0
        du = f"{u10}+{u48}"
        dec_used[du] = dec_used.get(du, 0) + 1
        if f10 is None or f48 is None:
            print(f"[{wi+1}] SKIP {w['id']} ({du})", flush=True)
            continue
        t0 = time.perf_counter()
        ok = True
        for (tag, r, c, tp, src) in MOSAIC_CFGS:
            n = r * c
            plan = p10 if src == "1080p" else p48
            fr = f10 if src == "1080p" else f48
            times = B.tile_times(n, WINDOW_S)
            sidx = B.frame_indices(times, plan["fps"], plan["nf"])
            if any(k not in fr for k in sidx):
                ok = False
                break
            img, sc = build_one(w["id"], f"ego10k-{src}", [fr[k] for k in sidx],
                                times, sidx, (r, c), tp, plan["fps"])
            B.save_mosaic(img, sc, os.path.join(MOSAIC_DIR, f"{w['id']}_{tag}.png"),
                          os.path.join(MOSAIC_DIR, f"{w['id']}_{tag}.json"))
        dtb = (time.perf_counter() - t0) * 1000.0
        if not ok:
            print(f"[{wi+1}] SKIP {w['id']} (frames missing)", flush=True)
            continue
        cost["build_ms"] += dtb
        for (tag, *_r) in MOSAIC_CFGS:
            build_ms[tag] += dtb / len(MOSAIC_CFGS)
        kept.append(w)
        cache10[w["id"]] = (f10, p10)
        cache48[w["id"]] = (f48, p48)
        meta[w["id"]] = {"y": w["y"], "factory": w["factory"], "reason": w["reason"]}
        if (wi + 1) % 10 == 0 or wi == len(wins) - 1:
            print(f"[{wi+1}/{len(wins)}] built {w['id']} "
                  f"{time.perf_counter()-t_all:.0f}s", flush=True)
    print(f"phase1 kept={len(kept)} decode={dec_used} "
          f"{time.perf_counter()-t_all:.0f}s", flush=True)
    if not kept:
        raise SystemExit("nothing decoded; aborting")

    emb = Embedder()
    print("EG2 loaded", flush=True)
    mfeat, enc_ms = {}, {}
    for (tag, *_r) in MOSAIC_CFGS:
        ids = [w["id"] for w in kept]
        t0 = time.perf_counter()
        vecs = emb.encode([Image.open(os.path.join(MOSAIC_DIR, f"{i}_{tag}.png")).convert("RGB") for i in ids])
        dte = (time.perf_counter() - t0) * 1000.0
        cost["mosaic_embed_ms"] += dte
        enc_ms[tag] = dte / max(1, len(ids))
        mfeat[tag] = (ids, vecs)
        print(f"  {tag}: {len(ids)} imgs {dte / max(1, len(ids)):.0f} ms/win", flush=True)

    meanfeat, mean_ms = {}, 0.0
    if not args.skip_mean:
        for n in MEAN_NS:
            key = f"mean1080_N{n}"
            vecs, tsum = [], 0.0
            for w in kept:
                f10, p10 = cache10[w["id"]]
                fp = [Image.fromarray(f10[k]).convert("RGB") for k in p10["want"][n]]
                t0 = time.perf_counter()
                E = emb.encode(fp)
                tsum += (time.perf_counter() - t0) * 1000.0
                vecs.append(_l2(E.mean(axis=0)[None, :])[0])
            meanfeat[key] = ([w["id"] for w in kept], np.stack(vecs))
            mean_ms = tsum / max(1, len(kept))
            cost["frame_embed_ms"] += tsum
    # free frame RAM before eval
    cache10.clear()
    cache48.clear()

    common = sorted(set.intersection(*[set(v[0]) for v in
                                       list(mfeat.values()) + list(meanfeat.values())]))
    y = np.array([meta[i]["y"] for i in common])
    facs = np.array([meta[i]["factory"] for i in common])
    feats = {}
    for k, (ids, V) in list(mfeat.items()) + list(meanfeat.items()):
        pos = {j: q for q, j in enumerate(ids)}
        feats[k] = np.stack([V[pos[i]] for i in common]).astype(np.float32)
    np.savez_compressed(os.path.join(RESULTS, "features.npz"), y=y,
                        groups=np.array(facs), ids=np.array(common), **feats)

    res = {k: evaluate(X, y, facs) for k, X in feats.items()}
    print("\n=== true-1080p mosaic (OOF AUROC [factory-boot CI]) ===")
    for k in sorted(res):
        r = res[k]
        print(f"{k:18s} {r['auroc']:.3f} {r['ci95']} bal {r['bal_acc']:.3f}")

    n = len(common)
    per_cfg = {}
    for (tag, r, c, tp, src) in MOSAIC_CFGS:
        per_cfg[tag] = {"canvas_px": canvas_px(r, c, tp), "grid": [r, c],
                        "tile_px": tp, "src": src,
                        "build_ms_per_win": round(build_ms[tag] / n, 1),
                        "embed_ms_per_win": round(enc_ms[tag], 1)}
    json.dump({"res": res, "per_cfg": per_cfg, "n": n,
               "ids": common, "meta": {i: meta[i] for i in common},
               "cost": {k: round(v, 1) for k, v in cost.items()},
               "decode_paths": dec_used,
               "mean_ms_per_win": round(mean_ms, 1)},
              open(os.path.join(RESULTS, "raw_eval.json"), "w"), indent=1)
    print("wrote", RESULTS, f"total {time.perf_counter()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
