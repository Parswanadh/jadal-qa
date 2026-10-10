"""E-mosaic-res (claim X3): low-res / dense mosaic tiles vs sequential frames.

Q3: if mosaic tiles are lower resolution (480p or smaller) can we pack MORE
frames per canvas so processing is FASTER at the SAME accuracy?

Builds (per gold window, single decode, deterministic):
  * tile-resolution ladder  : 3x3 at tile_px {224,160,112,96,64}
  * density ladder          : tile 112 at grids {2x2,3x3,4x4,5x5,6x6}
  * equal-canvas duel       : 3x3 tile224 (680x680) vs 4x4 tile168 (682x682)
  * control                 : mean of the SAME frames' EG2 embeddings
  * cheap ablation          : 4x4 tile112 timestamps off / tile order shuffled

Eval (same as group E2):
  in-domain leave-one-clip-out over the 7 IndEgo clips (150 gold windows),
  logistic C=0.01 on L2-normalised features, clip-bootstrap CI B=1000 seed0.

Regenerate: bash tools/pod/gpu_run.sh python3 eval/exp_mosaic_res/run_res.py
            [--limit N] [--no-artifacts]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from jadal.mosaic import build as B  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
EG2_DIR = os.environ.get(
    "JADAL_EG2_DIR",
    "/workspace/.cache/huggingface/hub/models--google--embeddinggemma-2/"
    "snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b")
RESULTS = os.path.join(ROOT, "eval", "results", "exp_mosaic_res")
ARTIFACTS = os.environ.get("JADAL_MOSAIC_ARTIFACTS", "/workspace/artifacts/mosaics_res")

WINDOW_S = 10.0
BATCH = 64

# --- configs -----------------------------------------------------------------
TILE_LADDER = [224, 160, 112, 96, 64]                 # grid 3x3 fixed
DENSITY_GRIDS = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]  # tile 112 fixed
DENSITY_TILE = 112
CANVAS_DUEL = [(3, 3, 224), (4, 4, 168)]              # ~equal canvas px
ABL_BASE = (4, 4, 112)
MEAN_NS = [4, 9, 16, 25, 36]


def mosaic_configs():
    """ordered list of (tag, rows, cols, tile_px, timestamps, shuffle)."""
    cfgs = []
    for tp in TILE_LADDER:
        cfgs.append((f"m_3x3_t{tp}", 3, 3, tp, True, False))
    for (r, c) in DENSITY_GRIDS:
        if (r, c) == (3, 3):
            continue  # already built at t112? no: t112 is in ladder; keep once
        cfgs.append((f"m_{r}x{c}_t{DENSITY_TILE}", r, c, DENSITY_TILE, True, False))
    cfgs.append((f"m_4x4_t168", 4, 4, 168, True, False))
    # ablation at best-guess density 4x4 t112
    cfgs.append(("m_4x4_t112_nots", 4, 4, 112, False, False))
    cfgs.append(("m_4x4_t112_shuf", 4, 4, 112, True, True))
    # dedupe by tag preserving order
    seen, out = set(), []
    for k in cfgs:
        if k[0] in seen:
            continue
        seen.add(k[0])
        out.append(k)
    return out


def canvas_px(rows, cols, tile_px):
    return (cols * tile_px + (cols + 1) * 2) * (rows * tile_px + (rows + 1) * 2)


# ------------------------------------------------------------------- data ----
def load_windows():
    man = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))
    wins = []
    for it in man["items"]:
        wins.append({"id": it["id"], "source": "indego", "path": it["path"],
                     "start_s": float(it["start_s"]), "clip": it["clip"],
                     "y": int(it["label"]), "reason": it.get("reason")})
    return wins


def native_fps(path):
    import cv2
    c = cv2.VideoCapture(path)
    fps = c.get(cv2.CAP_PROP_FPS) or 10.0
    c.release()
    return float(fps)


def decode_window(path, start_s, max_frames):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    frames = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        frames.append(fr[..., ::-1].copy())
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames {path}@{start_s}")
    return frames


def needed_indices(native, n_frames):
    idx = set()
    for n in MEAN_NS:
        t = B.tile_times(n, WINDOW_S)
        idx.update(B.frame_indices(t, native, n_frames))
    for (_, r, c, tp, ts, sh) in mosaic_configs():
        n = r * c
        t = B.tile_times(n, WINDOW_S)
        idx.update(B.frame_indices(t, native, n_frames))
    return sorted(idx)


# --------------------------------------------------------------- embedding ----
class Embedder:
    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.manual_seed(1234)
        np.random.seed(1234)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        assert torch.cuda.is_available(), "EmbeddingGemma-2 requires CUDA"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        torch.cuda.reset_peak_memory_stats()
        self.model = SentenceTransformer(str(EG2_DIR),
                                         model_kwargs={"torch_dtype": dtype},
                                         trust_remote_code=True)
        self.model.eval()
        self.torch = torch

    def encode(self, pils):
        if not pils:
            return np.zeros((0, 768), dtype=np.float32)
        with self.torch.inference_mode():
            e = self.model.encode(
                pils, batch_size=BATCH, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
                processing_kwargs={"image": {"max_soft_tokens": 140}})
        return np.asarray(e, dtype=np.float32)

    def peak_mb(self):
        return self.torch.cuda.max_memory_allocated() / 2 ** 20


# ------------------------------------------------------------------ eval ----
def _l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def loco_proba(X, y, clips, C=0.01):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    p = np.zeros(len(y))
    for c in sorted(set(clips)):
        tr, te = clips != c, clips == c
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
        m.fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def _boot_ci(y, p, groups, Bn=1000, seed=0):
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


def evaluate(X, y, clips):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    p = loco_proba(X, y, clips)
    return {"n": int(len(y)), "n_good": int(y.sum()),
            "n_bad": int((1 - y).sum()),
            "auroc": round(float(roc_auc_score(y, p)), 4),
            "ci95": _boot_ci(y, p, clips),
            "bal_acc": round(float(balanced_accuracy_score(
                y, (p > 0.5).astype(int))), 4)}


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-artifacts", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(ARTIFACTS, exist_ok=True)

    wins = load_windows()
    if args.limit:
        wins = wins[::max(1, len(wins) // args.limit)][:args.limit]
    cfgs = mosaic_configs()
    print(f"windows={len(wins)} configs={len(cfgs)}", flush=True)

    fps_cache = {}
    for w in wins:
        if w["path"] not in fps_cache:
            fps_cache[w["path"]] = native_fps(w["path"])
        w["native"] = fps_cache[w["path"]]

    emb = Embedder()
    print(f"model loaded; vram alloc {emb.peak_mb():.0f} MB", flush=True)

    mfeat = {k[0]: [] for k in cfgs}
    meanfeat = {f"mean_N{n}": [] for n in MEAN_NS}
    per_cfg_ms = {k[0]: 0.0 for k in cfgs}
    mean_ms = 0.0
    ys, clips, ids = [], [], []
    cost = {"decode_ms": 0.0, "frame_embed_ms": 0.0, "build_ms": 0.0,
            "mosaic_embed_ms": 0.0, "frames_decoded": 0}
    t_start = time.perf_counter()

    from PIL import Image
    for wi, w in enumerate(wins):
        native = w["native"]
        max_frames = int(round(WINDOW_S * native))
        t0 = time.perf_counter()
        frames = decode_window(w["path"], w["start_s"], max_frames)
        cost["decode_ms"] += (time.perf_counter() - t0) * 1000.0
        cost["frames_decoded"] += len(frames)

        # per-frame embeddings (control + potential tile reuse)
        uidx = needed_indices(native, len(frames))
        t0 = time.perf_counter()
        fpils = [Image.fromarray(frames[k]).convert("RGB") for k in uidx]
        fvecs = emb.encode(fpils)
        cost["frame_embed_ms"] += (time.perf_counter() - t0) * 1000.0
        fmap = {k: v for k, v in zip(uidx, fvecs)}

        # mean-of-frames control per density
        t0 = time.perf_counter()
        for n in MEAN_NS:
            t = B.tile_times(n, WINDOW_S)
            kidx = B.frame_indices(t, native, len(frames))
            E = np.stack([fmap[k] for k in kidx])
            meanfeat[f"mean_N{n}"].append(_l2(E.mean(axis=0)[None, :])[0])
        mean_ms += (time.perf_counter() - t0) * 1000.0

        # mosaics
        t0 = time.perf_counter()
        imgs = []
        for (tag, r, c, tp, ts, sh) in cfgs:
            img, _ = B.build_window(w["id"], w["source"], frames, native,
                                    n_tiles=r * c, tile_px=tp,
                                    timestamps=ts, shuffle_seed=(1 if sh else None))
            imgs.append(img)
        cost["build_ms"] += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        vecs = emb.encode(imgs)
        dt = (time.perf_counter() - t0) * 1000.0
        cost["mosaic_embed_ms"] += dt
        per = dt / max(1, len(cfgs))
        for (tag, *_rest), v in zip(cfgs, vecs):
            mfeat[tag].append(v)
            per_cfg_ms[tag] += per

        # artifact (best-guess density 4x4 t112)
        if not args.no_artifacts:
            ai = [i for i, k in enumerate(cfgs) if k[0] == "m_4x4_t112"][0]
            png = os.path.join(ARTIFACTS, f"{w['id']}.png")
            js = os.path.join(ARTIFACTS, f"{w['id']}.json")
            _, sc = B.build_window(w["id"], w["source"], frames, native,
                                   n_tiles=16, tile_px=112)
            B.save_mosaic(imgs[ai], sc, png, js)

        ys.append(w["y"])
        clips.append(w["clip"])
        ids.append(w["id"])
        if (wi + 1) % 25 == 0 or wi == len(wins) - 1:
            print(f"[{wi+1}/{len(wins)}] {w['id']} "
                  f"{time.perf_counter()-t_start:.0f}s", flush=True)

    y = np.array(ys)
    clips = np.array(clips)
    np.savez_compressed(
        os.path.join(RESULTS, "features.npz"),
        y=y, clips=clips, ids=np.array(ids),
        **{f"m_{k}": np.stack(v).astype(np.float32) for k, v in mfeat.items()},
        **{k: np.stack(v).astype(np.float32) for k, v in meanfeat.items()},
        per_cfg_ms=np.array([per_cfg_ms[k[0]] for k in cfgs]),
        cfg_tags=np.array([k[0] for k in cfgs]),
        mean_ms=np.array([mean_ms]),
        cost=json.dumps(cost))

    # eval every feature block (L2-normalised)
    allfeat = {}
    for k, v in mfeat.items():
        allfeat[k] = _l2(np.stack(v).astype(np.float32))
    for k, v in meanfeat.items():
        allfeat[k] = _l2(np.stack(v).astype(np.float32))

    res = {}
    for k, X in allfeat.items():
        res[k] = evaluate(X, y, clips)
    print("\n=== mosaic-res (AUROC [CI95]) ===")
    for k in sorted(res):
        r = res[k]
        print(f"{k:20s} {r['auroc']:.3f} {r['ci95']} bal {r['bal_acc']:.3f}")
    json.dump({"res": res, "cost": cost, "per_cfg_ms": per_cfg_ms,
               "mean_ms": mean_ms, "n": int(len(y))},
              open(os.path.join(RESULTS, "raw_eval.json"), "w"), indent=1)
    print("cost:", {k: round(v, 1) if isinstance(v, float) else v
                    for k, v in cost.items()})
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
