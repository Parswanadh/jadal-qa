"""E2 mosaic variants — build, embed (EmbeddingGemma-2), evaluate.

One entry point that, for every labelled window (150 IndEgo human + 60
Egocentric-10K human-labelled), decodes the native-rate frames once, builds the
mosaic variants, embeds them with EmbeddingGemma-2 (local HF cache, offline)
and evaluates every feature set with the SAME protocol:

  * in-domain: leave-one-clip-out over the 7 IndEgo clips + clip-bootstrap CI
  * out-of-domain: logistic frozen on all IndEgo, scored on the 60 OOD windows,
    AUROC + window-bootstrap CI (optimistic; n_bad = 9 -> exploratory)

Variants / registry ids:
  E2.01 mosaic embedding (N in the spec grids + fps {1,2,3,5,10})
  E2.02 mean-of-frames control on the SAME frames (matched folds)
  E2.03 shuffled tile order (N=9)
  E2.04 no timestamps (N=9)
  E2.05 single middle frame (1x1)
  E2.06 motion mosaic |I_t - I_{t-1}| (N=9)
  E2.07 pooled per-tile statistics concat(mean,std) of N=9 per-tile embeddings

Frame sampling rule (documented, same for all datasets):
  t_i = (i + 0.5) * window_s / N ,  src_frame_index = round(t_i * native_fps)

Regenerate: bash tools/pod/gpu_run.sh python3 eval/exp_mosaic/run_variants.py
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
RESULTS = os.path.join(ROOT, "eval", "results", "exp_mosaic")
ARTIFACTS = os.environ.get("JADAL_MOSAIC_ARTIFACTS", "/workspace/artifacts/mosaics")

WINDOW_S = 10.0
TILE_PX = 224
REF_N = 9
SPEC_NS = [4, 6, 9, 12, 16]           # canonical grids
FPS_NS = [10, 20, 30, 50, 100]        # fps {1,2,3,5,10} * 10 s
N_SET = SPEC_NS + FPS_NS
BATCH = 64


# ---------------------------------------------------------------- data ----
def load_windows():
    """[(window_id, source, path, start_s, native_fps, y, clip)] for both sets."""
    wins = []
    man = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))
    for it in man["items"]:
        wins.append({"id": it["id"], "source": "indego", "path": it["path"],
                     "start_s": float(it["start_s"]), "clip": it["clip"],
                     "y": int(it["label"]), "reason": it.get("reason")})
    q = json.load(open(os.path.join(BUNDLE, "queues", "label_queue_random.json")))
    lab = json.load(open(os.path.join(BUNDLE, "labels", "ood_labels.json")))["latest"]
    for s in q["segments"]:
        sid = s["segment_id"]
        v = lab.get(sid)
        if v is None:
            continue
        wins.append({"id": sid, "source": "ood", "path": s["path"],
                     "start_s": float(s.get("start_s", 0.0)),
                     "clip": s.get("factory", sid), "y": 1 if v["verdict"] == "good" else 0,
                     "reason": v.get("reason")})
    return wins


def native_fps(path):
    import cv2
    c = cv2.VideoCapture(path)
    fps = c.get(cv2.CAP_PROP_FPS) or 10.0
    c.release()
    return float(fps)


def decode_window(path, start_s, window_s, native, max_frames):
    """Sequential decode of the window at native rate; RGB uint8 list."""
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
        frames.append(fr[..., ::-1].copy())  # BGR -> RGB
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames {path}@{start_s}")
    return frames


def needed_indices(native, n_frames):
    """Union of cell-centred frame indices over every N in N_SET, plus middle."""
    idx = set()
    for n in N_SET:
        t = B.tile_times(n, WINDOW_S)
        idx.update(B.frame_indices(t, native, n_frames))
    mid = min(int(round(0.5 * WINDOW_S * native)), n_frames - 1)
    idx.add(mid)
    return sorted(idx)


# ------------------------------------------------------------ embedding ----
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
        with self.torch.inference_mode():
            e = self.model.encode(
                pils, batch_size=BATCH, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
                processing_kwargs={"image": {"max_soft_tokens": 140}})
        return np.asarray(e, dtype=np.float32)

    def peak_mb(self):
        return self.torch.cuda.max_memory_allocated() / 2 ** 20


# ---------------------------------------------------------------- eval ----
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


def frozen_proba(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=4000, C=0.01, class_weight="balanced")
    m.fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


def _boot_ci(y, p, groups, B=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    names = sorted(set(groups))
    out = []
    for _ in range(B):
        gs = rng.choice(names, len(names))
        idx = np.concatenate([np.where(groups == g)[0] for g in gs])
        if len(set(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    return [round(float(np.percentile(out, 2.5)), 4),
            round(float(np.percentile(out, 97.5)), 4)]


def evaluate(X, y, clips, ood_mask, tag):
    """LOCO on in-domain + frozen logistic on OOD. Returns result dict."""
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    ind = ~ood_mask
    Xi, yi, ci = X[ind], y[ind], np.asarray(clips)[ind]
    p = loco_proba(Xi, yi, ci)
    auroc = float(roc_auc_score(yi, p))
    ci95 = _boot_ci(yi, p, ci)
    bal = float(balanced_accuracy_score(yi, (p > 0.5).astype(int)))
    Xo, yo = X[ood_mask], y[ood_mask]
    po = frozen_proba(Xi, yi, Xo)
    auroc_o = float(roc_auc_score(yo, po)) if len(set(yo)) == 2 else float("nan")
    ci_o = _boot_ci(yo, po, np.arange(len(yo)))
    return {"tag": tag, "indego": {"n": int(len(yi)), "n_good": int(yi.sum()),
                                   "n_bad": int((1 - yi).sum()),
                                   "auroc": round(auroc, 4), "ci95": ci95,
                                   "bal_acc": round(bal, 4)},
            "ood": {"n": int(len(yo)), "n_good": int(yo.sum()),
                    "n_bad": int((1 - yo).sum()),
                    "auroc": round(auroc_o, 4), "ci95": ci_o}}


def cached_reference():
    """EG2 window-embedding logistic from the cached segments.npz (sanity)."""
    z = np.load(os.path.join(BUNDLE, "caches", "frames10k", "segments.npz"),
                allow_pickle=True)
    clips = np.array([str(c) for c in z["clips"]])
    y = np.array(z["labels"]).astype(int)
    E = z["F"].astype(np.float32)
    cnt = np.array(z["counts"]).astype(int)
    X = _l2(np.stack([E[k, :max(1, int(cnt[k]))].mean(axis=0) for k in range(len(y))]))
    p = loco_proba(X, y, clips, C=0.01)
    from sklearn.metrics import roc_auc_score
    return {"auroc": round(float(roc_auc_score(y, p)), 4),
            "ci95": _boot_ci(y, p, clips), "n": int(len(y)),
            "note": "cached 0.5s-spacing EG2 window mean, 2fps recipe"}


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke-test window cap")
    ap.add_argument("--no-artifacts", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if not args.no_artifacts:
        os.makedirs(ARTIFACTS, exist_ok=True)

    wins = load_windows()
    if args.limit:
        step = max(1, len(wins) // args.limit)
        wins = wins[::step][:args.limit]
    print(f"windows: {len(wins)} (indego {sum(w['source']=='indego' for w in wins)}, "
          f"ood {sum(w['source']=='ood' for w in wins)})", flush=True)

    # resolve native fps once per source path
    fps_cache = {}
    for w in wins:
        if w["path"] not in fps_cache:
            fps_cache[w["path"]] = native_fps(w["path"])
        w["native"] = fps_cache[w["path"]]

    emb = Embedder()
    print(f"model loaded; vram alloc {emb.peak_mb():.0f} MB", flush=True)

    # feature accumulation: tag -> list of vectors, plus bookkeeping
    feats = {}
    for n in N_SET:
        feats[f"mosaic_N{n}"] = []
        feats[f"meanframes_N{n}"] = []
    for t in ("shuffled", "no_ts", "single_middle", "motion", "pooled_stats"):
        feats[t] = []
    ys, clips, ood_mask, ids = [], [], [], []
    cost = {"decode_ms": 0.0, "build_ms": 0.0, "embed_ms": 0.0,
            "frames_decoded": 0, "n_windows": len(wins)}
    t_start = time.perf_counter()

    for wi, w in enumerate(wins):
        t0 = time.perf_counter()
        native = w["native"]
        max_frames = int(round(WINDOW_S * native))
        idx_need = needed_indices(native, max_frames)
        frames = decode_window(w["path"], w["start_s"], WINDOW_S, native,
                               max_frames)
        cost["decode_ms"] += (time.perf_counter() - t0) * 1000.0
        cost["frames_decoded"] += len(frames)

        # --- per-frame embeddings (control): raw frames, EG2 preprocessing ---
        from PIL import Image
        t0 = time.perf_counter()
        frame_pils = [Image.fromarray(frames[k]).convert("RGB") for k in idx_need]
        frame_emb = {k: e for k, e in zip(idx_need, emb.encode(frame_pils))}
        cost["embed_ms"] += (time.perf_counter() - t0) * 1000.0

        def frame_vec(n):
            t = B.tile_times(n, WINDOW_S)
            kidx = B.frame_indices(t, native, len(frames))
            E = np.stack([frame_emb[k] for k in kidx])
            return _l2(E.mean(axis=0)[None, :])[0]

        for n in N_SET:
            feats[f"meanframes_N{n}"].append(frame_vec(n))

        # --- mosaic images per N ---
        t0 = time.perf_counter()
        mosaic_imgs, sidecars = {}, {}
        for n in N_SET:
            img, sc = B.build_window(w["id"], w["source"], frames, native,
                                     n_tiles=n, tile_px=TILE_PX)
            mosaic_imgs[f"mosaic_N{n}"] = img
            sidecars[f"mosaic_N{n}"] = sc
        s_img, _ = B.build_window(w["id"], w["source"], frames, native,
                                  n_tiles=REF_N, tile_px=TILE_PX, shuffle_seed=1)
        nt_img, _ = B.build_window(w["id"], w["source"], frames, native,
                                   n_tiles=REF_N, tile_px=TILE_PX, timestamps=False)
        mo_img, _ = B.build_window(w["id"], w["source"], frames, native,
                                   n_tiles=REF_N, tile_px=TILE_PX, motion=True)
        mid_idx = min(int(round(0.5 * WINDOW_S * native)), len(frames) - 1)
        mid_img, mid_sc = B.build_mosaic(
            [frames[mid_idx]], [0.5 * WINDOW_S], [mid_idx], (1, 1), TILE_PX,
            window_id=w["id"], source=w["source"], fps_used=1.0 / WINDOW_S)
        # per-tile (REF_N) letterboxed tiles for the pooled-stats control
        tref = B.tile_times(REF_N, WINDOW_S)
        tidx = B.frame_indices(tref, native, len(frames))
        tile_pils = [B.letterbox_tile(frames[k], TILE_PX) for k in tidx]
        cost["build_ms"] += (time.perf_counter() - t0) * 1000.0

        # --- embed mosaics + variants + tiles in one batch ---
        t0 = time.perf_counter()
        keys = [f"mosaic_N{n}" for n in N_SET] + ["shuffled", "no_ts",
                                                  "single_middle", "motion"]
        pils = [mosaic_imgs[k] for k in [f"mosaic_N{n}" for n in N_SET]] + \
               [s_img, nt_img, mid_img, mo_img]
        vecs = emb.encode(pils)
        for k, v in zip(keys, vecs):
            feats[k].append(v)
        tile_vecs = emb.encode(tile_pils)
        pooled = np.concatenate([tile_vecs.mean(axis=0), tile_vecs.std(axis=0)])
        feats["pooled_stats"].append(pooled.astype(np.float32))
        cost["embed_ms"] += (time.perf_counter() - t0) * 1000.0

        # --- artefacts for every labelled window (reference N, tile 224) ---
        if not args.no_artifacts:
            png = os.path.join(ARTIFACTS, f"{w['id']}.png")
            js = os.path.join(ARTIFACTS, f"{w['id']}.json")
            B.save_mosaic(mosaic_imgs[f"mosaic_N{REF_N}"], sidecars[f"mosaic_N{REF_N}"],
                          png, js)

        ys.append(w["y"])
        clips.append(w["clip"])
        ood_mask.append(w["source"] == "ood")
        ids.append(w["id"])
        if (wi + 1) % 20 == 0 or wi == len(wins) - 1:
            print(f"[{wi+1}/{len(wins)}] {w['id']} y={w['y']} "
                  f"frames={len(frames)} fps={native:.0f} "
                  f"{time.perf_counter()-t_start:.0f}s", flush=True)

    y = np.array(ys)
    clips = np.array(clips)
    ood_mask = np.array(ood_mask)
    device = "A40"
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=ROOT, capture_output=True, text=True).stdout.strip()
    cmd = "bash tools/pod/gpu_run.sh python3 eval/exp_mosaic/run_variants.py"
    protocol = ("in-domain LOCO over 7 IndEgo clips, logistic C=0.01 on "
                "L2-normalised features, clip-bootstrap CI B=1000 seed0; "
                "OOD = logistic frozen on all IndEgo, window bootstrap "
                "(optimistic, 1 window/factory), n_bad=9 exploratory")

    results = {}
    for tag, rows in feats.items():
        X = _l2(np.stack([r.astype(np.float32) for r in rows]))
        results[tag] = evaluate(X, y, clips, ood_mask, tag)
    # pooled_stats is 1536-dim; eval on raw concat, no L2 (feature scale kept)
    Xp = np.stack([r.astype(np.float32) for r in feats["pooled_stats"]])
    results["pooled_stats"] = evaluate(Xp, y, clips, ood_mask, "pooled_stats")

    ref = cached_reference()
    n_ind = int((~ood_mask).sum())
    n_ood = int(ood_mask.sum())
    base = {"n": n_ind, "n_good": int(y[~ood_mask].sum()),
            "n_bad": int((1 - y[~ood_mask]).sum()),
            "label_source": "human (IndEgo gold manifest, one labeller)",
            "protocol": protocol, "device": device, "command": cmd,
            "git_sha": git_sha,
            "ood": {"n": n_ood, "n_good": int(y[ood_mask].sum()),
                    "n_bad": int((1 - y[ood_mask]).sum()),
                    "label_source": "human (Egocentric-10K, one labeller)",
                    "note": "OOD n_bad=9 -> exploratory, CI optimistic"},
            "cost": {k: (round(v, 1) if isinstance(v, float) else v)
                     for k, v in cost.items()},
            "vram_peak_mb": round(emb.peak_mb(), 0),
            "sample_rule": "t_i=(i+0.5)*10/N; src_frame_index=round(t_i*native_fps)",
            "tile_px": TILE_PX, "window_s": WINDOW_S,
            "cached_eg2_reference": ref}

    def meanframes_of(n):
        return results[f"meanframes_N{n}"]["indego"]["auroc"]

    # ---- E2.01 mosaic ----
    e201 = dict(base, experiment="E2.01", variant="mosaic embedding",
                tile_px_sweep_planned=[112, 160, 224], tile_px_used=TILE_PX,
                results={})
    for n in N_SET:
        r = results[f"mosaic_N{n}"]
        r = dict(r, n_frames=n, fps_used=n / WINDOW_S,
                 embed_passes=1, frames_decoded=n,
                 delta_vs_meanframes=round(
                     r["indego"]["auroc"] - meanframes_of(n), 4))
        e201["results"][f"N{n}"] = r
    json.dump(e201, open(os.path.join(RESULTS, "E2.01.json"), "w"), indent=1)

    # ---- E2.02 mean-of-frames ----
    e202 = dict(base, experiment="E2.02", variant="mean-of-frames control",
                results={})
    for n in N_SET:
        r = results[f"meanframes_N{n}"]
        r = dict(r, n_frames=n, fps_used=n / WINDOW_S, embed_passes=n,
                 frames_decoded=n)
        e202["results"][f"N{n}"] = r
    json.dump(e202, open(os.path.join(RESULTS, "E2.02.json"), "w"), indent=1)

    # ---- E2.03..E2.07 single variants (REF_N) ----
    singles = {"E2.03": ("shuffled", "shuffled tile order"),
               "E2.04": ("no_ts", "no timestamps"),
               "E2.05": ("single_middle", "single middle frame"),
               "E2.06": ("motion", "motion mosaic |I_t-I_{t-1}|"),
               "E2.07": ("pooled_stats", "pooled per-tile stats")}
    for eid, (tag, desc) in singles.items():
        r = dict(results[tag], experiment=eid, variant=desc, n_frames=REF_N,
                 embed_passes=(REF_N if tag in ("pooled_stats",) else 1))
        if tag == "single_middle":
            r["n_frames"] = 1
        # fps_used is derived from the number of tiles actually built
        r["fps_used"] = r["n_frames"] / WINDOW_S
        # paired comparison vs mosaic N=9 and mean-of-frames N=9
        r["vs_mosaic_N9"] = round(r["indego"]["auroc"] -
                                  results[f"mosaic_N{REF_N}"]["indego"]["auroc"], 4)
        r["vs_meanframes_N9"] = round(r["indego"]["auroc"] -
                                      meanframes_of(REF_N), 4)
        json.dump(dict(base, **r), open(os.path.join(RESULTS, f"{eid}.json"), "w"),
                  indent=1)

    # ---- console summary ----
    print("\n=== E2 summary (AUROC [CI95]) ===")
    for n in N_SET:
        m = results[f"mosaic_N{n}"]["indego"]
        f = results[f"meanframes_N{n}"]["indego"]
        mo = results[f"mosaic_N{n}"]["ood"]
        fo = results[f"meanframes_N{n}"]["ood"]
        print(f"N={n:>3} ({n/10:.1f}fps) mosaic {m['auroc']:.3f} {m['ci95']} "
              f"| mean {f['auroc']:.3f} {f['ci95']} | d={m['auroc']-f['auroc']:+.3f} "
              f"|| OOD mosaic {mo['auroc']:.3f} mean {fo['auroc']:.3f}")
    for tag in ("shuffled", "no_ts", "single_middle", "motion", "pooled_stats"):
        r = results[tag]["indego"]
        print(f"{tag:14s} indego {r['auroc']:.3f} {r['ci95']} "
              f"ood {results[tag]['ood']['auroc']:.3f}")
    print("cached EG2 reference:", ref)
    print("cost:", base["cost"], "vram_peak_mb", base["vram_peak_mb"])
    print("wrote results to", RESULTS)
    json.dump({"ids": ids, "y": y.tolist(), "clips": clips.tolist(),
               "ood_mask": ood_mask.tolist(),
               "features": {k: np.stack(v).astype(np.float16).shape
                            for k, v in feats.items()}},
              open(os.path.join(RESULTS, "E2.index.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
