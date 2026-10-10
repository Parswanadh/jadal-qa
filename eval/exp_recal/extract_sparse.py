#!/usr/bin/env python3
"""N5 — re-extract window features at the ADOPTED SPARSE settings.

The optimized runner (``eval/exp_latency_fast/run_fast.py``, branch m/latency-fast,
L3.01) scores a window from ONE NVDEC three-branch pass with per-branch fps
filters and a 4 s early exit:

  * pixel branch   ``fps=3.33``  -> 160x90 rgb24 -> ``cpu_window_features`` (26)
  * EG2 branch     ``fps=1.0``   -> 512x288 rgb24 -> mean of L2-normalised
                                     EmbeddingGemma-2 frame embeddings (768)
  * native branch  ``fps=0.8``   -> (regime descriptors only; not a head input)
  * ``dur = 4.0 s`` (E3.03 early exit)

pod/base does not carry the L3 runner, so ``decode_three_branch_cfg`` /
``frames_from_raw`` / ``encode_eg2`` below are copied VERBATIM from
``eval/exp_latency_fast/run_fast.py`` and ``eval/exp_rescore/run_e8.py`` so the
features are exactly what the optimized pipeline feeds the heads.  The scoring
code (``cpu_window_features``, EG2 mean) is imported from the repo.

Sets:
  --set gold    IndEgo gold 150 windows (manifest_gold.json)
  --set ood160  160 human-labelled Egocentric-10K windows (ood160_labels.json)

Output npz: ids, clips, labels, X_pixel (n,26), X_eg2 (n,768),
n_frames_pixel, n_frames_eg2, fps_pixel, names.

Run under the GPU slot:
  source /workspace/env.sh
  bash tools/pod/gpu_run.sh python3 eval/exp_recal/extract_sparse.py --set gold --workers 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from jadal.gpu.features import cpu_window_features  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
CONFIG = os.path.join(REPO, "configs", "ensemble_v1.json")
LAPTOP_DATA = "/home/parshu/projects/awing/data/"
POD_DATA = os.path.join(BUNDLE, "data") + "/"
W, H = 160, 90
EGW, EGH = 512, 288
POD_EG2 = ("/workspace/.cache/huggingface/hub/models--google--embeddinggemma-2"
           "/snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b")

# adopted sparse settings (identical to run_fast.py --pixel-fps 3.33 --eg2-fps 1 --early-exit-s 4)
SPARSE = {"dur": 4.0, "pixel_fps": 3.33, "eg2_fps": 1.0, "native_fps": 0.8}

_CFG = None
_OPTS = None
_MODEL = None


# ------------------------------------------------------------------ decode (copied from run_fast.py / run_e8.py)
def _drain(fd, sink, errors):
    try:
        with os.fdopen(fd, "rb", 0) as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                sink.append(chunk)
    except Exception as e:  # noqa: BLE001
        errors.append(f"{type(e).__name__}: {e}")


def decode_three_branch_cfg(path, start_s, dur, pixel_fps=0.0, eg2_fps=2.0,
                            native_fps=0.8, w=W, h=H, egw=EGW, egh=EGH,
                            timeout_s=300):
    """E8.01 three-branch NVDEC pass with parameterised per-branch fps (verbatim
    from run_fast.py)."""
    pipes = [os.pipe() for _ in range(3)]
    pfx = f"fps={pixel_fps:g}," if pixel_fps and pixel_fps > 0 else ""
    efx = f"fps={eg2_fps:g}," if eg2_fps and eg2_fps > 0 else ""
    nfx = f"fps={native_fps:g}," if native_fps and native_fps > 0 else ""
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-ss", f"{float(start_s):.3f}", "-t", f"{float(dur):.3f}", "-i", path,
        "-filter_complex",
        "[0:v]split=3[a][b][c];"
        f"[a]{pfx}scale_cuda={w}:{h}:format=nv12,hwdownload,format=nv12,format=rgb24[pix];"
        f"[b]{efx}scale_cuda={egw}:{egh}:format=nv12,hwdownload,format=nv12,format=rgb24[eg];"
        f"[c]{nfx}hwdownload,format=nv12,format=rgb24[nat]",
        "-map", "[pix]", "-f", "rawvideo", "-pix_fmt", "rgb24", f"pipe:{pipes[0][1]}",
        "-map", "[eg]", "-f", "rawvideo", "-pix_fmt", "rgb24", f"pipe:{pipes[1][1]}",
        "-map", "[nat]", "-f", "rawvideo", "-pix_fmt", "rgb24", f"pipe:{pipes[2][1]}",
    ]
    sinks = [[], [], []]
    errors = []
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, pass_fds=tuple(p[1] for p in pipes))
    for p in pipes:
        os.close(p[1])
    threads = []
    for i in range(3):
        t = threading.Thread(target=_drain, args=(pipes[i][0], sinks[i], errors),
                             daemon=True)
        t.start()
        threads.append(t)
    err = proc.stderr.read()
    rc = proc.wait(timeout=timeout_s)
    for t in threads:
        t.join(timeout=60)
    for p in pipes:
        try:
            os.close(p[0])
        except OSError:
            pass
    if rc != 0:
        raise RuntimeError(f"ffmpeg rc={rc}: {err[-300:].decode('utf-8', 'replace')}")
    return b"".join(sinks[0]), b"".join(sinks[1]), b"".join(sinks[2])


def frames_from_raw(buf, h, w):
    px = h * w * 3
    n = len(buf) // px
    if n == 0:
        return np.zeros((0, h, w, 3), dtype=np.uint8)
    return np.frombuffer(buf[:n * px], dtype=np.uint8).reshape(n, h, w, 3)


# ------------------------------------------------------------------ EG2 (copied from run_e8.py)
def _get_model(cfg):
    global _MODEL
    if _MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer
        v = cfg["views"]["eg2"]
        key = POD_EG2 if os.path.isdir(POD_EG2) else v["embed_model_dir"]
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        _MODEL = (key, SentenceTransformer(key, model_kwargs={"torch_dtype": dtype},
                                           trust_remote_code=True, device="cuda").eval())
    return _MODEL


def encode_eg2(rgb_frames, cfg):
    """frozen EG2 view: L2-normalised mean of per-frame embeddings (verbatim)."""
    if rgb_frames.shape[0] == 0:
        raise RuntimeError("no EG2 frames")
    import torch
    from PIL import Image
    _, model = _get_model(cfg)
    pils = [Image.fromarray(f) for f in rgb_frames]
    with torch.inference_mode():
        E = model.encode(pils, batch_size=32, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True,
                         processing_kwargs={"image": {"max_soft_tokens":
                                                      cfg["views"]["eg2"]["max_soft_tokens"]}})
    E = np.asarray(E, dtype=np.float64)
    m = E.mean(axis=0)
    return m / max(np.linalg.norm(m), 1e-12), int(E.shape[0])


# ------------------------------------------------------------------ items
def remap(p):
    if p.startswith(LAPTOP_DATA):
        return POD_DATA + p[len(LAPTOP_DATA):]
    return p


def load_items(setname):
    if setname == "gold":
        m = json.load(open(os.path.join(BUNDLE, "labels", "manifest_gold.json")))
        items = [{"id": it["id"], "path": remap(it["path"]), "start_s": it["start_s"],
                  "dur_s": round(it["end_s"] - it["start_s"], 2), "clip": it["clip"],
                  "label": int(it["label"]), "factory": it["clip"]}
                 for it in m["items"]]
    elif setname == "ood160":
        d = json.load(open(os.path.join(BUNDLE, "labels", "ood160_labels.json")))
        items = []
        for w in d["windows"]:
            vid = os.path.join(BUNDLE, w["video"])
            stem = os.path.splitext(os.path.basename(vid))[0]
            items.append({"id": stem, "path": vid, "start_s": 0.0, "dur_s": 10.0,
                          "clip": w.get("factory", ""), "factory": w.get("factory", ""),
                          "label": 1 if w["verdict"] == "good" else 0})
    else:
        raise SystemExit(f"unknown set {setname}")
    return items


def init_worker():
    global _CFG, _OPTS
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    _CFG = json.load(open(CONFIG))
    if os.path.isdir(POD_EG2):
        _CFG["views"]["eg2"]["embed_model_dir"] = POD_EG2
    _OPTS = SPARSE


def process(item):
    opts = _OPTS or SPARSE
    pix_raw, eg_raw, _ = decode_three_branch_cfg(
        item["path"], item["start_s"], opts["dur"],
        opts["pixel_fps"], opts["eg2_fps"], opts["native_fps"])
    pix = frames_from_raw(pix_raw, H, W).astype(np.float64) / 255.0
    if pix.shape[0] == 0:
        raise RuntimeError("no pixel frames")
    luma = 0.299 * pix[..., 0] + 0.587 * pix[..., 1] + 0.114 * pix[..., 2]
    fps_pix = min(pix.shape[0] / max(opts["dur"], 1e-6), 30.0)
    feats = cpu_window_features(luma, pix, fps=fps_pix)
    eg2m, n_eg = encode_eg2(frames_from_raw(eg_raw, EGH, EGW), _CFG)
    return {"id": item["id"], "feats": feats, "eg2": eg2m,
            "n_pix": int(pix.shape[0]), "n_eg": int(n_eg), "fps_pix": float(fps_pix)}


def _work(item):
    try:
        return {"ok": True, "row": process(item), "id": item["id"]}
    except Exception as ex:  # noqa: BLE001
        return {"ok": False, "id": item["id"], "error": f"{type(ex).__name__}: {ex}"[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=("gold", "ood160"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = load_items(args.set)
    if args.limit:
        items = items[:args.limit]
    names = json.load(open(CONFIG))["views"]["pixel"]["feature_names"]
    out = args.out or os.path.join(HERE, "..", "results", "exp_recal",
                                   f"sparse_{args.set}.npz")
    out = os.path.normpath(out)

    t0 = time.perf_counter()
    rows, errors = [], []
    if args.workers <= 1:
        init_worker()
        for k, it in enumerate(items, 1):
            r = _work(it)
            (rows if r["ok"] else errors).append(r)
            if k % 10 == 0 or k == len(items):
                el = time.perf_counter() - t0
                print(f"  {k}/{len(items)} ok={len(rows)} err={len(errors)} "
                      f"{el:.0f}s {k/max(el,1e-9):.2f} win/s", flush=True)
    else:
        import multiprocessing as mp
        with mp.Pool(args.workers, initializer=init_worker) as pool:
            for k, r in enumerate(pool.imap(_work, items, chunksize=1), 1):
                (rows if r["ok"] else errors).append(r)
                if k % 10 == 0 or k == len(items):
                    el = time.perf_counter() - t0
                    print(f"  {k}/{len(items)} ok={len(rows)} err={len(errors)} "
                          f"{el:.0f}s {k/max(el,1e-9):.2f} win/s", flush=True)

    order = {it["id"]: i for i, it in enumerate(items)}
    rows.sort(key=lambda r: order[r["id"]])
    ok = [r["row"] for r in rows]
    ids = np.array([r["id"] for r in ok])
    Xp = np.array([[float(r["feats"][n]) for n in names] for r in ok], dtype=np.float64)
    Xe = np.array([r["eg2"] for r in ok], dtype=np.float64)
    by_id = {it["id"]: it for it in items}
    clips = np.array([by_id[i]["clip"] for i in ids])
    labels = np.array([by_id[i]["label"] for i in ids], dtype=int)
    npix = np.array([r["n_pix"] for r in ok], dtype=int)
    neg = np.array([r["n_eg"] for r in ok], dtype=int)
    fps = np.array([r["fps_pix"] for r in ok], dtype=float)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, ids=ids, clips=clips, labels=labels, X_pixel=Xp,
                        X_eg2=Xe, n_frames_pixel=npix, n_frames_eg2=neg,
                        fps_pixel=fps, names=np.asarray(names))
    print(f"wrote {out}: n={len(ids)} errors={len(errors)} "
          f"wall={time.perf_counter()-t0:.0f}s pixel_frames_p50={int(np.median(npix))} "
          f"eg2_frames_p50={int(np.median(neg))}")
    if errors:
        print("errors:", json.dumps(errors[:5]))


if __name__ == "__main__":
    main()
