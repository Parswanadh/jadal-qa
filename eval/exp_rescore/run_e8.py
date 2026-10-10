#!/usr/bin/env python3
"""E8.01 — re-score all 1,546 Egocentric-10K windows (B2, track E8).

Per window, one ffmpeg/NVDEC pass with three branches:
  (a) 160x90 rgb24, every frame            -> 26 per-pixel features (frozen pixel view)
  (b) 512x288 rgb24, 2 fps (0.5 s grid)    -> EmbeddingGemma-2 mean embedding (frozen EG2 view)
  (c) native 1920x1080 rgb24, 0.8 fps (8 frames at the (i+0.5)/8 window grid)
                                           -> E4.01 regime descriptors -> deterministic router
Then: frozen logistic heads (configs/ensemble_v1.json) -> equal-weight probability
average -> profile operating point (profiles/*.json via jadal.regimes.router) -> contract line.

Label source: NONE. These windows are unlabelled; verdicts are DESCRIPTIVE.
IndEgo-trained models did not transfer out-of-domain (0.4-0.53 AUROC, n_bad=9),
so the accept/reject share is not a quality measure.

Usage:
  python3 eval/exp_rescore/run_e8.py --workers 3 --out /workspace/artifacts/windows_rescored.jsonl
  python3 eval/exp_rescore/run_e8.py --limit 8 --workers 1 --out /tmp/opencode/e8_try.jsonl
  python3 eval/exp_rescore/run_e8.py --mode control --n 6   # frozen-path decode control
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval", "exp_regimes"))

from contract import (HONESTY_NOTE, build_line, canonical_bytes, features_block,  # noqa: E402
                      header_line, reasons_for, route_from_decision, round_floats,
                      sha256_hex, verdict_for)
from descriptors import ffprobe_meta, frame_stats, motion_stats  # noqa: E402
from jadal.regimes.router import choose_profile  # noqa: E402
from jadal.ensemble.scorer import ensemble_prob  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
MANIFEST = os.path.join(BUNDLE, "queues", "manifest_scale.json")
CONFIG = os.path.join(REPO, "configs", "ensemble_v1.json")
FROZEN_CONFIG = os.path.join(BUNDLE, "caches", "ensemble", "ensemble_v1.json")
PROFILE_DIR = os.path.join(REPO, "profiles")
ARTIFACTS = "/workspace/artifacts"
DEVICE = "A40"
STAGE = "L1"  # funnel vocabulary: L0 rules, L1 model heads, L2 verdict
W, H = 160, 90
EGW, EGH = 512, 288
NATIVE_W, NATIVE_H = 1920, 1080
WINDOW_S = 10.0

_CFG = None
_CFG_FRAG = None
_PROFILE_CACHE = {}
_MODEL = None
_META_CACHE = {}
_LOCK = threading.Lock()


# --------------------------------------------------------------------------- io
def load_manifest():
    man = json.load(open(MANIFEST))
    items = man["items"]
    for it in items:
        it["window_id"] = it["id"]
    items = sorted(items, key=lambda i: (i["clip"], i["start_s"], i["window_id"]))
    return man, items


def profiles():
    out = {}
    for name in ("default", "clear_stationary", "blurry_lowlight",
                 "walking_logistics", "factory_1080p30"):
        p = os.path.join(PROFILE_DIR, name + ".json")
        if os.path.exists(p):
            out[name] = json.load(open(p))
    return out


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=REPO, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def ffmpeg_version():
    try:
        p = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
        return p.stdout.splitlines()[0].strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------- config
_NUMERIC_KEYS = ("views.eg2.mean", "views.eg2.scale", "views.eg2.coef",
                 "views.pixel.mean", "views.pixel.scale", "views.pixel.coef",
                 "views.pixel.intercept", "fusion", "threshold", "abstain_band",
                 "verdict_rule", "views.pixel.feature_names")


def _numeric_view(cfg):
    """The frozen head numbers only (path/command strings drop out)."""
    return {
        "views": {"eg2": {k: cfg["views"]["eg2"][k] for k in ("mean", "scale", "coef")},
                  "pixel": {k: cfg["views"]["pixel"][k] for k in
                            ("mean", "scale", "coef", "intercept", "feature_names")}},
        "fusion": cfg["fusion"], "threshold": cfg["threshold"],
        "abstain_band": cfg["abstain_band"], "verdict_rule": cfg["verdict_rule"],
    }


def load_config_checked():
    """Load configs/ensemble_v1.json and check it against the frozen pod artifact.

    `load_config` (jadal/ensemble/scorer.py) refuses a stale `.sha256` sidecar:
    the sidecar was written on the laptop, and the path-remap step changed
    `views.eg2.embed_model_dir` + the two command strings, so the raw digest no
    longer matches.  We therefore verify the *numbers* against the frozen
    `$JADAL_BUNDLE/caches/ensemble/ensemble_v1.json` and record both digests.
    """
    cfg = json.load(open(CONFIG))
    frozen = json.load(open(FROZEN_CONFIG))
    a, b = _numeric_view(cfg), _numeric_view(frozen)
    if canonical_bytes(a) != canonical_bytes(b):
        raise ValueError("config head numbers differ from the frozen pod artifact")
    frag = {
        "config_path": os.path.relpath(CONFIG, REPO),
        "config_sha256": sha256_hex(open(CONFIG, "rb").read()),
        "config_numeric_sha256": sha256_hex(canonical_bytes(a)),
        "frozen_bundle_config_sha256": sha256_hex(open(FROZEN_CONFIG, "rb").read()),
        "config_sidecar_stale": True,
        "config_sidecar_note": "raw .sha256 sidecar predates the pod path remap; "
                               "head numbers verified equal to the frozen bundle artifact",
    }
    return cfg, frag


# --------------------------------------------------------------------- decoding
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


def decode_three_branch(path, start_s, dur=WINDOW_S):
    """One NVDEC pass, three rawvideo pipes -> (pixel rgb24, eg2 rgb24, native rgb24)."""
    pipes = [os.pipe() for _ in range(3)]
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-ss", f"{float(start_s):.3f}", "-t", f"{float(dur):.3f}", "-i", path,
        "-filter_complex",
        "[0:v]split=3[a][b][c];"
        f"[a]scale_cuda={W}:{H}:format=nv12,hwdownload,format=nv12,format=rgb24[pix];"
        f"[b]fps=2,scale_cuda={EGW}:{EGH}:format=nv12,hwdownload,format=nv12,format=rgb24[eg];"
        "[c]fps=0.8,hwdownload,format=nv12,format=rgb24[nat]",
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
        t = threading.Thread(target=_drain, args=(pipes[i][0], sinks[i], errors), daemon=True)
        t.start()
        threads.append(t)
    err = proc.stderr.read()
    rc = proc.wait(timeout=300)
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


# -------------------------------------------------------------- descriptors (c)
def describe_from_native_frames(meta, frames, start_s, end_s, k=8):
    """E4.01 descriptor record from already-decoded native frames.

    Mirrors eval/exp_regimes/descriptors.py::describe_window exactly (same
    aggregation, same rounding, same `_sha256` recipe); the only difference is
    that frames are decoded by NVDEC with the same (i+0.5)/8 sampling grid
    instead of cv2 seeks. `frames` are rgb24 -> flipped to BGR for frame_stats.
    """
    good = [f for f in frames if f is not None]
    per = [frame_stats(f[..., ::-1].copy()) for f in good]
    if not per:
        raise ValueError("no native frames decoded")
    agg = {}
    for key in per[0]:
        agg[key] = round(float(np.mean([p[key] for p in per])), 6)
    agg["sharp_norm"] = round(agg["sharp_lapvar"] / (agg["luma_mean"] + 1e-3), 4)
    agg["clip_frac"] = round(agg["dark_frac"] + agg["bright_frac"], 6)
    mo = motion_stats([f[..., ::-1].copy() for f in frames])
    desc = {**meta, **agg, **{kk: round(v, 6) for kk, v in mo.items()},
            "n_frames_used": len(good), "n_frames_req": k}
    desc["_sha256"] = sha256_hex(json.dumps(desc, sort_keys=True))[:16]
    return desc


# ------------------------------------------------------------------- per window
def init_worker():
    global _CFG, _CFG_FRAG
    import torch
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _CFG, _CFG_FRAG = load_config_checked()


def _get_meta(path):
    if path not in _META_CACHE:
        _META_CACHE[path] = ffprobe_meta(path)
    return _META_CACHE[path]


def _get_model(cfg):
    global _MODEL
    if _MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer
        v = cfg["views"]["eg2"]
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        _MODEL = (v["embed_model_dir"], SentenceTransformer(
            v["embed_model_dir"], model_kwargs={"torch_dtype": dtype},
            trust_remote_code=True, device="cuda").eval())
    return _MODEL


def encode_eg2(rgb_frames, cfg):
    """frozen EG2 view: L2-normalised mean of per-frame embeddings."""
    if rgb_frames.shape[0] == 0:
        raise RuntimeError("no EG2 frames")
    from PIL import Image
    key, model = _get_model(cfg)
    pils = [Image.fromarray(f) for f in rgb_frames]
    with _get_model_torch_inference():
        E = model.encode(pils, batch_size=32, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True,
                         processing_kwargs={"image": {"max_soft_tokens":
                                                      cfg["views"]["eg2"]["max_soft_tokens"]}})
    E = np.asarray(E, dtype=np.float64)
    m = E.mean(axis=0)
    return m / max(np.linalg.norm(m), 1e-12), int(E.shape[0])


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _get_model_torch_inference():
    import torch
    return torch.inference_mode()


def score_window(item, profile_map):
    """Full path for one window -> (contract line, descriptor record)."""
    from jadal.gpu.features import cpu_window_features
    lat = {}
    t_all = time.perf_counter()
    path, s0 = item["path"], float(item["start_s"])
    s1 = float(item["end_s"])

    t = time.perf_counter()
    pix_raw, eg_raw, nat_raw = decode_three_branch(path, s0, s1 - s0)
    lat["decode"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    pix = frames_from_raw(pix_raw, H, W).astype(np.float64) / 255.0
    if pix.shape[0] == 0:
        raise RuntimeError("no pixel frames")
    luma = 0.299 * pix[..., 0] + 0.587 * pix[..., 1] + 0.114 * pix[..., 2]
    fps_pix = min(pix.shape[0] / max(s1 - s0, 1e-6), 30.0)
    feats = cpu_window_features(luma, pix, fps=fps_pix)
    lat["pixel_features"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    meta = _get_meta(path)
    meta = {k: v for k, v in meta.items()}
    desc = describe_from_native_frames(meta, list(frames_from_raw(nat_raw, NATIVE_H, NATIVE_W)),
                                      s0, s1)
    lat["descriptors"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    eg2_mean, n_eg = encode_eg2(frames_from_raw(eg_raw, EGH, EGW), _CFG)
    lat["eg2_embed"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    prob, p_e, p_g = ensemble_prob(eg2_mean, feats, _CFG)
    decision = choose_profile(desc)
    route = route_from_decision(decision, desc)
    prof = profile_map.get(route["profile"], profile_map["default"])
    accept_thr = float(prof["thresholds"]["accept"])
    reject_thr = float(prof["thresholds"]["reject"])
    verdict = verdict_for(prob, accept_thr, reject_thr)
    reasons = reasons_for(prob, verdict, accept_thr, reject_thr, route, feats)
    features, fdig = features_block(feats, eg2_mean, pix.shape[0], n_eg, fps_pix, p_e, p_g)
    eg2_rounded = [round(float(x), 4) for x in eg2_mean]
    lat["score"] = (time.perf_counter() - t) * 1000.0
    lat["total"] = (time.perf_counter() - t_all) * 1000.0

    versions = {
        "recipe": "ensemble_v1",
        **_CFG_FRAG,
        "fusion": "equal_prob_average",
        "config_sha256": sha256_hex(open(CONFIG, "rb").read()),
        "eg2_model": "google/embeddinggemma-2@914f7f8",
        "profile": route["profile"],
        "profile_version": prof["version"],
        "profile_margin": route["margin"],
        "profile_thresholds": {"accept": accept_thr, "reject": reject_thr},
        "router_reasons": route["profile_reasons"],
        "unseen_regime": route["unseen_regime"],
        "descriptors_sha256": sha256_hex(canonical_bytes(desc)),
        "decode_path": "nvdec(cuda)+scale_cuda(bilinear)->rgb24",
        "sampling": {"window_s": WINDOW_S, "pixel_fps": round(fps_pix, 4),
                     "eg2_step_s": 0.5, "descriptor_frames": 8},
        "deterministic": True,
        "device": DEVICE,
        "git_sha": git_sha(),
    }
    line = build_line(item, verdict, STAGE, reasons, prob, fdig, features,
                      versions, lat)
    return line, desc, eg2_rounded


def _worker(args):
    item, profile_map = args
    try:
        line, desc, eg2 = score_window(item, profile_map)
        return {"ok": True, "line": line, "desc": desc, "eg2": eg2,
                "window_id": item["window_id"]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "window_id": item["window_id"], "clip": item["clip"],
                "error": f"{type(e).__name__}: {e}"[:300]}


# -------------------------------------------------------------- frozen control
def control_frozen_path(n=6, seed=7):
    """Compare the frozen (swscale/cv2) pixel decode path with the NVDEC path."""
    from jadal.ensemble.scorer import pixel_features_for_window
    from jadal.gpu.features import cpu_window_features
    _, items = load_manifest()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(items), size=min(n, len(items)), replace=False)
    cfg, _ = load_config_checked()
    rows = []
    for i in sorted(idx.tolist()):
        it = items[i]
        pix_raw, _, _ = decode_three_branch(it["path"], it["start_s"],
                                             it["end_s"] - it["start_s"])
        pix = frames_from_raw(pix_raw, H, W).astype(np.float64) / 255.0
        luma = 0.299 * pix[..., 0] + 0.587 * pix[..., 1] + 0.114 * pix[..., 2]
        f_fast = cpu_window_features(luma, pix, fps=min(pix.shape[0] / 10.0, 30.0))
        f_frozen, meta = pixel_features_for_window(it["path"], it["start_s"], it["end_s"], cfg)
        # EG2 mean not recomputed (GPU expensive): compare with the frozen p_pixel head only
        order = cfg["views"]["pixel"]["feature_names"]
        from jadal.ensemble.scorer import head_prob
        from jadal.gpu.features import cpu_window_features as _c  # noqa: F401
        pf = np.array([float(f_fast[k]) for k in order], dtype=np.float64)
        pz = np.array([float(f_frozen[k]) for k in order], dtype=np.float64)
        rows.append({
            "window_id": it["window_id"], "clip": it["clip"],
            "n_frames_fast": int(pix.shape[0]), "n_frames_frozen": int(meta["n_frames"]),
            "p_pixel_fast": round(head_prob(pf, cfg["views"]["pixel"]), 4),
            "p_pixel_frozen": round(head_prob(pz, cfg["views"]["pixel"]), 4),
            "feat_abs_delta_max": round(float(np.max(np.abs(pf - pz))), 4),
            "feat_abs_delta_mean": round(float(np.mean(np.abs(pf - pz))), 4),
        })
    d = [abs(r["p_pixel_fast"] - r["p_pixel_frozen"]) for r in rows]
    return {
        "what": "NVDEC+scale_cuda(bilinear) fast decode path vs frozen "
                "ffmpeg_cpu(swscale bicubic) path, pixel view only",
        "n": len(rows), "seed": seed,
        "p_pixel_abs_delta_max": round(max(d), 4) if d else None,
        "p_pixel_abs_delta_mean": round(float(np.mean(d)), 4) if d else None,
        "rows": rows,
        "note": "features are NOT bit-identical across decode paths; documented, "
                "not claimed identical (verdicts are descriptive)",
    }


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ARTIFACTS, "windows_rescored.jsonl"))
    ap.add_argument("--mode", choices=("run", "control"), default="run")
    ap.add_argument("--resume", action="store_true",
                    help="skip windows already present in <out>.partial")
    ap.add_argument("--n", type=int, default=6, help="control sample size")
    args = ap.parse_args()

    if args.mode == "control":
        res = control_frozen_path(args.n)
        out = os.path.join(ARTIFACTS, "exp_rescore", "control_frozen_path.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, "w"), indent=1, sort_keys=True)
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
        print("wrote", out)
        return

    man, items = load_manifest()
    if args.limit:
        items = items[:args.limit]
    pmap = profiles()
    os.makedirs(os.path.join(ARTIFACTS, "exp_rescore"), exist_ok=True)
    partial = args.out + ".partial"
    done = {}
    if args.resume and os.path.exists(partial):
        for l in open(partial):
            try:
                r = json.loads(l)
            except Exception:  # noqa: BLE001
                continue
            done[r["window_id"]] = r if "line" in r else {"line": r}
        print(f"resume: {len(done)} windows already done", flush=True)
        items = [i for i in items if i["window_id"] not in done]
    t0 = time.perf_counter()
    results = list(done.values())
    errors = []
    if args.workers <= 1:
        init_worker()
        for it in items:
            r = _worker((it, pmap))
            (results if r["ok"] else errors).append(r)
            if r["ok"]:
                with open(partial, "a") as f:
                    f.write(json.dumps({"line": r["line"], "desc": r["desc"],
                                        "eg2": r["eg2"]}, sort_keys=True) + "\n")
            print(f"  {len(results)+len(errors)}/{len(items)} {r['window_id']} "
                  f"{'' if r['ok'] else r['error']}", flush=True)
    else:
        import multiprocessing as mp
        with mp.Pool(args.workers, initializer=init_worker) as pool:
            with open(partial, "a") as fh:
                for k, r in enumerate(pool.imap(_worker, [(it, pmap) for it in items],
                                                chunksize=1), 1):
                    (results if r["ok"] else errors).append(r)
                    if r["ok"]:
                        fh.write(json.dumps({"line": r["line"], "desc": r["desc"],
                                             "eg2": r["eg2"]}, sort_keys=True) + "\n")
                        fh.flush()
                    if k % 25 == 0 or k == len(items):
                        el = time.perf_counter() - t0
                        print(f"  {k}/{len(items)} ok={len(results)} err={len(errors)} "
                              f"{el:.0f}s {k/el:.2f} win/s eta="
                              f"{(len(items)-k)/max(k/el,1e-9)/60:.1f}min", flush=True)
    wall = time.perf_counter() - t0
    order = {w["window_id"]: i for i, w in enumerate(items)}
    results.sort(key=lambda r: order[r["window_id"]])

    results = [r for r in results if r.get("line")]
    lines = [r["line"] for r in results]
    body = b"".join(canonical_bytes(l) + b"\n" for l in lines)
    vdist = Counter(l["verdict"] for l in lines)
    pdist = Counter(l["versions"]["profile"] for l in lines)
    lat = np.array([[l["latency_ms"][k] for k in
                     ("decode", "pixel_features", "descriptors", "eg2_embed", "total")]
                    for l in lines]) if lines else np.zeros((0, 5))
    hdr = header_line(len(lines), "ensemble_v1 (EG2+pixel equal-weight) + "
                                  "profiles/router (E4.04)",
                      dict(sorted(pdist.items())), dict(sorted(vdist.items())),
                      DEVICE, git_sha(), sha256_hex(body))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(json.dumps(hdr, sort_keys=True) + "\n")
        for l in lines:
            f.write(json.dumps(l, sort_keys=True) + "\n")
    # descriptors sidecar (full router input per window, for re-verification)
    dpath = os.path.join(ARTIFACTS, "exp_rescore", "descriptors.jsonl")
    with open(dpath, "w") as f:
        for r in results:
            f.write(json.dumps({"window_id": r["window_id"], "desc": r["desc"]},
                               sort_keys=True) + "\n")
    dsig = sha256_hex(open(dpath, "rb").read())

    # EG2 mean vectors (float32, 4-dp rounded to match features{}), 1546 x 768
    eg_path = os.path.join(ARTIFACTS, "exp_rescore", "eg2_mean_l2.npy")
    if results:
        np.save(eg_path, np.array([r["eg2"] for r in results], dtype=np.float64))
    eg_sig = sha256_hex(open(eg_path, "rb").read()) if results else None
    summary = {
        "n_windows": len(lines), "n_requested": len(items), "n_errors": len(errors),
        "errors": errors[:20],
        "wall_s": round(wall, 1),
        "windows_per_s": round(len(lines) / wall, 4) if wall else None,
        "ms_per_window_mean": round(float(lat[:, 4].mean()), 1) if len(lat) else None,
        "ms_per_window_p50": round(float(np.percentile(lat[:, 4], 50)), 1) if len(lat) else None,
        "ms_per_window_p95": round(float(np.percentile(lat[:, 4], 95)), 1) if len(lat) else None,
        "latency_ms_mean": {k: round(float(lat[:, i].mean()), 1) for i, k in enumerate(
            ("decode", "pixel_features", "descriptors", "eg2_embed", "total"))} if len(lat) else {},
        "verdict_distribution": dict(sorted(vdist.items())),
        "profile_distribution": dict(sorted(pdist.items())),
        "descriptors_sidecar": {"path": dpath, "sha256": dsig},
        "eg2_mean_sidecar": {"path": eg_path, "sha256": eg_sig,
                             "shape": [len(results), 768] if results else None,
                             "note": "float64, 4-dp rounded to match features{}.eg2_mean_sha256"},
        "contract": {"path": args.out, "body_sha256": sha256_hex(body),
                     "n_lines": len(lines) + 1},
        "workers": args.workers, "device": DEVICE, "git_sha": git_sha(),
        "honesty_note": HONESTY_NOTE,
    }
    sp = os.path.join(ARTIFACTS, "exp_rescore", "run_summary.json")
    json.dump(round_floats(summary), open(sp, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, indent=1))
    print("wrote", args.out, sp, dpath)


if __name__ == "__main__":
    main()
