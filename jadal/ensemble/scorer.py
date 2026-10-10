"""Frozen two-view ensemble scorer (config ensemble_v1).

score(video_path, start_s, end_s, config_path=None) ->
  {usable_prob, verdict accept|reject|review, reasons[], margin,
   features_digest, versions}.

Views (frozen, numpy-only heads):
  EG2: sample every 0.5 s (cv2 seek, PIL thumb max-side 512), EmbeddingGemma-2
       encode (GPU), L2-normalised mean -> StandardScaler + logistic.
  pixel: ffmpeg CPU decode 160x90 rgb24, luma=0.299R+0.587G+0.114B float64,
       jadal.gpu.features:cpu_window_features -> StandardScaler + logistic.
Fusion: equal-weight probability average. usable_prob = round4(prob-average).
Verdict: p>0.5 accept; p<0.4 reject; in [0.4,0.6] review (frozen abstain band).
Determinism: same machine+device -> identical digests (EG2 GPU numerics are
NOT claimed identical across devices).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.normpath(os.path.join(HERE, "..", "..", "configs", "ensemble_v1.json"))

_Q = 4

_MODEL = None  # cached (model_dir, SentenceTransformer)


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=os.path.dirname(HERE)).stdout.strip()
    except Exception:
        return "unknown"


def load_config(path=None):
    path = path or DEFAULT_CONFIG
    with open(path) as f:
        cfg = json.load(f)
    sha_file = path + ".sha256"
    if os.path.exists(sha_file):
        expect = open(sha_file).read().split()[0]
        blob = json.dumps(cfg, indent=1).encode()
        actual = hashlib.sha256(blob).hexdigest()
        if actual != expect:
            raise ValueError(f"config sha256 mismatch: {actual} != {expect}")
    return cfg, path


def head_prob(x, h):
    """Numpy-only logistic head. x: (d,) float64; h: config head dict."""
    mean = np.asarray(h["mean"], dtype=np.float64)
    scale = np.asarray(h["scale"], dtype=np.float64)
    coef = np.asarray(h["coef"], dtype=np.float64)
    z = (np.asarray(x, dtype=np.float64) - mean) / scale
    return float(1.0 / (1.0 + np.exp(-(z @ coef + float(h["intercept"])))))


def ensemble_prob(eg2_mean_l2, pixel_feats, cfg):
    """Pure-numpy fusion from ready feature vectors. Returns (p, p_eg2, p_px)."""
    v = cfg["views"]
    p_e = head_prob(eg2_mean_l2, v["eg2"])
    order = v["pixel"]["feature_names"]
    x = np.array([float(pixel_feats[k]) for k in order], dtype=np.float64)
    p_g = head_prob(x, v["pixel"])
    w = cfg["fusion"].get("weights", [0.5, 0.5])
    return float(w[0] * p_e + w[1] * p_g), p_e, p_g


def decode_pixel_window(path, start_s, end_s, w=160, h=90):
    """ffmpeg CPU decode [start_s,end_s) at w,h rgb24 -> (luma, rgb, fps)."""
    import sys
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "eval", "gpu")))
    dur = round(float(end_s) - float(start_s), 2)
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
           "-ss", str(float(start_s)), "-t", str(dur), "-i", path,
           "-vf", f"scale={w}:{h},format=rgb24",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.DEVNULL, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg rc={p.returncode}: {p.stderr[-300:].decode('utf-8', 'replace')}")
    px = w * h * 3
    n = len(p.stdout) // px
    if n == 0:
        raise RuntimeError(f"no frames decoded from {path}@{start_s}")
    rgb = np.frombuffer(p.stdout[:n * px], dtype=np.uint8).astype(np.float64) / 255.0
    rgb = rgb.reshape(n, h, w, 3)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return luma, rgb, min(n / dur, 30.0)


def pixel_features_for_window(path, start_s, end_s, cfg):
    from jadal.gpu.features import cpu_window_features
    d = cfg["views"]["pixel"]["decode"]
    luma, rgb, fps = decode_pixel_window(path, start_s, end_s, d["w"], d["h"])
    feats = cpu_window_features(luma, rgb, fps=fps)
    return feats, {"n_frames": int(luma.shape[0]), "fps": float(fps)}


def sample_frames_eg2(path, start_s, end_s, step_s=0.5, max_side=512):
    """Mirror frames10k/eval/frames10k/embed_segments.py sampling (CPU)."""
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = total / fps if fps else 0.0
        pils = []
        for t in np.arange(float(start_s), min(float(end_s), dur), step_s):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).convert("RGB")
            img.thumbnail((max_side, max_side), Image.BILINEAR)
            pils.append(img)
        return pils
    finally:
        cap.release()


def eg2_mean_for_window(path, start_s, end_s, cfg):
    """EG2 L2-normalised mean embedding. Requires CUDA + model dir."""
    global _MODEL
    import torch
    from sentence_transformers import SentenceTransformer
    v = cfg["views"]["eg2"]
    key = v["embed_model_dir"]
    if _MODEL is None or _MODEL[0] != key:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        _MODEL = (key, SentenceTransformer(key, model_kwargs={"torch_dtype": dtype},
                                           trust_remote_code=True).eval())
    pils = sample_frames_eg2(path, start_s, end_s, v["frame_step_s"])
    if not pils:
        raise RuntimeError(f"no EG2 frames sampled from {path}@{start_s}")
    with torch.inference_mode():
        E = _MODEL[1].encode(pils, batch_size=32, show_progress_bar=False,
                             convert_to_numpy=True, normalize_embeddings=True,
                             processing_kwargs={"image": {"max_soft_tokens": v["max_soft_tokens"]}})
    E = np.asarray(E, dtype=np.float64)
    m = E.mean(axis=0)
    m = m / max(np.linalg.norm(m), 1e-12)
    return m, {"n_frames": int(len(pils))}


def features_digest(pixel_feats, eg2_mean, meta):
    canon = json.dumps({
        "pixel": {k: round(float(v), _Q) for k, v in sorted(pixel_feats.items())},
        "eg2_mean": [round(float(v), _Q) for v in np.asarray(eg2_mean).ravel()],
        "meta": meta}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def score(video_path, start_s, end_s, config_path=None):
    cfg, used = load_config(config_path)
    with open(used, "rb") as f:
        cfg_sha = hashlib.sha256(f.read()).hexdigest()
    feats, pmeta = pixel_features_for_window(video_path, start_s, end_s, cfg)
    eg2m, emeta = eg2_mean_for_window(video_path, start_s, end_s, cfg)
    p, p_e, p_g = ensemble_prob(eg2m, feats, cfg)
    prob = round(p, _Q)
    lo, hi = cfg["abstain_band"]
    thr = cfg["threshold"]
    if prob > thr:
        verdict = "accept"
    elif prob < lo:
        verdict = "reject"
    else:
        verdict = "review"
    reasons = []
    if verdict == "review":
        reasons.append({"code": "LOW_MARGIN",
                        "evidence": {"feature": "usable_prob", "value": prob,
                                     "op": "in", "threshold": hi - lo}})
    # documented heuristics (NOT validated operating points): extreme physics
    if float(feats.get("dark_frac", 0)) > 0.5:
        reasons.append({"code": "HEURISTIC_DARK",
                        "evidence": {"feature": "dark_frac", "value": round(float(feats["dark_frac"]), _Q),
                                     "op": ">", "threshold": 0.5}})
    if float(feats.get("frozen_frac", 0)) > 0.5:
        reasons.append({"code": "HEURISTIC_FROZEN",
                        "evidence": {"feature": "frozen_frac", "value": round(float(feats["frozen_frac"]), _Q),
                                     "op": ">", "threshold": 0.5}})
    meta = {"n_frames_pixel": pmeta["n_frames"], "n_frames_eg2": emeta["n_frames"]}
    return {"usable_prob": prob, "verdict": verdict, "reasons": reasons,
            "margin": round(float(p - 0.5), _Q),
            "p_eg2": round(p_e, _Q), "p_pixel": round(p_g, _Q),
            "features_digest": features_digest(feats, eg2m, meta),
            "versions": {"config": "ensemble_v1", "config_sha256": cfg_sha,
                         "git_sha": _git_sha(),
                         "fusion": "equal_prob_average",
                         "note": "EG2 GPU numerics not claimed identical across devices"}}
