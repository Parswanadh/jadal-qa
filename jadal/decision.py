"""L1 decider: EmbeddingGemma-2 embeddings + linear-probe heads (issue #7).

``Decider.decide(path, signals) -> Decisions`` per ``docs/SPEC.md``::

    {"hands_ok": {"p_yes": float}, "active_work": {"p_yes": float},
     "step": {"choice": str|null, "probs": {str: float}, "answer_conf": float},
     "model_versions": {str: str}}

Backbone: EmbeddingGemma 2 from ``/home/parshu/projects/awing/data/eg2``
via sentence-transformers, bfloat16 (NEVER fp16), eval mode,
``torch.use_deterministic_algorithms(True, warn_only=True)``,
``cudnn.benchmark=False``, fixed seeds. FIXED frame sampling (4 frames at
fixed fractions of the segment, OpenCV decode). The image token budget is
configurable (default 140).

Preprocessing (SHARED with training — ``eval/train_heads.py`` imports
``preprocess_bgr``/``preprocess_pil`` from this module, so there is no
train/serve skew): BGR/RGB frame -> RGB PIL -> thumbnail so the longest
side is ``PREPROCESS_MAX_SIDE`` (512 px). Training embeds single still
frames; serving therefore embeds each sampled frame separately, applies
each logistic head per frame to get per-frame LOGITS, and averages the
logits (documented aggregation choice) before sigmoid/temperature.

Determinism note (measured in ``eval/results/bench_eg2.json``): the same
frame embedded alone vs inside a batch of 5 differs by ~1e-3 (not
bit-equal), so this module NEVER batches: every frame is encoded with
batch size 1, making a segment identical alone or inside a batch of 5.

Heads: scikit-learn logistic regression on frozen embeddings for
``hand_count >= 1`` (``hands_ok``) and ``active_labor == yes``
(``active_work``), with a single temperature each fitted on the CALIBRATION
split (see ``jadal.certify.fit_temperature``). Head files live under
``models/`` (git-ignored); their sha256 and provenance are recorded in
``model_versions``. Heads trained on a DEV subset carry provenance
``DEV_ONLY`` and are REFUSED unless env ``JADAL_ALLOW_DEV_HEADS=1``.
Frame embeddings are cached on disk (``cache_dir=None`` disables the
cache); the key includes the backbone weight hash, image token budget,
and preprocessing parameters.

STEP head: nearest-prototype over the ECU assembly steps, built with the
SAME per-frame sampling/preprocessing/pooling as serving
(mean of the sampled frame embeddings, renormalised); wired into
``step{choice,probs,answer_conf}`` when the prototypes file is present.

NOT DONE:
- No Laya fusion (Laya is NOT required for this baseline).
- If the clip cannot be decoded, both heads return ``p_yes = 0.0``
  (``frames_decoded == "0"`` in model_versions marks this; L2 marks
  UNDECODABLE from L0 signals); if head files are absent, both return 0.5
  with ``model_versions["heads"] = "missing"``.
"""
from __future__ import annotations

import hashlib
import io
import os
import platform
from pathlib import Path
from typing import Any, Callable

import numpy as np

EG2_DIR = "/home/parshu/projects/awing/data/eg2"
SEED = 1234
N_FRAMES = 4
FRAME_FRACTIONS = (0.125, 0.375, 0.625, 0.875)
DEFAULT_BUDGET = 140
EMBED_DIM = 768
PREPROCESS_MAX_SIDE = 512  # shared train/serve thumbnail (I1)
CACHE_VERSION = "v2"

HANDS_HEAD = "decider_hands.npz"
WORK_HEAD = "decider_work.npz"
STEP_PROTO = "decider_step_prototypes.npz"

DEV_PROVENANCE = "DEV_ONLY"
ALLOW_DEV_ENV = "JADAL_ALLOW_DEV_HEADS"

_backbone_id_cache: dict[str, str] = {}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# shared preprocessing (train AND serve — import from here, do not duplicate)
# --------------------------------------------------------------------------
def preprocess_pil(img):
    """RGB PIL -> RGB PIL thumbnailed so the longest side is PREPROCESS_MAX_SIDE."""
    from PIL import Image

    img = img.convert("RGB")
    img.thumbnail((PREPROCESS_MAX_SIDE, PREPROCESS_MAX_SIDE), Image.BILINEAR)
    return img


def preprocess_bgr(bgr: np.ndarray):
    """BGR uint8 frame -> preprocessed RGB PIL (same pixels as preprocess_pil)."""
    import cv2
    from PIL import Image

    rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)
    return preprocess_pil(Image.fromarray(rgb))


def backbone_id(model_dir: str | Path = EG2_DIR) -> str:
    """Stable backbone identity: sha256 (16 hex) of model.safetensors, cached."""
    key = str(model_dir)
    if key not in _backbone_id_cache:
        st = Path(model_dir) / "model.safetensors"
        if st.exists():
            _backbone_id_cache[key] = "sha256:" + file_sha256(st)[:16]
        else:
            _backbone_id_cache[key] = "weights_missing"
    return _backbone_id_cache[key]


def frame_cache_key(png_bytes: bytes, *, backbone: str, budget: int) -> str:
    """Cache key over the EXACT model input bytes + backbone + budget + recipe."""
    h = hashlib.sha256()
    h.update(CACHE_VERSION.encode())
    h.update(b"|thumb=")
    h.update(str(PREPROCESS_MAX_SIDE).encode())
    h.update(b"|backbone=")
    h.update(backbone.encode())
    h.update(b"|budget=")
    h.update(str(int(budget)).encode())
    h.update(b"|png=")
    h.update(png_bytes)
    return h.hexdigest()


# --------------------------------------------------------------------------
# heads
# --------------------------------------------------------------------------
def save_head(path: str | Path, *, coef: np.ndarray, intercept: float,
              temperature: float, meta: dict | None = None) -> str:
    """Save a logistic head; returns the file sha256."""
    payload: dict[str, Any] = {
        "coef": np.asarray(coef, dtype=np.float64),
        "intercept": np.float64(intercept),
        "temperature": np.float64(temperature),
    }
    for k, v in (meta or {}).items():
        payload[f"meta_{k}"] = np.asarray(str(v))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return file_sha256(path)


def load_head(path: str | Path) -> dict[str, Any]:
    z = np.load(str(path), allow_pickle=False)
    meta = {}
    for k in z.files:
        if k.startswith("meta_"):
            meta[k[5:]] = str(z[k])
    return {
        "coef": np.asarray(z["coef"], dtype=np.float64).reshape(-1),
        "intercept": float(z["intercept"]),
        "temperature": float(z["temperature"]),
        "meta": meta,
        "sha256": file_sha256(path),
    }


def head_logit(head: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Raw (unscaled) logits for rows of X."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    return (X @ head["coef"] + head["intercept"]).reshape(-1)


def head_predict_proba(head: dict[str, Any], X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    return _sigmoid(head_logit(head, X) / head["temperature"])


def head_provenance(head: dict[str, Any]) -> str:
    return head.get("meta", {}).get("provenance", "unknown-legacy")


def save_step_prototypes(path: str | Path, *, steps: list[str],
                         prototypes: np.ndarray, temperature: float,
                         meta: dict | None = None) -> str:
    """Save nearest-prototype STEP head; returns the file sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "steps": np.asarray(steps),
        "prototypes": np.asarray(prototypes, dtype=np.float64),
        "temperature": np.float64(temperature),
    }
    for k, v in (meta or {}).items():
        payload[f"meta_{k}"] = np.asarray(str(v))
    np.savez(path, **payload)
    return file_sha256(path)


def load_step_prototypes(path: str | Path) -> dict[str, Any]:
    z = np.load(str(path), allow_pickle=True)
    meta = {}
    for k in z.files:
        if k.startswith("meta_"):
            meta[k[5:]] = str(z[k])
    return {
        "steps": [str(s) for s in z["steps"]],
        "prototypes": np.asarray(z["prototypes"], dtype=np.float64),
        "temperature": float(z["temperature"]),
        "meta": meta,
        "sha256": file_sha256(path),
    }


def step_predict_proba(proto: dict[str, Any], seg: np.ndarray) -> dict[str, float]:
    """Softmax over cosine similarities / temperature (segment embedding unit-norm)."""
    P = proto["prototypes"]
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    Pn = P / np.maximum(norms, 1e-12)
    sims = Pn @ np.asarray(seg, dtype=np.float64).reshape(-1)
    scaled = sims / proto["temperature"]
    scaled = scaled - scaled.max()
    e = np.exp(scaled)
    probs = e / e.sum()
    return {s: float(p) for s, p in sorted(zip(proto["steps"], probs))}


# --------------------------------------------------------------------------
# frame sampling
# --------------------------------------------------------------------------
def sample_frames_bgr(path: str, n: int = N_FRAMES,
                      fractions: tuple = FRAME_FRACTIONS) -> list[np.ndarray]:
    """Decode n frames at fixed fractions of the clip (BGR uint8). Empty if undecodable."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames: list[np.ndarray] = []
        for f in fractions[:n]:
            idx = min(max(int(total * f), 0), max(total - 1, 0)) if total > 0 else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        return frames
    finally:
        cap.release()


class Decider:
    """L1 decider. Constructor takes the shared Config (pipeline-compatible)."""

    def __init__(self, config: Any = None, *,
                 model_dir: str | Path = EG2_DIR,
                 heads_dir: str | Path | None = None,
                 cache_dir: str | Path | None = None,
                 image_token_budget: int = DEFAULT_BUDGET,
                 embed_fn: Callable[[Any], np.ndarray] | None = None,
                 frame_source: Callable[[str], list] | None = None,
                 use_cache: bool = True) -> None:
        self.config = config
        if config is None or getattr(config, "float_decimals", None) is None:
            self.decimals = 4
        else:
            self.decimals = int(config.float_decimals)
        self.model_dir = Path(model_dir)
        root = Path(__file__).resolve().parent.parent
        self.heads_dir = Path(heads_dir) if heads_dir else root / "models"
        # use_cache=False DISABLES the embedding cache (determinism tests).
        if use_cache:
            self.cache_dir = Path(cache_dir) if cache_dir is not None else self.heads_dir / ".embed_cache"
        else:
            self.cache_dir = None
        self.image_token_budget = int(image_token_budget)
        self._embed_fn = embed_fn  # injected fake embedder (tests)
        self._frame_source = frame_source  # injected frame supplier (tests)
        self._model = None
        self._heads: dict[str, dict[str, Any] | None] | None = None

    # -- lazy backbone ----------------------------------------------------
    def _get_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.manual_seed(SEED)
            np.random.seed(SEED)
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            if not torch.cuda.is_available():
                raise RuntimeError("Decider backbone requires CUDA (no CPU fallback for EG2).")
            model = SentenceTransformer(
                str(self.model_dir),
                model_kwargs={"torch_dtype": dtype},
                trust_remote_code=True,
            )
            model.eval()
            self._model = model
        return self._model

    def _embed_pil(self, img) -> np.ndarray:
        import torch

        model = self._get_model()
        with torch.inference_mode():
            emb = model.encode(
                img,
                batch_size=1,  # NEVER batch: batch-of-5 differs by ~1e-3 (see bench)
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                processing_kwargs={"image": {"max_soft_tokens": self.image_token_budget}},
            )
        return np.asarray(emb, dtype=np.float64).reshape(-1)

    def embed_frame(self, bgr: np.ndarray) -> np.ndarray:
        """Embed one BGR frame with the SHARED preprocessing + versioned cache."""
        img = preprocess_bgr(bgr)
        if self.cache_dir is None:
            if self._embed_fn is not None:
                return np.asarray(self._embed_fn(img), dtype=np.float64).reshape(-1)
            return self._embed_pil(img)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        key = frame_cache_key(
            buf.getvalue(), backbone=backbone_id(self.model_dir),
            budget=self.image_token_budget)
        hit = self.cache_dir / f"{key}.npy"
        if hit.exists():
            return np.load(str(hit))
        if self._embed_fn is not None:
            emb = np.asarray(self._embed_fn(img), dtype=np.float64).reshape(-1)
        else:
            emb = self._embed_pil(img)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(hit), emb)
        except OSError:
            pass
        return emb

    # -- heads ------------------------------------------------------------
    def _get_heads(self) -> dict[str, dict[str, Any] | None]:
        if self._heads is None:
            self._heads = {}
            for name, fname in (("hands_ok", HANDS_HEAD), ("active_work", WORK_HEAD)):
                p = self.heads_dir / fname
                if not p.exists():
                    self._heads[name] = None
                    continue
                head = load_head(p)
                prov = head_provenance(head)
                if prov in (DEV_PROVENANCE, "unknown-legacy") and os.environ.get(ALLOW_DEV_ENV) != "1":
                    raise RuntimeError(
                        f"refusing {prov} head {p}: retrain on the fixed splits "
                        f"or set {ALLOW_DEV_ENV}=1")
                self._heads[name] = head
            sp = self.heads_dir / STEP_PROTO
            self._heads["step"] = load_step_prototypes(sp) if sp.exists() else None
        return self._heads

    def _versions(self) -> dict[str, str]:
        import torch

        heads = self._get_heads()
        parts = []
        for name in ("hands_ok", "active_work"):
            h = heads[name]
            parts.append(f"{name}={'missing' if h is None else h['sha256'][:16]}")
        provs = sorted({head_provenance(heads[n]) for n in ("hands_ok", "active_work")
                        if heads[n] is not None})
        dtype = "torch.bfloat16" if torch.cuda.is_bf16_supported() else "torch.float32"
        backbone = (f"embeddinggemma2:budget={self.image_token_budget}"
                    f":dtype={dtype}:torch={torch.__version__}"
                    f":python={platform.python_version()}"
                    f":backbone={backbone_id(self.model_dir)}")
        sp = heads.get("step")
        return {
            "backbone": backbone,
            "heads": ",".join(parts),
            "heads_provenance": ",".join(provs) if provs else "missing",
            "granularity": "per single frame",
            "step": "none" if sp is None else f"prototypes:{sp['sha256'][:16]}",
        }

    # -- main entry -------------------------------------------------------
    def decide(self, path: str, signals: dict) -> dict:
        if self._frame_source is not None:
            frames = [np.asarray(f) for f in self._frame_source(str(path))]
        else:
            frames = sample_frames_bgr(path)
        if not frames:
            return self._round({
                "hands_ok": {"p_yes": 0.0},
                "active_work": {"p_yes": 0.0},
                "step": {"choice": None, "probs": {}, "answer_conf": 0.0},
                "model_versions": {**self._versions(), "frames_decoded": "0"},
            })
        # Per-frame embeddings (shared preprocessing), per-frame logits for
        # hands/work averaged (train uses single frames); mean-pooled segment
        # embedding (renormalised) for the STEP prototype head.
        embs = np.stack([self.embed_frame(f) for f in frames], axis=0)
        heads = self._get_heads()
        probs = {}
        for name in ("hands_ok", "active_work"):
            h = heads[name]
            if h is None:
                probs[name] = 0.5
            else:
                probs[name] = float(_sigmoid(head_logit(h, embs).mean() / h["temperature"]).reshape(-1)[0])
        seg = embs.mean(axis=0)
        n = np.linalg.norm(seg)
        if n > 0:
            seg = seg / n
        step_proto = heads.get("step")
        if step_proto is None:
            step = {"choice": None, "probs": {}, "answer_conf": 0.0}
        else:
            p = step_predict_proba(step_proto, seg)
            choice = max(p, key=p.get)
            step = {"choice": choice, "probs": p, "answer_conf": p[choice]}
        return self._round({
            "hands_ok": {"p_yes": probs["hands_ok"]},
            "active_work": {"p_yes": probs["active_work"]},
            "step": step,
            "model_versions": {**self._versions(), "frames_decoded": str(len(frames))},
        })

    def _round(self, obj):
        if isinstance(obj, float):
            return round(obj, self.decimals)
        if isinstance(obj, list):
            return [self._round(v) for v in obj]
        if isinstance(obj, dict):
            return {k: self._round(v) for k, v in obj.items()}
        return obj
