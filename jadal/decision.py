"""L1 decider: EmbeddingGemma-2 embeddings + linear-probe heads (issue #7).

``Decider.decide(path, signals) -> Decisions`` per ``docs/SPEC.md``::

    {"hands_ok": {"p_yes": float}, "active_work": {"p_yes": float},
     "step": {"choice": str|null, "probs": {str: float}, "answer_conf": float},
     "model_versions": {str: str}}

Backbone: EmbeddingGemma 2 from ``/home/parshu/projects/awing/data/eg2``
via sentence-transformers, bfloat16 (NEVER fp16), eval mode,
``torch.use_deterministic_algorithms(True, warn_only=True)``,
``cudnn.benchmark=False``, fixed seeds. Per-frame batch size 1 with a
FIXED frame sampling (4 frames at fixed fractions of the segment, OpenCV
decode -> PIL). The image token budget is configurable (default 140).

Determinism note (measured in ``eval/results/bench_eg2.json``): the same
frame embedded alone vs inside a batch of 5 differs by ~1e-3 (not
bit-equal), so this module NEVER batches: every frame is encoded with
batch size 1, making a segment identical alone or inside a batch of 5.

Heads: scikit-learn logistic regression on frozen L2-normalised segment
embeddings (mean of frame embeddings) for ``hand_count >= 1``
(``hands_ok``) and ``active_labor == yes`` (``active_work``), with a
single temperature each fitted on the calibration split (see
``jadal.certify.fit_temperature``). Head files live under ``models/``
(git-ignored); their sha256 is recorded in ``model_versions``.
Frame embeddings are cached on disk under ``models/.embed_cache/``
keyed by sha256 of the resized frame RGB bytes.

NOT DONE:
- STEP head: ``step`` is ``{"choice": None, "probs": {}, "answer_conf": 0.0}``
  until the IndEgo stretch goal lands (nearest-prototype/logistic over the
  14 ECU assembly steps, wired here).
- No Laya fusion (Laya is NOT required for this baseline).
- If the clip cannot be decoded, both heads return ``p_yes = 0.0``
  (L2 marks UNDECODABLE); if head files are absent, both return 0.5
  with ``model_versions["heads"] = "missing"``.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

EG2_DIR = "/home/parshu/projects/awing/data/eg2"
SEED = 1234
N_FRAMES = 4
FRAME_FRACTIONS = (0.125, 0.375, 0.625, 0.875)
DEFAULT_BUDGET = 140
EMBED_DIM = 768
CACHE_MAX_SIDE = 256

HANDS_HEAD = "decider_hands.npz"
WORK_HEAD = "decider_work.npz"


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


def save_head(path: str | Path, *, coef: np.ndarray, intercept: float,
              temperature: float, meta: dict | None = None) -> str:
    """Save a logistic head; returns the file sha256."""
    payload: dict[str, Any] = {
        "coef": np.asarray(coef, dtype=np.float64),
        "intercept": np.float64(intercept),
        "temperature": np.float64(temperature),
    }
    for k, v in (meta or {}).items():
        payload[f"meta_{k}"] = np.asarray(v)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return file_sha256(path)


def load_head(path: str | Path) -> dict[str, Any]:
    z = np.load(str(path), allow_pickle=False)
    return {
        "coef": np.asarray(z["coef"], dtype=np.float64).reshape(-1),
        "intercept": float(z["intercept"]),
        "temperature": float(z["temperature"]),
        "sha256": file_sha256(path),
    }


def head_predict_proba(head: dict[str, Any], X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    logits = X @ head["coef"] + head["intercept"]
    return _sigmoid(logits / head["temperature"])


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


def frame_cache_key(bgr: np.ndarray) -> str:
    """Stable cache key: sha256 of RGB bytes resized to CACHE_MAX_SIDE."""
    import cv2

    h, w = bgr.shape[:2]
    s = CACHE_MAX_SIDE / max(h, w)
    if s < 1.0:
        small = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = bgr
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def bgr_to_pil(bgr: np.ndarray):
    import cv2
    from PIL import Image

    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


class Decider:
    """L1 decider. Constructor takes the shared Config (pipeline-compatible)."""

    def __init__(self, config: Any = None, *,
                 model_dir: str | Path = EG2_DIR,
                 heads_dir: str | Path | None = None,
                 cache_dir: str | Path | None = None,
                 image_token_budget: int = DEFAULT_BUDGET,
                 embed_fn: Callable[[Any], np.ndarray] | None = None) -> None:
        self.config = config
        self.decimals = int(getattr(config, "float_decimals", 4) or 4)
        self.model_dir = Path(model_dir)
        root = Path(__file__).resolve().parent.parent
        self.heads_dir = Path(heads_dir) if heads_dir else root / "models"
        self.cache_dir = Path(cache_dir) if cache_dir else self.heads_dir / ".embed_cache"
        self.image_token_budget = int(image_token_budget)
        self._embed_fn = embed_fn  # injected fake embedder (tests)
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
        key = frame_cache_key(bgr)
        hit = self.cache_dir / f"{key}.npy"
        if hit.exists():
            return np.load(str(hit))
        if self._embed_fn is not None:
            emb = np.asarray(self._embed_fn(bgr), dtype=np.float64).reshape(-1)
        else:
            emb = self._embed_pil(bgr_to_pil(bgr))
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
                self._heads[name] = load_head(p) if p.exists() else None
        return self._heads

    def _versions(self) -> dict[str, str]:
        heads = self._get_heads()
        parts = []
        for name in ("hands_ok", "active_work"):
            h = heads[name]
            parts.append(f"{name}={'missing' if h is None else h['sha256'][:16]}")
        st = self.model_dir / "model.safetensors"
        backbone = f"embeddinggemma2:budget={self.image_token_budget}"
        if st.exists():
            backbone += f":safetensors_bytes={st.stat().st_size}"
        else:
            backbone += ":weights_missing"
        return {
            "backbone": backbone,
            "heads": ",".join(parts),
        }

    # -- main entry -------------------------------------------------------
    def decide(self, path: str, signals: dict) -> dict:
        frames = sample_frames_bgr(path) if self._embed_fn is None else _test_frames(path, self._embed_fn)
        if not frames:
            return self._round({
                "hands_ok": {"p_yes": 0.0},
                "active_work": {"p_yes": 0.0},
                "step": {"choice": None, "probs": {}, "answer_conf": 0.0},
                "model_versions": {**self._versions(), "frames_decoded": "0"},
            })
        embs = np.stack([self.embed_frame(f) for f in frames], axis=0)
        seg = embs.mean(axis=0)
        n = np.linalg.norm(seg)
        if n > 0:
            seg = seg / n
        heads = self._get_heads()
        probs = {}
        for name in ("hands_ok", "active_work"):
            h = heads[name]
            probs[name] = float(head_predict_proba(h, seg)[0]) if h is not None else 0.5
        return self._round({
            "hands_ok": {"p_yes": probs["hands_ok"]},
            "active_work": {"p_yes": probs["active_work"]},
            "step": {"choice": None, "probs": {}, "answer_conf": 0.0},
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


def _test_frames(path: str, embed_fn: Callable) -> list:
    """Test hook: when an embed_fn is injected, read frame count from a sidecar.

    Tests write ``<path>.frames.npz`` (array ``frames``, uint8 NHWC) next to a
    dummy path; without the sidecar, one synthetic frame is used. Real video
    is never read in this mode, keeping CPU tests hermetic.
    """
    sidecar = Path(str(path) + ".frames.npz")
    if sidecar.exists():
        z = np.load(str(sidecar))
        return [np.asarray(f) for f in z["frames"]]
    rng = np.random.default_rng(int(hashlib.md5(str(path).encode()).hexdigest()[:8], 16))
    return [rng.integers(0, 256, size=(64, 64, 3)).astype(np.uint8)]
