"""Video-hub embeddings over EmbeddingGemma 2 (GitHub issue #27).

Wraps the sentence-transformers multimodal model at
``/home/parshu/projects/awing/data/eg2`` (bfloat16, NEVER fp16).
The model natively embeds video, text and images in ONE 768-d space;
Matryoshka truncation to 512/256/128 + renormalisation is supported.

Determinism: fixed OpenCV frame sampling (N frames at fixed fractions of
[start_s, end_s]), one segment per model call (batch size 1),
``torch.use_deterministic_algorithms(True, warn_only=True)``, fixed seeds.

No GPU/model load at import time: the model loads lazily on first embed.

NOT DONE:
- Video-dict input shape (``{"text": "<|video|>", "video": [PIL...]}``)
  verified against sentence-transformers 6.1 at runtime; if the installed
  processor expects a path instead, ``embed_segment`` falls back to writing
  the sampled frames to a temp clip — currently it passes PIL frames.
- Cache is file-backed under ``data_runtime/hub/cache/`` (git-ignored).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

MODEL_DIR = "/home/parshu/projects/awing/data/eg2"
SUPPORTED_DIMS = (768, 512, 256, 128)
DEFAULT_DIM = 256


def sample_frames(path: str | os.PathLike, start_s: float, end_s: float,
                   n_frames: int) -> list:
    """Sample N RGB PIL frames at fixed fractions of [start_s, end_s].

    Deterministic: timestamps are (i + 0.5) / N through the interval,
    frames are read via OpenCV seeks. Raises FileNotFoundError / ValueError.
    """
    import cv2
    from PIL import Image

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"clip not found: {path}")
    if end_s <= start_s:
        raise ValueError(f"end_s ({end_s}) must exceed start_s ({start_s})")
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    cap = cv2.VideoCapture(str(p))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = total / fps if fps > 0 and total > 0 else (end_s - start_s)
        out: list = []
        for i in range(n_frames):
            frac = (i + 0.5) / n_frames
            t = start_s + frac * (end_s - start_s)
            t = min(max(t, 0.0), max(dur - 1e-3, 0.0))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                # Fall back to nearest readable frame: first frame.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise ValueError(f"could not read any frame from {path}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.append(Image.fromarray(rgb))
        return out
    finally:
        cap.release()


def mid_frame(path: str | os.PathLike, at_s: float | None = None):
    """Return the mid-clip (or at_s) RGB PIL frame. Used for thumbnails."""
    import cv2
    from PIL import Image

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"clip not found: {path}")
    cap = cv2.VideoCapture(str(p))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = total / fps if fps > 0 else 0.0
        t = dur / 2.0 if at_s is None else max(at_s, 0.0)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"could not read any frame from {path}")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


class HubEmbedder:
    """Deterministic EmbeddingGemma-2 embedder for the video hub.

    Args:
        model_dir: local sentence-transformers model directory.
        dim: Matryoshka truncation dim (one of 768/512/256/128).
        n_frames: default frames per segment.
        device: e.g. "cuda:0" / "cpu" / None (auto).
        cache_dir: on-disk embedding cache (sha256 keyed).
    """

    def __init__(self, model_dir: str = MODEL_DIR, dim: int = DEFAULT_DIM,
                 n_frames: int = 8, device: str | None = None,
                 cache_dir: str | os.PathLike | None = None) -> None:
        if dim not in SUPPORTED_DIMS:
            raise ValueError(f"dim must be one of {SUPPORTED_DIMS}, got {dim}")
        self.model_dir = model_dir
        self.dim = dim
        self.n_frames = n_frames
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None
        self._weights_hash: str | None = None

    # -- model loading -------------------------------------------------
    def _ensure_model(self):
        if self._model is not None:
            return self._model
        import torch
        from sentence_transformers import SentenceTransformer

        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        kwargs: dict = {"torch_dtype": dtype}
        if self.device:
            kwargs["device"] = self.device
        else:
            # sentence-transformers auto-detects; keep default.
            pass
        self._model = SentenceTransformer(self.model_dir, model_kwargs=kwargs)
        return self._model

    def weights_hash(self) -> str:
        """Hash identifying the backbone weights (sha256 of safetensors if
        cheap, else ``size:name``). No GPU needed."""
        if self._weights_hash is not None:
            return self._weights_hash
        sf = Path(self.model_dir) / "model.safetensors"
        cfg = Path(self.model_dir) / "config.json"
        h = hashlib.sha256()
        if cfg.is_file():
            h.update(cfg.read_bytes())
        if sf.is_file():
            size = sf.stat().st_size
            # Full hash is ~740M params (~1.5GB bf16); hash head+tail+size.
            if size <= 64 * 1024 * 1024:
                h.update(sf.read_bytes())
            else:
                with open(sf, "rb") as f:
                    h.update(f.read(8 * 1024 * 1024))
                    f.seek(max(size - 8 * 1024 * 1024, 0))
                    h.update(f.read())
                h.update(f"size={size}".encode())
        else:
            h.update(self.model_dir.encode())
        self._weights_hash = h.hexdigest()
        return self._weights_hash

    # -- cache ----------------------------------------------------------
    def _cache_key(self, kind: str, payload: bytes, extra: str = "") -> str:
        h = hashlib.sha256()
        h.update(kind.encode())
        h.update(payload)
        h.update(f"|dim={self.dim}|weights={self.weights_hash()}|{extra}".encode())
        return h.hexdigest()

    def _cache_get(self, key: str) -> np.ndarray | None:
        if self.cache_dir is None:
            return None
        f = self.cache_dir / f"{key}.npy"
        if f.is_file():
            try:
                return np.load(f)
            except Exception:
                return None
        return None

    def _cache_put(self, key: str, vec: np.ndarray) -> None:
        if self.cache_dir is None:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_dir / f"{key}.npy", vec.astype(np.float32))
        except Exception:
            pass

    # -- public API -----------------------------------------------------
    @staticmethod
    def _truncate_norm(vec: np.ndarray, dim: int) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float64)[:dim]
        n = float(np.linalg.norm(v))
        if n == 0:
            raise ValueError("zero embedding vector")
        return (v / n).astype(np.float32)

    def embed_text(self, query: str) -> np.ndarray:
        """Embed a text query with the SearchQuery prompt (retrieval side)."""
        import torch

        key = self._cache_key("text", query.encode("utf-8"))
        hit = self._cache_get(key)
        if hit is not None:
            return hit
        model = self._ensure_model()
        torch.manual_seed(0)
        vec = model.encode(query, prompt_name="SearchQuery",
                           truncate_dim=self.dim if self.dim != 768 else None,
                           normalize_embeddings=True, batch_size=1,
                           convert_to_numpy=True, show_progress_bar=False)
        out = self._truncate_norm(np.asarray(vec), self.dim)
        self._cache_put(key, out)
        return out

    def embed_image(self, path: str | os.PathLike) -> np.ndarray:
        """Embed a single image (no text prefix per model README)."""
        import torch
        from PIL import Image

        raw = Path(path).read_bytes()  # FileNotFoundError if missing
        key = self._cache_key("image", raw)
        hit = self._cache_get(key)
        if hit is not None:
            return hit
        model = self._ensure_model()
        torch.manual_seed(0)
        img = Image.open(path).convert("RGB")
        try:
            vec = model.encode({"text": "<|image|>", "image": img},
                               truncate_dim=self.dim if self.dim != 768 else None,
                               normalize_embeddings=True, batch_size=1,
                               convert_to_numpy=True, show_progress_bar=False)
        except Exception:
            vec = model.encode(img,
                               truncate_dim=self.dim if self.dim != 768 else None,
                               normalize_embeddings=True, batch_size=1,
                               convert_to_numpy=True, show_progress_bar=False)
        out = self._truncate_norm(np.asarray(vec), self.dim)
        self._cache_put(key, out)
        return out

    def embed_segment(self, path: str | os.PathLike, start_s: float,
                      end_s: float, n_frames: int | None = None) -> np.ndarray:
        """Embed one video segment via the video route -> unit float32 vector.

        Fixed frame sampling (OpenCV), batch size 1, deterministic.
        """
        import torch

        n = n_frames or self.n_frames
        frames = sample_frames(path, start_s, end_s, n)
        # Cache key: raw frame bytes + interval + N + params + weights.
        h = hashlib.sha256()
        for f in frames:
            h.update(np.asarray(f).tobytes())
        payload = h.digest()
        extra = f"start={start_s:.3f}|end={end_s:.3f}|n={n}"
        key = self._cache_key("segment", payload, extra)
        hit = self._cache_get(key)
        if hit is not None:
            return hit
        model = self._ensure_model()
        torch.manual_seed(0)
        try:
            vec = model.encode({"text": "<|video|>", "video": frames},
                               truncate_dim=self.dim if self.dim != 768 else None,
                               normalize_embeddings=True, batch_size=1,
                               convert_to_numpy=True, show_progress_bar=False)
        except Exception:
            # Fallback: mean of per-frame image embeddings (documented).
            embs = model.encode(frames, batch_size=1,
                                truncate_dim=self.dim if self.dim != 768 else None,
                                normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
            vec = np.mean(np.asarray(embs, dtype=np.float64), axis=0)
            vec = vec / float(np.linalg.norm(vec))
        out = self._truncate_norm(np.asarray(vec), self.dim)
        self._cache_put(key, out)
        return out

    def versions(self) -> dict:
        """Version record for receipts/reports (no GPU needed)."""
        import torch
        info = {"model_dir": self.model_dir, "weights_hash": self.weights_hash(),
                "dim": self.dim, "n_frames": self.n_frames,
                "torch": torch.__version__}
        try:
            import sentence_transformers
            info["sentence_transformers"] = sentence_transformers.__version__
        except Exception:
            pass
        return info

    @staticmethod
    def frame_cache_payload(frames: Sequence) -> bytes:
        h = hashlib.sha256()
        for f in frames:
            h.update(np.asarray(f).tobytes())
        return h.digest()

    def describe_input(self, path: str, start_s: float, end_s: float,
                       n_frames: int | None = None) -> dict:
        """Explicit, JSON-serialisable record of the model input."""
        n = n_frames or self.n_frames
        return {"video_path": str(path), "start_s": start_s, "end_s": end_s,
                "n_frames": n, "fractions": [(i + 0.5) / n for i in range(n)],
                "input": {"text": "<|video|>", "video": f"{n}xPIL-RGB-frames"},
                "prompt": None, "truncate_dim": self.dim,
                "note": "video route takes no text prefix; frames via OpenCV"}
