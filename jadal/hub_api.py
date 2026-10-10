"""FastAPI APIRouter for the video hub (GitHub issue #27).

Mount (integration owns this; hub never touches jadal/service.py)::

    from jadal.hub_api import create_hub_router
    app.include_router(create_hub_router(index, embed_fn))

No GPU/model load at import time. The router takes an Index and an
embed function via dependency injection so tests use fakes.

Segment-id grammar (ONE grammar for eval ids, API paths and queries):
``SEGMENT_ID_RE = ^[A-Za-z0-9][A-Za-z0-9_.#\\-]*$`` — letters, digits,
``_ . # -`` (so eval ids like ``User_1_412_2010_3_480#0003`` work; in URLs
the ``#`` MUST be percent-encoded as ``%23``, e.g.
``/hub/thumb/User_1_412_2010_3_480%230003.jpg``, otherwise the browser
treats it as a fragment). ``/``, ``..`` and ``%`` are rejected even after
URL-decoding, and the id must exist in the index — no path traversal.

Endpoints (all under prefix /hub):
- GET /hub/search?q=&k=&site=&worker=&verdict=&reason=&clip=&step=&t0=&t1=
- GET /hub/similar?segment_id=&k=
- GET /hub/clusters?threshold= (413 when N > the index's max_clusters_n)
- GET /hub/thumb/{segment_id}.jpg  (mid-frame JPEG, cached; id validated
  against the index — no path traversal)
- GET /hub/stats

NOT DONE:
- similar-by-upload (query clip bytes) — needs an upload path; out of scope.
- auth/rate limits — inherit from the mounting service.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from jadal.hub import ClusterLimitError, Index

# One safe id grammar: start alnum, then alnum + _ . # - (no / % control).
SEGMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.#\-]*$")


def valid_segment_id(sid: str) -> bool:
    """Grammar check + traversal guards (applied AFTER URL-decoding)."""
    return bool(SEGMENT_ID_RE.match(sid)) and ".." not in sid \
        and "/" not in sid and "%" not in sid and "\\" not in sid


def _filters_from(site=None, worker=None, verdict=None, reason=None,
                  clip=None, step=None, t0=None, t1=None) -> dict:
    f: dict = {}
    if site is not None:
        f["site"] = site
    if worker is not None:
        f["worker"] = worker
    if verdict is not None:
        f["verdict"] = verdict
    if reason is not None:
        f["reason"] = reason
    if clip is not None:
        f["clip"] = clip
    if step is not None:
        f["step"] = step
    if t0 is not None:
        f["t0"] = t0
    if t1 is not None:
        f["t1"] = t1
    return f


def default_hub_dir() -> Path:
    return Path(os.environ.get("JADAL_HUB_DIR", "data_runtime/hub"))


def create_hub_router(index: Index, embed_fn: Callable | None = None,
                      thumb_cache_dir: str | Path | None = None) -> APIRouter:
    """Build the /hub router over an existing Index (dependency injection)."""
    router = APIRouter(prefix="/hub")
    thumbs = Path(thumb_cache_dir) if thumb_cache_dir else \
        default_hub_dir() / "thumbs"
    efn = embed_fn if embed_fn is not None else getattr(index, "_embed_fn", None)

    @router.get("/search")
    def search(q: str = Query(...), k: int = Query(5, ge=1, le=100),
               site: str | None = None, worker: str | None = None,
               verdict: str | None = None, reason: str | None = None,
               clip: str | None = None, step: str | None = None,
               t0: float | None = None, t1: float | None = None):
        if efn is None:
            raise HTTPException(500, "hub has no embed_fn configured")
        try:
            vec = efn(q)
        except Exception:
            # Deliberately generic: never leak exception text/paths.
            raise HTTPException(500, "hub embedding failed")
        return index.search_by_vector(
            vec, k=k, filters=_filters_from(site, worker, verdict, reason,
                                            clip, step, t0, t1))

    @router.get("/similar")
    def similar(segment_id: str = Query(...),
                k: int = Query(5, ge=1, le=100)):
        if not valid_segment_id(segment_id):
            raise HTTPException(404, "unknown segment")
        try:
            return index.search_similar(segment_id=segment_id, k=k)
        except KeyError:
            raise HTTPException(404, "unknown segment")

    @router.get("/clusters")
    def clusters(threshold: float = Query(0.95, ge=0.0, le=1.0)):
        try:
            return index.dedup_clusters(threshold=threshold)
        except ClusterLimitError as e:
            raise HTTPException(413, str(e))

    @router.get("/stats")
    def stats():
        return {"n_segments": len(index), "dim": index.dim,
                "index_hash": index.index_hash()}

    @router.get("/thumb/{segment_id}.jpg")
    def thumb(segment_id: str):
        # Grammar + traversal guards, then the id must exist in the index.
        if not valid_segment_id(segment_id):
            raise HTTPException(404, "unknown segment")
        meta = index.get_meta(segment_id)
        if meta is None:
            raise HTTPException(404, "unknown segment")
        src = meta.get("path")
        if not src:
            raise HTTPException(404, "segment has no stored path")
        # Never join user input into a path: the file comes from the index.
        src_p = Path(src)
        if not src_p.is_file():
            raise HTTPException(404, "clip file not available")
        thumbs.mkdir(parents=True, exist_ok=True)
        # Collision-free cached filename: sanitised stem + sha1 of the id.
        safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", segment_id)
        tag = hashlib.sha1(segment_id.encode()).hexdigest()[:8]
        out = thumbs / f"{safe}.{tag}.jpg"
        if not out.is_file():
            try:
                from jadal.hub_embed import mid_frame
                img = mid_frame(src_p)
                img.save(out, "JPEG", quality=85)
            except FileNotFoundError:
                raise HTTPException(404, "clip file not available")
            except (IOError, ValueError):
                raise HTTPException(404, "clip frame not readable")
            except Exception:
                raise HTTPException(500, "thumbnail generation failed")
        return FileResponse(out, media_type="image/jpeg")

    return router
