"""FastAPI APIRouter for the video hub (GitHub issue #27).

Mount (integration owns this; hub never touches jadal/service.py)::

    from jadal.hub_api import create_hub_router
    app.include_router(create_hub_router(index, embed_fn))

No GPU/model load at import time. The router takes an Index and an
embed function via dependency injection so tests use fakes.

Endpoints (all under prefix /hub):
- GET /hub/search?q=&k=&site=&worker=&verdict=&reason=&clip=&step=&t0=&t1=
- GET /hub/similar?segment_id=&k=
- GET /hub/clusters?threshold=
- GET /hub/thumb/{segment_id}.jpg  (mid-frame JPEG, cached; id validated
  against the index — no path traversal)
- GET /hub/stats

NOT DONE:
- similar-by-upload (query clip bytes) — needs an upload path; out of scope.
- auth/rate limits — inherit from the mounting service.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from jadal.hub import Index

SEGMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


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


def create_hub_router(index: Index, embed_fn: Callable | None = None,
                      thumb_cache_dir: str | Path | None = None) -> APIRouter:
    """Build the /hub router over an existing Index (dependency injection)."""
    router = APIRouter(prefix="/hub")
    thumbs = Path(thumb_cache_dir) if thumb_cache_dir else \
        Path("data_runtime/hub/thumbs")
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
        except Exception as e:
            raise HTTPException(500, f"embed failed: {e}")
        return index.search_by_vector(
            vec, k=k, filters=_filters_from(site, worker, verdict, reason,
                                            clip, step, t0, t1))

    @router.get("/similar")
    def similar(segment_id: str = Query(...),
                k: int = Query(5, ge=1, le=100)):
        try:
            return index.search_similar(segment_id=segment_id, k=k)
        except KeyError:
            raise HTTPException(404, f"unknown segment_id: {segment_id}")

    @router.get("/clusters")
    def clusters(threshold: float = Query(0.95, ge=0.0, le=1.0)):
        return index.dedup_clusters(threshold=threshold)

    @router.get("/stats")
    def stats():
        return {"n_segments": len(index), "dim": index.dim,
                "index_hash": index.index_hash()}

    @router.get("/thumb/{segment_id}.jpg")
    def thumb(segment_id: str):
        # No path traversal: id must match the allowlist AND exist in index.
        if not SEGMENT_ID_RE.match(segment_id) or ".." in segment_id \
                or "/" in segment_id:
            raise HTTPException(404, "unknown segment")
        meta = index._meta.get(segment_id)
        if meta is None:
            raise HTTPException(404, "unknown segment")
        src = meta.get("path")
        if not src:
            raise HTTPException(404, "segment has no stored path")
        # Resolve strictly inside the filesystem; never join user input.
        src_p = Path(src)
        if not src_p.is_file():
            raise HTTPException(404, "clip file not available")
        thumbs.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", segment_id)
        out = thumbs / f"{safe}.jpg"
        if not out.is_file():
            try:
                from jadal.hub_embed import mid_frame
                img = mid_frame(src_p)
                img.save(out, "JPEG", quality=85)
            except FileNotFoundError:
                raise HTTPException(404, "clip file not available")
            except Exception as e:
                raise HTTPException(500, f"thumb failed: {e}")
        return FileResponse(out, media_type="image/jpeg")

    return router
