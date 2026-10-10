"""FastAPI service for Jadal QA (GitHub issue #3).

Runs the pipeline from ``jadal/pipeline.py`` behind the endpoints in
docs/SPEC.md, persists state in SQLite (``data_runtime/jadal.db`` by default),
closes a batch every ``batch_size`` segments or after ``batch_max_age_s`` seconds
per site, watches ``data_runtime/bucket/`` for docked clips, pushes batch events
on ``/ws`` and serves the ``web/`` dashboard at ``/``.

Run with::

    /home/parshu/envs/jadal/bin/python -m uvicorn jadal.service:app

Env knobs (all optional): ``JADAL_BATCH_MAX_AGE_S`` (30), ``JADAL_WATCH_BUCKET``
(1), ``JADAL_WATCH_SITE`` (site-sim), ``JADAL_WATCH_WORKER`` (w-sim),
``JADAL_WATCH_POLL_S`` (0.25). No network calls happen at import time; the only
I/O at import is creating the SQLite file.

NOT DONE:
- L0/L1/segmenter/certify/receipts are the stubs in jadal/stubs.py until those
  branches merge (pipeline resolves the real modules automatically once they do).
- /stats/risk_coverage needs calibration points (eval module); with no rows it
  returns [] -- it is not a measurement of anything yet.
- /sop/search, /stats/sop_gaps and /pico/status return empty/offline payloads
  until the sop/ and pico/ modules (other branches) exist.
- A real bucket watcher is a polling thread (inotify later); the web/ dashboard
  is served only if the directory exists.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from jadal import pipeline, store as store_mod
from jadal.config import Config, DEFAULT


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "")


class IngestEvent(BaseModel):
    path: str
    worker_id: str
    site_id: str


class _Service:
    """All runtime state for one app instance (config, store, queues, threads)."""

    def __init__(self, config: Config, *, batch_max_age_s: float,
                 watch_bucket: bool, watch_site: str, watch_worker: str,
                 watch_poll_s: float) -> None:
        self.cfg = config
        self.store = store_mod.Store(config.db_path)
        self.batch_max_age_s = batch_max_age_s
        self.watch_bucket = watch_bucket
        self.watch_site = watch_site
        self.watch_worker = watch_worker
        self.watch_poll_s = watch_poll_s
        self._lock = threading.RLock()
        self._pending: dict[str, list[dict]] = {}   # site -> [{"segment", "ts"}]
        self._ingested: set[tuple[str, str]] = set()  # (resolved path, sha256)
        self._seen_files: set[str] = set()
        self._ws_clients: set[WebSocket] = set()
        self._recent: deque[dict] = deque(maxlen=50)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = threading.Event()
        self._threads_started = False
        self.started_utc = datetime.now(timezone.utc)

    # -- ingest ------------------------------------------------------------
    def handle_clip(self, path: str | Path, worker_id: str, site_id: str) -> dict:
        clip = Path(path)
        if not clip.is_file():
            raise FileNotFoundError(str(clip))
        key = (str(clip.resolve()), self._sha(clip))
        with self._lock:
            if key in self._ingested:
                return {"duplicate": True, "segments": [],
                        "pending": self.pending_count(), "batch": None}
            self._ingested.add(key)
        segments = pipeline.process_clip(clip, worker_id, site_id, self.cfg)
        with self._lock:
            buf = self._pending.setdefault(site_id, [])
            for seg in segments:
                buf.append({"segment": seg, "ts": time.monotonic()})
            triggered = len(buf) >= int(self.cfg.batch_size)
        batch = self.flush_site(site_id) if triggered else None
        return {"duplicate": False,
                "segments": [s["segment_id"] for s in segments],
                "pending": self.pending_count(), "batch": batch}

    @staticmethod
    def _sha(clip: Path) -> str:
        import hashlib
        h = hashlib.sha256()
        with open(clip, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    def pending_count(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._pending.values())

    # -- batches -----------------------------------------------------------
    def flush_site(self, site_id: str) -> dict | None:
        with self._lock:
            buf = self._pending.get(site_id) or []
            if not buf:
                return None
            segments = [entry["segment"] for entry in buf]
            self._pending[site_id] = []
        batch, receipts = pipeline.run_batch(segments, self.cfg)
        rows = [dict(seg, created_utc=batch["created_utc"]) for seg in segments]
        self.store.save_batch(batch, receipts, rows)
        self._emit_batch(batch, receipts)
        return batch

    # -- websocket ---------------------------------------------------------
    def _emit_batch(self, batch: dict, receipts: list[dict]) -> None:
        event = {"type": "batch", **batch,
                 "receipt_ids": [r["receipt_id"] for r in receipts]}
        with self._lock:
            self._recent.append(event)
            clients = list(self._ws_clients)
        if not clients:
            return
        loop = self._loop
        if loop is None or not loop.is_running():
            return

        async def _send() -> None:
            for ws in clients:
                try:
                    await ws.send_json(event)
                except Exception:
                    self._ws_clients.discard(ws)

        try:
            asyncio.run_coroutine_threadsafe(_send(), loop)
        except RuntimeError:
            pass

    def register_ws(self, ws: WebSocket, loop) -> None:
        with self._lock:
            self._ws_clients.add(ws)
            if self._loop is None:
                self._loop = loop

    def unregister_ws(self, ws: WebSocket) -> None:
        with self._lock:
            self._ws_clients.discard(ws)

    def recent_events(self) -> list[dict]:
        with self._lock:
            return list(self._recent)

    # -- background threads ------------------------------------------------
    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop
        with self._lock:
            if self._threads_started:
                return
            self._threads_started = True
        threading.Thread(target=self._worker_loop, name="jadal-worker",
                         daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _worker_loop(self) -> None:
        while not self._stop.wait(self.watch_poll_s):
            try:
                self._tick()
            except Exception:
                pass  # the watcher must never take the app down

    def _tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            due = [site for site, buf in self._pending.items()
                   if buf and now - buf[0]["ts"] > self.batch_max_age_s]
        for site in due:
            self.flush_site(site)
        if self.watch_bucket:
            self._watch_bucket_tick()

    def _watch_bucket_tick(self) -> None:
        bucket = Path(self.cfg.bucket_dir)
        if not bucket.is_dir():
            return
        for clip in sorted(bucket.iterdir()):
            if not clip.is_file() or clip.name.startswith("."):
                continue
            if clip.name in self._seen_files:
                continue
            self._seen_files.add(clip.name)
            try:
                self.handle_clip(clip, self.watch_worker, self.watch_site)
            except Exception:
                continue


def create_app(config: Config | None = None, *, batch_max_age_s: float | None = None,
               watch_bucket: bool | None = None, watch_site: str | None = None,
               watch_worker: str | None = None,
               watch_poll_s: float | None = None) -> FastAPI:
    """Build the FastAPI app; inject a Config (and overrides) for tests."""
    cfg = config or DEFAULT
    svc = _Service(
        cfg,
        batch_max_age_s=(batch_max_age_s if batch_max_age_s is not None
                         else _env_float("JADAL_BATCH_MAX_AGE_S", 30.0)),
        watch_bucket=(watch_bucket if watch_bucket is not None
                      else _env_flag("JADAL_WATCH_BUCKET", True)),
        watch_site=(watch_site or os.environ.get("JADAL_WATCH_SITE", "site-sim")),
        watch_worker=(watch_worker or os.environ.get("JADAL_WATCH_WORKER", "w-sim")),
        watch_poll_s=(watch_poll_s if watch_poll_s is not None
                      else _env_float("JADAL_WATCH_POLL_S", 0.25)),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        svc.start(loop=asyncio.get_running_loop())
        try:
            yield
        finally:
            svc.stop()

    app = FastAPI(title="Jadal QA service", version=cfg.pipeline_version,
                  lifespan=lifespan)
    app.state.service = svc

    # -- health ------------------------------------------------------------
    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "pipeline_version": cfg.pipeline_version,
            "pending_segments": svc.pending_count(),
            "started_utc": svc.started_utc.isoformat(),
        }

    # -- ingest ------------------------------------------------------------
    @app.post("/ingest/event")
    def ingest_event(event: IngestEvent) -> dict:
        try:
            return svc.handle_clip(event.path, event.worker_id, event.site_id)
        except FileNotFoundError:
            raise HTTPException(status_code=400,
                                detail=f"clip not found: {event.path}")
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(exc))

    # -- batches / receipts ------------------------------------------------
    @app.get("/batches")
    def list_batches() -> dict:
        return {"batches": svc.store.list_batches()}

    @app.get("/batches/{batch_id}")
    def get_batch(batch_id: str) -> dict:
        batch = svc.store.get_batch(batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="batch not found")
        batch["receipt_ids"] = [r["receipt_id"]
                                for r in svc.store.get_receipts_for_batch(batch_id)]
        return batch

    @app.get("/receipts/{receipt_id}")
    def get_receipt(receipt_id: str) -> dict:
        receipt = svc.store.get_receipt(receipt_id)
        if receipt is None:
            raise HTTPException(status_code=404, detail="receipt not found")
        return receipt

    @app.post("/receipts/{receipt_id}/reverify")
    def reverify(receipt_id: str) -> dict:
        receipt = svc.store.get_receipt(receipt_id)
        if receipt is None:
            raise HTTPException(status_code=404, detail="receipt not found")
        segment = svc.store.get_segment(receipt["verdict"]["segment_id"])
        if segment is None:
            raise HTTPException(status_code=409, detail="segment row missing")
        return pipeline.reverify_receipt(receipt, segment, svc.cfg)

    # -- stats / sop / voice / pico ----------------------------------------
    @app.get("/stats/risk_coverage")
    def stats_risk_coverage() -> list[dict]:
        return pipeline.risk_coverage(svc.store.calibration_points())

    @app.get("/stats/sop_gaps")
    def stats_sop_gaps() -> list[dict]:
        mod = _sop_module()
        if mod is None:
            return []
        for name in ("sop_gaps", "gaps"):
            fn = getattr(mod, name, None)
            if callable(fn):
                try:
                    return list(fn())
                except TypeError:
                    continue
        return []

    @app.get("/sop/search")
    def sop_search(q: str = "", lang: str = "") -> dict:
        mod = _sop_module()
        if mod is not None:
            fn = getattr(mod, "search", None)
            if callable(fn):
                try:
                    return {"results": list(fn(q, lang))}
                except TypeError:
                    return {"results": list(fn(q))}
        return {"results": []}

    @app.get("/voice/{reason_code}/{lang}")
    def voice(reason_code: str, lang: str):
        wav = Path(cfg.voicebank_dir) / lang / f"{reason_code}.wav"
        if not wav.is_file():
            raise HTTPException(status_code=404, detail="no voice message")
        return FileResponse(str(wav), media_type="audio/wav")

    @app.get("/pico/status")
    def pico_status() -> dict:
        try:
            from jadal import pico_link
        except Exception:
            return {"connected": False, "reason": "jadal.pico_link not installed"}
        try:
            link = pico_link.PicoLink(cfg.pico_port, cfg.pico_baud, fake=True)
            if hasattr(link, "status"):
                return dict(link.status())
            return {"connected": bool(getattr(link, "connected", False))}
        except Exception as exc:
            return {"connected": False, "reason": f"pico_link unavailable: {exc}"}

    # -- websocket ---------------------------------------------------------
    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        svc.start(asyncio.get_running_loop())
        svc.register_ws(websocket, asyncio.get_running_loop())
        try:
            await websocket.send_json({"type": "hello",
                                       "recent": len(svc.recent_events())})
            for event in svc.recent_events():
                await websocket.send_json(event)
            while True:
                msg = await websocket.receive_text()
                if msg == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            pass
        finally:
            svc.unregister_ws(websocket)

    # -- static dashboard ---------------------------------------------------
    web_dir = Path(cfg.web_dir)
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    else:
        @app.get("/")
        def root() -> dict:
            return {"service": "jadal-qa", "docs": "/docs",
                    "dashboard": "not built yet (web/ missing)"}

    return app


def _sop_module():
    """sop/ is owned by another branch; degrade to empty results without it."""
    try:
        from sop import sop_index  # type: ignore
        return sop_index
    except Exception:
        return None


app = create_app()
