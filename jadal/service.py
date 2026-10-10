"""FastAPI service for Jadal QA (GitHub issue #3).

Runs the pipeline from ``jadal/pipeline.py`` behind the endpoints in
docs/SPEC.md, persists state in SQLite (``data_runtime/jadal.db`` by default),
closes a batch every ``batch_size`` segments or after ``batch_max_age_s`` seconds
per site, watches ``data_runtime/bucket/`` for docked clips, pushes batch events
on ``/ws`` and serves the ``web/`` dashboard at ``/``.

Run with::

    /home/parshu/envs/jadal/bin/python -m uvicorn jadal.service:app

Env knobs (all optional; ``jadal/config.py`` is read-only so the time trigger
lives here): ``JADAL_BATCH_MAX_AGE_S`` (30.0 s), ``JADAL_WATCH_BUCKET`` (1),
``JADAL_WATCH_SITE`` (site-sim), ``JADAL_WATCH_WORKER`` (w-sim),
``JADAL_WATCH_POLL_S`` (0.25 s), ``JADAL_USE_STUBS`` (1 = allow pipeline stubs).
No network calls happen at import time; the SQLite file is created lazily.

NOT DONE:
- L0/L1/segmenter are stubs (hash-derived fakes, whole clip = one segment)
  until the real modules merge; set JADAL_USE_STUBS=1 to run on stubs.
- /stats/risk_coverage needs calibration points (store rows, else
  eval/results/calibration.json); with neither it returns [] -- "no measured
  data", not a measurement of anything.
- /sop/search returns empty payloads until sop/ merges; /stats/sop_gaps
  counts real stored-verdict reasons until then.
- /pico/status connects on startup when a port is configured; without hardware
  it reports connected=false.
"""
from __future__ import annotations

import asyncio
import logging
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
from jadal.config import LANGS, REASON_CODES, Config, DEFAULT

log = logging.getLogger("jadal.service")


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
        self._store: store_mod.Store | None = None
        self.batch_max_age_s = batch_max_age_s
        self.watch_bucket = watch_bucket
        self.watch_site = watch_site
        self.watch_worker = watch_worker
        self.watch_poll_s = watch_poll_s
        self._lock = threading.RLock()
        self._pending: dict[str, list[dict]] = {}   # site -> [{"segment", "ts"}]
        self._ingested: set[tuple[str, str]] = set()  # (real path, sha256)
        self._errors: dict[str, str] = {}           # real path -> last error
        self._watch_backoff: dict[str, dict] = {}   # clip name -> {"attempts", "next_ts"}
        self._stability: dict[str, tuple[int, int]] = {}  # name -> (size, mtime_ns)
        self._done_files: set[str] = set()
        self._ws_clients: set[WebSocket] = set()
        self._recent: deque[dict] = deque(maxlen=50)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = threading.Event()
        self._threads_started = False
        self._pico = None
        self.started_utc = datetime.now(timezone.utc)

    @property
    def store(self) -> store_mod.Store:
        if self._store is None:
            with self._lock:
                if self._store is None:
                    self._store = store_mod.Store(self.cfg.db_path)
        return self._store

    # -- ingest ------------------------------------------------------------
    def _clip_in_bucket(self, clip: Path) -> Path | None:
        """Resolved clip path, or None when it escapes cfg.bucket_dir (C1)."""
        try:
            resolved = Path(os.path.realpath(clip))
            bucket = Path(os.path.realpath(self.cfg.bucket_dir))
        except OSError:
            return None
        try:
            resolved.relative_to(bucket)
        except ValueError:
            return None
        return resolved

    def handle_clip(self, path: str | Path, worker_id: str, site_id: str) -> dict:
        clip = Path(path)
        if not clip.is_file():
            raise FileNotFoundError(str(clip))
        resolved = self._clip_in_bucket(clip)
        if resolved is None:
            raise PermissionError(f"clip outside bucket: {clip}")
        key = (str(resolved), self._sha(resolved))
        with self._lock:
            if key in self._ingested:
                return {"duplicate": True, "segments": [],
                        "pending": self.pending_count(), "batch": None}
        # process first; only a success marks the clip ingested (I3)
        try:
            segments = pipeline.process_clip(resolved, worker_id, site_id, self.cfg)
        except Exception as exc:
            with self._lock:
                self._errors[str(resolved)] = f"{type(exc).__name__}: {exc}"
            log.warning("ingest failed for %s: %s", resolved, exc)
            raise
        with self._lock:
            self._ingested.add(key)
            self._errors.pop(str(resolved), None)
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
    def flush_site(self, site_id: str, *, limit: int | None = None) -> dict | None:
        """Close one batch of at most batch_size segments (I4); rest stays queued.

        A failing batch never drops its segments: they are requeued at the
        FRONT of the site buffer (original order) and the error propagates so
        the caller sees the 500 while the segments stay pending for retry.
        """
        n = int(self.cfg.batch_size if limit is None else limit)
        with self._lock:
            buf = self._pending.get(site_id) or []
            if not buf:
                return None
            take, rest = buf[:n], buf[n:]
            segments = [entry["segment"] for entry in take]
            self._pending[site_id] = rest
        try:
            batch, receipts = pipeline.run_batch(segments, self.cfg)
        except Exception:
            with self._lock:
                buf = self._pending.get(site_id) or []
                self._pending[site_id] = (
                    [{"segment": s, "ts": time.monotonic()} for s in segments]
                    + buf)
            raise
        rows = [dict(seg, created_utc=batch["created_utc"]) for seg in segments]
        self.store.save_batch(batch, receipts, rows)
        self._emit_batch(batch, receipts)
        # a clip that yielded many segments may leave a full batch behind
        with self._lock:
            more = len(self._pending.get(site_id) or []) >= int(self.cfg.batch_size)
        if more:
            self.flush_site(site_id)
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
        self._connect_pico()
        with self._lock:
            if self._threads_started:
                return
            self._threads_started = True
        threading.Thread(target=self._worker_loop, name="jadal-worker",
                         daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _connect_pico(self) -> None:
        try:
            from jadal import pico_link
        except Exception as exc:
            log.info("pico_link not installed: %s", exc)
            return
        try:
            self._pico = pico_link.PicoLink()
            self._pico.connect(self.cfg.pico_port)
            log.info("pico connected on %s", self.cfg.pico_port)
        except Exception as exc:
            self._pico = None
            log.info("pico not connected (%s); status will report offline",
                     exc)

    def pico_status(self) -> dict:
        if self._pico is None:
            return {"connected": False,
                    "reason": f"no pico on {self.cfg.pico_port}"}
        try:
            return dict(self._pico.status())
        except Exception as exc:
            return {"connected": False, "reason": f"pico error: {exc}"}

    def _worker_loop(self) -> None:
        while not self._stop.wait(self.watch_poll_s):
            try:
                self._tick()
            except Exception as exc:
                log.warning("worker tick failed: %s", exc)

    def _tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            due = [site for site, buf in self._pending.items()
                   if buf and now - buf[0]["ts"] > self.batch_max_age_s]
        for site in due:
            try:
                self.flush_site(site)
            except Exception as exc:
                log.warning("timeout flush for %s failed: %s", site, exc)
        if self.watch_bucket:
            try:
                self._watch_bucket_tick()
            except Exception as exc:
                log.warning("bucket watch failed: %s", exc)

    def _watch_bucket_tick(self) -> None:
        bucket = Path(self.cfg.bucket_dir)
        if not bucket.is_dir():
            return
        now = time.monotonic()
        for clip in sorted(bucket.iterdir()):
            if not clip.is_file() or clip.name.startswith("."):
                continue
            if clip.suffix == ".part" or clip.name in self._done_files:
                continue
            # retry-with-backoff: a bad file waits 2^attempts s (cap 30 s)
            # before the next attempt instead of spinning every poll.
            with self._lock:
                gate = self._watch_backoff.get(clip.name)
            if gate is not None and now < gate["next_ts"]:
                continue
            try:
                stat = clip.stat()
                fingerprint = (stat.st_size, stat.st_mtime_ns)
            except OSError:
                continue
            # ingest only once the same (size, mtime) is seen twice (I2)
            if self._stability.get(clip.name) != fingerprint:
                self._stability[clip.name] = fingerprint
                continue
            del self._stability[clip.name]
            try:
                self.handle_clip(clip, self.watch_worker, self.watch_site)
            except PermissionError:
                self._done_files.add(clip.name)
            except Exception as exc:
                # leave it out of _done_files so a later poll retries (I3),
                # but back off exponentially so a poison file cannot spin.
                with self._lock:
                    attempts = self._watch_backoff.get(clip.name, {}).get("attempts", 0) + 1
                    self._watch_backoff[clip.name] = {
                        "attempts": attempts,
                        "next_ts": time.monotonic() + min(2.0 ** attempts, 30.0),
                    }
                    self._errors[str(Path(os.path.realpath(clip)))] = (
                        f"{type(exc).__name__}: {exc}")
                log.warning("watcher ingest of %s failed (attempt %d): %s",
                            clip.name, attempts, exc)
                continue
            else:
                with self._lock:
                    self._watch_backoff.pop(clip.name, None)
                self._done_files.add(clip.name)


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
        except PermissionError:
            raise HTTPException(status_code=400,
                                detail="clip is outside the bucket directory")
        except RuntimeError as exc:
            # missing pipeline implementation and similar config errors
            log.warning("ingest rejected: %s", exc)
            raise HTTPException(status_code=500, detail="ingest unavailable")
        except Exception as exc:
            log.warning("ingest failed: %s", exc)
            raise HTTPException(status_code=500, detail="ingest failed")

    @app.get("/ingest/errors")
    def ingest_errors() -> dict:
        """Last ingest error per clip path (failed watcher retries land here)."""
        with svc._lock:
            return {"errors": dict(svc._errors)}

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
        points = svc.store.calibration_points()
        if points:
            return pipeline.risk_coverage(points)
        # real eval results when the eval module has produced them; [] is the
        # honest "no measured data" state -- never fabricated points.
        return pipeline.risk_coverage_from_eval()

    @app.get("/stats/sop_gaps")
    def stats_sop_gaps() -> list[dict]:
        mod = _sop_module()
        if mod is None:
            return pipeline.sop_gaps_from_verdicts(svc.store.list_batches())
        for name in ("sop_gaps", "gaps"):
            fn = getattr(mod, name, None)
            if callable(fn):
                try:
                    return list(fn())
                except TypeError:
                    continue
        return pipeline.sop_gaps_from_verdicts(svc.store.list_batches())

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
        # allowlist first; never build a path from unvalidated input (C2)
        if lang not in LANGS or reason_code not in REASON_CODES:
            raise HTTPException(status_code=404, detail="no voice message")
        wav = Path(cfg.voicebank_dir) / lang / f"{reason_code}.wav"
        if not wav.is_file():
            raise HTTPException(status_code=404, detail="no voice message")
        return FileResponse(str(wav), media_type="audio/wav")

    @app.get("/pico/status")
    def pico_status() -> dict:
        return svc.pico_status()

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
