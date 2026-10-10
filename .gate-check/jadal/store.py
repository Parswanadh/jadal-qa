"""SQLite persistence for the Jadal service.

Tables (all JSON payloads stored as canonical text):
- ``segments``  one row per produced Segment
- ``batches``   one row per closed batch (verdicts embedded, per the SPEC batch schema)
- ``receipts``  one row per receipt, linked to its batch and segment
- ``calibration`` (conf, correct) evaluation points for /stats/risk_coverage

The store is intentionally dumb: no business logic, no rounding, no hashing. All
values are inserted exactly as the pipeline produced them so re-verify can compare
byte-for-byte.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    segment_id   TEXT PRIMARY KEY,
    source_clip  TEXT NOT NULL,
    worker_id    TEXT NOT NULL,
    site_id      TEXT NOT NULL,
    start_s      REAL NOT NULL,
    end_s        REAL NOT NULL,
    path         TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    created_utc  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id       TEXT PRIMARY KEY,
    site_id        TEXT NOT NULL,
    batch_decision TEXT NOT NULL,
    rule           TEXT NOT NULL,
    k              INTEGER NOT NULL,
    n              INTEGER NOT NULL,
    created_utc    TEXT NOT NULL,
    segment_ids    TEXT NOT NULL,
    verdicts       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS receipts (
    receipt_id     TEXT PRIMARY KEY,
    segment_id     TEXT NOT NULL,
    batch_id       TEXT NOT NULL,
    receipt_sha256 TEXT NOT NULL,
    created_utc    TEXT NOT NULL,
    receipt        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calibration (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    conf    REAL NOT NULL,
    correct INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_receipts_batch ON receipts(batch_id);
CREATE INDEX IF NOT EXISTS idx_batches_site ON batches(site_id);
"""


def init_db(db_path: str | Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(SCHEMA)


class Store:
    """Thin data-access wrapper; one connection per call (thread-safe)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        init_db(self.db_path)

    # -- internals ---------------------------------------------------------
    @contextlib.contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _dumps(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))

    # -- writes ------------------------------------------------------------
    def save_batch(self, batch: dict, receipts: list[dict],
                   segments: list[dict]) -> None:
        """Persist a closed batch, its verdicts, its receipts and its segments."""
        with self._conn() as conn:
            for seg in segments:
                conn.execute(
                    """INSERT OR REPLACE INTO segments
                       (segment_id, source_clip, worker_id, site_id, start_s, end_s,
                        path, sha256, created_utc)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (seg["segment_id"], seg["source_clip"], seg["worker_id"],
                     seg["site_id"], float(seg["start_s"]), float(seg["end_s"]),
                     seg["path"], seg["sha256"],
                     seg.get("created_utc", batch["created_utc"])),
                )
            conn.execute(
                """INSERT OR REPLACE INTO batches
                   (batch_id, site_id, batch_decision, rule, k, n, created_utc,
                    segment_ids, verdicts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (batch["batch_id"], batch["site_id"], batch["batch_decision"],
                 batch["rule"], int(batch["k"]), int(batch["n"]),
                 batch["created_utc"], self._dumps(batch["segment_ids"]),
                 self._dumps(batch["verdicts"])),
            )
            for receipt in receipts:
                conn.execute(
                    """INSERT OR REPLACE INTO receipts
                       (receipt_id, segment_id, batch_id, receipt_sha256, created_utc, receipt)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (receipt["receipt_id"], receipt["verdict"]["segment_id"],
                     batch["batch_id"], receipt["receipt_sha256"],
                     receipt["created_utc"], self._dumps(receipt)),
                )

    def add_calibration(self, points: list[tuple[float, int]]) -> None:
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO calibration (conf, correct) VALUES (?, ?)", points)

    # -- reads -------------------------------------------------------------
    def _row_to_batch(self, row: sqlite3.Row) -> dict:
        return {
            "batch_id": row["batch_id"],
            "site_id": row["site_id"],
            "segment_ids": json.loads(row["segment_ids"]),
            "verdicts": json.loads(row["verdicts"]),
            "batch_decision": row["batch_decision"],
            "rule": row["rule"],
            "k": row["k"],
            "n": row["n"],
            "created_utc": row["created_utc"],
        }

    def get_batch(self, batch_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM batches WHERE batch_id = ?",
                               (batch_id,)).fetchone()
        return self._row_to_batch(row) if row else None

    def list_batches(self, site_id: str | None = None) -> list[dict]:
        sql = "SELECT * FROM batches"
        params: tuple[Any, ...] = ()
        if site_id:
            sql += " WHERE site_id = ?"
            params = (site_id,)
        sql += " ORDER BY created_utc ASC, batch_id ASC"
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_batch(r) for r in rows]

    def get_receipt(self, receipt_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT receipt FROM receipts WHERE receipt_id = ?",
                               (receipt_id,)).fetchone()
        return json.loads(row["receipt"]) if row else None

    def get_receipts_for_batch(self, batch_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT receipt FROM receipts WHERE batch_id = ? ORDER BY receipt_id",
                (batch_id,)).fetchall()
        return [json.loads(r["receipt"]) for r in rows]

    def get_segment(self, segment_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM segments WHERE segment_id = ?",
                               (segment_id,)).fetchone()
        if not row:
            return None
        return {
            "segment_id": row["segment_id"],
            "source_clip": row["source_clip"],
            "worker_id": row["worker_id"],
            "site_id": row["site_id"],
            "start_s": row["start_s"],
            "end_s": row["end_s"],
            "path": row["path"],
            "sha256": row["sha256"],
            "created_utc": row["created_utc"],
        }

    def calibration_points(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT conf, correct FROM calibration ORDER BY id").fetchall()
        return [{"conf": r["conf"], "correct": r["correct"]} for r in rows]
