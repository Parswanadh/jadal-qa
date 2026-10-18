"""Shared runner helpers for GRID.V02/V04 (sheet caching + parallel calls).

Sheets are built ONCE per (window, config) and cached as PNG bytes under
/tmp/opencode/dsh/sheets, so:
  * V04's repeat runs reuse the EXACT same PNG bytes (the only variation is the
    model), and
  * a re-run after a crash does not rebuild or re-call anything already done.

PNGs are dataset frames: they live under /tmp/opencode only and are never
committed.  Dataset frames and videos must not reach git.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "dsh"))
sys.path.insert(0, os.path.join(ROOT, "eval", "exp_vlm"))

import vlm_common as V  # noqa: E402
import vlm_mosaic as M  # noqa: E402

SHEET_DIR = os.environ.get("JADAL_SHEET_DIR", "/tmp/opencode/dsh/sheets")
CALLS_DIR = os.path.join(ROOT, "eval", "results", "exp_vlm")
_LOCK = threading.Lock()

#: The `quality` prompt asks for `tile <n> @ <t> :: <text>`, but the model in
#: practice answers `#<n>  <MM:SS.s>: <text>` (it echoes the sheet label).  The
#: frozen `vlm_mosaic.parse_tile_lines` only matches the requested form, so it
#: returns {} on real replies.  This tolerant parser accepts both and is used for
#: the per-tile evidence recorded by this stream.  Raised to the lead rather than
#: changing the frozen tool, which V01/V03 also share.
_TILE_RE = re.compile(
    r"^\s*(?:tile\s*#?|#)\s*(\d+)\s*(?:@\s*)?"
    r"([0-9]{1,2}:[0-9]{2}(?:\.[0-9]+)?\s*s?)?\s*(?:::|:|-|—)?\s*(.+?)\s*$", re.I)


def parse_tile_lines_tolerant(text: str):
    """{tile_number: (timestamp, description)} from either tile-line format.

    First occurrence of a tile number wins, so a stray numbered line cannot
    overwrite real evidence.
    """
    out = {}
    if not text:
        return out
    for line in text.splitlines():
        m = _TILE_RE.match(line)
        if not m:
            continue
        n = int(m.group(1))
        desc = (m.group(3) or "").strip()
        if n >= 1 and desc and not desc.startswith("{"):
            out.setdefault(n, ((m.group(2) or "").strip(), desc))
    return out


def cache_paths(tag: str, window: str):
    os.makedirs(SHEET_DIR, exist_ok=True)
    base = os.path.join(SHEET_DIR, f"{tag}_{window}")
    return base + ".png", base + ".json"


def sheet_cached(tag: str, row: dict, **cfg):
    """(png_bytes, sheet_doc) for one window, building and caching on first use."""
    png_p, doc_p = cache_paths(tag, row["window"])
    if os.path.exists(png_p) and os.path.exists(doc_p):
        with open(png_p, "rb") as f:
            png = f.read()
        doc = json.load(open(doc_p))
        if doc.get("png_sha256") == hashlib.sha256(png).hexdigest():
            return png, doc
    png, doc = V.sheet_for(row["path"], float(row.get("start_s", 0.0)), 10.0, **cfg)
    doc = dict(doc)
    doc["window"] = row["window"]
    doc["tag"] = tag
    with open(png_p, "wb") as f:
        f.write(png)
    with open(doc_p, "w") as f:
        json.dump(doc, f, sort_keys=True)
    return png, doc


def call_one(row: dict, png: bytes, doc: dict, prompt_name: str, *, run: int,
             tag: str, ledger: str, retries: int = 1):
    """One model call with a bounded retry.  Returns the (last) call record."""
    extra = {"window": row["window"], "tag": tag, "run": run,
             "png_sha256_cached": doc["png_sha256"]}
    rec = M.ask(png, prompt_name, ledger=ledger, extra=extra)
    attempts = 1
    while not rec.get("ok") and attempts <= retries:
        time.sleep(2.0 * attempts)
        rec = M.ask(png, prompt_name, ledger=ledger,
                    extra={**extra, "retry_of_attempt": attempts})
        attempts += 1
    rec["attempts"] = attempts
    rec["window"] = row["window"]
    rec["tag"] = tag
    rec["run"] = run
    return rec


def parallel_run(items, fn, workers: int = 3, on_done=None):
    """Run fn(item) over items with a small thread pool; results keep input order.

    Deliberately small: the machine is shared and the provider may rate-limit.
    """
    out = [None] * len(items)

    def wrapped(i_item):
        i, item = i_item
        res = fn(item)
        if on_done:
            with _LOCK:
                on_done(i, len(items), res)
        return i, res

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, res in ex.map(wrapped, list(enumerate(items))):
            out[i] = res
    return out
