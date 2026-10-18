"""Shared run helpers for the GRID.V track (sub-streams V-A and V-B merged).

V-B side: the pre-registered V03 arm table, sheet building for a named arm, a hard
call-budget guard wired to the per-run JSONL ledger, and ledger-based RESUME so a
crash or a timeout never re-spends a call.
V-A side: PNG sheet caching (one build per window/config, reused byte-identically
across repeat runs) and a small parallel runner.

The two sub-streams independently created this module under the same path; the merge
kept BOTH APIs rather than dropping either, because run_v01.py/run_v03.py need the
guard/resume half and grid_v02.py/grid_v04.py/grid_v06.py need the cache half.

Nothing here prints, logs or stores the API key: it is read in-process by
`portal/muse_video.py::_post` and never surfaces.  PNGs are dataset frames and live
under /tmp/opencode only; they are never committed.
"""
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "tools", "dsh"), os.path.join(ROOT, "eval", "exp_vlm")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vlm_common as V  # noqa: E402
import vlm_mosaic as M  # noqa: E402
import vlm_mosaic as VM  # noqa: E402
import vlm_sheet as S  # noqa: E402

PREREG = json.load(open(os.path.join(ROOT, "experiments", "GRID.VLM", "PREREG.json")))

V03_ARMS = {
    "a_no_timestamps":       {"labels": False},
    "b_shuffled_order":      {"shuffle_seed": 12345},
    "c_single_middle_frame": {"n_tiles": 1, "cols": 1},
    "d6_tiles":              {"n_tiles": 6, "cols": 3},
    "d24_tiles":             {"n_tiles": 24, "cols": 6},
    "e_json_only_prompt":    {"prompt": "quality_json_only"},
}
V03_REFERENCE = "e_json_only_prompt"


# ----------------------------------------------------------------- budget ----
#: the sub-stream's own total ceiling: V01's plan (20) + V03's cap (122).  The
#: PREREG's caps are PER EXPERIMENT, and this ledger is shared between the two, so
#: a per-experiment guard must filter by arm -- counting the whole ledger against
#: V03's 122 would stop 20 calls early (that mistake happened once; see
#: docs/reports/grid_vlm_b.md).
SUBSTREAM_TOTAL_CAP = 142


def guard(ledger: str, cap: int, will_add: int = 1, arms=None):
    """Raise before a call that would exceed a pre-registered cap.

    ``arms`` : when given, only ledger records whose ``arm`` is in this set count
    against ``cap`` (per-experiment accounting on a shared ledger).  The
    sub-stream total is checked as well, so no combination can exceed 142.
    """
    rows = [r for r in ledger_rows(ledger) if not r.get("text_only")]
    if arms is not None:
        rows = [r for r in rows if r.get("arm") in arms]
    n_exp = len(rows)
    n_all = sum(1 for r in ledger_rows(ledger) if not r.get("text_only"))
    if n_exp + will_add > cap:
        raise RuntimeError(
            f"BUDGET STOP: this experiment already has {n_exp} calls and its cap is {cap}; "
            f"{will_add} more would exceed it. Reporting instead of spending.")
    if n_all + will_add > SUBSTREAM_TOTAL_CAP:
        raise RuntimeError(
            f"BUDGET STOP (sub-stream total): the ledger has {n_all} image calls and the "
            f"sub-stream ceiling is {SUBSTREAM_TOTAL_CAP}; {will_add} more would exceed it.")
    return {"experiment": n_exp, "substream_total": n_all}


def done_pairs(ledger: str):
    """{(window, arm)} already recorded OK -- used to resume without re-spending."""
    out = set()
    if not os.path.exists(ledger):
        return out
    with open(ledger) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("ok") and e.get("window") is not None:
                out.add((e["window"], e.get("arm")))
    return out


def ledger_rows(ledger: str):
    if not os.path.exists(ledger):
        return []
    rows = []
    with open(ledger) as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


# ------------------------------------------------------------------ sheet ----
def sheet_for_window(win: dict, *, n_tiles=12, cols=4, labels=True, shuffle_seed=None,
                     sheet_dir=None, tag=""):
    """Build (or reuse) one sheet.  Returns (png_bytes, doc)."""
    sheet_dir = sheet_dir or V.SHEET_DIR
    os.makedirs(sheet_dir, exist_ok=True)
    stem = f"{tag}{win['window']}_n{n_tiles}_c{cols}_l{int(labels)}_s{shuffle_seed}"
    png_path = os.path.join(sheet_dir, stem + ".png")
    side_path = os.path.join(sheet_dir, stem + ".json")
    if os.path.exists(png_path) and os.path.exists(side_path):
        with open(png_path, "rb") as f:
            png = f.read()
        doc = json.load(open(side_path))
        if doc.get("png_sha256") == hashlib.sha256(png).hexdigest():
            return png, doc
    png, doc = V.sheet_for(win["path"], float(win.get("start_s") or 0.0), 10.0,
                           n_tiles=n_tiles, cols=cols, labels=labels,
                           shuffle_seed=shuffle_seed, cache_key=stem)
    with open(png_path, "wb") as f:
        f.write(png)
    with open(side_path, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    return png, doc


def grey_sheet(n_tiles=12, cols=4, sheet_dir=None):
    """Flat mid-grey sheet with NO content and NO labels (the negative control)."""
    sheet_dir = sheet_dir or V.SHEET_DIR
    os.makedirs(sheet_dir, exist_ok=True)
    p = os.path.join(sheet_dir, "null_grey_n%d_c%d.png" % (n_tiles, cols))
    if os.path.exists(p):
        with open(p, "rb") as f:
            return f.read()
    frames = [np.full((240, 426, 3), 128, np.uint8) for _ in range(n_tiles)]
    img, _meta = S.build_sheet(frames, S.cell_times(n_tiles, 10.0), cols=cols,
                               labels=False)
    png = S.png_bytes(img)
    with open(p, "wb") as f:
        f.write(png)
    return png


# ---------------------------------------------------------------- metrics ----
def fmt_ts(t: float) -> str:
    """The sheet's own timestamp format, MM:SS.s."""
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def norm_ts(s: str) -> str:
    """Normalise a reply timestamp to MM:SS.s; '' when unparseable."""
    s = str(s).strip().rstrip("s").rstrip("S").strip()
    parts = s.split(":")
    try:
        if len(parts) == 2:
            mm, ss = parts
        elif len(parts) == 3:
            hh, mm, ss = parts
            mm = str(int(mm) + 60 * int(hh))
        else:
            return ""
        return f"{int(mm):02d}:{float(ss):04.1f}"
    except (ValueError, TypeError):
        return ""


def expected_tiles(sidecar_tiles) -> dict:
    """{1-based tile number: 'MM:SS.s'} -- ground truth from OUR builder's sidecar."""
    return {int(t["index"]) + 1: fmt_ts(float(t["t_s"])) for t in sidecar_tiles}


def read_tiles_metrics(reply_tiles: dict, sidecar_tiles) -> dict:
    """Exact-match and coverage of a read_tiles reply against the sidecar."""
    exp = expected_tiles(sidecar_tiles)
    n_expected = len(exp)
    got = {}
    for num, (ts, _desc) in reply_tiles.items():
        got[int(num)] = norm_ts(ts)
    hits = {n: (n in got and got[n] == exp[n]) for n in exp}
    covered = {n: (n in got) for n in exp}
    n_lines = len(reply_tiles)
    n_exact = sum(1 for n in exp if hits[n])
    return {
        "n_expected_tiles": n_expected,
        "n_returned_lines": n_lines,
        "n_tiles_covered": sum(1 for n in exp if covered[n]),
        "tile_coverage_rate": round(sum(1 for n in exp if covered[n]) / n_expected, 4),
        "n_exact_number_and_timestamp": n_exact,
        "exact_match_rate_per_sheet_tile": round(n_exact / n_expected, 4),
        "exact_match_rate_per_returned_line": (round(n_exact / n_lines, 4) if n_lines else None),
        "n_number_ok_timestamp_wrong": sum(1 for n in exp if covered[n] and not hits[n]),
        "n_out_of_range_lines": sum(1 for n in got if n not in exp),
        "per_tile": [{"tile": n, "expected_ts": exp[n], "got_ts": got.get(n),
                      "exact": bool(hits[n])} for n in sorted(exp)],
        "reply_numbers": sorted(got),
    }


def quality_scores(rec: dict):
    """(usable_score, progress_score) from one quality reply; (None, None) if unparsed.

    progress_score: advancing=1.0, cannot_tell=0.5, no_relevant_work=0.0.
    """
    q = rec.get("quality")
    if not q:
        return None, None
    usable, _code, prog, _expl = q
    return (1.0 if usable else 0.0,
            {"advancing": 1.0, "cannot_tell": 0.5, "no_relevant_work": 0.0}.get(prog))


def usable_share(records) -> float:
    us = [r for r in records if r.get("quality") is not None]
    return round(sum(1 for r in us if r["quality"][0]) / len(us), 4) if us else float("nan")


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
