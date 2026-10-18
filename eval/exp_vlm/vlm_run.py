"""Shared run helpers for GRID.V01 / GRID.V03 (sub-stream V-B).

Three jobs:
  * build the sheet for a named arm WITHOUT re-deriving anything (the arm table is
    fixed by `experiments/GRID.VLM/PREREG.json`);
  * a hard call-budget guard: every call goes through `guard()`, which reads the
    per-run JSONL ledger and refuses to exceed the pre-registered cap;
  * ledger-based RESUME, so a crash or a timeout never re-spends a call: a
    (window, arm) pair already recorded with ok=True is skipped.

Nothing here prints, logs or stores the API key: the key is read in-process by
`portal/muse_video.py::_post` and never surfaces.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "tools", "dsh")):
    if p not in sys.path:
        sys.path.insert(0, p)

import vlm_mosaic as VM  # noqa: E402
import vlm_sheet as S  # noqa: E402
from eval.exp_vlm import vlm_common as V  # noqa: E402

PREREG = json.load(open(os.path.join(ROOT, "experiments", "GRID.VLM", "PREREG.json")))

#: V03 arms exactly as pre-registered.  ``prompt`` defaults to the plain quality
#: prompt; only arm (e) changes it.  ``labels``/``shuffle_seed``/``n_tiles``/``cols``
#: are the sheet switches.
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
