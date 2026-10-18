#!/usr/bin/env python3
"""Independent re-derivation of the GRID.V01 reading numbers.

Deliberately imports NOTHING from `eval/exp_vlm/vlm_run.py`: it re-parses the raw
replies from the ledger with its own regex, re-reads the sheet sidecars itself, and
recomputes the exact-match and coverage rates from scratch.  If the two disagree,
the disagreement is the finding.

Run: JADAL_VLM_LEDGER=<ledger> python3 eval/exp_vlm/verify_v01.py
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEDGER = os.environ.get("JADAL_VLM_LEDGER", "/tmp/opencode/dsh/vlm_ledger_b.jsonl")
SHEETS = os.environ.get("JADAL_SHEET_DIR", "/tmp/opencode/dsh/sheets")
OUT = os.path.join(ROOT, "eval", "results", "exp_vlm", "GRID.V01_verify.json")

# my own line pattern, written from the reply text rather than copied from the caller
LINE = re.compile(r"tile\s*#?(\d+)\s*@\s*(\d{1,2}:\d{2}(?:\.\d)?)\s*s?\s*::", re.I)


def own_fmt(t):
    m = int(t // 60)
    s = t % 60
    return "%02d:%04.1f" % (m, s)


def own_norm(x):
    parts = x.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        return "%02d:%04.1f" % (int(parts[0]), float(parts[1]))
    except ValueError:
        return None


def main():
    rows = [json.loads(l) for l in open(LEDGER) if l.strip()]
    v01 = [r for r in rows if r.get("arm") == "v01_read_tiles"]
    tot_tiles = tot_lines = tot_exact = tot_cov = 0
    per = []
    for r in v01:
        wid = r["window"]
        side_path = os.path.join(SHEETS, "v01_%s_n12_c4_l1_sNone.json" % wid)
        if not os.path.exists(side_path):
            per.append({"window": wid, "error": "sidecar missing, not counted"})
            continue
        side = json.load(open(side_path))
        if side.get("png_sha256") != r.get("sheet_sha"):
            per.append({"window": wid, "error": "PNG sha mismatch: the reply is not for this sheet"})
            continue
        truth = {}
        for t in side["tiles"]:
            truth[int(t["index"]) + 1] = own_fmt(float(t["t_s"]))
        got = {}
        for m in LINE.finditer(r.get("raw") or ""):
            got[int(m.group(1))] = own_norm(m.group(2))
        n_tiles = len(truth)
        cov = sum(1 for n in truth if n in got)
        exact = sum(1 for n in truth if n in got and got[n] == truth[n])
        tot_tiles += n_tiles
        tot_lines += len(got)
        tot_cov += cov
        tot_exact += exact
        per.append({"window": wid, "n_tiles": n_tiles, "lines": len(got),
                    "covered": cov, "exact": exact,
                    "coverage": round(cov / n_tiles, 4),
                    "exact_rate": round(exact / n_tiles, 4)})

    res = {
        "id": "GRID.V01_verify",
        "method": "independent re-parse of the raw ledger replies with a locally written "
                  "regex, against independently re-read sidecars; imports nothing from "
                  "eval/exp_vlm/vlm_run.py",
        "n_windows_counted": sum(1 for p in per if "error" not in p),
        "sheet_tiles": tot_tiles, "returned_lines": tot_lines,
        "tiles_covered": tot_cov, "exact_number_and_timestamp": tot_exact,
        "tile_coverage_rate": round(tot_cov / tot_tiles, 4) if tot_tiles else None,
        "exact_match_rate_per_sheet_tile": round(tot_exact / tot_tiles, 4) if tot_tiles else None,
        "exact_match_rate_per_returned_line": round(tot_exact / tot_lines, 4) if tot_lines else None,
        "per_window": per,
        "ledger": LEDGER,
        "note": "a disagreement with GRID.V01.json is the finding; a match is a reproduction "
                "of the arithmetic, not a validation of the protocol",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("n_windows_counted", "sheet_tiles", "returned_lines",
                                          "tiles_covered", "exact_number_and_timestamp",
                                          "tile_coverage_rate", "exact_match_rate_per_sheet_tile",
                                          "exact_match_rate_per_returned_line")}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
