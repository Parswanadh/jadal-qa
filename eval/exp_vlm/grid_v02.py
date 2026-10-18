#!/usr/bin/env python3
"""GRID.V02 — the quality verdict of a frozen vision model on 160 factory sheets.

One 4x3 contact sheet (12 tiles, 426x240, `#N  MM:SS.s`) per pre-registered
OOD-160 window, one frozen `quality` prompt call per window, no training.

Protocol: the window list, the prompt sha256 and the decision rules are frozen in
`experiments/GRID.VLM/PREREG.json`.  The model is PROBABILISTIC: nothing here is
deterministic and no number is an accuracy claim.  OOD-160 is n=160, n_bad=22, of
which 19 are NO_ACTIVE_WORK, 47 factories, ONE human labeller — every number below
carries that context.

Resume: a call already present in the ledger for the same (window, run, png sha,
prompt sha) is REUSED, never repeated.  The exact number of calls made is read from
the ledger and written into `cost`.

Run: JADAL_VLM_LEDGER=<ledger> python3 eval/exp_vlm/grid_v02.py --workers 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "eval", "exp_vlm"))
sys.path.insert(0, os.path.join(ROOT, "tools", "dsh"))

import common as C  # noqa: E402
import vlm_common as V  # noqa: E402
import vlm_mosaic as M  # noqa: E402
import vlm_run as R  # noqa: E402

PREREG = os.path.join(ROOT, "experiments", "GRID.VLM", "PREREG.json")
OUT = os.path.join(ROOT, "eval", "results", "exp_vlm")
SEED, N_BOOT = 0, 2000
TAG = "v02"
PROMPT = "quality"
CONTEXT = ("OOD-160: n=160 (138 good / 22 bad), 19 of the 22 bad are NO_ACTIVE_WORK, "
           "47 factories, ONE human labeller. The model is probabilistic.")


def load_prereg():
    return json.load(open(PREREG))


def local_ledger():
    return os.environ.get("JADAL_VLM_LEDGER", M.DEFAULT_LEDGER)


def ledger_done(ledger: str, prompt_name: str):
    """Completed calls keyed by (window, run, png_sha256, prompt_sha256).

    ``run`` defaults to 1 when a record has no run field (an ad-hoc or smoke call
    with the same frozen prompt and the same sheet bytes is a legitimate run 1 and
    must be reused rather than paid for twice).
    """
    done = {}
    if not os.path.exists(ledger):
        return done
    with open(ledger) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if not e.get("ok") or e.get("prompt_name") != prompt_name:
                continue
            w = e.get("extra", {}).get("window") or e.get("window")
            run = e.get("extra", {}).get("run", e.get("run"))
            if w is None:
                continue
            done[(w, 1 if run is None else run)] = e
    return done


def parse_row(rec):
    """Normalise one call record into the fields the metrics need."""
    p = rec.get("parsed")
    usable = None
    if isinstance(p, dict) and isinstance(p.get("usable"), bool):
        usable = bool(p["usable"])
    else:
        q = M.parse_quality(rec.get("raw", ""))
        if q is not None:
            usable, code, prog, expl = q
            p = {"usable": usable, "reason_code": code, "task_progress": prog,
                 "explanation": expl}
    code = (p or {}).get("reason_code", "UNPARSED")
    prog = (p or {}).get("task_progress", "cannot_tell")
    return {"ok": bool(rec.get("ok")), "usable": usable, "reason_code": code,
            "task_progress": prog, "explanation": (p or {}).get("explanation", ""),
            "seconds": rec.get("seconds"), "png_sha256": rec.get("png_sha256"),
            "raw": rec.get("raw", ""), "ts": rec.get("ts"),
            # tolerant parser: the model answers `#N MM:SS.s: text`, not the
            # requested `tile N @ t :: text`, so the frozen parser returns {}
            "tile_lines": R.parse_tile_lines_tolerant(rec.get("raw", "")),
            "n_tile_lines": len(R.parse_tile_lines_tolerant(rec.get("raw", ""))),
            "tile_lines_frozen_parser": len(M.parse_tile_lines(rec.get("raw", "")))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="smoke only; not the real run")
    ap.add_argument("--tag", default=TAG)
    args = ap.parse_args()

    pre = load_prereg()
    ids = pre["windows"]["V02_160"]
    if args.limit:
        ids = ids[:args.limit]
    ledger = local_ledger()
    rows = {r["window"]: r for r in V.ood_windows()}
    missing = [i for i in ids if i not in rows or not rows[i]["path"]]
    if missing:
        raise SystemExit(f"windows without a clip: {missing[:5]}")

    done = ledger_done(ledger, PROMPT)
    started = time.time()
    print(f"V02: {len(ids)} windows; {len(done)} successful calls already in the ledger",
          flush=True)

    results = {}

    def work(wid):
        row = rows[wid]
        png, doc = R.sheet_cached(args.tag, row)
        reuse = done.get((wid, 1))
        if reuse and reuse.get("png_sha256") == doc["png_sha256"]:
            rec = dict(reuse)
            rec["window"] = wid
            rec["reused"] = True
        else:
            rec = R.call_one(row, png, doc, PROMPT, run=1, tag=args.tag, ledger=ledger)
            rec["reused"] = False
        rec["sheet_sha256"] = doc["png_sha256"]
        return parse_row(rec)

    def on_done(i, n, res):
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  {i+1}/{n}", flush=True)

    out = R.parallel_run(ids, work, workers=args.workers, on_done=on_done)
    for wid, parsed in zip(ids, out):
        results[wid] = parsed
    wall = time.time() - started

    counts = M.count_calls(ledger)
    secs = [r["seconds"] for r in results.values() if r.get("seconds")]

    # ---------------------------------------------------------------- metrics --
    y = np.array([rows[w]["y"] for w in ids])
    fac = np.array([rows[w]["factory"] for w in ids])
    reason = np.array([rows[w]["reason"] for w in ids], dtype=object)
    p26 = np.array([rows[w]["p26"] for w in ids], float)

    usable = np.array([results[w]["usable"] for w in ids], dtype=object)
    parsed_mask = np.array([u is not None for u in usable])
    u = np.array([bool(x) if x is not None else False for x in usable])
    advancing = np.array([results[w]["task_progress"] == "advancing" for w in ids], float)
    score_u = u.astype(float)
    score_adv = advancing
    score_both = np.where(u, 0.5, 0.0) + np.where(advancing.astype(bool) & u, 0.5, 0.0)

    def auc_block(mask, label):
        yy, ss = y[mask], score_u[mask]
        if len(set(yy)) < 2:
            return {"label": label, "n": int(mask.sum()), "note": "one class only"}
        a = C.fast_auc(yy, ss)
        lo, hi, nb = C.group_boot_ci(yy, ss, fac[mask], n_boot=N_BOOT, seed=SEED)
        wlo, whi, _ = C.window_boot_ci(yy, ss, n_boot=N_BOOT, seed=SEED)
        return {"label": label, "n": int(mask.sum()), "n_bad": int((yy == 0).sum()),
                "auroc_usable": round(a, 4), "ci95": [round(lo, 3), round(hi, 3)],
                "ci95_type": f"factory-grouped bootstrap ({N_BOOT} draws, seed {SEED})",
                "ci95_window_optimistic": [round(wlo, 3), round(whi, 3)],
                "auroc_advancing": round(C.fast_auc(yy, score_adv[mask]), 4),
                "auroc_both": round(C.fast_auc(yy, score_both[mask]), 4)}

    allm = np.ones(len(ids), bool)
    blocks = {"all_160": auc_block(allm, "all 160 windows")}
    naw = reason == "NO_ACTIVE_WORK"
    if naw.sum() >= 10:
        sel = naw | (y == 1)
        blocks["no_active_work_vs_good"] = auc_block(sel, "19 NO_ACTIVE_WORK bad vs all good")
    hn = reason == "HANDS_NOT_VISIBLE"
    if hn.sum() >= 2:
        blocks["hands_not_visible_vs_good"] = {
            "note": "n_bad too small for a CI to mean anything; descriptive only",
            "n_bad": int(hn.sum()),
            "auroc_usable": round(C.fast_auc(y[hn | (y == 1)], score_u[hn | (y == 1)]), 4)}

    good, bad = y == 1, y == 0
    # Balanced accuracy with GOOD as the positive class:
    #   sensitivity = good accepted = 1 - FRR_good ; specificity = bad rejected = recall_bad.
    # The first version of this file used (~u[good]).mean() for the sensitivity term and
    # printed 0.3208 instead of 0.7701; the unit test now pins the convention.
    sens = float(u[good].mean())
    spec = float((~u[bad]).mean())
    natural = {
        "accept_definition": "usable == true",
        "share_all_answers_usable_true": round(float(u.mean()), 4),
        "share_parsed": round(float(parsed_mask.mean()), 4),
        "recall_of_bad_rejected": round(spec, 4),
        "n_bad": int(bad.sum()),
        "false_reject_rate_on_good": round(float((~u[good]).mean()), 4),
        "n_good": int(good.sum()),
        "sensitivity_good_accepted": round(sens, 4),
        "balanced_accuracy": round(0.5 * (sens + spec), 4),
        "accepted_bad": int(u[bad].sum()), "rejected_good": int((~u[good]).sum()),
        "ba_convention": "balanced accuracy = 0.5 * (good accepted + bad rejected), "
                         "positive class = good",
    }
    paired = C.paired_group_boot_diff(y, score_u, p26, fac, n_boot=N_BOOT, seed=SEED)
    paired["auroc_vlm_usable"] = round(C.fast_auc(y, score_u), 4)
    paired["auroc_shipped_26"] = round(C.fast_auc(y, p26), 4)
    paired["shipped_note"] = ("shipped head rejects 53 of 138 good windows at its 0.25 band; "
                              "its AUROC on these windows is 0.5878 [0.450, 0.728]")

    # confusion: human reason (rows) x model reason_code (cols)
    hr = [r if r else "GOOD" for r in reason]
    mr = [results[w]["reason_code"] for w in ids]
    conf = {}
    for h, m in zip(hr, mr):
        conf.setdefault(h, Counter())[m] += 1
    conf = {k: dict(v) for k, v in sorted(conf.items())}

    payload = {
        "id": "GRID.V02", "n": int(len(ids)), "n_bad": int(bad.sum()),
        "label_source": C.load_ood160()["label_source"],
        "protocol": ("one frozen-prompt `quality` call per pre-registered window on a 4x3 "
                     "12-tile 426x240 contact sheet built by tools/dsh/vlm_sheet.py "
                     f"({pre['sheet']['sheet_version']}); window list, prompt sha256 and "
                     "decision rules frozen in experiments/GRID.VLM/PREREG.json before the run; "
                     "no training, no fitting, no threshold selection"),
        "model": M.MODEL_LABEL, "model_id": M.MODEL_ID,
        "prompt_name": PROMPT, "prompt_sha256": M.PROMPT_SHA256[PROMPT],
        "context_required_with_every_number": CONTEXT,
        "auroc_blocks": blocks,
        "natural_threshold": natural,
        "paired_vs_shipped_head": paired,
        "confusion_human_reason_x_model_code": conf,
        "tile_line_parsing": {
            "note": "the frozen vlm_mosaic.parse_tile_lines matches only the REQUESTED "
                    "`tile N @ t :: text` form; the model answers in the sheet-label form "
                    "`#N MM:SS.s: text`, so the frozen parser returns 0 lines on every real "
                    "reply. The tolerant parser in eval/exp_vlm/vlm_run.py is used for the "
                    "per-tile evidence recorded here.",
            "replies_with_frozen_parser_lines": int(sum(
                1 for w in ids if results[w]["tile_lines_frozen_parser"] > 0)),
            "tile_lines_tolerant_total": int(sum(results[w]["n_tile_lines"] for w in ids)),
            "expected_lines_if_all_12_tiles_described": int(12 * len(ids))},
        "timing": {"seconds_mean": round(float(np.mean(secs)), 2) if secs else None,
                   "seconds_median": round(float(np.median(secs)), 2) if secs else None,
                   "total_wall_s": round(wall, 1),
                   "calls_in_ledger_total": counts},
        "ci95": blocks["all_160"]["ci95"] if "ci95" in blocks["all_160"] else None,
        "ci95_type": f"factory-grouped bootstrap ({N_BOOT} draws, seed {SEED})",
        "device": "CPU (sheet build) + OpenCode Go API (image calls); no GPU used",
        "cost": f"{counts['image']} image calls in the ledger, {len(ids)} V02 windows, "
                f"{args.workers} workers, {round(wall,1)} s wall",
        "command": "JADAL_VLM_LEDGER=<ledger> python3 eval/exp_vlm/grid_v02.py --workers 3",
        "git_sha": C.git_sha(), "seed": SEED,
        "config_sha256": C.config_sha256({"windows": ids, "prompt": M.PROMPT_SHA256[PROMPT],
                                          "sheet": pre["sheet"], "model": M.MODEL_ID,
                                          "seed": SEED}),
        "per_window": {w: {"usable": results[w]["usable"],
                           "reason_code": results[w]["reason_code"],
                           "task_progress": results[w]["task_progress"],
                           "explanation": results[w]["explanation"],
                           "seconds": results[w]["seconds"],
                           "raw": results[w]["raw"],
                           "png_sha256": results[w]["png_sha256"],
                           "tile_lines": {str(k): list(v) for k, v in
                                          sorted(results[w]["tile_lines"].items())},
                           "n_tile_lines": results[w]["n_tile_lines"],
                           "tile_lines_frozen_parser": results[w]["tile_lines_frozen_parser"],
                           "y": int(rows[w]["y"]), "factory": rows[w]["factory"],
                           "human_reason": rows[w]["reason"],
                           "p26_shipped": float(rows[w]["p26"])} for w in ids},
    }
    C.save_results(os.path.join(OUT, "GRID.V02.json"), payload)
    print(json.dumps({"auroc_usable": blocks["all_160"]["auroc_usable"],
                      "ci95": blocks["all_160"]["ci95"],
                      "share_usable": natural["share_all_answers_usable_true"],
                      "BA": natural["balanced_accuracy"],
                      "paired_vs_shipped": {k: paired[k] for k in ("delta", "lo", "hi", "excludes_zero")},
                      "calls": counts}, indent=1))
    print("wrote", os.path.join(OUT, "GRID.V02.json"))


if __name__ == "__main__":
    main()
