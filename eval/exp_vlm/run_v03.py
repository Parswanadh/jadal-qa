#!/usr/bin/env python3
"""GRID.V03 -- ablations: which part of the sheet (if any) carries the signal?

20 pre-registered stratified windows (10 bad / 10 good, all 10 bad are
NO_ACTIVE_WORK, 14 factories), SIX arms x 20 windows = 120 calls, PLUS two
negative controls on a contentless flat-grey sheet (1 `quality` + 1 `read_tiles`)
= 122 calls, which is exactly the pre-registered V03 cap.

Reference arm for the paired comparisons: **arm (e) `e_json_only_prompt`** -- the
same default 12-tile labelled sheet as the primary configuration, with only the
reasoning step removed.  It is the closest thing inside this budget to V02's
primary arm; the true primary (`quality` on the 12-tile sheet) is V02's job and is
NOT re-bought here.  The comparison "12 tiles vs 1 tile" therefore compares arm (c)
against arm (e) and inherits a PROMPT confound (c uses `quality`, e uses
`quality_json_only`); that confound is stated with every such number.

Score definitions (both reported):
  * `usable`         -- 1.0 if the reply's JSON says usable=true else 0.0.  A binary
                        score, so its AUROC is exactly balanced accuracy at 0.5;
                        reported because it is what a deployment would consume.
  * `progress_score` -- advancing=1.0 / cannot_tell=0.5 / no_relevant_work=0.0, a
                        3-level graded score and the more informative summary.

Run: python3 eval/exp_vlm/run_v03.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "tools", "dsh"), os.path.join(ROOT, "eval")):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as C  # noqa: E402
import vlm_mosaic as VM  # noqa: E402
from eval.exp_vlm import vlm_common as V  # noqa: E402
from eval.exp_vlm import vlm_run as R  # noqa: E402

CAP = 122
#: which ledger arms count against V03's cap (per-experiment accounting: this
#: ledger is shared with V01, whose 20 calls must not eat V03's budget)
V03_LEDGER_ARMS = set(R.V03_ARMS) | {"null_grey_quality", "null_grey_read_tiles"}
IDS = R.PREREG["windows"]["V03_20_stratified"]
OUT = os.path.join(ROOT, "eval", "results", "exp_vlm", "GRID.V03.json")
LEDGER = os.environ.get("JADAL_VLM_LEDGER", "/tmp/opencode/dsh/vlm_ledger_b.jsonl")
SEED, N_BOOT = 0, 2000


def run_arms(rows, already):
    for arm, cfg in R.V03_ARMS.items():
        prompt = cfg.get("prompt", "quality")
        n_tiles = cfg.get("n_tiles", 12)
        cols = cfg.get("cols", 4)
        labels = cfg.get("labels", True)
        seed = cfg.get("shuffle_seed")
        for wid in IDS:
            if (wid, arm) in already:
                continue
            win = rows[wid]
            png, doc = R.sheet_for_window(win, n_tiles=n_tiles, cols=cols, labels=labels,
                                          shuffle_seed=seed, tag="v03_")
            R.guard(LEDGER, CAP, 1, arms=V03_LEDGER_ARMS)
            rec = VM.ask(png, prompt, ledger=LEDGER,
                         extra={"window": wid, "arm": arm, "y": win["y"],
                                "factory": win["factory"], "reason": win["reason"],
                                "sheet_sha": doc["png_sha256"], "n_tiles": doc["n_tiles"],
                                "cols": doc["cols"], "labels": doc["labels"],
                                "shuffle_seed": doc.get("shuffle_seed"), "prompt_name": prompt})
            print(f"  {arm:22s} {wid}: ok={rec['ok']} "
                  f"{'usable=' + str(rec['quality'][0]) if rec.get('quality') else 'UNPARSED'} "
                  f"({rec.get('seconds')}s)", flush=True)


def run_nulls():
    already = R.done_pairs(LEDGER)
    grey = R.grey_sheet()
    for arm, prompt in (("null_grey_quality", "quality"), ("null_grey_read_tiles", "read_tiles")):
        if ("null_grey", arm) in already:
            continue
        R.guard(LEDGER, CAP, 1, arms=V03_LEDGER_ARMS)
        rec = VM.ask(grey, prompt, ledger=LEDGER,
                     extra={"window": "null_grey", "arm": arm, "y": None,
                            "factory": None, "reason": None,
                            "sheet_sha": VM.png_sha256(grey), "prompt_name": prompt,
                            "negative_control": True})
        print(f"  {arm:22s} ok={rec['ok']} parsed={rec.get('quality')} "
              f"lines={len(rec.get('tiles') or {})} ({rec.get('seconds')}s)", flush=True)


def analyse(recs, y_of, fac_of):
    """Per-arm scores + AUROC + paired vs the reference arm."""
    arms = {}
    for arm in list(R.V03_ARMS):
        by_w = {}
        for r in recs:
            if r.get("arm") == arm and r.get("window") != "null_grey":
                u, p = R.quality_scores(r)
                by_w[r["window"]] = {"usable": u, "progress": p, "ok": r.get("ok"),
                                     "seconds": r.get("seconds"), "raw": r.get("raw"),
                                     "sheet_sha256": r.get("sheet_sha"),
                                     "reason_code": (r["quality"][1] if r.get("quality") else None),
                                     "task_progress": (r["quality"][2] if r.get("quality") else None)}
        arms[arm] = by_w
    out = {}
    for arm, by_w in arms.items():
        ids = [w for w in IDS if by_w.get(w, {}).get("usable") is not None]
        y = np.asarray([y_of[w] for w in ids])
        fac = np.asarray([fac_of[w] for w in ids])
        su = np.asarray([by_w[w]["usable"] for w in ids], float)
        sp = np.asarray([by_w[w]["progress"] for w in ids], float)
        a_u = C.fast_auc(y, su) if len(set(su.tolist())) > 1 else float("nan")
        a_p = C.fast_auc(y, sp) if len(set(sp.tolist())) > 1 else float("nan")
        ci_u = ([round(v, 3) for v in C.group_boot_ci(y, su, fac, n_boot=N_BOOT, seed=SEED)[:2]]
                if len(set(su.tolist())) > 1 else None)
        ci_p = ([round(v, 3) for v in C.group_boot_ci(y, sp, fac, n_boot=N_BOOT, seed=SEED)[:2]]
                if len(set(sp.tolist())) > 1 else None)
        bad, good = y == 0, y == 1
        out[arm] = {
            "n_parsed": len(ids), "n_unparsed": len(IDS) - len(ids),
            "n_bad": int(bad.sum()), "n_good": int(good.sum()),
            "auroc_usable_binary": round(a_u, 4), "ci95_usable": ci_u,
            "auroc_progress_graded": round(a_p, 4), "ci95_progress": ci_p,
            "usable_true_share": R.usable_share([{"quality": (by_w[w]["usable"] == 1.0,
                                                              None, None, None)}
                                                 for w in ids]) if ids else None,
            "recall_bad_call_unusable": (round(float((su[bad] == 0).mean()), 4)
                                         if bad.sum() else None),
            "false_reject_good_call_unusable": (round(float((su[good] == 0).mean()), 4)
                                                if good.sum() else None),
            "mean_seconds": (round(float(np.mean([by_w[w]["seconds"] for w in ids
                                                  if by_w[w]["seconds"]])), 2) if ids else None),
            "_ids": ids, "_y": y, "_fac": fac, "_su": su, "_sp": sp,
        }
    ref = R.V03_REFERENCE
    if ref in out:
        for arm, d in out.items():
            common = [w for w in d["_ids"] if w in out[ref]["_ids"]]
            if arm == ref or not common:
                d["paired_vs_reference"] = None
                continue
            ix = [d["_ids"].index(w) for w in common]
            rx = [out[ref]["_ids"].index(w) for w in common]
            yy = np.asarray([y_of[w] for w in common])
            ff = np.asarray([fac_of[w] for w in common])
            du = C.paired_group_boot_diff(yy, d["_su"][ix], out[ref]["_su"][rx], ff,
                                          n_boot=N_BOOT, seed=SEED)
            dp = C.paired_group_boot_diff(yy, d["_sp"][ix], out[ref]["_sp"][rx], ff,
                                          n_boot=N_BOOT, seed=SEED)
            d["paired_vs_reference"] = {
                "reference": ref, "n_windows_paired": len(common),
                "delta_usable": round(du["delta"], 4),
                "ci95_usable": [round(du["lo"], 3), round(du["hi"], 3)],
                "excludes_zero_usable": du["excludes_zero"],
                "delta_progress": round(dp["delta"], 4),
                "ci95_progress": [round(dp["lo"], 3), round(dp["hi"], 3)],
                "excludes_zero_progress": dp["excludes_zero"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = {r["window"]: r for r in V.ood_windows()}
    already = R.done_pairs(LEDGER)

    run_arms(rows, already)
    run_nulls()

    recs = R.ledger_rows(LEDGER)
    y_of = {w: rows[w]["y"] for w in IDS}
    fac_of = {w: rows[w]["factory"] for w in IDS}
    arms = analyse(recs, y_of, fac_of)
    for d in arms.values():
        for k in ("_ids", "_y", "_fac", "_su", "_sp"):
            d.pop(k, None)

    nulls = {r["arm"]: {"ok": r.get("ok"), "quality": r.get("quality"),
                        "n_tile_lines": len(r.get("tiles") or {}),
                        "raw": r.get("raw"), "seconds": r.get("seconds"),
                        "sheet_sha256": r.get("sheet_sha")}
             for r in recs if r.get("negative_control")}
    null_q = nulls.get("null_grey_quality", {})
    null_verdict = None
    if null_q.get("quality"):
        usable = null_q["quality"][0]
        null_verdict = ("INVALIDATES THE TRACK: the contentless sheet read as usable=true, so the "
                        "model is not reading the image"
                        if usable else
                        "PASS: the contentless sheet was not read as usable "
                        "(usable=false), consistent with the model using the image")
    calls = VM.count_calls(LEDGER)
    v03_calls = sum(1 for r in recs if r.get("arm") in V03_LEDGER_ARMS)
    v01_calls = sum(1 for r in recs if r.get("arm") == "v01_read_tiles")

    payload = {
        "id": "GRID.V03", "status": "done",
        "n": len(IDS), "n_bad": 10, "n_good": 10, "n_factories": len(set(fac_of.values())),
        "label_source": "human labels, ONE labeller (Egocentric-10K OOD-160 subset); "
                        "all 10 bad windows are NO_ACTIVE_WORK",
        "protocol": "20-window pre-registered stratified subset (10 bad / 10 good -- BALANCED, "
                    "therefore NOT representative of the base rate), six arms x 20 windows on the "
                    "SAME windows, plus two contentless-sheet negative controls; "
                    "factory-grouped bootstrap CI (2000 draws, seed 0)",
        "reference_arm": R.V03_REFERENCE,
        "reference_why": "arm (e) is the default 12-tile labelled sheet with only the reasoning "
                         "step removed, the closest in-budget analogue of V02's primary arm; the "
                         "true primary (quality on 12 tiles) is V02's job and was not re-bought",
        "prompt_confound": "arms a,b,c,d6,d24 use the `quality` prompt while arm e uses "
                           "`quality_json_only`, so a c-vs-e 'one frame vs twelve frames' "
                           "comparison is confounded with the prompt; read it beside the "
                           "within-prompt arms (c vs d6 vs d24)",
        "arms": arms,
        "negative_controls": {"records": nulls, "verdict": null_verdict},
        "auroc_note": "auroc_usable_binary is a BINARY score, so it equals balanced accuracy at "
                      "0.5 and is a coarse summary; auroc_progress_graded uses the 3-level "
                      "task_progress field and is the more informative number",
        "power_note": "10 bad windows across 14 factories: arm-to-arm intervals are very wide and "
                      "most differences will not be established. A paired interval that includes "
                      "zero is the expected outcome, not a failure.",
        "ci95": None,
        "ci95_type": "none at experiment level; per-arm CI is factory-grouped bootstrap "
                     "(2000 draws, seed 0) over the 14 factories of this subset",
        "model_id": VM.MODEL_ID, "model_label": VM.MODEL_LABEL,
        "prompt_sha256": {k: VM.PROMPT_SHA256[k] for k in ("quality", "quality_json_only")},
        "device": "API call (OpenCode Go Responses); sheets built on CPU",
        "cost": f"V03 made {v03_calls} image API calls (cap 122 = 120 arm calls + 2 "
                f"negative-control calls); the shared sub-stream ledger also holds V01's "
                f"{v01_calls} calls, total {calls['total']} image calls; {calls['text']} text calls",
        "command": "JADAL_VLM_LEDGER=<ledger> python3 eval/exp_vlm/run_v03.py",
        "git_sha": C.git_sha(), "seed": SEED,
        "config_sha256": C.config_sha256({"windows": IDS, "arms": R.V03_ARMS, "cap": CAP,
                                          "reference": R.V03_REFERENCE}),
        "config": {"windows": IDS, "arms": R.V03_ARMS, "cap": CAP,
                   "reference": R.V03_REFERENCE},
        "honesty": [
            "the model is probabilistic; nothing here is deterministic",
            "the usable=true share of ALL answers is reported per arm (documented success bias)",
            "one labeller; a ranker/reviewer result, not an accuracy claim",
            "the subset is balanced, so no rate here is a base rate",
        ],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    C.save_results(OUT, payload)
    print(json.dumps({"finite": {k: {kk: vv for kk, vv in v.items()
                                     if kk in ("n_parsed", "auroc_usable_binary", "ci95_usable",
                                               "auroc_progress_graded", "ci95_progress",
                                               "usable_true_share", "paired_vs_reference")}
                                for k, v in arms.items()},
                      "null_verdict": null_verdict, "cost": payload["cost"]}, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
