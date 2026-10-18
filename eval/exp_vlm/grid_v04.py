#!/usr/bin/env python3
"""GRID.V04 — run-to-run variance of the frozen vision model.

Three runs of the SAME frozen `quality` prompt on the SAME 40 pre-registered
windows, using the SAME cached PNG bytes every time, so the only source of
variation is the model.  Run 1 is the V02 call, which is REUSED (not paid for
again); runs 2 and 3 are 80 new calls (PREREG cap 120).

Reports: unanimous share across the 3 runs over `usable` and over `task_progress`,
Fleiss' kappa and the mean pairwise Cohen's kappa, the per-run AUROC with a
factory-grouped CI, and the spread.  The question this answers is whether a single
call can be treated as a measurement.

Run: JADAL_VLM_LEDGER=<ledger> python3 eval/exp_vlm/grid_v04.py --workers 3
"""
from __future__ import annotations

import argparse
import itertools
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
from grid_v02 import CONTEXT, PROMPT, ledger_done, load_prereg, parse_row  # noqa: E402

OUT = os.path.join(ROOT, "eval", "results", "exp_vlm")
SEED, N_BOOT = 0, 2000
RUNS = (1, 2, 3)
SHEET_TAG = "v02"          # deliberately the same cache as V02: identical PNG bytes


def fleiss_kappa(counts: np.ndarray) -> float:
    """counts: (n_subjects, n_categories) integer rater counts (equal raters)."""
    n, _k = counts.shape
    m = counts.sum(axis=1)
    if len(set(m.tolist())) != 1 or m[0] < 2:
        return float("nan")
    m0 = int(m[0])
    p_i = (np.sum(counts.astype(float) ** 2, axis=1) - m0) / (m0 * (m0 - 1))
    p_bar = float(p_i.mean())
    p_j = counts.sum(axis=0) / (n * m0)
    p_e = float(np.sum(p_j ** 2))
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else float("nan")


def cohen_kappa(a, b) -> float:
    a, b = list(a), list(b)
    cats = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = 0.0
    for c in cats:
        pe += (a.count(c) / n) * (b.count(c) / n)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pre = load_prereg()
    ids = pre["windows"]["V04_40"]
    if args.limit:
        ids = ids[:args.limit]
    ledger = os.environ.get("JADAL_VLM_LEDGER", M.DEFAULT_LEDGER)
    rows = {r["window"]: r for r in V.ood_windows()}
    done = ledger_done(ledger, PROMPT)
    started = time.time()
    print(f"V04: {len(ids)} windows x runs {RUNS}; ledger already has {len(done)} calls",
          flush=True)

    runs = {wid: {} for wid in ids}
    sheets = {}
    # run 1 is reused from V02 whenever the sheet bytes match
    todo = []
    for wid in ids:
        row = rows[wid]
        png, doc = R.sheet_cached(SHEET_TAG, row)
        sheets[wid] = (png, doc)
        r1 = done.get((wid, 1))
        if r1 and r1.get("png_sha256") == doc["png_sha256"]:
            runs[wid][1] = parse_row(r1)
        else:
            todo.append((wid, 1))
    for wid in ids:
        for run in (2, 3):
            if (wid, run) not in done:
                todo.append((wid, run))
    print(f"  {len(todo)} new calls needed, "
          f"{sum(1 for w in ids if 1 in runs[w])} run-1 replies reused", flush=True)

    def work(item):
        wid, run = item
        png, doc = sheets[wid]
        rec = R.call_one(rows[wid], png, doc, PROMPT, run=run, tag="v04", ledger=ledger)
        return (wid, run, parse_row(rec))

    def on_done(i, n, res):
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  {i+1}/{n}", flush=True)

    for wid, run, parsed in R.parallel_run(todo, work, workers=args.workers, on_done=on_done):
        runs[wid][run] = parsed
    wall = time.time() - started
    counts = M.count_calls(ledger)

    complete = [w for w in ids if all(r in runs[w] for r in RUNS)]
    y = np.array([rows[w]["y"] for w in complete])
    fac = np.array([rows[w]["factory"] for w in complete])

    u = np.array([[bool(runs[w][r]["usable"]) for r in RUNS] for w in complete])
    tp = np.array([[runs[w][r]["task_progress"] for r in RUNS] for w in complete], dtype=object)

    def cat_counts(mat, cats):
        idx = {c: i for i, c in enumerate(cats)}
        out = np.zeros((len(mat), len(cats)), int)
        for i, rowv in enumerate(mat):
            for v in rowv:
                out[i, idx[v]] += 1
        return out

    u_cats = [False, True]
    uc = cat_counts(u, u_cats)
    tp_cats = sorted({v for rowv in tp for v in rowv})
    tc = cat_counts(tp, tp_cats)

    def unanim(mat):
        same = sum(1 for rowv in mat if len(set(rowv.tolist())) == 1)
        return round(same / len(mat), 4) if len(mat) else None

    pairwise_u = [cohen_kappa(u[:, i], u[:, j]) for i, j in itertools.combinations(range(3), 2)]
    pairwise_t = [cohen_kappa(tp[:, i], tp[:, j]) for i, j in itertools.combinations(range(3), 2)]

    per_run = {}
    for k, r in enumerate(RUNS):
        s = u[:, k].astype(float)
        a = C.fast_auc(y, s)
        lo, hi, _ = C.group_boot_ci(y, s, fac, n_boot=N_BOOT, seed=SEED)
        per_run[f"run{r}"] = {"auroc_usable": round(a, 4), "ci95": [round(lo, 3), round(hi, 3)],
                              "share_usable": round(float(s.mean()), 4),
                              "n_accepted_bad": int(s[y == 0].sum()),
                              "n_rejected_good": int((1 - s[y == 1]).sum())}
    aucs = [per_run[f"run{r}"]["auroc_usable"] for r in RUNS]

    # majority-vote combination as the natural way to use repeated calls
    maj = (u.sum(axis=1) >= 2).astype(float)
    a_maj = C.fast_auc(y, maj)
    lo, hi, _ = C.group_boot_ci(y, maj, fac, n_boot=N_BOOT, seed=SEED)

    payload = {
        "id": "GRID.V04", "n": int(len(complete)), "n_bad": int((y == 0).sum()),
        "label_source": C.load_ood160()["label_source"],
        "protocol": ("3 independent runs of the frozen `quality` prompt on the same "
                     "pre-registered 40 windows, reusing the EXACT cached PNG bytes for "
                     "every run, so the only variation is the model; run 1 is the V02 call "
                     "reused, runs 2-3 are new calls; no training, no fitting"),
        "model": M.MODEL_LABEL, "model_id": M.MODEL_ID, "prompt_name": PROMPT,
        "prompt_sha256": M.PROMPT_SHA256[PROMPT],
        "context_required_with_every_number": CONTEXT,
        "windows_requested": len(ids), "windows_complete": len(complete),
        "runs": list(RUNS),
        "usable": {"categories": ["false", "true"],
                   "unanimous_share": unanim(u),
                   "fleiss_kappa": round(fleiss_kappa(uc), 4),
                   "mean_pairwise_cohen_kappa": round(float(np.nanmean(pairwise_u)), 4),
                   "pairwise_cohen_kappa": [round(v, 4) for v in pairwise_u],
                   "per_window_votes_true": [int(v) for v in u.sum(axis=1)]},
        "task_progress": {"categories": [str(c) for c in tp_cats],
                          "unanimous_share": unanim(tp),
                          "fleiss_kappa": round(fleiss_kappa(tc), 4),
                          "mean_pairwise_cohen_kappa": round(float(np.nanmean(pairwise_t)), 4),
                          "pairwise_cohen_kappa": [round(v, 4) for v in pairwise_t]},
        "per_run": per_run,
        "auroc_spread_across_runs": round(float(max(aucs) - min(aucs)), 4),
        "majority_vote_of_3": {"auroc_usable": round(a_maj, 4), "ci95": [round(lo, 3), round(hi, 3)],
                              "share_usable": round(float(maj.mean()), 4)},
        "single_run_treatable_as_a_measurement": None,   # filled below, explicitly
        "timing": {"total_wall_s": round(wall, 1), "calls_in_ledger_total": counts},
        "ci95": [round(lo, 3), round(hi, 3)],
        "ci95_type": f"factory-grouped bootstrap ({N_BOOT} draws, seed {SEED})",
        "device": "CPU (sheet build) + OpenCode Go API (image calls); no GPU used",
        "cost": f"{counts['image']} image calls in the ledger total; V04 added {len(todo)}; "
                f"{len(complete)} windows x 3 runs; {round(wall,1)} s wall",
        "command": "JADAL_VLM_LEDGER=<ledger> python3 eval/exp_vlm/grid_v04.py --workers 3",
        "git_sha": C.git_sha(), "seed": SEED,
        "config_sha256": C.config_sha256({"windows": ids, "runs": list(RUNS),
                                          "prompt": M.PROMPT_SHA256[PROMPT],
                                          "model": M.MODEL_ID, "seed": SEED}),
        "per_window": {w: {f"run{r}": {"usable": runs[w][r]["usable"],
                                       "reason_code": runs[w][r]["reason_code"],
                                       "task_progress": runs[w][r]["task_progress"],
                                       "seconds": runs[w][r]["seconds"],
                                       "raw": runs[w][r]["raw"]} for r in RUNS} | {
                           "y": int(rows[w]["y"]), "factory": rows[w]["factory"]}
                       for w in complete},
    }

    uk = payload["usable"]["fleiss_kappa"]
    tk = payload["task_progress"]["fleiss_kappa"]
    uu = payload["usable"]["unanimous_share"]
    spread = payload["auroc_spread_across_runs"]
    if uu is not None and uk >= 0.8 and spread <= 0.05:
        verdict = ("YES for `usable` on this subset: unanimous share "
                   f"{uu}, Fleiss kappa {uk}, AUROC spread {spread} across 3 runs. "
                   "Still a ranker/reviewer until a second human labeller agrees.")
    elif uu is not None and (uk < 0.6 or spread > 0.10):
        verdict = ("NO. The model is not stable enough to treat one call as a measurement: "
                   f"unanimous share {uu}, Fleiss kappa {uk}, AUROC spread {spread} across "
                   "3 runs. Use repeated calls with a majority vote, and only as triage.")
    else:
        verdict = ("PARTLY: unanimous share "
                   f"{uu}, Fleiss kappa {uk}, AUROC spread {spread} across 3 runs. One call "
                   "carries real run-to-run noise; repeat and majority-vote before acting on "
                   "a single window's verdict.")
    payload["single_run_treatable_as_a_measurement"] = verdict
    payload["task_progress_kappa_note"] = (
        f"task_progress Fleiss kappa {tk} over {len(tp_cats)} categories; the {len(complete)}-window "
        "subset is NOT the full OOD-160 and is not representative of the base rate")

    C.save_results(os.path.join(OUT, "GRID.V04.json"), payload)
    print(json.dumps({k: payload[k] for k in
                      ("usable", "task_progress", "per_run", "auroc_spread_across_runs",
                       "majority_vote_of_3", "single_run_treatable_as_a_measurement", "cost")},
                     indent=1)[:2600])
    print("wrote", os.path.join(OUT, "GRID.V04.json"))


if __name__ == "__main__":
    main()
