#!/usr/bin/env python3
"""GRID.V06 — the vision model's role in the funnel, two pre-registered roles.

(a) VETO ON HIGH-CONFIDENCE ACCEPTS — OOD-160, **zero new calls** (reuses the V02
    replies).  The shipped head accepts at p >= 0.75.  Among the windows it accepts,
    how many are actually bad, and how many of those does the model veto
    (`usable == false`)?  Policy comparison at 100 % coverage:
      P_head       accept iff p26 >= 0.75
      P_head_veto  accept iff p26 >= 0.75 AND model usable
    with a **paired factory-clustered** bootstrap of the balanced-accuracy difference.

(b) RESOLVER FOR THE REVIEW BAND — IndEgo-150, fresh calls, hard cap 120.
    Head scores are the frozen 26 features through `common.loco_proba` (the same
    leave-one-clip-out recipe as AUTO.00, whose forced-0.5 baseline is reproduced
    first).  Band = 0.25 < p < 0.75.  Policies at 100 % coverage:
      P_forced05     accept iff p >= 0.5                       (AUTO.00 baseline)
      P_band         accept iff p >= 0.75, else reject          (band thrown away)
      P_band_resolve accept iff p >= 0.75; model decides the band
      P_full_veto    as P_band_resolve, but the model may also veto a
                     high-confidence accept
    Band (72) + high-confidence accepts (43) = 115 <= 120, so BOTH are called and
    both roles of the model are measured in-domain.

The model is PROBABILISTIC and V04 showed a single call is not a measurement
(Fleiss kappa ~0.52, AUROC spread 0.071): every number here is repeat-needed.
One labeller.  Run: JADAL_VLM_LEDGER=<ledger> python3 eval/exp_vlm/grid_v06.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "eval", "exp_vlm"))
sys.path.insert(0, os.path.join(ROOT, "tools", "dsh"))

import common as C  # noqa: E402
import vlm_common as V  # noqa: E402
import vlm_mosaic as M  # noqa: E402
import vlm_run as R  # noqa: E402
from grid_v02 import CONTEXT, PROMPT, ledger_done, parse_row  # noqa: E402

OUT = os.path.join(ROOT, "eval", "results", "exp_vlm")
SEED, N_BOOT = 0, 2000
V06_CAP = 120
HI, LO = 0.75, 0.25


# ------------------------------------------------------------------ metrics --
def _group_index(groups):
    uniq = sorted(set(groups))
    m = {g: i for i, g in enumerate(uniq)}
    gi = np.asarray([m[g] for g in groups])
    return uniq, [np.where(gi == k)[0] for k in range(len(uniq))]


def bal_acc(y, accept):
    """Balanced accuracy, positive class = good (y == 1)."""
    y = np.asarray(y)
    accept = np.asarray(accept, bool)
    good, bad = y == 1, y == 0
    return float(0.5 * (accept[good].mean() + (~accept[bad]).mean()))


def rates(y, accept):
    y = np.asarray(y)
    accept = np.asarray(accept, bool)
    good, bad = y == 1, y == 0
    return {
        "n": int(len(y)), "n_good": int(good.sum()), "n_bad": int(bad.sum()),
        "balanced_accuracy": round(bal_acc(y, accept), 4),
        "false_accept_rate": round(float(accept[bad].mean()), 4),
        "false_reject_rate": round(float((~accept[good]).mean()), 4),
        "accept_share": round(float(accept.mean()), 4),
        "accepted_bad": int(accept[bad].sum()), "rejected_good": int((~accept[good]).sum()),
    }


def paired_group_boot_rate(y, a, b, groups, fn, n_boot=N_BOOT, seed=SEED):
    """Paired cluster bootstrap of fn(y, a) - fn(y, b) on the same windows."""
    y = np.asarray(y)
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    uniq, idx_by = _group_index(groups)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by[k] for k in pick])
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(fn(y[idx], a[idx]) - fn(y[idx], b[idx]))
    if not vals:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_boot": 0}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"delta": round(float(fn(y, a) - fn(y, b)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0), "n_boot": len(vals),
            "ci95_type": f"paired cluster bootstrap ({n_boot} draws, seed {seed})"}


# ---------------------------------------------------------------- role (a) --
def role_a():
    """OOD-160 veto on the shipped head's high-confidence accepts.  No calls."""
    o = C.load_ood160()
    y = o["y"]
    fac = o["factory"]
    p26 = o["p26"]
    v02 = json.load(open(os.path.join(OUT, "GRID.V02.json")))
    vlm = {w: v["usable"] for w, v in v02["per_window"].items()}
    ids = o["ids"]
    usable = np.array([bool(vlm[w]) for w in ids])

    head_accept = p26 >= HI
    head_veto = head_accept & usable
    good, bad = y == 1, y == 0
    acc_bad = head_accept & bad
    voted = {
        "n_head_accepts": int(head_accept.sum()),
        "head_accepts_good": int((head_accept & good).sum()),
        "head_accepts_bad": int(acc_bad.sum()),
        "head_false_accepts_vetoed_by_vlm": int((acc_bad & ~usable).sum()),
        "share_head_false_accepts_vetoed": (round(float((~usable[acc_bad]).mean()), 4)
                                            if acc_bad.sum() else None),
        "good_accepts_lost_to_veto": int((head_accept & good & ~usable).sum()),
        "false_reject_cost_on_good_the_head_accepted": (
            round(float((~usable[head_accept & good]).mean()), 4)
            if (head_accept & good).sum() else None),
        "head_accepts_in_band_not_counted": int(((p26 < HI) & (p26 > LO)).sum()),
    }
    pol = {
        "P_head_accept_iff_p>=0.75": rates(y, head_accept),
        "P_head_veto": rates(y, head_veto),
    }
    paired = paired_group_boot_rate(y, head_veto, head_accept, fac, bal_acc)
    return {"veto_table": voted, "policies": pol, "paired_ba_difference": paired,
            "context": CONTEXT,
            "note": "zero new calls: every reply is the V02 run-1 reply already in the "
                    "ledger; the band (0.25 < p26 < 0.75) is rejected under both policies, "
                    "so this measures the veto only, not a resolver"}


# ---------------------------------------------------------------- role (b) --
def role_b(workers: int, ledger: str, limit: int = 0):
    g = C.load_gold()
    ids, y, clip = g["ids"], g["y"], g["clip"]
    p = C.loco_proba(g["X"], y, clip, C=0.3)
    band = (p > LO) & (p < HI)
    hi = p >= HI
    lo = p <= LO
    forced = p >= 0.5

    targets = [ids[i] for i in np.where(band | hi)[0]]
    if limit:
        targets = targets[:limit]
    rows = {ids[i]: {"window": ids[i], "path": g["path"][i],
                     "start_s": float(g["start_s"][i]), "clip": clip[i]} for i in range(len(ids))}

    ledger = ledger
    before = M.count_calls(ledger)
    if before["image"] + len(targets) > before["image"] + V06_CAP:
        raise SystemExit("would exceed the V06 cap")
    done = ledger_done(ledger, PROMPT)
    todo = [w for w in targets if (w, 1) not in done]
    print(f"V06(b): band {int(band.sum())} + high-confidence accepts {int(hi.sum())} "
          f"= {len(targets)} target windows; {len(targets)-len(todo)} already in the ledger; "
          f"{len(todo)} new calls (cap {V06_CAP})", flush=True)
    if len(todo) > V06_CAP:
        raise SystemExit("would exceed the V06 cap")

    sheets = {}
    for w in targets:
        sheets[w] = R.sheet_cached("v06", rows[w])
    started = time.time()

    def work(w):
        png, doc = sheets[w]
        reuse = done.get((w, 1))
        if reuse and reuse.get("png_sha256") == doc["png_sha256"]:
            rec = dict(reuse)
        else:
            rec = R.call_one(rows[w], png, doc, PROMPT, run=1, tag="v06", ledger=ledger)
        rec["window"] = w
        return w, parse_row(rec)

    def on_done(i, n, res):
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  {i+1}/{n}", flush=True)

    res = R.parallel_run(todo, work, workers=workers, on_done=on_done)
    replies = {w: pr for w, pr in res}
    for w in targets:
        if w not in replies:
            reuse = done.get((w, 1))
            if reuse:
                replies[w] = parse_row(reuse)
    wall = time.time() - started
    after = M.count_calls(ledger)
    # The honest V06 expenditure is every image call tagged 'v06' in the ledger, NOT
    # `after - before` of this invocation: a re-run after a crash reuses the replies and
    # would otherwise report zero calls for 115 calls actually paid for.
    v06_calls = 0
    if os.path.exists(ledger):
        with open(ledger) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if not e.get("text_only") and e.get("tag") == "v06":
                    v06_calls += 1

    # only the 115 target windows (band + high-confidence accepts) have a reply; the
    # low-confidence rejects never need one because both policies reject them outright.
    vlm_ok = np.zeros(len(ids), bool)
    have = np.zeros(len(ids), bool)
    for i, w in enumerate(ids):
        if w in replies:
            vlm_ok[i] = bool(replies[w]["usable"])
            have[i] = True
    missing_replies = [w for w in targets if w not in replies]
    if missing_replies:
        raise SystemExit(f"no reply for target windows: {missing_replies[:5]}")

    p_band_resolve = np.where(hi, True, np.where(lo, False, vlm_ok))
    p_full_veto = np.where(band, vlm_ok, hi & vlm_ok)
    acc_in_band = band & hi

    def sub(mask):
        return {"n": int(mask.sum()), "n_bad": int((y[mask] == 0).sum()),
                "n_good": int((y[mask] == 1).sum())}

    pol = {
        "P_forced05_AUTO00_baseline": rates(y, forced),
        "P_band_throws_the_band_away": rates(y, hi),
        "P_band_resolve": rates(y, p_band_resolve),
        "P_full_veto": rates(y, p_full_veto),
    }
    veto_tbl = {
        "n_high_confidence_accepts": int(hi.sum()),
        "head_accepts_bad": int((hi & (y == 0)).sum()),
        "head_false_accepts_vetoed_by_vlm": int((hi & (y == 0) & ~vlm_ok).sum()),
        "share_head_false_accepts_vetoed": (round(float((~vlm_ok[hi & (y == 0)]).mean()), 4)
                                            if (hi & (y == 0)).sum() else None),
        "good_accepts_lost_to_veto": int((hi & (y == 1) & ~vlm_ok).sum()),
        "band_windows": int(band.sum()), "band_bad": int((y[band] == 0).sum()),
        "band_good": int((y[band] == 1).sum()),
        "vlm_says_usable_inside_band": int(vlm_ok[band].sum()),
        "vlm_says_usable_all_targets": int(vlm_ok[band | hi].sum()),
        "share_usable_all_targets": round(float(vlm_ok[band | hi].mean()), 4),
    }
    paired = {
        "band_resolve_minus_forced05": paired_group_boot_rate(
            y, p_band_resolve, forced, clip, bal_acc),
        "full_veto_minus_forced05": paired_group_boot_rate(
            y, p_full_veto, forced, clip, bal_acc),
        "band_resolve_minus_band_thrown_away": paired_group_boot_rate(
            y, p_band_resolve, hi, clip, bal_acc),
    }
    secs = [replies[w]["seconds"] for w in replies if replies[w].get("seconds")]
    return {
        "band_size": int(band.sum()), "accepts_size": int(hi.sum()),
        "targets": len(targets), "new_calls": int(len(todo)),
        "v06_calls_total_from_ledger_tag": int(v06_calls),
        "new_calls_this_invocation": int(len(todo)),
        "ledger_before": before, "ledger_after": after,
        "windows_with_a_reply": int(have.sum()),
        "head_scores": "common.loco_proba on the frozen 26 features (C=0.3, LOCO over 7 clips)",
        "forced05_reproduces_AUTO00": {
            "here": pol["P_forced05_AUTO00_baseline"]["balanced_accuracy"],
            "AUTO00_documented": 0.740,
            "matches": abs(pol["P_forced05_AUTO00_baseline"]["balanced_accuracy"] - 0.740) < 0.005},
        "veto_table": veto_tbl, "policies": pol, "paired_ba_differences": paired,
        "band_windows_meta": [{"window": w, "y": int(y[ids.index(w)]),
                               "p_head": round(float(p[ids.index(w)]), 4),
                               "in_band": bool(band[ids.index(w)]),
                               "vlm_usable": bool(vlm_ok[ids.index(w)])} for w in targets],
        "seconds_mean": round(float(np.mean(secs)), 2) if secs else None,
        "seconds_median": round(float(np.median(secs)), 2) if secs else None,
        "wall_s_this_invocation": round(wall, 1),
        "wall_note": ("wall_s is THIS invocation only; when the replies are already in the "
                      "ledger it is ~0 because nothing is called. The 115 calls themselves "
                      "took seconds_mean per call, measured when they were made."),
        "per_window": {w: {"usable": replies[w]["usable"],
                           "reason_code": replies[w]["reason_code"],
                           "task_progress": replies[w]["task_progress"],
                           "explanation": replies[w]["explanation"],
                           "seconds": replies[w]["seconds"],
                           "raw": replies[w]["raw"],
                           "tile_lines": {str(k): list(v) for k, v in
                                          sorted(replies[w]["tile_lines"].items())},
                           "n_tile_lines": replies[w]["n_tile_lines"],
                           "png_sha256": replies[w]["png_sha256"],
                           "y": int(y[ids.index(w)]), "clip": clip[ids.index(w)],
                           "p_head": round(float(p[ids.index(w)]), 4),
                           "in_band": bool(band[ids.index(w)])} for w in targets},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-calls", action="store_true")
    args = ap.parse_args()
    ledger = os.environ.get("JADAL_VLM_LEDGER", M.DEFAULT_LEDGER)

    t0 = time.time()
    a = role_a()
    print("role (a) done, zero calls", flush=True)
    b = role_b(args.workers, ledger, args.limit) if not args.skip_calls else {}
    pre = json.load(open(os.path.join(ROOT, "experiments", "GRID.VLM", "PREREG.json")))

    payload = {
        "id": "GRID.V06",
        # the top-level n/n_bad describe ROLE (a), which is the OOD-160 veto question;
        # role (b) carries its own n's inside role_b_indego_resolver.
        "n": 160, "n_bad": 22, "n_good": 138,
        "n_datasets": "role (a) OOD-160 (n=160, n_bad=22); role (b) IndEgo-150 (n=150, n_bad=50)",
        "label_source": ("OOD-160: human labels, ONE labeller (138 good / 22 bad). "
                         "IndEgo-150: human gold, ONE labeller (100 good / 50 bad)."),
        "protocol": ("role (a) reuses the V02 run-1 replies on all 160 OOD windows (zero new "
                     "calls) and compares P_head (accept iff p26>=0.75) with P_head_veto "
                     "(accept iff p26>=0.75 AND usable) by a paired factory-clustered bootstrap "
                     "of balanced accuracy; role (b) calls the frozen prompt once on the 72 "
                     "IndEgo review-band windows and the 43 high-confidence accepts (115 calls, "
                     "cap 120) and compares P_forced05 (AUTO.00), P_band, P_band_resolve and "
                     "P_full_veto at 100 % coverage with paired clip-clustered bootstraps"),
        "model": M.MODEL_LABEL, "model_id": M.MODEL_ID, "prompt_name": PROMPT,
        "prompt_sha256": M.PROMPT_SHA256[PROMPT],
        "context_required_with_every_number": CONTEXT,
        "role_a_ood_veto": a,
        "role_b_indego_resolver": b,
        "tile_line_parser_used": ("the tolerant parser in eval/exp_vlm/vlm_run.py: the shared "
                                  "vlm_mosaic.parse_tile_lines (even after the beb742e fix) "
                                  "requires a timestamp and recovered only 372 of 1920 lines, "
                                  "because 113 of 160 replies omit the timestamp entirely"),
        "probabilistic_caveat": ("the model is probabilistic and GRID.V04 measured Fleiss kappa "
                                 "~0.52 with a 0.071 AUROC spread across identical repeat runs, "
                                 "so EVERY number here is repeat-needed: this is a ranker / "
                                 "reviewer, not an automatic verdict, and it is one labeller"),
        "ci95": (b.get("paired_ba_differences", {})
                 .get("band_resolve_minus_forced05", {}).get("ci95")),
        "ci95_type": f"paired cluster bootstrap ({N_BOOT} draws, seed {SEED})",
        "device": "CPU (sheet build) + OpenCode Go API (image calls); no GPU used",
        "cost": (f"role (a) 0 new calls; role (b) {b.get('v06_calls_total_from_ledger_tag', 0)} "
                 f"image calls made for V06 (cap {V06_CAP}; band 72 + high-confidence accepts 43 "
                 f"= 115 targets; this invocation made {b.get('new_calls_this_invocation', 0)} new "
                 f"calls and reused the rest); ledger before "
                 f"{b.get('ledger_before', {}).get('image')} / after "
                 f"{b.get('ledger_after', {}).get('image')}; wall {round(time.time()-t0,1)} s"),
        "command": "JADAL_VLM_LEDGER=<ledger> python3 eval/exp_vlm/grid_v06.py --workers 3",
        "git_sha": C.git_sha(), "seed": SEED,
        "config_sha256": C.config_sha256({"role_a": "veto on p26>=0.75, reuse V02",
                                          "role_b": {"band": [LO, HI], "targets": 115,
                                                     "prompt": M.PROMPT_SHA256[PROMPT],
                                                     "model": M.MODEL_ID},
                                          "seed": SEED}),
        "prereg_note": f"PREREG frozen_utc {pre['frozen_utc']}; V06 budget {pre['call_budget']['V06']}",
    }
    C.save_results(os.path.join(OUT, "GRID.V06.json"), payload)
    print(json.dumps({"role_a_veto": a["veto_table"],
                      "role_a_policies": {k: v["balanced_accuracy"] for k, v in a["policies"].items()},
                      "role_a_paired": a["paired_ba_difference"],
                      "role_b_veto": b.get("veto_table"),
                      "role_b_policies": {k: v["balanced_accuracy"] for k, v in b.get("policies", {}).items()},
                      "role_b_paired": b.get("paired_ba_differences"),
                      "calls": b.get("v06_calls_total")}, indent=1))
    print("wrote", os.path.join(OUT, "GRID.V06.json"))


if __name__ == "__main__":
    main()
