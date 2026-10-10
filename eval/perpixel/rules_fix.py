"""Quality-stage rule list with FRR cap (orchestrator review fix #1).

Old objective (max train Youden, unconstrained) chose thresholds rejecting
every good window out-of-clip (GOOD kept 0.0, bal 0.47). New objective per
rule: maximise train Youden J SUBJECT TO train-good FRR <= 0.15; fallback to
min-FRR grid point (reported). Rules monotone (fire BAD only). Stage = OR of
rules with train J > 0. Nested LOCO; vs trivial accept-all/reject-all
(bal 0.5); 'primary' only if the stage beats both.

Output: eval/results/perpixel/rulelist.json (+ frozen thresholds for scorer)
Regenerate: OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/perpixel/rules_fix.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from jadal.perpixel import physics as P  # noqa: E402

SPEC_DIR = "/home/parshu/projects/jadal-qa-wt/spectral_data"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results", "perpixel")

QUALITY_REASONS = {"TOO_DARK", "UNDECODABLE"}
FRR_CAP = 0.15
Q = 4

CANDS = [
    ("dark", "th", "p05", "<", None),
    ("overexp", "th", "p95", ">", None),
    ("blur", "th", "blur_norm", "<", None),
    ("ten", "ph", "ten_norm", "<", None),
    ("frozen", "th", "frozen_idx", "<", [1.5, 2.0, 3.0, 5.0, 10.0]),
    ("madfroz", "ph", "mad_frozen_frac", ">", [0.05, 0.1, 0.2, 0.4, 0.6]),
    ("ffroz", "ph", "frm_frozen_frac", ">", [0.05, 0.1, 0.2, 0.4, 0.6]),
    ("pbytes", "ph", "p_size_med", "<", None),
    ("clipfrac", "ph", "clip_frac", ">", [0.05, 0.1, 0.2, 0.4, 0.6]),
    ("lowent", "ph", "hist_ent", "<", None),
    ("lowsnr", "ph", "snr_db", "<", None),
    ("still", "th", "still_run", ">", [0.5, 0.7, 0.8, 0.9, 0.95]),
]


def load_all():
    gm = json.load(open(os.path.join(SPEC_DIR, "manifest_gold.json")))
    items = {it["id"]: it for it in gm["items"]}
    zt = np.load(os.path.join(RES, "theory_feats.npz"), allow_pickle=False)
    meta = {json.loads(m)["id"]: json.loads(m)["float"] for m in zt["meta"]}
    zp = np.load(os.path.join(RES, "physics_feats.npz"), allow_pickle=False)
    pmap = {str(i): np.asarray(zp["X"][j], dtype=np.float64)
            for j, i in enumerate([str(x) for x in zp["ids"]])}
    order = sorted(set(items) & set(meta) & set(pmap))
    return items, order, meta, pmap


def col(src, feat, order, meta, pmap):
    if src == "th":
        return np.round(np.array([meta[i][feat] for i in order]), Q)
    v = np.array([pmap[i][P.PHYSICS_FEATURES.index(feat)] for i in order])
    return np.round(np.where(np.isfinite(v), v, np.nan), Q)


def hits(v, op, t):
    fin = np.isfinite(v)
    h = np.zeros(len(v), bool)
    h[fin] = (v[fin] < t) if op == "<" else (v[fin] > t)
    return h


def grid_vals(vtr, fixed):
    if fixed is not None:
        return [float(t) for t in fixed]
    v = np.asarray(vtr, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 4:
        return []
    return sorted(set(round(float(np.percentile(v, q)), Q)
                      for q in (2, 5, 10, 20, 30)))


def fit_rule_fixed(vtr, ybtr, op, fixed):
    P = ybtr.sum()
    N = len(ybtr) - P
    if P == 0 or N == 0:
        return None
    grid = [float(t) for t in fixed] if fixed is not None else grid_vals(
        vtr, None)
    best_ok, best_any = None, None
    for t in grid:
        h = hits(vtr, op, t)
        tpr = float(h[ybtr == 1].mean())
        frr = float(h[ybtr == 0].mean())
        J = tpr - frr
        rec = (t, J, frr)
        if best_any is None or (frr, -J) < (best_any[2], -best_any[1]):
            best_any = rec
        if frr <= FRR_CAP and (best_ok is None or J > best_ok[1]):
            best_ok = rec
    if best_ok is not None and best_ok[1] > 0:
        return {"thr": best_ok[0], "J": round(best_ok[1], 4),
                "train_frr": round(best_ok[2], 4), "fallback": False}
    if best_any is not None:
        return {"thr": best_any[0], "J": round(best_any[1], 4),
                "train_frr": round(best_any[2], 4), "fallback": True}
    return None


def apply_stage(order, meta, pmap, idx, rules):
    H = np.zeros(len(idx), bool)
    ev = [""] * len(idx)
    for name, r in rules.items():
        if r is None:
            continue
        v = col(r["src"], r["feature"], order, meta, pmap)[idx]
        h = hits(v, r["op"], r["thr"])
        for j in np.where(h & ~H)[0]:
            ev[j] = (f"{name}:{r['feature']} {r['op']} {r['thr']} "
                     f"(value {v[j]:.4g})")
        H |= h
    return H, ev


def main():
    items, order, meta, pmap = load_all()
    yb = np.array([0 if items[i]["label"] == 1 else 1 for i in order])
    clips = [items[i]["clip"] for i in order]
    rea = [items[i]["reason"] for i in order]
    is_q = np.array([r in QUALITY_REASONS for r in rea])
    good = (yb == 0) & ~is_q  # good windows (quality task negatives)
    print(f"n={len(order)} quality_pos={is_q.sum()} good={good.sum()}",
          flush=True)
    uclips = sorted(set(clips))
    dec = np.zeros(len(order), bool)
    fold_thr = {}
    for tc in uclips:
        tr = np.array([c != tc for c in clips])
        pool = tr & (is_q | (yb == 0))
        rules = {}
        for name, src, feat, op, fixed in CANDS:
            f = fit_rule_fixed(col(src, feat, order, meta, pmap)[pool],
                               yb[pool], op, fixed)
            if f is not None:
                f.update({"feature": feat, "op": op, "src": src})
            rules[name] = f
        fold_thr[tc] = {k: (None if v is None else v["thr"])
                        for k, v in rules.items()}
        te = np.where(~tr)[0]
        H, _ = apply_stage(order, meta, pmap, te, rules)
        dec[te] = H
    mask = is_q | (yb == 0)
    P, N = yb[mask].sum(), (yb[mask] == 0).sum()
    sens = float(dec[mask][yb[mask] == 1].mean())
    frr = float(dec[mask][yb[mask] == 0].mean())
    bal = (sens + 1 - frr) / 2
    rec = {}
    for r in sorted(set((x or "GOOD") for x in rea)):
        m = np.array([((x or "GOOD") == r) for x in rea]) & mask
        if m.sum():
            rec[r] = {"recall_bad": float(dec[m].mean()) if r != "GOOD"
                      else None,
                      "kept": float((~dec[m]).mean()) if r == "GOOD" else None,
                      "n": int(m.sum())}
    print(f"LOCO quality stage: sens={sens:.3f} FRR={frr:.3f} bal={bal:.3f}",
          flush=True)
    print("per-reason:", json.dumps(rec)[:600], flush=True)
    print("trivial accept-all bal=0.5 reject-all bal=0.5", flush=True)
    # frozen refit on ALL
    pool_all = is_q | (yb == 0)
    frozen = {}
    for name, src, feat, op, fixed in CANDS:
        f = fit_rule_fixed(col(src, feat, order, meta, pmap)[pool_all],
                           yb[pool_all], op, fixed)
        if f is not None:
            f.update({"feature": feat, "op": op, "src": src,
                      "thr_int": int(round(f["thr"] * 10 ** Q))})
        frozen[name] = f
    active = {k: v for k, v in frozen.items()
              if v is not None and not v["fallback"] and v["J"] > 0}
    # M2: LOCO for the frozen {lowent, blur, ten} stage (same capped fits,
    # restricted candidate set), stored for freeze_v2 to read.
    SUB = ["lowent", "blur", "ten"]
    sub = {k: CANDS[[c[0] for c in CANDS].index(k)] for k in SUB}
    dec2 = np.zeros(len(order), bool)
    for tc in uclips:
        tr = np.array([c != tc for c in clips])
        pool = tr & (is_q | (yb == 0))
        te = np.where(~tr)[0]
        for name, (nm, src, feat, op, fixed) in sub.items():
            f = fit_rule_fixed(col(src, feat, order, meta, pmap)[pool],
                               yb[pool], op, fixed)
            if f is None:
                continue
            dec2[te] |= hits(col(src, feat, order, meta, pmap)[te], op,
                             f["thr"])
    ss = float(dec2[mask][yb[mask] == 1].mean())
    ff = float(dec2[mask][yb[mask] == 0].mean())
    out = {"frr_cap": FRR_CAP,
           "loco": {"sens": sens, "frr_good": frr, "bal_acc": bal,
                    "per_reason": rec,
                    "baselines": {"accept_all": 0.5, "reject_all": 0.5},
                    "beats_trivial": bool(bal > 0.5)},
           "stage_lowent_blur_ten": {"sens": ss, "frr": ff,
                                           "bal": (ss + 1 - ff) / 2},
           "fold_thresholds": fold_thr, "frozen": frozen,
           "active_rules": sorted(active),
           "primary_claim": ("quality stage only; activity stays a score; "
                             "rule list is NOT the primary decider unless "
                             "it beats trivial baselines out-of-clip") if True
           else ""}
    with open(os.path.join(RES, "rulelist.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("active:", sorted(active), "wrote rulelist.json", flush=True)


if __name__ == "__main__":
    main()
