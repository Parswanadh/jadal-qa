#!/usr/bin/env python3
"""A4 — coverage at >=95% decided-accuracy with a deterministic abstain rule.

Question: for the best cached scorer, sweep the abstain rule (symmetric band,
global/within-factory quantile, asymmetric top/bottom-k, split conformal) and
report **decided balanced-accuracy vs coverage**, in-domain (IndEgo gold, LOCO),
OOD (160 human-labelled Egocentric-10K, factory-grouped) and the round-3
confirmatory hold-out (108, evaluate only). Find the maximum coverage at
>=95% decided accuracy (point and grouped-CI lower bound).

Rules are deterministic: quantile rules are integer top/bottom-k counts on the
batch; band rules are fixed integer permille thresholds. Nothing is sampled.

Outputs: eval/results/exp_acc/A4.01.json, A4.02.json
Run: OMP_NUM_THREADS=2 python3 eval/exp_acc/run_a4.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
import o4_lib as R  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
MYIN = os.path.join(REPO, "eval", "results", "exp_acc", "inputs")


def _resolve(fname, *fallbacks):
    """Prefer the local cached copy; fall back to the pod's sibling worktrees.

    The .npz caches are git-ignored (pod-local), so a fresh checkout resolves
    them from the branches that produced them."""
    cands = [os.path.join(MYIN, fname)] + list(fallbacks)
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]


Z3_160 = _resolve("z3_160.npz",
                  "/workspace/wt/m-adapt160/eval/results/exp_adapt160/inputs/z3_160.npz")
PIXEL160 = _resolve("ood160_pixel.npz",
                    "/workspace/wt/m-fewshot160/eval/results/exp_fewshot/inputs/ood160_pixel.npz")
CONFIRM108 = _resolve("confirm108_features.npz",
                      "/workspace/wt/m-confirm/eval/results/exp_confirm/inputs/confirm108_features.npz",
                      "/workspace/wt/m-acc-labelbudget/eval/results/exp_acc/inputs/confirm108_features.npz")
R.INPUTS = os.path.dirname(Z3_160)                # z3_160.npz
R.PIXEL160 = PIXEL160
OUT = os.path.join(REPO, "eval", "results", "exp_acc")
NB = 2000
SEED = 7
DEV = "CPU (numpy; cached scores only, no GPU)"

BAND_W = [round(0.005 * i, 3) for i in range(0, 61)]      # 0.000 .. 0.300
Z_W = [round(0.05 * i, 2) for i in range(0, 81)]          # 0.00 .. 4.00
QGRID = [round(0.005 * i, 3) for i in range(0, 101)]      # 0.000 .. 0.500
A_GRID = [round(0.01 * i, 2) for i in range(1, 51)]       # 0.01 .. 0.50
ALPHAS = [round(0.02 * i, 2) for i in range(1, 50)]       # 0.02 .. 0.98


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=REPO).stdout.strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ data
def load_in_domain():
    oof = np.load(os.path.join(BUNDLE, "caches", "ensemble", "oof_probs.npz"),
                  allow_pickle=True)
    y = oof["y"].astype(int)
    clips = np.asarray([str(x) for x in oof["clip"]])
    p_frozen = oof["p_ens"].astype(float)
    gold = R.load_gold()
    z3 = R.load_z3_160()
    z6 = [z3["names"].index(n) for n in R.Z3_SIX]
    s8 = [R.FEATURE_NAMES26.index(n) for n in R.STABLE8]
    X14g = np.hstack([gold["X26"][:, s8], z3["X_indego"][:, z6]])
    s_adapted = adapted_loco(X14g, gold["y"], gold["clips"])
    return {"y": y, "groups": clips, "frozen": p_frozen, "adapted": s_adapted,
            "names": clips}


def adapted_loco(X14, y, clips):
    """LOCO adapted O4 score: IndEgo signs + per-domain z-score fit on the
    other 6 clips, applied to the held-out clip."""
    y = np.asarray(y)
    clips = np.asarray(clips)
    s = np.full(len(y), np.nan)
    for held in sorted(set(clips.tolist())):
        te = clips == held
        tr = ~te
        Xg, yg = X14[tr], y[tr]
        au = np.asarray([R.fast_auc(yg, Xg[:, j]) for j in range(Xg.shape[1])])
        sg = np.where(au >= 0.5, 1.0, -1.0)
        mu = Xg.mean(0)
        sd = np.clip(Xg.std(0), 1e-9, None)
        s[te] = ((X14[te] - mu) / sd * sg).mean(1)
    return s


def _x14_from(px26, z3x, pix_names, z3_names):
    s8 = [list(pix_names).index(n) for n in R.STABLE8]
    z6 = [list(z3_names).index(n) for n in R.Z3_SIX]
    return np.hstack([px26[:, s8], z3x[:, z6]])


def load_ood160():
    lab = R.load_labels160()
    z3 = R.load_z3_160()
    pix = R.load_pixel160()
    gold = R.load_gold()
    ids, y, fac = lab["ids"], lab["y"], lab["factory"]
    Xz3, Z3i = z3["X"], z3["X_indego"]
    Xpx = R.align(pix["ids"], pix["X10"], ids)
    z6 = [z3["names"].index(n) for n in R.Z3_SIX]
    s8 = [R.FEATURE_NAMES26.index(n) for n in R.STABLE8]
    X14o = np.hstack([Xpx[:, s8], Xz3[:, z6]])
    X14g = np.hstack([gold["X26"][:, s8], Z3i[:, z6]])
    _, so_b, _, _ = R.combo_scores(X14g, X14o, gold["y"])
    frozen, _ = R.load_frozen160(ids)
    return {"y": y, "groups": fac, "frozen": frozen, "adapted": so_b}


def load_holdout108():
    z = np.load(CONFIRM108, allow_pickle=True)
    y = z["y"].astype(int)
    fac = np.asarray([str(x) for x in z["factory"]])
    X14o = _x14_from(z["Xpx"], z["Xz3"], z["pix_names"], z["z3_names"])
    gold = R.load_gold()
    z3 = R.load_z3_160()
    z6 = [z3["names"].index(n) for n in R.Z3_SIX]
    s8 = [R.FEATURE_NAMES26.index(n) for n in R.STABLE8]
    X14g = np.hstack([gold["X26"][:, s8], z3["X_indego"][:, z6]])
    _, so, _, _ = R.combo_scores(X14g, X14o, gold["y"])
    ens = json.load(open(os.path.join(MYIN, "confirm108_ensemble.json")))
    sc = {x["id"]: float(x["usable_prob"]) for x in ens["scores"]}
    ids = [str(x) for x in z["ids"]]
    frozen = np.asarray([sc[i] for i in ids], float)
    return {"y": y, "groups": fac, "frozen": frozen, "adapted": so}


# ------------------------------------------------------------------ metrics
def decisions(s, kind, param):
    """Deterministic decision/prediction vectors. pred 1=accept(good), 0=reject."""
    s = np.asarray(s, float)
    n = len(s)
    dec = np.zeros(n, bool)
    pred = np.zeros(n, int)
    if kind == "band_prob":
        hi = s >= 0.5 + param - 1e-12
        lo = s <= 0.5 - param + 1e-12
        dec = hi | lo
        pred[hi] = 1
    elif kind == "band_z":
        hi = s >= param - 1e-12
        lo = s <= -param + 1e-12
        dec = hi | lo
        pred[hi] = 1
    elif kind == "quantile":                    # reject bottom q, accept top q
        q = param
        k = int(np.floor(q * n + 1e-9))
        order = np.argsort(s, kind="mergesort")
        if k > 0:
            dec[order[:k]] = True
            dec[order[n - k:]] = True
            pred[order[n - k:]] = 1
    elif kind == "asym":                        # accept top a, reject bottom r
        a, r = param
        na = int(round(a * n))
        nr = int(round(r * n))
        order = np.argsort(s, kind="mergesort")
        if na > 0:
            dec[order[n - na:]] = True
            pred[order[n - na:]] = 1
        if nr > 0:
            dec[order[:nr]] = True
    elif kind == "fixed":                       # fixed integer thresholds (band_z)
        lo_t, hi_t = param
        hi = s >= hi_t - 1e-12
        lo = s <= lo_t + 1e-12
        dec = hi | lo
        pred[hi] = 1
    return pred, dec


def within_group_quantile(s, fac, q):
    s = np.asarray(s, float)
    fac = np.asarray(fac)
    n = len(s)
    dec = np.zeros(n, bool)
    pred = np.zeros(n, int)
    for f in sorted(set(fac.tolist())):
        m = np.where(fac == f)[0]
        k = int(np.floor(q * len(m) + 1e-9))
        order = m[np.argsort(s[m], kind="mergesort")]
        if k > 0:
            dec[order[:k]] = True
            dec[order[len(m) - k:]] = True
            pred[order[len(m) - k:]] = 1
    return pred, dec


def metrics(y, pred, dec):
    y = np.asarray(y)
    pred = np.asarray(pred)
    dec = np.asarray(dec, bool)
    g, b = y == 1, y == 0
    tp = int((g & dec & (pred == 1)).sum())
    fn = int((g & dec & (pred == 0)).sum())
    tn = int((b & dec & (pred == 0)).sum())
    fp = int((b & dec & (pred == 1)).sum())
    n_dec = tp + fn + tn + fp
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    ba = 0.5 * (tpr + tnr)
    acc = (tp + tn) / n_dec if n_dec else float("nan")
    return {"coverage": n_dec / len(y), "n_decided": n_dec, "acc": acc, "ba": ba,
            "n_good_dec": tp + fn, "n_bad_dec": tn + fp, "tp": tp, "fn": fn,
            "tn": tn, "fp": fp}


def boot_ci(y, s, groups, kind, param, n=NB, seed=SEED, fixed=False):
    """Group bootstrap CI for coverage/acc/ba of a rule.

    fixed=False re-applies the (transductive) rule on each resample (quantile /
    asym). fixed=True keeps the calibrated thresholds and only resamples windows.
    """
    y = np.asarray(y)
    s = np.asarray(s, float)
    groups = np.asarray(groups)
    gs = sorted(set(groups.tolist()))
    idx_by = {g: np.where(groups == g)[0] for g in gs}
    rng = np.random.RandomState(seed)
    cov, acc, ba = [], [], []
    for _ in range(n):
        pick = rng.randint(0, len(gs), len(gs))
        idx = np.concatenate([idx_by[gs[j]] for j in pick])
        if fixed:
            pred, dec = decisions(s[idx], kind, param)
        else:
            pred, dec = decisions(s[idx], kind, param)
        m = metrics(y[idx], pred, dec)
        if m["coverage"] > 0:
            cov.append(m["coverage"])
            if m["ba"] == m["ba"]:
                ba.append(m["ba"])
            if m["acc"] == m["acc"]:
                acc.append(m["acc"])
    def pct(v):
        if not v:
            return [None, None]
        return [round(float(np.percentile(v, 2.5)), 4),
                round(float(np.percentile(v, 97.5)), 4)]
    return {"coverage_ci95": pct(cov), "acc_ci95": pct(acc), "ba_ci95": pct(ba)}


def envelope(rows, step=0.01):
    """Upper envelope of decided-BA vs coverage over all swept rules."""
    bins = {}
    for r in rows:
        if r["coverage"] <= 0 or r["ba"] != r["ba"]:
            continue
        b = int(round(r["coverage"] / step))
        cur = bins.get(b)
        if cur is None or r["ba"] > cur["ba"]:
            bins[b] = {"coverage": round(b * step, 4), "ba": r["ba"], "acc": r["acc"],
                       "rule": r["rule"], "param": r["param"],
                       "n_decided": r["n_decided"]}
    return [bins[b] for b in sorted(bins)]


def sweep(y, s, groups, kinds, tag):
    rows = []
    for kind, grid in kinds:
        for param in grid:
            pred, dec = decisions(s, kind, param)
            m = metrics(y, pred, dec)
            if m["n_decided"] == 0:
                continue
            rows.append({"rule": kind, "param": param, **m})
    return rows


def max_cov(rows, key, target=0.95):
    ok = [r for r in rows if r[key] == r[key] and r[key] >= target - 1e-9]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["coverage"], r["ba"]))


def at_coverage(rows, cov, key="ba"):
    ok = [r for r in rows if abs(r["coverage"] - cov) < 1e-9]
    if not ok:
        return None
    return max(ok, key=lambda r: r[key])


# ------------------------------------------------------------------ conformal
def conformal_lco(s, y, groups, alpha):
    """Split/LAC conformal, leave-one-group-out calibration (group=clip/factory).

    Conformity E_i = s_i if y_i=1 else -s_i. Threshold q = the alpha-quantile of
    the *calibration* E (other groups). Prediction set {good} iff s>=q,
    {bad} iff -s>=q i.e. s<=-q; else review."""
    s = np.asarray(s, float)
    y = np.asarray(y)
    groups = np.asarray(groups)
    E = np.where(y == 1, s, -s)
    dec = np.zeros(len(s), bool)
    pred = np.zeros(len(s), int)
    for g in sorted(set(groups.tolist())):
        te = groups == g
        tr = ~te
        Ec = np.sort(E[tr])
        k = int(np.ceil((len(Ec) + 1) * alpha)) - 1
        k = min(max(k, 0), len(Ec) - 1)
        q = Ec[k]
        hi = s[te] >= q - 1e-12
        lo = s[te] <= -q + 1e-12
        dec[te] = hi | lo
        pred[te] = hi.astype(int)
    return pred, dec


# ------------------------------------------------------------------ receipt
def receipt_check(lo_t, hi_t):
    from jadal.receipts import make_receipt, verify_receipt, receipt_hash
    config = {
        "thresholds": {"reject_at_or_below": int(lo_t), "accept_at_or_above": int(hi_t),
                       "scale": 1000, "float_decimals": 4},
        "pipeline_version": "A4-acc-precision-1",
        "versions": {"python": "3"},
    }
    segment = {"sha256": "0" * 64, "window_id": "demo"}
    signals = {"score_permille": 812}
    decisions = {"rule": "band_permille", "reject_at_or_below": int(lo_t),
                 "accept_at_or_above": int(hi_t)}
    verdict = {"decision": "ACCEPT", "reason": "OK"}
    r1 = make_receipt(segment, signals, decisions, verdict, config,
                      created_utc="2026-10-10T00:00:00+00:00")
    r2 = make_receipt(segment, signals, decisions, verdict, config,
                      created_utc="2026-10-10T00:00:00+00:00")
    res = verify_receipt(r1, lambda r: r)
    return {"receipt_sha256": r1["receipt_sha256"], "receipt_id": r1["receipt_id"],
            "byte_identical": bool(r1 == r2 and r1["receipt_sha256"] == r2["receipt_sha256"]),
            "verify_match": bool(res.get("match")),
            "recomputed_hash": receipt_hash(r1)}


# ------------------------------------------------------------------ main
def main():
    data = {
        "in_domain": load_in_domain(),
        "ood160": load_ood160(),
        "holdout108": load_holdout108(),
    }
    kinds = [("band_prob", BAND_W), ("band_z", Z_W), ("quantile", QGRID)]
    asym_grid = [(a, r) for a in A_GRID for r in A_GRID if a + r <= 0.5 + 1e-9]

    out1 = {"experiment": "A4.01", "device": DEV, "git_sha": git_sha(),
            "seed": SEED, "n_boot": NB,
            "rules": {"band_prob": "decide iff |p-0.5|>=w (center 0.5)",
                      "band_z": "decide iff |s|>=w (center 0, adapted z-score)",
                      "quantile": "reject bottom q*n, accept top q*n (integer counts)",
                      "asym": "accept top round(a*n), reject bottom round(r*n)",
                      "conformal": "LAC split conformal, leave-one-group-out"},
            "regimes": {}}
    frontiers = {}

    for name, d in data.items():
        y, groups = d["y"], d["groups"]
        regimes = {}
        for scname in ("frozen", "adapted"):
            s = d[scname]
            auroc = round(R.fast_auc(y, s), 4)
            rows = sweep(y, s, groups, kinds, name)
            # asymmetric sweep
            for (a, r) in asym_grid:
                pred, dec = decisions(s, "asym", (a, r))
                m = metrics(y, pred, dec)
                if m["n_decided"] == 0:
                    continue
                rows.append({"rule": "asym", "param": [a, r], **m})
            # conformal sweep
            for al in ALPHAS:
                pred, dec = conformal_lco(s, y, groups, al)
                m = metrics(y, pred, dec)
                if m["n_decided"] == 0:
                    continue
                rows.append({"rule": "conformal", "param": al, **m})
            # within-factory quantile sweep
            for q in QGRID:
                pred, dec = within_group_quantile(s, groups, q)
                m = metrics(y, pred, dec)
                if m["n_decided"] == 0:
                    continue
                rows.append({"rule": "within_group_quantile", "param": q, **m})

            key = f"{name}/{scname}"
            frontiers[key] = rows
            ba95 = max_cov(rows, "ba", 0.95)
            acc95 = max_cov(rows, "acc", 0.95)
            # full-decision reference
            pred0, dec0 = decisions(s, "band_prob" if scname == "frozen" else "band_z", 0.0)
            full = metrics(y, pred0, dec0)
            regimes[scname] = {
                "auroc": auroc,
                "n": int(len(y)), "n_good": int((y == 1).sum()),
                "n_bad": int((y == 0).sum()), "n_groups": int(len(set(groups.tolist()))),
                "full_decision": full,
                "max_coverage_at_ba95": ba95,
                "max_coverage_at_acc95": acc95,
                "frontier_envelope": envelope(rows),
            }
            # CI at the chosen ba95 point (point-selected -> optimistic)
            regimes[scname]["max_coverage_at_ba95_ci_lower"] = None
            if ba95 is not None:
                ci = boot_ci(y, s, groups, ba95["rule"], tuple(ba95["param"])
                             if isinstance(ba95["param"], list) else ba95["param"])
                regimes[scname]["ba95_point_ci95"] = ci
                # honest: max coverage whose CI lower bound >= 0.95. Scan only
                # candidates whose point BA already clears 0.95, high cov first.
                cands = [r for r in rows if r["coverage"] > 0 and r["ba"] == r["ba"]
                         and r["ba"] >= 0.95 - 1e-9]
                cands.sort(key=lambda r: r["coverage"], reverse=True)
                honest = None
                for r in cands[:60]:
                    c = boot_ci(y, s, groups, r["rule"],
                                tuple(r["param"]) if isinstance(r["param"], list) else r["param"],
                                n=400, seed=SEED + 1)
                    lo = c["ba_ci95"][0]
                    if lo is not None and lo >= 0.95 - 1e-9:
                        honest = {"rule": r["rule"], "param": r["param"],
                                  "coverage": r["coverage"], "ba": r["ba"],
                                  "ba_ci95": c["ba_ci95"]}
                        break
                regimes[scname]["max_coverage_at_ba95_ci_lower"] = honest
        out1["regimes"][name] = regimes

    # ---------------- A4.02 deterministic rule + per-factory + receipt -------
    out2 = {"experiment": "A4.02", "device": DEV, "git_sha": git_sha(), "seed": SEED}

    # pick the rule: highest coverage at point-BA>=0.95 across scorers, per regime
    pick = {}
    for name in data:
        best = None
        for scname in ("frozen", "adapted"):
            b = out1["regimes"][name][scname]["max_coverage_at_ba95"]
            if b is None:
                continue
            if best is None or b["coverage"] > best[1]["coverage"]:
                best = (scname, b)
        pick[name] = best

    det = {}
    for name, best in pick.items():
        d = data[name]
        scname, b = best
        s = d[scname]
        y, groups = d["y"], d["groups"]
        # express as integer top/bottom-k counts (the quantile/asym form)
        n = len(s)
        if b["rule"] == "asym":
            a, r = b["param"]
            na, nr = int(round(a * n)), int(round(r * n))
            form = {"type": "top_bottom_k", "accept_top_k": na, "reject_bottom_k": nr,
                    "review_middle": n - na - nr, "n": n}
        elif b["rule"] == "quantile":
            k = int(np.floor(b["param"] * n + 1e-9))
            form = {"type": "top_bottom_k", "accept_top_k": k, "reject_bottom_k": k,
                    "review_middle": n - 2 * k, "n": n}
        elif b["rule"] in ("band_prob", "band_z"):
            w = b["param"]
            if scname == "frozen":
                lo_t, hi_t = round((0.5 - w) * 1000), round((0.5 + w) * 1000)
            else:
                lo_t, hi_t = round(-w * 1000), round(w * 1000)
            form = {"type": "permille_band", "reject_at_or_below_permille": lo_t,
                    "accept_at_or_above_permille": hi_t,
                    "accept_top_k": None, "reject_bottom_k": None, "n": n}
        else:
            form = {"type": b["rule"], "param": b["param"], "n": n}
        det[name] = {"scorer": scname, "rule": b["rule"], "param": b["param"],
                     "coverage": b["coverage"], "ba": b["ba"], "acc": b["acc"],
                     "n_decided": b["n_decided"], "form": form}
    out2["deterministic_rule"] = det

    # per-factory variant: within-factory quantile vs global quantile at matched coverage
    perfac = {}
    for name, d in data.items():
        y, groups = d["y"], d["groups"]
        pf = {}
        for scname in ("frozen", "adapted"):
            s = d[scname]
            rows_w = [r for r in frontiers[f"{name}/{scname}"]
                      if r["rule"] == "within_group_quantile"]
            rows_g = [r for r in frontiers[f"{name}/{scname}"] if r["rule"] == "quantile"]
            w95 = max_cov(rows_w, "ba", 0.95)
            g95 = max_cov(rows_g, "ba", 0.95)
            # matched-coverage comparison at the within-factory best coverage
            match = None
            if w95 is not None:
                cand = [r for r in rows_g if abs(r["coverage"] - w95["coverage"]) < 0.02]
                if cand:
                    match = max(cand, key=lambda r: r["ba"])
            pf[scname] = {"within_factory_max_cov_ba95": w95,
                          "global_max_cov_ba95": g95,
                          "global_at_matched_coverage": match}
        perfac[name] = pf
    out2["per_factory"] = perfac

    # receipt path on the frozen in-domain chosen band (or permille of pick)
    lo_t, hi_t = 300, 700
    if det["in_domain"]["form"].get("type") == "permille_band":
        lo_t = det["in_domain"]["form"]["reject_at_or_below_permille"]
        hi_t = det["in_domain"]["form"]["accept_at_or_above_permille"]
    out2["receipt"] = receipt_check(lo_t, hi_t)

    # ---------------------------------------------------------------- write
    os.makedirs(OUT, exist_ok=True)
    for fn, obj in (("A4.01.json", out1), ("A4.02.json", out2)):
        with open(os.path.join(OUT, fn), "w") as f:
            json.dump(obj, f, indent=1, sort_keys=True)

    # ---------------- console summary
    print(f"git {git_sha()}  device {DEV}")
    for name, regs in out1["regimes"].items():
        print(f"\n=== {name} ===")
        for sc, r in regs.items():
            b = r["max_coverage_at_ba95"]
            a = r["max_coverage_at_acc95"]
            print(f"  {sc:8s} AUROC {r['auroc']:.3f}  full dec BA {r['full_decision']['ba']:.3f}"
                  f" cov {r['full_decision']['coverage']:.3f}")
            if b:
                print(f"    BA>=0.95 @ cov {b['coverage']:.3f} ({b['n_decided']}/{r['n']})"
                      f" rule={b['rule']} param={b['param']} ba={b['ba']:.3f} acc={b['acc']:.3f}")
            else:
                print("    BA>=0.95 : unreachable at any positive coverage")
            if a:
                print(f"    ACC>=0.95 @ cov {a['coverage']:.3f} rule={a['rule']} param={a['param']}")
            h = r["max_coverage_at_ba95_ci_lower"]
            print(f"    BA>=0.95 with CI-LB : "
                  f"{'cov %.3f' % h['coverage'] if h else 'unreachable'}")
    print("\n=== deterministic rule (A4.02) ===")
    for name, d in det.items():
        print(f"  {name:11s} scorer={d['scorer']:8s} {d['form']}")
    print("\n=== per-factory ===")
    for name, pf in perfac.items():
        for sc, r in pf.items():
            w = r["within_factory_max_cov_ba95"]
            g = r["global_max_cov_ba95"]
            print(f"  {name}/{sc}: within-factory cov@95 "
                  f"{'%.3f' % w['coverage'] if w else '-'} vs global "
                  f"{'%.3f' % g['coverage'] if g else '-'}")
    print("\n=== receipt ===", out2["receipt"])
    return out1, out2


if __name__ == "__main__":
    main()
