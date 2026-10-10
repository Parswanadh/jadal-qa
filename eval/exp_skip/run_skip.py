#!/usr/bin/env python3
"""SKIP (Q2): can we skip *sections anywhere* (middle/end/evenly) and still get the
same clip verdict at lower cost?  Extends E3 (whose "skip the start" was NOT supported).

Protocol: per-clip subsampling simulation on the 150 human-gold IndEgo windows,
ordered by start_s within clip.  Ground truth = bad fraction over ALL gold windows
of the clip ("full coverage").  Threshold = 0.30 (bad fraction) for the clip verdict.

Three sub-experiments:
  SKIP.01  section-drop rules (drop 1 or 2 of 5 equal sections; START/MIDDLE/END/EVENLY)
  SKIP.02  stratified sparse probe (first window of each kept section) vs the same
           number of contiguous windows; + dense-where-uncertain (densify the section
           whose probe score is closest to threshold)
  SKIP.03  best-cell selection (cheapest policy with verdict agreement >= 0.90 and
           bad-fraction error <= 0.05) + per-clip detail

CPU-only: uses cached labels + caches/ensemble/oof_probs.npz (p_ens). No GPU.

Regen:
  OMP_NUM_THREADS=2 python3 eval/exp_skip/run_skip.py --out eval/results/exp_skip
"""
import argparse, json, os
import numpy as np

MANIFEST = "/workspace/jadal_pod_bundle/labels/manifest_gold.json"
OOF = "/workspace/jadal_pod_bundle/caches/ensemble/oof_probs.npz"
THRESH = 0.30          # primary clip verdict threshold (required by E3/Q2)
THRESH2 = 0.50         # discriminative secondary (all clips >0.30 -> degenerate)
NSEC = 5


# ------------------------------------------------------------------ data
def load_gold():
    m = json.load(open(MANIFEST))
    oof = np.load(OOF, allow_pickle=True)
    assert list(oof["ids"]) == [it["id"] for it in m["items"]], "oof order != manifest"
    clips = {}
    for i, it in enumerate(m["items"]):
        clips.setdefault(it["clip"], []).append(i)
    starts = np.array([it["start_s"] for it in m["items"]], dtype=float)
    for c in clips:                          # within-clip order = time order
        clips[c] = sorted(clips[c], key=lambda i: starts[i])
    y = np.array([it["label"] for it in m["items"]], int)   # 1 = good
    p = np.asarray(oof["p_ens"], float)                     # P(good)
    return m, clips, y, 1 - y, p


def clip_bootstrap_ci(vals, n_rep=2000, seed=11):
    rng = np.random.RandomState(seed)
    v = np.asarray(vals, float); n = len(v)
    b = [float(np.mean(rng.choice(v, n, replace=True))) for _ in range(n_rep)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


# ------------------------------------------------------------------ policies
def sections(n):
    """5 equal sections as within-clip position arrays."""
    return [s for s in np.array_split(np.arange(n), NSEC) if len(s)]


def _keep(n, drop_pos):
    drop = set(int(i) for i in drop_pos)
    return sorted(i for i in range(n) if i not in drop)


def drop_sections(n, k, rule, seed=0):
    if k <= 0:
        return list(range(n))
    ss = sections(n)
    k = min(k, len(ss))
    if rule == "start":
        drop = np.concatenate(ss[:k])
    elif rule == "end":
        drop = np.concatenate(ss[len(ss) - k:])
    elif rule == "middle":
        c = [(s.mean() / (n - 1)) if n > 1 else 0.5 for s in ss]
        which = sorted(range(len(ss)), key=lambda i: (abs(c[i] - 0.5), i))[:k]
        drop = np.concatenate([ss[i] for i in which])
    elif rule == "evenly":
        res = {(seed + j) % NSEC for j in range(k)}      # evenly spaced residue classes
        drop = [i for i in range(n) if (i % NSEC) in res]
    else:
        raise ValueError(rule)
    return _keep(n, drop)


def strat_probe(n, k=0, rule="start", seed=0, first_only=True):
    """Probe first window of each kept section (k=0 -> keep all sections)."""
    kept = set(drop_sections(n, k, rule, seed))
    ss = sections(n)
    S = [int(s[0]) for s in ss if int(s[0]) in kept]
    if not first_only:                                   # probe whole kept sections
        S = sorted(kept)
    return sorted(set(S))


def contig_probe(n, m):
    return sorted(set(range(min(m, n))))                 # first m contiguous windows


def dense_uncertain(n, p_bad, base_k=0, base_rule="start", base_seed=0,
                    n_densify=1, target=THRESH):
    """Sparse stratified probe, then fully process the section(s) whose probe score
    (model P(bad)) is closest to the decision threshold.  Returns probed positions,
    and records which sections were densified."""
    ss = sections(n)
    keep = drop_sections(n, base_k, base_rule, base_seed)
    keep_set = set(keep)
    S = set(int(s[0]) for s in ss if int(s[0]) in keep_set)
    # per-section probe score from currently probed windows
    def sec_score(sec):
        idx = [i for i in sec if i in S]
        return float(np.mean([p_bad[i] for i in idx])) if idx else 0.5
    order = sorted(range(len(ss)), key=lambda j: (abs(sec_score(ss[j]) - target), j))
    dens = []
    for j in order[:n_densify]:
        for i in ss[j]:
            if i in keep_set:
                S.add(int(i)); 
        dens.append(j)
    return sorted(S), dens


# ------------------------------------------------------------------ evaluation
def eval_clip(bad, p_good, S, thresh=THRESH):
    n = len(bad)
    S = sorted(set(S))
    frac = len(S) / n
    truth = float(np.mean(bad))
    tv = "reject" if truth >= thresh else "accept"
    est_L = float(np.mean(bad[S]))
    est_M = float(np.mean(1.0 - p_good[S]))
    return dict(frac=frac, truth=truth, truth_v=tv, n_probed=len(S), n=n,
                est_L=est_L, ver_L="reject" if est_L >= thresh else "accept",
                err_L=abs(est_L - truth), agree_L=int(("reject" if est_L >= thresh else "accept") == tv),
                est_M=est_M, ver_M="reject" if est_M >= thresh else "accept",
                err_M=abs(est_M - truth), agree_M=int(("reject" if est_M >= thresh else "accept") == tv))


def agg(rows, thresh=THRESH):
    eL = [r["err_L"] for r in rows]; eM = [r["err_M"] for r in rows]
    return dict(
        mean_frac=float(np.mean([r["frac"] for r in rows])),
        mean_err_L=float(np.mean(eL)), ci_L=clip_bootstrap_ci(eL),
        mean_err_M=float(np.mean(eM)), ci_M=clip_bootstrap_ci(eM),
        agree_L=float(np.mean([r["agree_L"] for r in rows])),
        agree_M=float(np.mean([r["agree_M"] for r in rows])),
        max_err_L=float(np.max(eL)), n_agree_L=int(sum(r["agree_L"] for r in rows)),
        n_clips=len(rows))


def policy_rows(policy_fn, clips, bad, p, thresh=THRESH):
    rows = []
    for c in sorted(clips):
        idx = clips[c]
        S = policy_fn(len(idx), p[idx])
        r = eval_clip(bad[idx], p[idx], S, thresh)
        r["clip"] = c
        rows.append(r)
    return rows


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/exp_skip")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m, clips, y, bad, p = load_gold()
    p_bad = 1.0 - p

    # ---- section position profile (descriptive; is bad concentrated anywhere?) ----
    prof = np.zeros(NSEC); cnt = np.zeros(NSEC); sec_rows = []
    for c in sorted(clips):
        idx = clips[c]; b = bad[idx]
        for j, s in enumerate(np.array_split(np.arange(len(b)), NSEC)):
            if len(s):
                prof[j] += float(np.mean(b[s])); cnt[j] += 1
                sec_rows.append(dict(clip=c, sec=j, n=len(s), bad_frac=float(np.mean(b[s]))))
    profile = [dict(section=j + 1, mean_bad_frac=float(prof[j] / cnt[j]), n_clips=int(cnt[j]))
               for j in range(NSEC)]
    # weighted bad rate per section (window-weighted)
    wprof = np.zeros(NSEC); wcnt = np.zeros(NSEC)
    for r in sec_rows:
        wprof[r["sec"]] += r["bad_frac"] * r["n"]; wcnt[r["sec"]] += r["n"]

    # ================= SKIP.01: section-drop rules =================
    drop_pols = {"full": lambda n, pb: list(range(n))}
    for k in (1, 2):
        for rule in ("start", "middle", "end"):
            drop_pols[f"drop{k}-{rule}"] = (lambda n, pb, k=k, r=rule: drop_sections(n, k, r))
        for seed in (0, 1, 2):
            drop_pols[f"drop{k}-evenly-s{seed}"] = (lambda n, pb, k=k, s=seed: drop_sections(n, k, "evenly", s))
    skip01 = {}
    for name, fn in drop_pols.items():
        rows = policy_rows(fn, clips, bad, p)
        skip01[name] = agg(rows)
        skip01[name]["per_clip"] = {r["clip"]: dict(frac=round(r["frac"], 3),
                                                    err_L=round(r["err_L"], 3),
                                                    ver_L=r["ver_L"], agree_L=r["agree_L"])
                                    for r in rows}

    # ================= SKIP.02: stratified vs contiguous + dense =================
    skip02 = {}
    nsec_med = int(round(np.median([len(sections(len(clips[c]))) for c in clips])))
    # stratified probe, all sections
    r0 = policy_rows(lambda n, pb: strat_probe(n, 0, "start"), clips, bad, p)
    skip02["strat-probe(1/section)"] = agg(r0)
    m_used = int(round(np.mean([len(strat_probe(len(clips[c]), 0, "start")) for c in clips])))
    r1 = policy_rows(lambda n, pb: contig_probe(n, m_used), clips, bad, p)
    skip02[f"contig-probe(first {m_used})"] = agg(r1)
    # stratified after dropping 1 & 2 sections (start/middle/end/evenly)
    for k in (1, 2):
        for rule in ("start", "middle", "end"):
            name = f"strat-after-drop{k}-{rule}"
            skip02[name] = agg(policy_rows(lambda n, pb, k=k, r=rule: strat_probe(n, k, r), clips, bad, p))
        for seed in (0, 1, 2):
            name = f"strat-after-drop{k}-evenly-s{seed}"
            skip02[name] = agg(policy_rows(lambda n, pb, k=k, r="evenly", s=seed: strat_probe(n, k, r, s), clips, bad, p))
    # dense-where-uncertain
    for label, kw in {"dense-from-strat(x1)": dict(base_k=0, n_densify=1),
                      "dense-from-strat(x2)": dict(base_k=0, n_densify=2),
                      "dense-after-drop1-start(x1)": dict(base_k=1, base_rule="start", n_densify=1)}.items():
        rows = []
        for c in sorted(clips):
            idx = clips[c]
            S, dens = dense_uncertain(len(idx), p_bad[idx], **kw)
            r = eval_clip(bad[idx], p[idx], S)
            r["clip"] = c; r["densified"] = [d + 1 for d in dens]
            rows.append(r)
        skip02[label] = agg(rows)
        skip02[label]["per_clip_densified"] = {r["clip"]: r["densified"] for r in rows}
    # equal-budget uniform reference: every-k grids
    for k in (2, 3, 4, 5):
        skip02[f"uniform-every-{k}"] = agg(policy_rows(lambda n, pb, k=k: list(range(0, n, k)), clips, bad, p))

    # ================= SKIP.03: best-cell + per-clip =================
    allpol = {}
    for name, v in skip01.items():      allpol[name] = ("SKIP.01", v)
    for name, v in skip02.items():
        if isinstance(v, dict) and "mean_frac" in v: allpol[name] = ("SKIP.02", v)
    # representative per-clip detail at 0.30 and 0.50 for key policies
    keysets = {
        "full": lambda n, pb: list(range(n)),
        "drop1-start": lambda n, pb: drop_sections(n, 1, "start"),
        "drop1-middle": lambda n, pb: drop_sections(n, 1, "middle"),
        "drop1-end": lambda n, pb: drop_sections(n, 1, "end"),
        "drop1-evenly-s0": lambda n, pb: drop_sections(n, 1, "evenly", 0),
        "drop2-start": lambda n, pb: drop_sections(n, 2, "start"),
        "drop2-middle": lambda n, pb: drop_sections(n, 2, "middle"),
        "drop2-end": lambda n, pb: drop_sections(n, 2, "end"),
        "drop2-evenly-s0": lambda n, pb: drop_sections(n, 2, "evenly", 0),
        "strat-probe(1/section)": lambda n, pb: strat_probe(n, 0, "start"),
        "uniform-every-2": lambda n, pb: list(range(0, n, 2)),
    }
    detail = {}
    for name, fn in keysets.items():
        d = {}
        for c in sorted(clips):
            idx = clips[c]
            S = fn(len(idx), p[idx])
            r30 = eval_clip(bad[idx], p[idx], S, THRESH)
            r50 = eval_clip(bad[idx], p[idx], S, THRESH2)
            d[c] = dict(n=len(idx), frac=round(r30["frac"], 3), truth=round(r30["truth"], 3),
                        est_L=round(r30["est_L"], 3), est_M=round(r30["est_M"], 3),
                        ver30=r30["ver_L"], ver50=r50["ver_L"],
                        agree30=r30["agree_L"], agree50=r50["agree_L"], err_L=round(r30["err_L"], 3))
        detail[name] = d
    # pick best cell
    cands = []
    for name, (src, v) in allpol.items():
        cands.append((name, src, v["mean_frac"], v["mean_err_L"], v["ci_L"], v["agree_L"]))
    ok = [c for c in cands if c[5] >= 0.90 and c[3] <= 0.05]
    ok.sort(key=lambda c: c[2])
    best = ok[0] if ok else None
    # also best at <=0.07 / <=0.10 tolerances (robustness)
    def best_at(err_tol, agree_min=0.90):
        o = sorted([c for c in cands if c[5] >= agree_min and c[3] <= err_tol], key=lambda c: c[2])
        return o[0] if o else None
    frontier = sorted([dict(policy=c[0], source=c[1], frac=round(c[2], 4),
                            err_L=round(c[3], 4), ci=c[4], agree_L=c[5]) for c in cands],
                      key=lambda d: d["frac"])

    meta = dict(
        n_windows=len(y), n_good=int(y.sum()), n_bad=int((1 - y).sum()), n_clips=len(clips),
        label_source="IndEgo human gold 150 windows / 7 clips (labels/manifest_gold.json)",
        **{"clip_bad_verdict_at_0.30": {c: ("reject" if np.mean(bad[clips[c]]) >= THRESH else "accept")
                                        for c in sorted(clips)},
           "clip_truth_bad_frac": {c: round(float(np.mean(bad[clips[c]])), 4) for c in sorted(clips)}},
        note=("Clip truth bad fractions: desk 0.45, ecu_full 0.10, ecu_late 0.29, logistics4 0.18, "
              "organise 0.53, pallet 0.65, pallet2 0.67 -> at the required 0.30 threshold 4 clips "
              "reject / 3 accept, so verdict agreement is discriminative but thin (n=7 clips). "
              "A secondary threshold 0.50 (2 reject / 5 accept) is also reported."),
        thresh=THRESH, thresh_secondary=THRESH2, n_sections=NSEC)

    skip01_out = dict(meta=meta, policy="section-drop rules (drop 1 or 2 of 5 sections)",
                      protocol="per-clip subsampling on gold-window time order; truth = bad fraction over all gold windows; CI = clip bootstrap (2000)",
                      section_position_profile=profile,
                      section_position_profile_window_weighted=[float(wprof[j] / wcnt[j]) for j in range(NSEC)],
                      per_policy=skip01)
    skip02_out = dict(meta=meta, policy="stratified probe / contiguous probe / dense-where-uncertain",
                      protocol="stratified probe = first window of each kept section; contig = first m windows; dense = fully process section with probe score closest to 0.30",
                      per_policy=skip02)
    skip03_out = dict(meta=meta, policy="best-cell selection + per-clip detail",
                      protocol="select cheapest policy with mean verdict agreement@0.30 >= 0.90 and mean bad-fraction error <= 0.05",
                      best_cell=dict(policy=best[0], source=best[1], mean_frac=round(best[2], 4),
                                     mean_err_L=round(best[3], 4), ci_L=best[4], agree_L=best[5]) if best else None,
                      best_at_agree90_err05=best,
                      best_at_agree90_err07=best_at(0.07), best_at_agree90_err10=best_at(0.10),
                      frontier_by_cost=frontier, per_clip_detail=detail)

    for fname, obj in (("SKIP.01.json", skip01_out),
                       ("SKIP.02.json", skip02_out),
                       ("SKIP.03.json", skip03_out)):
        json.dump(obj, open(os.path.join(a.out, fname), "w"), indent=1)
        print("wrote", os.path.join(a.out, fname))

    # ------- console summary -------
    print("\n=== section position profile (mean bad frac by section) ===")
    for r in profile: print(f"  sec{r['section']}: {r['mean_bad_frac']:.3f}  (n_clips={r['n_clips']})")
    print("\n=== SKIP.01 section-drop ===")
    for name in ["full", "drop1-start", "drop1-middle", "drop1-end", "drop1-evenly-s0",
                 "drop2-start", "drop2-middle", "drop2-end", "drop2-evenly-s0"]:
        v = skip01[name]; print(f"  {name:18s} frac={v['mean_frac']:.3f} err_L={v['mean_err_L']:.3f} "
                                f"CI={[round(x,3) for x in v['ci_L']]} agree_L={v['agree_L']:.2f}")
    print("\n=== SKIP.02 probe / dense ===")
    for name, v in skip02.items():
        if "mean_frac" in v: print(f"  {name:32s} frac={v['mean_frac']:.3f} err_L={v['mean_err_L']:.3f} agree_L={v['agree_L']:.2f}")
    print("\n=== SKIP.03 best cell ===")
    print("  best:", skip03_out["best_cell"])
    print("\nwrote SKIP.01..03.json")


if __name__ == "__main__":
    main()
