"""E3 temporal sampling policy: simulate window-subsampling policies per IndEgo clip.

Protocols:
  E3.01  sparse probe (every k-th), skip-first-M, random sparse, CUSUM/Page-Hinkley
          change-point-driven probing; label-oracle + OOF-model estimates.
  E3.02  OOD position bias: does position in the Egocentric-10K clip predict bad?
  E3.03  early exit within a window: EG2-prefix LOCO accuracy vs observed seconds.
  E3.04  compute accounting from E3.01/E3.03 fractions.

Regen:
  PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python \
    eval/exp_temporal/run_e3.py --out eval/results/exp_temporal
(run with cwd=/home/parshu/projects/jadal-qa-wt/exp-temporal)
"""
import argparse, json, os, sys
import numpy as np

RNG = np.random.RandomState(7)

MANIFEST = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"
OOF = "/home/parshu/projects/jadal-qa-wt/ensemble-views/eval/results/ensemble/oof_probs.npz"
SEG = "/home/parshu/projects/jadal-qa-wt/frames10k/eval/results/frames10k/human_segments/segments.npz"
Q1 = "/home/parshu/projects/jadal-qa-wt/scale-data/eval/results/scale/label_queue_random.json"
Q2 = "/home/parshu/projects/jadal-qa-wt/scale-data/eval/results/scale/label_queue_random2.json"
OOD1 = "/home/parshu/projects/jadal-qa-wt/ood_labels/labels.json"
OOD2 = "/home/parshu/projects/jadal-qa-wt/ood_labels/labels2.json"
MANIFEST_SCALE = "/home/parshu/projects/jadal-qa-wt/scale-data/eval/scale/manifest_scale.json"
THRESH = 0.30  # clip accept/reject bad-fraction threshold


def load_gold():
    m = json.load(open(MANIFEST))
    oof = np.load(OOF, allow_pickle=True)
    assert list(oof["ids"]) == [it["id"] for it in m["items"]], "oof order != manifest"
    clips = {}
    for i, it in enumerate(m["items"]):
        clips.setdefault(it["clip"], []).append(i)
    y = np.array([it["label"] for it in m["items"]], dtype=int)  # 1=good
    bad = 1 - y
    p = np.asarray(oof["p_ens"], dtype=float)  # P(good); verify orientation below
    starts = np.array([it["start_s"] for it in m["items"]], dtype=float)
    return m, clips, y, bad, p, starts


def check_orientation(y, p):
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, p)
    assert auc > 0.5, f"p_ens orientation wrong: AUROC={auc}"
    return auc


def clip_bootstrap_ci(vals, n_rep=2000, seed=11):
    rng = np.random.RandomState(seed)
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    boots = [float(np.mean(rng.choice(vals, size=n, replace=True))) for _ in range(n_rep)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------------- policies (each returns sorted probed index list, positions within clip order) ----------------
def pol_every_k(n, k):
    return list(range(0, n, k))

def pol_skip_first_M(starts, M):
    idx = [i for i, s in enumerate(starts) if s >= M]
    return idx if idx else [len(starts) - 1]

def pol_random(n, frac, seed):
    rng = np.random.RandomState(seed)
    m = max(1, int(round(n * frac)))
    return sorted(rng.choice(n, size=m, replace=False).tolist())

def pol_cusum(scores, k0=4, delta=0.10, h=0.50, max_run=3):
    """Sequential Page-Hinkley on probed P(good): sparse step k0, densify on
    downward alarm (toward bad). Genuinely sequential: only probed scores used."""
    n = len(scores)
    S, probed, mean, cnt = [], [], 0.0, 0
    cum = 0.0
    i = 0
    dens_left = 0
    while i < n:
        probed.append(i)
        p = float(scores[i])
        cnt += 1
        mean += (p - mean) / cnt
        cum = max(0.0, cum + (mean - p - delta))
        if cum > h:
            dens_left = max_run
            cum = 0.0
        if dens_left > 0:
            dens_left -= 1
            i += 1
        else:
            i += k0
    return sorted(set(probed))


def first_bad_run(bad):
    """start index of first maximal bad run, or None if clip all-good."""
    for i, b in enumerate(bad):
        if b == 1 and (i == 0 or bad[i - 1] == 0):
            return i
    return None


def eval_policy_clip(y_bad, p_good, S, thresh=THRESH):
    """Label-oracle (L) and model (M) estimates from probed set S."""
    n = len(y_bad)
    S = sorted(set(S))
    frac = len(S) / n
    truth_frac = float(np.mean(y_bad))
    truth_verdict = "reject" if truth_frac >= thresh else "accept"
    # L: oracle labels
    est_L = float(np.mean(y_bad[S]))
    ver_L = "reject" if est_L >= thresh else "accept"
    # M: model bad prob = 1 - P(good)
    est_M = float(np.mean(1.0 - p_good[S]))
    ver_M = "reject" if est_M >= thresh else "accept"
    # detection delay of first bad stretch (in windows; spacing ~14.6 s)
    run = first_bad_run(y_bad)
    if run is None:
        dL = dM = 0.0
        missL = missM = 0
    else:
        hitsL = [j for j in S if j >= run and y_bad[j] == 1]
        dL = float(hitsL[0] - run) if hitsL else float("inf")
        missL = 0 if hitsL else 1
        hitsM = [j for j in S if j >= run and (1.0 - p_good[j]) >= 0.5]
        dM = float(hitsM[0] - run) if hitsM else float("inf")
        missM = 0 if hitsM else 1
    return dict(frac=frac, truth_frac=truth_frac, truth_verdict=truth_verdict,
                est_L=est_L, ver_L=ver_L, err_L=abs(est_L - truth_frac),
                agree_L=int(ver_L == truth_verdict),
                est_M=est_M, ver_M=ver_M, err_M=abs(est_M - truth_frac),
                agree_M=int(ver_M == truth_verdict),
                delay_L=dL, miss_L=missL, delay_M=dM, miss_M=missM,
                n_probed=len(S), n=n)


def run_e301(outdir):
    m, clips, y, bad, p, starts = load_gold()
    orient_auc = check_orientation(y, p)
    order = sorted(clips)
    policies = {}
    for k in (2, 3, 4, 6, 8):
        policies[f"every-{k}"] = ("k", k)
    for Msec in (30, 60, 120, 240):
        policies[f"skip-first-{Msec}s"] = ("skip", Msec)
    for k in (2, 4):
        policies[f"random-1/{k}-s7"] = ("rand", k)
    policies["cusum-default"] = ("cusum", (0.10, 0.50))
    policies["full"] = ("full", None)

    per_clip, per_policy = {}, {}
    for pname, (kind, par) in policies.items():
        rows = []
        for clip in order:
            idx = clips[clip]
            ss = starts[idx]
            nn = len(idx)
            if kind == "k":
                S = pol_every_k(nn, par)
            elif kind == "skip":
                S = pol_skip_first_M(ss, par)
            elif kind == "rand":
                S = pol_random(nn, 1.0 / par, 7)
            elif kind == "cusum":
                d, h = par
                S = pol_cusum(p[idx], delta=d, h=h)
            else:
                S = list(range(nn))
            r = eval_policy_clip(bad[idx], p[idx], S)
            r.update(policy=pname, clip=clip)
            rows.append(r)
            per_clip.setdefault(clip, {})[pname] = r
        errs_L = [r["err_L"] for r in rows]
        errs_M = [r["err_M"] for r in rows]
        dL = [r["delay_L"] for r in rows if np.isfinite(r["delay_L"])]
        dM = [r["delay_M"] for r in rows if np.isfinite(r["delay_M"])]
        lo, hi = clip_bootstrap_ci(errs_L)
        loM, hiM = clip_bootstrap_ci(errs_M)
        per_policy[pname] = dict(
            mean_frac=float(np.mean([r["frac"] for r in rows])),
            mean_err_L=float(np.mean(errs_L)), ci_L=[lo, hi],
            mean_err_M=float(np.mean(errs_M)), ci_M=[loM, hiM],
            agree_L=float(np.mean([r["agree_L"] for r in rows])),
            agree_M=float(np.mean([r["agree_M"] for r in rows])),
            mean_delay_L=float(np.mean(dL)) if dL else None,
            mean_delay_M=float(np.mean(dM)) if dM else None,
            miss_L=int(sum(r["miss_L"] for r in rows)),
            miss_M=int(sum(r["miss_M"] for r in rows)))

    # LOCO tuning for CUSUM (delta,h): tune on 6 clips (min mean_err_M), test held-out
    grid = [(d, h) for d in (0.05, 0.10, 0.20) for h in (0.30, 0.50, 1.00)]
    loco = {}
    for held in order:
        train = [c for c in order if c != held]
        best, best_cost = None, 1e9
        for (d, h) in grid:
            errs, fr = [], []
            for clip in train:
                idx = clips[clip]
                S = pol_cusum(p[idx], delta=d, h=h)
                r = eval_policy_clip(bad[idx], p[idx], S)
                errs.append(r["err_M"])
                fr.append(r["frac"])
            cost = float(np.mean(errs)) + 0.05 * float(np.mean(fr))  # error first, cost tiebreak
            if cost < best_cost:
                best_cost, best = cost, (d, h)
        d, h = best
        idx = clips[held]
        S = pol_cusum(p[idx], delta=d, h=h)
        r = eval_policy_clip(bad[idx], p[idx], S)
        loco[held] = dict(delta=d, h=h, test_err_M=r["err_M"], test_err_L=r["err_L"],
                          test_frac=r["frac"], agree_M=r["agree_M"], agree_L=r["agree_L"])
    mu = float(np.mean([v["test_err_M"] for v in loco.values()]))
    lo, hi = clip_bootstrap_ci([v["test_err_M"] for v in loco.values()])
    res = dict(n_windows=150, n_clips=7, label_source="human gold (IndEgo)",
               protocol="per-clip subsampling simulation on gold-window order; "
                        "full=gold windows of clip (NOTE: gold is a sparse sample of each clip, "
                        "not exhaustive coverage); clip verdict threshold bad-frac>=0.30; "
                        "OOF p_ens orientation AUROC=%.3f; CI=clip bootstrap (2000)" % orient_auc,
               ci="clip-bootstrap 95%", per_policy=per_policy, per_clip=per_clip,
               cusum_loco=dict(mean_test_err_M=mu, ci_M=[lo, hi],
                               mean_test_frac=float(np.mean([v["test_frac"] for v in loco.values()])),
                               folds=loco, grid=["d=%.2f,h=%.2f" % g for g in grid]),
               command="PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python eval/exp_temporal/run_e3.py --out eval/results/exp_temporal")
    json.dump(res, open(os.path.join(outdir, "E3.01_policies.json"), "w"), indent=1)
    print("E3.01: orient AUROC=%.3f" % orient_auc)
    for pname, s in per_policy.items():
        print(f"  {pname:16s} frac={s['mean_frac']:.2f} errL={s['mean_err_L']:.3f}{s['ci_L']} agrL={s['agree_L']:.2f} "
              f"errM={s['mean_err_M']:.3f} agrM={s['agree_M']:.2f} dlyL={s['mean_delay_L']} missL={s['miss_L']} missM={s['miss_M']}")
    print(f"  cusum-LOCO mean test errM={mu:.3f} [{lo:.3f},{hi:.3f}]")
    return res


def run_e302(outdir):
    from scipy.stats import mannwhitneyu
    try:
        from sklearn.linear_model import LogisticRegression
        have_sk = True
    except ImportError:
        have_sk = False
    ood = json.load(open(OOD1))
    latest = ood.get("latest", ood)
    q1 = json.load(open(Q1))["segments"]
    qpos = {s["segment_id"]: (s["source_start_s"], s["clip"]) for s in q1}
    man = json.load(open(MANIFEST_SCALE))
    dur = {}
    for it in man["items"]:
        dur.setdefault(it["clip"], it["duration_s"])
    rows = []
    for sid, v in latest.items():
        if sid in qpos and isinstance(v, dict) and "verdict" in v:
            ss, clip = qpos[sid]
            rows.append(dict(y=int(v["verdict"] == "bad"), pos=ss,
                             pos_norm=ss / dur[clip] if clip in dur and dur[clip] else None,
                             dur=dur.get(clip)))
    n2 = []
    if os.path.exists(OOD2):
        ood2 = json.load(open(OOD2))
        latest2 = ood2.get("latest", ood2)
        q2 = json.load(open(Q2))["segments"]
        qpos2 = {s["segment_id"]: (s["source_start_s"], s["clip"]) for s in q2}
        for sid, v in latest2.items():
            if sid in qpos2 and isinstance(v, dict) and "verdict" in v:
                ss, clip = qpos2[sid]
                n2.append(dict(y=int(v["verdict"] == "bad"), pos=ss,
                               pos_norm=ss / dur[clip] if clip in dur and dur[clip] else None))
    pos = np.array([r["pos"] for r in rows])
    yb = np.array([r["y"] for r in rows])
    pn = np.array([r["pos_norm"] for r in rows])
    n_bad = int(yb.sum())
    out = dict(n_windows=len(rows), n_bad=n_bad, n_good=int((1 - yb).sum()),
               labels2_present=bool(n2), n2=len(n2),
               label_source="human OOD labels (labels.json) + queue source_start_s + manifest durations",
               protocol="Mann-Whitney bad-vs-good positions (raw s + normalised); logistic bad~pos_norm")
    try:
        u_raw = mannwhitneyu(pos[yb == 1], pos[yb == 0], alternative="two-sided")
        u_norm = mannwhitneyu(pn[yb == 1], pn[yb == 0], alternative="two-sided")
        out["mw_raw_U"] = float(u_raw.statistic)
        out["mw_raw_p"] = float(u_raw.pvalue)
        out["mw_norm_U"] = float(u_norm.statistic)
        out["mw_norm_p"] = float(u_norm.pvalue)
    except Exception as e:
        out["mw_error"] = str(e)
    out["median_pos_bad_s"] = float(np.median(pos[yb == 1])) if n_bad else None
    out["median_pos_good_s"] = float(np.median(pos[yb == 0]))
    out["median_posnorm_bad"] = float(np.median(pn[yb == 1])) if n_bad else None
    out["median_posnorm_good"] = float(np.median(pn[yb == 0]))
    if have_sk and n_bad >= 2:
        lr = LogisticRegression(C=1.0).fit(pn.reshape(-1, 1), yb)
        out["logreg_coef_posnorm"] = float(lr.coef_[0, 0])
        out["logreg_intercept"] = float(lr.intercept_[0])
    out["command"] = "PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python eval/exp_temporal/run_e3.py --out eval/results/exp_temporal"
    json.dump(out, open(os.path.join(outdir, "E3.02_ood_position.json"), "w"), indent=1)
    print("E3.02: n=%d n_bad=%d mw_raw_p=%.3f mw_norm_p=%.3f med_bad=%.0fs med_good=%.0fs" % (
        len(rows), n_bad, out.get("mw_raw_p", -1), out.get("mw_norm_p", -1),
        out["median_pos_bad_s"] or -1, out["median_pos_good_s"]))
    return out


def run_e303(outdir):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score
    seg = np.load(SEG, allow_pickle=True)
    F = np.asarray(seg["F"], dtype=np.float32)  # (150,21,768) fp16 2fps
    counts = np.asarray(seg["counts"], dtype=int)  # valid frames per window (20-21; rest is NaN pad)
    m, clips, y, bad, p, starts = load_gold()
    assert list(seg["ids"]) == [it["id"] for it in m["items"]]
    prefixes = [4, 8, 12, 16, 21]  # frames @2fps -> ~2,4,6,8,10.5 s
    order = sorted(clips)
    idx_of = {c: clips[c] for c in order}
    curve = []
    for pr in prefixes:
        mlen = np.minimum(pr, counts)
        E = np.stack([F[i, :mlen[i], :].mean(axis=0) for i in range(len(y))]).astype(np.float64)
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        oof_pred = np.zeros(len(y))
        for held in order:
            tr = np.concatenate([idx_of[c] for c in order if c != held])
            te = idx_of[held]
            sc = StandardScaler().fit(E[tr])
            lr = LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000).fit(sc.transform(E[tr]), y[tr])
            oof_pred[te] = lr.predict_proba(sc.transform(E[te]))[:, 1]
        auc = float(roc_auc_score(y, oof_pred))
        acc = float(accuracy_score(y, (oof_pred >= 0.5).astype(int)))
        curve.append(dict(prefix_frames=pr, seconds=pr / 2.0, auroc=auc, acc=acc))
        print(f"  prefix {pr:2d}fr (~{pr/2:.1f}s): AUROC={auc:.3f} acc={acc:.3f}")
    res = dict(n_windows=150, label_source="human gold (IndEgo)",
               protocol="LOCO logistic (StandardScaler, C=0.01 balanced) on L2-normalised mean "
                        "of first m EG2 frames (2 fps); pooled OOF AUROC/acc; recipe mirrors E0 baseline",
               curve=curve,
               command="PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python eval/exp_temporal/run_e3.py --out eval/results/exp_temporal")
    json.dump(res, open(os.path.join(outdir, "E3.03_early_exit.json"), "w"), indent=1)
    return res


def run_e304(outdir, r01, r03):
    man = json.load(open(MANIFEST_SCALE))
    total_h = man["total_hours"]
    n_win = man["n_windows"]
    tab = []
    for pname, s in r01["per_policy"].items():
        f = s["mean_frac"]
        tab.append(dict(policy=pname, window_frac=round(f, 4),
                        proc_hours=round(total_h * f, 3), speedup=round(1 / f, 2)))
    early = {c["seconds"]: c for c in r03["curve"]}
    combo = []
    for pname in ("every-2", "every-3", "every-4", "cusum-default"):
        f = r01["per_policy"][pname]["mean_frac"]
        for sec in (4.0, 6.0):
            wf = sec / 10.0
            tot = f * wf
            combo.append(dict(policy=pname, seconds=sec, total_frac=round(tot, 4),
                              proc_hours=round(total_h * tot, 3), speedup=round(1 / tot, 2)))
    res = dict(corpus_hours=total_h, corpus_windows=n_win, window_s=man["window_s"],
               label_source="unlabelled Egocentric-10K manifest (arithmetic on E3.01/E3.03 fractions)",
               protocol="processed_hours = corpus_hours x window_frac x (seconds/10); "
                        "assumes per-window cost scales with windows probed and, for early exit, "
                        "with seconds observed (decode-bound regime; feature cost sublinear — upper bound)",
               per_policy=tab, with_early_exit=combo,
               command="PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python eval/exp_temporal/run_e3.py --out eval/results/exp_temporal")
    json.dump(res, open(os.path.join(outdir, "E3.04_compute.json"), "w"), indent=1)
    for row in tab:
        print(f"  {row['policy']:16s} frac={row['window_frac']:.2f} hrs={row['proc_hours']:.2f} x{row['speedup']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="all")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r01 = r03 = None
    if a.only in ("all", "e301"):
        r01 = run_e301(a.out)
    if a.only in ("all", "e302"):
        run_e302(a.out)
    if a.only in ("all", "e303"):
        r03 = run_e303(a.out)
    if a.only in ("all", "e304"):
        if r01 is None:
            r01 = json.load(open(os.path.join(a.out, "E3.01_policies.json")))
        if r03 is None:
            r03 = json.load(open(os.path.join(a.out, "E3.03_early_exit.json")))
        run_e304(a.out, r01, r03)


if __name__ == "__main__":
    main()
