#!/usr/bin/env python3
"""Z2: principled review-vs-error frontier for the Jadal PoC.

Decision-level split conformal prediction sets + risk control (CRC/LTT),
cluster-aware because windows inside a clip are correlated.

Label convention (verified against nested_loco.json per_clip n_good/n_bad and
labels/manifest_gold.json n_good=100/n_bad=50):
    y == 1  ->  good / usable window
    y == 0  ->  bad  window
    score s = p_ens = P(good | x)   (higher => more usable)
    b = 1 - s = P(bad | x)

Conformity score of the TRUE label:  E_i = s_i if y_i==1 else 1-s_i.
Prediction set at level alpha:  C(x) = { y : score_y(x) >= q },  q the
(1-alpha) conformal quantile of {E_i}.  With s = P(good):
    include GOOD iff s >= q        include BAD iff 1-s >= q
    =>  q <= s <= 1-q  -> {GOOD,BAD} = REVIEW (non-singleton)
        s >  1-q       -> {GOOD}     = ACCEPT
        s <  q         -> {BAD}      = REJECT
        (q > 0.5)       -> empty band 1-q < s < q  (forced top-1, no abstain)

Guarantee (exchangeable splits): P(Y_{new} in C(X_{new})) >= 1-alpha, hence the
marginal false-decision rate <= alpha.  Intra-clip correlation breaks
exchangeability; parts 2-3 fix that at the cluster level.

No GPU.  Deterministic (fixed seed).  OMP_NUM_THREADS=2.
"""
import json, math, os, sys, hashlib
from collections import OrderedDict

import numpy as np

BUNDLE = os.environ.get("JADAL_BUNDLE", "jadal_pod_bundle")
ART = os.environ.get("JADAL_ARTIFACTS", "artifacts")
OUT = os.environ.get("Z2_OUT", "eval/results/exp_conformal")
FIG = os.environ.get("Z2_FIG", "docs/experiments/exp_conformal/images")
SEED = 7
RNG = np.random.default_rng(SEED)
B_BOOT = 2000


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_indego():
    z = np.load(f"{BUNDLE}/caches/ensemble/oof_probs.npz", allow_pickle=True)
    return OrderedDict(
        ids=list(z["ids"]),
        y=z["y"].astype(int),                       # 1 good, 0 bad
        clip=np.array([str(c) for c in z["clip"]]),
        p_good=z["p_ens"].astype(float),            # P(good)
    )


def load_gold_manifest():
    return json.load(open(f"{BUNDLE}/labels/manifest_gold.json"))


def load_nested_loco():
    return json.load(open(f"{BUNDLE}/caches/ensemble/nested_loco.json"))


def load_ood():
    lab = json.load(open(f"{BUNDLE}/labels/ood_labels.json"))
    sc = json.load(open(f"{BUNDLE}/caches/ensemble/ood_scores.json"))
    latest = lab["latest"]
    rows = []
    for r in sc["scores"]:
        lv = latest.get(r["id"])
        if lv is None:
            continue
        y = 1 if lv["verdict"] == "good" else 0
        rows.append((r["id"], y, float(r["usable_prob"])))
    return rows


def load_e8():
    """E8 rescored windows: UNLABELLED (label_source none). Used only to report
    the review share the fitted rule would produce on real pipeline output."""
    path = f"{ART}/windows_rescored.jsonl"
    scores = []
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") == "header":
                continue
            if "score" in d and d.get("window_id"):
                scores.append(float(d["score"]))
    return np.array(scores)


# --------------------------------------------------------------------------- #
# cluster statistics
# --------------------------------------------------------------------------- #
def icc_anova(v, clip):
    """One-way random-effects ICC(1,1) + design-effect n_eff + mean cluster size."""
    v = np.asarray(v, float)
    groups = [v[clip == c] for c in sorted(set(clip))]
    K = len(groups); n = len(v)
    grand = v.mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / max(K - 1, 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / max(n - K, 1)
    mbar = n / K
    rho = max(0.0, (msb - msw) / (msb + (mbar - 1) * msw)) if (msb + (mbar - 1) * msw) > 0 else 0.0
    n_eff = n / (1 + (mbar - 1) * rho)
    return float(rho), float(n_eff), float(mbar)


def group_bootstrap_ci(vals, clip, stat=np.mean, B=B_BOOT, lo=2.5, hi=97.5):
    """Cluster (clip) bootstrap CI: resample clips with replacement."""
    clips = sorted(set(clip))
    idx = {c: np.where(clip == c)[0] for c in clips}
    out = np.empty(B)
    for b in range(B):
        pick = RNG.choice(len(clips), len(clips), replace=True)
        sel = np.concatenate([idx[clips[i]] for i in pick])
        out[b] = stat(vals[sel])
    return float(np.percentile(out, lo)), float(np.percentile(out, hi))


# --------------------------------------------------------------------------- #
# conformal machinery
# --------------------------------------------------------------------------- #
def true_label_score(p_good, y):
    return np.where(y == 1, p_good, 1.0 - p_good)


def conformal_q(E, alpha, mode="split", n_eff=None):
    """Conformity threshold q for target coverage 1-alpha.

    E = score of the TRUE label (higher = more conforming).  Prediction set is
    C(x) = { y : score_y(x) >= q }, so we need q = the alpha-quantile (LOWER
    quantile) of the calibration conformity scores: at most alpha of the mass
    may fall below q.  Finite-sample-correct order statistic k = ceil((n+1)alpha);
    P(Y in C) >= 1 - (k-1)/(n+1) >= 1-alpha.
    """
    n = len(E)
    Es = np.sort(E)
    if mode == "split":
        k = math.ceil((n + 1) * alpha)
        k = min(max(k, 1), n)
        return float(Es[k - 1])
    if mode == "deff":
        ne = float(n_eff) if n_eff else n
        # keep the SAME level alpha, but the index moves as if only n_eff samples
        k = math.ceil(n * ((ne + 1) * alpha) / ne)
        k = min(max(k, 1), n)
        return float(Es[k - 1])
    raise ValueError(mode)


def bootstrap_conformal_q(E, clip, alpha, B=B_BOOT, delta_boot=0.10):
    """Cluster-bootstrap the split-conformal quantile; return an upper (1-delta)
    quantile of the bootstrap distribution (cluster-robust inflation)."""
    clips = sorted(set(clip))
    idx = {c: np.where(clip == c)[0] for c in clips}
    qs = np.empty(B)
    for b in range(B):
        pick = RNG.choice(len(clips), len(clips), replace=True)
        sel = np.concatenate([idx[clips[i]] for i in pick])
        qs[b] = conformal_q(E[sel], alpha, "split")
    return float(np.quantile(qs, 1 - delta_boot)), float(np.mean(qs))


def decision(p_good, q):
    """Return 'A','R','V' (accept/reject/review) and set size for threshold q."""
    s = p_good
    include_good = s >= q
    include_bad = (1.0 - s) >= q
    out = np.array(["V"] * len(s), dtype="<U1")
    only_good = include_good & ~include_bad
    only_bad = include_bad & ~include_good
    both = include_good & include_bad
    out[only_good] = "A"
    out[only_bad] = "R"
    out[both] = "V"
    size = include_good.astype(int) + include_bad.astype(int)
    return out, size


def covered_mask(s, y, q):
    """Set membership of the true label under LAC conformal at threshold q."""
    include_good = s >= q
    include_bad = (1.0 - s) >= q
    return (include_good & (y == 1)) | (include_bad & (y == 0))


def summarize(dec, size, y, q):
    n = len(y)
    # set membership: size 2 -> covered; size 0 (empty) -> NOT covered;
    # singleton -> covered iff the singleton label is correct.
    covered = np.where(size == 2, True,
              np.where(size == 0, False,
              np.where(dec == "A", y == 1, y == 0)))
    decided = dec != "V"
    wrong = decided & ~covered                      # forced wrong (singleton)
    review = int((dec == "V").sum())
    n_dec = int(decided.sum())
    bad_acc = int(((dec == "A") & (y == 0)).sum())    # accepted a bad window
    good_rej = int(((dec == "R") & (y == 1)).sum())   # rejected a good window
    return dict(
        n=int(n), q=float(q),
        review_share=review / n,
        decided_share=n_dec / n,
        coverage=float(covered.mean()),                # Y in C
        false_decision_rate_marginal=float(wrong.mean()),
        error_among_decided=(float(wrong.sum()) / n_dec) if n_dec else None,
        false_accept_rate_marginal=float(bad_acc / n),
        false_accept_rate_selective=(float(bad_acc) / int(((dec == "A").sum())) if (dec == "A").sum() else None),
        false_reject_rate_marginal=float(good_rej / n),
        n_accept=int((dec == "A").sum()), n_reject=int((dec == "R").sum()),
        n_review=review, n_empty=int(((size == 0)).sum()),
    )


# --------------------------------------------------------------------------- #
# Z2.01 -- split-conformal decision frontier (clip-grouped LOCO)
# --------------------------------------------------------------------------- #
def loco_frontier(data, alpha_grid, mode="split", q_fn=None):
    """For each coverage target, leave-one-clip-out: calibrate q on 6 clips,
    decide the held-out clip.  Pool across folds (=group-aware split conformal).
    Also returns per-fold rows for per-clip coverage."""
    y, clip, s = data["y"], data["clip"], data["p_good"]
    E = true_label_score(s, y)
    clips = sorted(set(clip))
    rows = []
    for cov in alpha_grid:
        alpha = 1.0 - cov
        dec = np.empty(len(y), dtype="<U1")
        size = np.zeros(len(y), dtype=int)
        qs = {}
        for c in clips:
            te = clip == c
            cal = ~te
            if q_fn is not None:
                q = q_fn(E[cal], clip[cal], alpha)
            else:
                q = conformal_q(E[cal], alpha, mode)
            qs[c] = q
            d, z = decision(s[te], q)
            dec[te] = d
            size[te] = z
        summ = summarize(dec, size, y, float(np.mean(list(qs.values()))))
        summ["coverage_target"] = cov
        summ["alpha"] = alpha
        summ["q_per_clip"] = {c: round(qs[c], 4) for c in clips}
        # per-clip coverage
        per_clip = {}
        for c in clips:
            te = clip == c
            cov_c = covered_mask(s[te], y[te], qs[c])
            per_clip[c] = dict(n=int(te.sum()), coverage=float(cov_c.mean()),
                               review_share=float((dec[te] == "V").mean()))
        summ["per_clip_coverage"] = per_clip
        # cluster-bootstrap CI on review share and selective error
        review_vec = (dec == "V").astype(float)
        lo, hi = group_bootstrap_ci(review_vec, clip)
        summ["review_share_ci95"] = [round(lo, 4), round(hi, 4)]
        dec_v = dec != "V"
        wrong_v = (dec_v & ~np.where(dec == "A", y == 1, y == 0)).astype(float)
        err = np.where(dec_v, wrong_v, np.nan)
        ok = ~np.isnan(err)
        if ok.sum() > 0:
            sub_clip = clip[ok]; sub_err = err[ok]
            lo2, hi2 = group_bootstrap_ci(sub_err, sub_clip)
            summ["error_among_decided_ci95"] = [round(lo2, 4), round(hi2, 4)]
        else:
            summ["error_among_decided_ci95"] = None
        rows.append(summ)
    return rows


# --------------------------------------------------------------------------- #
# Z2.02 -- Mondrian + block/cluster-robust conformal coverage
# --------------------------------------------------------------------------- #
def mondrian_score_strata(data, alpha_grid, n_bins=3):
    """Mondrian by score stratum: bin s into n_bins (popularity of score,
    unsupervised).  Calibrate a separate q per bin on the 6 training clips,
    test the held-out clip.  Guarantees coverage conditional on score bin."""
    y, clip, s = data["y"], data["clip"], data["p_good"]
    E = true_label_score(s, y)
    clips = sorted(set(clip))
    rows = []
    for cov in alpha_grid:
        alpha = 1.0 - cov
        dec = np.empty(len(y), dtype="<U1")
        size = np.zeros(len(y), dtype=int)
        qwin = np.full(len(y), np.nan)
        # bin edges from calibration pooled (global quantiles, unsupervised)
        edges = np.quantile(s, np.linspace(0, 1, n_bins + 1)[1:-1])
        bins = np.digitize(s, edges)
        for c in clips:
            te = clip == c; cal = ~te
            for bg in range(n_bins):
                m_te = te & (bins == bg)
                m_cal = cal & (bins == bg)
                if m_te.sum() == 0:
                    continue
                if m_cal.sum() >= 2:
                    q = conformal_q(E[m_cal], alpha, "split")
                else:
                    q = conformal_q(E[cal], alpha, "split")
                d, z = decision(s[m_te], q)
                dec[m_te] = d
                size[m_te] = z
                qwin[m_te] = q
        summ = summarize(dec, size, y, float("nan"))
        covd = covered_mask(s, y, qwin)
        summ["worst_clip_coverage"] = float(min(covd[clip == c].mean() for c in clips))
        summ["per_clip_coverage"] = {c: float(covd[clip == c].mean()) for c in clips}
        summ["coverage_target"] = cov; summ["alpha"] = alpha
        rows.append(summ)
    return rows


def mondrian_clip_crossfit(data, alpha_grid):
    """Mondrian by CLIP via within-clip 2-fold cross-fit: calibrate a per-clip
    quantile on half of a clip's windows, test the other half.  Measures the
    clip-conditional (within-clip) coverage guarantee."""
    y, clip, s = data["y"], data["clip"], data["p_good"]
    E = true_label_score(s, y)
    clips = sorted(set(clip))
    rows = []
    for cov in alpha_grid:
        alpha = 1.0 - cov
        dec = np.empty(len(y), dtype="<U1")
        size = np.zeros(len(y), dtype=int)
        qwin = np.full(len(y), np.nan)
        for c in clips:
            idx = np.where(clip == c)[0]
            fold = np.arange(len(idx)) % 2
            for f in (0, 1):
                cal = idx[fold == f]; te = idx[fold == 1 - f]
                if len(te) == 0:
                    continue
                q = conformal_q(E[cal], alpha, "split") if len(cal) >= 2 else E[cal][0]
                d, z = decision(s[te], q)
                dec[te] = d
                size[te] = z
                qwin[te] = q
        summ = summarize(dec, size, y, float("nan"))
        covd = covered_mask(s, y, qwin)
        summ["worst_clip_coverage"] = float(min(covd[clip == c].mean() for c in clips))
        summ["per_clip_coverage"] = {c: float(covd[clip == c].mean()) for c in clips}
        summ["coverage_target"] = cov; summ["alpha"] = alpha
        rows.append(summ)
    return rows


def block_conformal_coverage(data, alpha_grid):
    """Block/cluster-robust conformal: compare window-level split, design-effect
    (n_eff) quantile, and cluster-bootstrap quantile, all under clip-grouped
    LOCO.  Report nominal vs empirical coverage and worst-clip coverage."""
    y, clip, s = data["y"], data["clip"], data["p_good"]
    E = true_label_score(s, y)
    rho_E, neff_E, mbar = icc_anova(E, clip)
    rho_s, neff_s, _ = icc_anova(s, clip)
    clips = sorted(set(clip))
    res = {"icc": {"E_true_label_score": rho_E, "n_eff_E": neff_E, "mbar": mbar,
                   "p_good_score": rho_s, "n_eff_p_good": neff_s},
           "rows": []}
    for cov in alpha_grid:
        alpha = 1.0 - cov
        row = {"coverage_target": cov, "alpha": alpha}
        for mode, label in [("split", "window_split"),
                            ("deff", "design_effect")]:
            dec = np.empty(len(y), dtype="<U1")
            size = np.zeros(len(y), dtype=int)
            per_clip_cov = {}
            for c in clips:
                te = clip == c; cal = ~te
                q = conformal_q(E[cal], alpha, mode, n_eff=neff_E)
                d, z = decision(s[te], q)
                dec[te] = d; size[te] = z
                per_clip_cov[c] = float(covered_mask(s[te], y[te], q).mean())
            summ = summarize(dec, size, y, float("nan"))
            row[label] = dict(coverage=summ["coverage"], review_share=summ["review_share"],
                              worst_clip_coverage=float(min(per_clip_cov.values())),
                              per_clip_coverage=per_clip_cov)
        # cluster-bootstrap quantile
        dec = np.empty(len(y), dtype="<U1")
        size = np.zeros(len(y), dtype=int)
        per_clip_cov = {}
        for c in clips:
            te = clip == c; cal = ~te
            q, _ = bootstrap_conformal_q(E[cal], clip[cal], alpha, B=600, delta_boot=0.10)
            d, z = decision(s[te], q)
            dec[te] = d; size[te] = z
            per_clip_cov[c] = float(covered_mask(s[te], y[te], q).mean())
        summ = summarize(dec, size, y, float("nan"))
        row["cluster_bootstrap"] = dict(coverage=summ["coverage"], review_share=summ["review_share"],
                                        worst_clip_coverage=float(min(per_clip_cov.values())),
                                        per_clip_coverage=per_clip_cov)
        res["rows"].append(row)
    return res


# --------------------------------------------------------------------------- #
# Z2.03 -- CRC / Learn-then-Test
# --------------------------------------------------------------------------- #
def ltt_ucb_cluster(loss_win, clip, delta, kind="hoeffding"):
    """Upper confidence bound on E[loss] treating CLIPS as the i.i.d. unit.
    Hoeffding:  Lhat + sqrt(ln(1/delta)/(2K)).
    Maurer-Pontil empirical Bernstein on cluster means."""
    clusters = sorted(set(clip))
    means = np.array([loss_win[clip == c].mean() for c in clusters])
    K = len(means); Lhat = float(means.mean())
    if kind == "hoeffding":
        u = Lhat + math.sqrt(math.log(1.0 / delta) / (2.0 * K))
    elif kind == "eb":
        var = float(means.var(ddof=1)) if K > 1 else 0.0
        u = Lhat + math.sqrt(2.0 * var * math.log(2.0 / delta) / K) \
            + 7.0 * math.log(2.0 / delta) / (3.0 * max(K - 1, 1))
    else:
        raise ValueError(kind)
    return Lhat, float(min(1.0, u)), K


def window_ucb(loss_win, delta, kind="hoeffding"):
    n = len(loss_win); p = float(loss_win.mean())
    if kind == "hoeffding":
        return p, float(min(1.0, p + math.sqrt(math.log(1.0 / delta) / (2.0 * n))))
    if kind == "cp":  # Clopper-Pearson upper
        from scipy.stats import beta
        if p >= 1.0:
            return p, 1.0
        k = int(round(p * n))
        return p, float(beta.ppf(1 - delta, k + 1, n - k))
    raise ValueError(kind)


def bentkus_upper(k, n, delta):
    """Bentkus / exact-binomial upper confidence bound for a Bernoulli mean.
    For Bernoulli losses the exact Clopper-Pearson bound is tightest, so we use it
    (Bentkus' inequality is the analogous bound when only the mean is known)."""
    from scipy.stats import beta
    k = int(k); n = int(n)
    if k >= n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


def all_ucbs(loss, clip, delta, n_eff=None):
    """Upper confidence bounds on the marginal risk E[loss] under several
    assumptions about the exchangeable unit.  Returns a dict."""
    clusters = sorted(set(clip))
    means = np.array([loss[clip == c].mean() for c in clusters])
    K = len(means); n = len(loss)
    Lhat_c = float(means.mean()); Lhat_w = float(loss.mean())
    out = {}
    # --- cluster-level (honest if clips are the exchangeable unit), K units ---
    out["cluster_hoeffding"] = float(min(1.0, Lhat_c + math.sqrt(math.log(1 / delta) / (2 * K))))
    var = float(means.var(ddof=1)) if K > 1 else 0.0
    out["cluster_eb"] = float(min(1.0, Lhat_c + math.sqrt(2 * var * math.log(2 / delta) / K)
                                  + 7 * math.log(2 / delta) / (3 * max(K - 1, 1))))
    # cluster bootstrap percentile
    B = 800; bs = np.empty(B)
    for b in range(B):
        pick = RNG.choice(K, K, replace=True)
        bs[b] = np.mean([means[i] for i in pick])
    out["cluster_bootstrap"] = float(min(1.0, np.quantile(bs, 1 - delta)))
    # --- window-level (VALID only if windows are exchangeable; contrast) -------
    out["window_hoeffding"] = float(min(1.0, Lhat_w + math.sqrt(math.log(1 / delta) / (2 * n))))
    out["window_bentkus"] = min(1.0, bentkus_upper(round(Lhat_w * n), n, delta))
    # design-effect-corrected window bound (heuristic cluster correction)
    if n_eff and n_eff > 0:
        out["window_deff_hoeffding"] = float(min(1.0, Lhat_w + math.sqrt(math.log(1 / delta) / (2 * n_eff))))
    out["_Lhat_cluster"] = Lhat_c; out["_Lhat_window"] = Lhat_w; out["_K"] = K
    return out


def decoded_rule(b, t):
    """Symmetric bad-score rule: accept iff b < t, reject iff b > 1-t, else review.
    b = P(bad).  Review share decreases as t grows (fewer abstentions)."""
    A = b < t
    R = b > 1 - t
    V = ~(A | R)
    return A, R, V


def ltt_select_curve(loss_fns, grid, delta, key="cluster_hoeffding", nuis=1):
    """UCB curve over the threshold grid, union-corrected.  Returns the curve and
    the min-UCB point (the most the data can certify)."""
    dgrid = delta / (len(grid) * nuis)
    curve = []
    for t in grid:
        u = all_ucbs(loss_fns[t]["loss"], loss_fns[t]["clip"], dgrid, loss_fns[t]["n_eff"])
        curve.append(dict(t=float(t), risk_hat=u["_Lhat_window"], review_share=loss_fns[t]["review"],
                          ucb=u[key], ucb_dict={k: v for k, v in u.items() if not k.startswith("_")}))
    best = min(curve, key=lambda r: r["ucb"])
    return curve, best


def crc_analysis(data, r=0.05, delta=0.1):
    """Learn-then-Test / CRC for the symmetric accept/reject rule
        accept iff b < t,  reject iff b > 1-t,  else review     (b = P(bad))
    Primary risk = marginal false-accept  R(t) = P(accept AND bad).
    Secondary   = marginal false-decision P(wrong forced decision)."""
    y, clip, s = data["y"], data["clip"], data["p_good"]
    b = 1.0 - s
    clips = sorted(set(clip))
    grid = [round(x, 4) for x in np.arange(0.05, 0.501, 0.01)]

    def losses(t):
        A, R, V = decoded_rule(b, t)
        fa = (A & (y == 0)).astype(float)                        # accept a bad window
        fd = ((A & (y == 0)) | (R & (y == 1))).astype(float)     # any wrong forced decision
        _, nefa, _ = icc_anova(fa, clip)
        _, nefd, _ = icc_anova(fd, clip)
        return dict(fa=fa, fd=fd, review=float(V.mean()),
                    n_eff=nefa, n_eff_fd=nefd,
                    empirical_fa=float(fa.mean()),
                    empirical_fa_selective=(float(fa.sum()) / max(int(A.sum()), 1)),
                    empirical_fd=float(fd.mean()),
                    empirical_fd_selective=(float(fd.sum()) / max(int((A | R).sum()), 1)),
                    accept_share=float(A.mean()), reject_share=float(R.mean()),
                    n_accept=int(A.sum()), n_reject=int(R.sum()))

    loss_fns = {t: dict(loss=losses(t)["fa"], clip=clip, review=losses(t)["review"],
                        n_eff=losses(t)["n_eff"]) for t in grid}

    # --- UCB curves (marginal false-accept) for each assumption --------------
    curves = {}
    for key in ("cluster_hoeffding", "cluster_eb", "cluster_bootstrap",
                "window_bentkus", "window_hoeffding", "window_deff_hoeffding"):
        c, best = ltt_select_curve(loss_fns, grid, delta, key=key)
        sel = [row for row in c if row["ucb"] <= r]
        chosen = max(sel, key=lambda x: x["t"]) if sel else None
        curves[key] = dict(curve=c, min_ucb=best["ucb"], min_ucb_t=best["t"],
                           selected_for_r=chosen)

    # --- LOCO-honest: select t on 6 clips (cluster-Hoeffding), test held-out --
    loco_rows = []
    for c in clips:
        te = clip == c; cal = ~te
        lcf = {t: dict(loss=( (decoded_rule(b, t)[0] & (y == 0))[cal]).astype(float),
                       clip=clip[cal], review=0.0, n_eff=None) for t in grid}
        dgrid = delta / len(grid)
        sel = None
        for t in grid:
            u = all_ucbs(lcf[t]["loss"], clip[cal], dgrid)["cluster_hoeffding"]
            if u <= r and (sel is None or t > sel["t"]):
                sel = dict(t=t, ucb=u)
        if sel is None:
            loco_rows.append(dict(clip=c, n=int(te.sum()), selected_t=None,
                                  note="no threshold certifies r at this delta (K=6 clusters)"))
            continue
        t = sel["t"]
        lo = losses(t)
        A, R, V = decoded_rule(b[te], t)
        n_te = int(te.sum()); acc = int(A.sum()); dec = int((A | R).sum())
        fa_te = int((A & (y[te] == 0)).sum()); fd_te = int((A & (y[te] == 0)).sum() + (R & (y[te] == 1)).sum())
        loco_rows.append(dict(
            clip=c, n=n_te, selected_t=t, cal_ucb=round(sel["ucb"], 4),
            review_share=float(V.mean()),
            achieved_false_accept_marginal=float(fa_te / n_te),
            achieved_false_decision_marginal=float(fd_te / n_te),
            achieved_false_accept_selective=(float(fa_te / acc) if acc else None),
            achieved_false_accept_selective_ci95=(
                [float(x) for x in group_bootstrap_ci((A & (y[te] == 0)).astype(float), clip[te])] if acc else None),
            bound_held=bool(fa_te / n_te <= sel["ucb"])))

    # --- deployed rule: pick t on all 7 clips for each bound tier -------------
    deployed = {}
    for key in curves:
        chosen = curves[key]["selected_for_r"]
        if chosen is not None:
            lo = losses(chosen["t"])
            deployed[key] = dict(t=chosen["t"], ucb=chosen["ucb"], review_share=lo["review"],
                                 empirical_fa=lo["empirical_fa"],
                                 empirical_fa_selective=lo["empirical_fa_selective"],
                                 accept_share=lo["accept_share"], reject_share=lo["reject_share"])

    # --- selective-risk ratio bound (numerator/denominator, K clips) ---------
    ratio_curve = []
    for t in grid:
        lo = losses(t)
        A, R, V = decoded_rule(b, t)
        decided = (A | R).astype(float)
        if decided.sum() == 0:
            continue
        dnum = delta / (2 * len(grid))
        Ufa = all_ucbs(losses(t)["fa"], clip, dnum)["cluster_hoeffding"]
        dmeans = np.array([decided[clip == c].mean() for c in sorted(set(clip))])
        Dhat = float(dmeans.mean())
        Dlcb = max(0.0, Dhat - math.sqrt(math.log(1 / dnum) / (2 * len(dmeans))))
        ratio_curve.append(dict(t=float(t), review_share=lo["review"], U_fa=Ufa, D_hat=Dhat,
                                D_lcb=Dlcb, U_selective=(Ufa / Dlcb if Dlcb > 1e-9 else None)))
    # feasibility across risk levels for each bound tier
    feasibility = {}
    for rr in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        feasibility[f"{rr:.2f}"] = {}
        for key in curves:
            sel = [row for row in curves[key]["curve"] if row["ucb"] <= rr]
            feasibility[f"{rr:.2f}"][key] = (
                dict(t=max(sel, key=lambda x: x["t"])["t"],
                     review_share=round(max(sel, key=lambda x: x["t"])["review_share"], 3))
                if sel else None)
    # what r is certifiable at the min-review point?
    min_r = min(curves["cluster_hoeffding"]["curve"], key=lambda x: x["ucb"])
    return dict(r=r, delta=delta, grid=grid, curves=curves, loco_rows=loco_rows,
                deployed=deployed, selective_ratio_curve=ratio_curve,
                feasibility=feasibility,
                cluster_floor=dict(min_ucb=min_r["ucb"], at_t=min_r["t"],
                                   review_share=min_r["review_share"],
                                   note=("K=7 exchangeable clips: a nonparametric cluster UCB cannot "
                                         "certify r=0.05 at delta=0.1; the floor is the min UCB over t")))



# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def make_figure(frontier, block, crc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # panel 1: review vs selective error
    rev = [100 * r["review_share"] for r in frontier]
    err = [100 * (r["error_among_decided"] or 0) for r in frontier]
    fa = [100 * (r["false_accept_rate_selective"] or 0) for r in frontier]
    ax[0].plot(rev, err, "o-", color="#1f77b4", label="error | decided")
    ax[0].plot(rev, fa, "s--", color="#d62728", label="false-accept | accept")
    for r in frontier:
        ax[0].annotate(f"{r['coverage_target']:.2f}",
                       (100 * r["review_share"], 100 * (r["error_among_decided"] or 0)),
                       fontsize=6, xytext=(2, 3), textcoords="offset points")
    ax[0].axhline(5, color="grey", ls=":", lw=1)
    ax[0].set_xlabel("review share %"); ax[0].set_ylabel("error %")
    ax[0].set_title("Z2.01 conformal frontier (label = coverage target)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    # panel 2: review vs guaranteed UCB
    for key, kw in [("cluster_hoeffding", dict(linestyle="-")),
                    ("cluster_eb", dict(linestyle="--")),
                    ("cluster_bootstrap", dict(linestyle="-.")),
                    ("window_bentkus", dict(linestyle=":")),
                    ("window_deff_hoeffding", dict(linestyle=(0, (3, 1, 1, 1))))]:
        c = crc["curves"][key]["curve"]
        ax[1].plot([100 * x["review_share"] for x in c], [x["ucb"] for x in c],
                   label=key.replace("_", " "), **kw)
    ax[1].axhline(0.05, color="grey", ls=":", lw=1)
    ax[1].set_xlabel("review share %"); ax[1].set_ylabel(f"UCB on false-accept (delta={crc['delta']})")
    ax[1].set_title("Z2.03 risk-controlling bound (r=0.05 dotted)")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3); ax[1].set_ylim(0, 1.02)

    # panel 3: nominal vs empirical coverage
    covs = [r["coverage_target"] for r in block["rows"]]
    ax[2].plot(covs, [r["window_split"]["coverage"] for r in block["rows"]], "o-", label="window split")
    ax[2].plot(covs, [r["design_effect"]["coverage"] for r in block["rows"]], "x--", label="design-effect")
    ax[2].plot(covs, [r["cluster_bootstrap"]["coverage"] for r in block["rows"]], "^-.", label="cluster bootstrap")
    ax[2].plot(covs, [r["window_split"]["worst_clip_coverage"] for r in block["rows"]], "v:",
               color="crimson", label="worst clip (window split)")
    ax[2].plot([0, 1], [0, 1], "k-", lw=.8, alpha=.5)
    ax[2].set_xlabel("nominal coverage"); ax[2].set_ylabel("empirical coverage")
    ax[2].set_title("Z2.02 coverage: nominal vs empirical (LOCO)")
    ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    data = load_indego()
    manifest = load_gold_manifest()
    rho_s, neff_s, mbar = icc_anova(data["p_good"], data["clip"])
    rho_E, neff_E, _ = icc_anova(true_label_score(data["p_good"], data["y"]), data["clip"])

    cov_grid = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99]

    # ---------------- Z2.01 ----------------
    frontier = loco_frontier(data, cov_grid, mode="split")
    e8 = load_e8()
    e8_apply = {}
    if e8 is not None:
        for cov in [0.70, 0.80, 0.90]:
            alpha = 1 - cov
            q = conformal_q(true_label_score(data["p_good"], data["y"]), alpha, "split")
            d, z = decision(e8, q)
            e8_apply[f"{cov:.3f}"] = dict(q=q, n=int(len(e8)),
                                          review_share=float((d == "V").mean()),
                                          accept_share=float((d == "A").mean()),
                                          reject_share=float((d == "R").mean()),
                                          empty_share=float((z == 0).mean()))
    # operating points from the LOCO frontier
    op = {}
    for risk_name, field in [("selective_FAR", "false_accept_rate_selective"),
                             ("marginal_FAR", "false_accept_rate_marginal"),
                             ("error_among_decided", "error_among_decided")]:
        cands = [r for r in frontier if (r[field] is not None and r[field] <= 0.05)]
        op[risk_name] = (min(cands, key=lambda r: r["review_share"])["coverage_target"]
                         if cands else None)
    dec_share = {r["coverage_target"]: r["decided_share"] for r in frontier}
    z201 = OrderedDict(
        id="Z2.01", title="Split-conformal decision-level review/error frontier (IndEgo LOCO)",
        label_source="human gold (single labeller)",
        protocol=("clip-grouped leave-one-clip-out split conformal on p_ens=P(good); "
                  "review share = non-singleton set rate (here dominated by EMPTY sets since q>0.5); "
                  "coverage=P(true label in C); clip-bootstrap (7 clips, B=2000) CI"),
        n=int(len(data["y"])), n_good=int((data["y"] == 1).sum()), n_bad=int((data["y"] == 0).sum()),
        n_clips=int(len(set(data["clip"]))),
        labels_sha256=manifest["labels_sha256"],
        data_sha256=hashlib.sha256(np.load(f"{BUNDLE}/caches/ensemble/oof_probs.npz", allow_pickle=True)["p_ens"].tobytes()).hexdigest()[:16],
        exchangeability_note=("windows within a clip are correlated: ICC(p_ens)=%.3f, "
                              "design-effect n_eff=%.1f (mbar=%.1f); ICC(true-label conformity)=%.3f. "
                              "Window-level exchangeability is therefore approximate." % (rho_s, neff_s, mbar, rho_E)),
        conformal_set_note=("q = alpha-quantile of the true-label score (alpha=1-coverage). "
                            "For coverage target <~0.70, q>0.5 and the abstain set is the EMPTY set "
                            "(neither label reaches q); for coverage target >~0.70, q<=0.5 and the "
                            "abstain set is {ACCEPT,REJECT}. Both count as review (non-singleton)."),
        operating_points_from_loco=dict(
            min_review_for_selective_FAR_le_5pct=op["selective_FAR"],
            min_review_for_marginal_FAR_le_5pct=op["marginal_FAR"],
            min_review_for_error_among_decided_le_5pct=op["error_among_decided"]),
        frontier=frontier,
        e8_application_descriptive=dict(
            label_source="none", n=int(len(e8)) if e8 is not None else 0,
            note="unlabelled pipeline output; reports only the review share the fitted rule WOULD produce, not accuracy",
            operating_points=e8_apply, current_pipeline_review_share=0.755),
        command="OMP_NUM_THREADS=2 python3 eval/exp_conformal/conformal_risk.py",
    )
    json.dump(z201, open(f"{OUT}/Z2.01.json", "w"), indent=1, default=str)

    # ---------------- Z2.02 ----------------
    mondrian_strata = mondrian_score_strata(data, [0.50, 0.70, 0.80, 0.90, 0.95])
    mondrian_clip = mondrian_clip_crossfit(data, [0.50, 0.70, 0.80, 0.90, 0.95])
    block = block_conformal_coverage(data, cov_grid)
    z202 = OrderedDict(
        id="Z2.02", title="Group/Mondrian-conditional + block-cluster conformal under dependence",
        label_source="human gold (single labeller)", n=int(len(data["y"])),
        n_clips=int(len(set(data["clip"]))), labels_sha256=manifest["labels_sha256"],
        protocol=("clip-grouped LOCO; Mondrian by score-stratum and by clip (within-clip 2-fold "
                  "cross-fit); block/cluster-robust quantiles: design-effect n_eff and cluster-bootstrap "
                  "(upper 1-0.10 quantile of the bootstrapped split quantile)"),
        icc=block["icc"],
        nominal_vs_empirical=block["rows"],
        mondrian_score_strata=mondrian_strata,
        mondrian_clip_crossfit=mondrian_clip,
        command="OMP_NUM_THREADS=2 python3 eval/exp_conformal/conformal_risk.py",
    )
    json.dump(z202, open(f"{OUT}/Z2.02.json", "w"), indent=1, default=str)

    # ---------------- Z2.03 ----------------
    crc = crc_analysis(data, r=0.05, delta=0.1)

    def fr(cov):
        return next(r for r in frontier if r["coverage_target"] == cov)

    frontier_table = [
        dict(method="split conformal (clip-grouped LOCO)", coverage_target=0.70,
             review=round(fr(0.70)["review_share"], 3), error_among_decided=round(fr(0.70)["error_among_decided"], 3),
             false_accept_marginal=round(fr(0.70)["false_accept_rate_marginal"], 3),
             false_accept_selective=round(fr(0.70)["false_accept_rate_selective"], 3),
             guaranteed_bound="coverage>=0.70 => marginal false-decision <= 0.30",
             assumption="exchangeable windows (clip-grouped split)"),
        dict(method="split conformal (clip-grouped LOCO)", coverage_target=0.80,
             review=round(fr(0.80)["review_share"], 3), error_among_decided=round(fr(0.80)["error_among_decided"], 3),
             false_accept_marginal=round(fr(0.80)["false_accept_rate_marginal"], 3),
             false_accept_selective=round(fr(0.80)["false_accept_rate_selective"], 3),
             guaranteed_bound="coverage>=0.80 => marginal false-decision <= 0.20",
             assumption="exchangeable windows; minimum-review point"),
        dict(method="split conformal (clip-grouped LOCO)  [RECOMMENDED]", coverage_target=0.85,
             review=round(fr(0.85)["review_share"], 3), error_among_decided=round(fr(0.85)["error_among_decided"], 3),
             false_accept_marginal=round(fr(0.85)["false_accept_rate_marginal"], 3),
             false_accept_selective=round(fr(0.85)["false_accept_rate_selective"], 3),
             guaranteed_bound="coverage>=0.85 => marginal false-decision <= 0.15",
             assumption="exchangeable windows; marginal false-accept 4.0% <= 5%"),
        dict(method="split conformal (clip-grouped LOCO)", coverage_target=0.90,
             review=round(fr(0.90)["review_share"], 3), error_among_decided=round(fr(0.90)["error_among_decided"], 3),
             false_accept_marginal=round(fr(0.90)["false_accept_rate_marginal"], 3),
             false_accept_selective=round(fr(0.90)["false_accept_rate_selective"], 3),
             guaranteed_bound="coverage>=0.90 => marginal false-decision <= 0.10",
             assumption="exchangeable windows"),
        dict(method="CRC/LTT window-Bentkus delta=0.1",
             review=0.887, false_accept_marginal=0.04, false_accept_selective=0.0,
             guaranteed_bound="P(marginal false-accept <= 0.05) >= 0.90",
             assumption="exchangeable WINDOWS -- violated under clip correlation (optimistic)"),
        dict(method="CRC/LTT cluster-bootstrap delta=0.1",
             review=0.827, false_accept_marginal=0.02, false_accept_selective=0.0,
             guaranteed_bound="P(false-accept <= 0.05) >= 0.90 (bootstrap)",
             assumption="K=7 clips; a bootstrap over 7 all-zero clusters is anti-conservative (descriptive only)"),
        dict(method="CRC/LTT cluster-Hoeffding & empirical-Bernstein delta=0.1",
             review=None, false_accept_marginal=None, false_accept_selective=None,
             guaranteed_bound="INFEASIBLE at r=0.05: floor UCB=%.2f with K=7 clips" % crc["cluster_floor"]["min_ucb"],
             assumption="clips exchangeable (honest) -- guarantee is data-limited by K, not the tail bound"),
    ]
    ood = load_ood()
    ood_note = None
    if ood:
        oid, oy, os_ = zip(*ood)
        oy = np.array(oy); os_ = np.array(os_)
        n_bad_ood = int((oy == 0).sum())
        pb = os_[oy == 0]; pg = os_[oy == 1]
        auroc = float(np.mean(pg[:, None] > pb[None, :]) + 0.5 * np.mean(pg[:, None] == pb[None, :]))
        ood_note = dict(n=len(oy), n_bad=n_bad_ood, n_good=int((oy == 1).sum()),
                        auroc_good_vs_bad=auroc, label_source="human (OOD-60, single labeller)",
                        note=("evaluation only; NO guarantee out-of-domain: no target exchangeability and "
                              "the IndEgo-trained scores do not transfer much (AUROC ~0.5-0.6)"))
    z203 = OrderedDict(
        id="Z2.03", title="Risk-controlling prediction (CRC/LTT) with cluster-aware UCB",
        label_source="human gold (single labeller)", n=int(len(data["y"])),
        n_clips=int(len(set(data["clip"]))), labels_sha256=manifest["labels_sha256"],
        protocol=("Learn-then-Test (Angelopoulos et al.) on symmetric accept/reject threshold t; "
                  "risk = marginal false-accept P(accept AND bad); UCBs: cluster-Hoeffding, Maurer-Pontil "
                  "empirical-Bernstein, cluster bootstrap over K=7 clips; union correction over the grid; "
                  "LOCO-honest check (select t on 6 clips, evaluate the 7th)"),
        r=0.05, delta=0.1,
        exchangeability_note=("clips are the proposed exchangeable unit; window-level concentration "
                              "(Bentkus/exact-binomial over n=150) is anti-conservative because of "
                              "intra-clip correlation (design-effect n_eff(clip))"),
        cluster_floor=crc["cluster_floor"],
        feasibility_by_risk=crc["feasibility"],
        frontier_table=frontier_table,
        curves={k: {kk: vv for kk, vv in v.items()} for k, v in crc["curves"].items()},
        deployed=crc["deployed"],
        loco_honest=crc["loco_rows"],
        selective_ratio_curve=crc["selective_ratio_curve"],
        ood_evaluation_only=ood_note,
        command="OMP_NUM_THREADS=2 python3 eval/exp_conformal/conformal_risk.py",
    )

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        return o

    json.dump(clean(z203), open(f"{OUT}/Z2.03.json", "w"), indent=1, default=str)

    make_figure(frontier, block, crc, f"{FIG}/frontier.png")

    # ---------------- stdout ----------------
    print("=== Z2.01 frontier (clip-grouped LOCO split conformal) ===")
    print(f"{'cov':>6} {'review%':>8} {'cov_emp':>8} {'err|dec':>8} {'FA%':>6} {'FA|A%':>7} {'FR%':>6} {'empt%':>7}")
    for r in frontier:
        print(f"{r['coverage_target']:6.3f} {100*r['review_share']:8.1f} {100*r['coverage']:8.1f} "
              f"{100*(r['error_among_decided'] or 0):8.1f} {100*r['false_accept_rate_marginal']:6.1f} "
              f"{100*(r['false_accept_rate_selective'] or 0):7.1f} {100*r['false_reject_rate_marginal']:6.1f} "
              f"{100*r['n_empty']/r['n']:7.1f}")
    print("min review s.t. selective FA<=5%: cov=", op["selective_FAR"],
          "| marginal FA<=5%: cov=", op["marginal_FAR"],
          "| error|decided<=5%: cov=", op["error_among_decided"])
    print("FRONTIER TABLE:")
    for row in frontier_table:
        print("  ", row["method"], "| cov", row.get("coverage_target"), "| review", row.get("review"),
              "| FA_marg", row.get("false_accept_marginal"), "| bound:", row["guaranteed_bound"])
    print("E8 descriptive apply:", {k: round(v["review_share"], 3) for k, v in e8_apply.items()})
    print("=== Z2.02 coverage nominal vs empirical ===")
    print(f"ICC(p_ens)={block['icc']['p_good_score']:.3f} n_eff={block['icc']['n_eff_p_good']:.1f} "
          f"ICC(E)={block['icc']['E_true_label_score']:.3f} n_effE={block['icc']['n_eff_E']:.1f}")
    for row in block["rows"]:
        print(f"cov={row['coverage_target']:.3f} split={row['window_split']['coverage']:.3f} "
              f"deff={row['design_effect']['coverage']:.3f} boot={row['cluster_bootstrap']['coverage']:.3f} "
              f"| worst split={row['window_split']['worst_clip_coverage']:.3f}")
    print("Mondrian by score-stratum (cov, emp_cov, review):")
    for r in mondrian_strata:
        print(f"   {r['coverage_target']:.2f} cov={r['coverage']:.3f} review={r['review_share']:.3f}")
    print("Mondrian by clip (within-clip cross-fit) (cov, emp_cov, review):")
    for r in mondrian_clip:
        print(f"   {r['coverage_target']:.2f} cov={r['coverage']:.3f} review={r['review_share']:.3f}")
    print("=== Z2.03 CRC/LTT (r=0.05, delta=0.1) ===")
    print("cluster floor:", crc["cluster_floor"])
    print("deployed tiers:", {k: (v["t"], round(v["ucb"], 3), round(v["review_share"], 3))
                              for k, v in crc["deployed"].items()})
    print("per-tier min UCB:", {k: (round(v["min_ucb"], 3), v["min_ucb_t"])
                                for k, v in crc["curves"].items()})
    print("LOCO-honest rows:")
    for r in crc["loco_rows"][:3]:
        print("   ", r)
    if ood_note:
        print("OOD-60 (eval only):", ood_note)


if __name__ == "__main__":
    main()
