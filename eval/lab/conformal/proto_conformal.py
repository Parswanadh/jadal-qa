"""Direction 4: distribution-free risk control for small n (CPU-only).

What it does (read-only on shared data):
  1. Loads gold manifest (150 windows, 7 clips) + Arm-A features (150 rows).
  2. Builds a tiny baseline score s(x)=P(usable) via LEAVE-ONE-CLIP-OUT (LOCO)
     cross-fitted L2 logistic on 4 cheap signals. No in-sample optimism.
  3. Split-conformal / conformal-risk-control (CRC) thresholds for the
     false-reject rate P(s<tau | label=good) at alpha in {0.05, 0.10},
     calibrated on a pre-committed subset of clips, tested on held-out clips.
  4. Clopper-Pearson honest bounds (pooled + Kish cluster-adjusted + per-clip),
     ICC/design-effect estimates, cluster bootstrap CIs.
  5. 86-vs-84 Bonferroni fixed-sequence arithmetic check.
  6. Power calc: extra labels needed; active-labelling sketch inputs.

Outputs: results/*.json (this script) + REPORT.md (written separately).
"""
import json, math, os, re, time
import numpy as np

rng = np.random.default_rng(0)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

MANIFEST = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"
ARM_A = "/home/parshu/projects/jadal-qa-wt/spectral_data/arms/A/gold.json"

# ---------------- load ----------------
man = json.load(open(MANIFEST))
items = man["items"]
arm = json.load(open(ARM_A))["items"]

FEATS = ["blur_var", "brightness_mean", "motion_mean", "hands_visible_frac"]

def wearer_of(path):
    m = re.search(r"User_(\d+)", path or "")
    return "user_" + m.group(1) if m else "unknown"

rows = []
for it in items:
    f = arm[it["id"]]["features"]
    rows.append({
        "id": it["id"], "clip": it["clip"], "label": it["label"],
        "reason": it.get("reason"), "wearer": wearer_of(it.get("path", "")),
        "x": np.array([math.log1p(f["blur_var"]), f["brightness_mean"],
                        f["motion_mean"], f["hands_visible_frac"]], float),
    })
X = np.stack([r["x"] for r in rows])
y = np.array([r["label"] for r in rows])
clips = np.array([r["clip"] for r in rows])
uniq_clips = sorted(set(clips.tolist()))
print("n =", len(rows), "n_good =", int(y.sum()), "n_bad =", int((1 - y).sum()))
print("clips:", {c: (int((clips == c).sum()), int(y[clips == c].sum())) for c in uniq_clips})
print("wearers:", sorted(set(r["wearer"] for r in rows)))
json.dump({"n": len(rows), "n_good": int(y.sum()), "n_bad": int((1 - y).sum()),
           "clips": {c: {"n": int((clips == c).sum()), "n_good": int(y[clips == c].sum())} for c in uniq_clips},
           "wearers": sorted(set(r["wearer"] for r in rows))},
          open(f"{RES}/00_data_summary.json", "w"), indent=2)

# ---------------- LOCO cross-fitted scores ----------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

scores = np.zeros(len(rows))
for c in uniq_clips:
    te = clips == c
    tr = ~te
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    clf = LogisticRegression(C=1.0, max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
    scores[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
auc = roc_auc_score(y, scores)
print(f"LOCO AUROC = {auc:.4f}")
json.dump({"loco_auroc": auc, "scores": {r["id"]: float(s) for r, s in zip(rows, scores)},
           "features": FEATS, "model": "L2-logistic(C=1), LOCO cross-fitted"},
          open(f"{RES}/01_scores.json", "w"), indent=2)

# ---------------- (5) 86-vs-84 arithmetic check ----------------
from scipy.stats import beta as beta_dist, binom

def n_zero_fail(p, delta, m=1):
    return math.ceil(math.log(delta / m) / math.log(1 - p))

arith = {"grid": [], "cp_zero_fail_ub": {}}
for p in (0.05, 0.035, 0.03):
    for delta in (0.05,):
        for m in (1, 2, 3, 4, 6):
            arith["grid"].append({"p": p, "delta": delta, "m_bonf": m,
                                  "n_req_zero_fail": n_zero_fail(p, delta, m)})
for n in (84, 86):
    arith["cp_zero_fail_ub"][str(n)] = 1 - 0.05 ** (1.0 / n)  # 1-sided 95% CP with k=0
json.dump(arith, open(f"{RES}/02_arithmetic.json", "w"), indent=2)
print("CP zero-fail UB n=84:", arith["cp_zero_fail_ub"]["84"], " n=86:", arith["cp_zero_fail_ub"]["86"])

# ---------------- helpers ----------------
def cp_upper(k, n, delta=0.05):
    """1-sided 1-delta Clopper-Pearson upper bound."""
    if n == 0: return float("nan")
    if k >= n: return 1.0
    return float(beta_dist.ppf(1 - delta, k + 1, n - k))

def cp_two(k, n, alpha=0.05):
    if n == 0: return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)

def hb_pvalue(k, n, alpha):
    """Hoeffding-Bentkus p-value for H0: risk>=alpha given k/n observed (LTT)."""
    if n == 0: return 1.0
    phat = k / n
    h1 = phat * math.log(phat / alpha) + (1 - phat) * math.log((1 - phat) / (1 - alpha)) if 0 < phat < 1 else (math.log(1 / alpha) if phat == 0 else float("inf"))
    hoeff = math.exp(-n * h1) if phat <= alpha else 1.0
    bentkus = float(binom.cdf(k, n, alpha)) * math.e if k <= n * alpha else 1.0
    return min(hoeff, bentkus, 1.0)

# ---------------- (a) split-conformal / CRC, clip-grouped ----------------
# Pre-committed split BEFORE looking at scores: 4 biggest clips = calibration,
# 3 smallest = test (sizes from manifest counts, fixed alphabetically on ties).
sizes = sorted(uniq_clips, key=lambda c: (int((clips == c).sum()), c), reverse=True)
CAL_CLIPS, TEST_CLIPS = sizes[:4], sizes[4:]
print("CAL:", CAL_CLIPS, " TEST:", TEST_CLIPS)
cal_good = np.array([(r["clip"] in CAL_CLIPS) and r["label"] == 1 for r in rows])
tst_good = np.array([(r["clip"] in TEST_CLIPS) and r["label"] == 1 for r in rows])
tst_all = np.array([r["clip"] in TEST_CLIPS for r in rows])
n_cal = int(cal_good.sum())

crc_out = {"cal_clips": CAL_CLIPS, "test_clips": TEST_CLIPS, "n_cal_good": n_cal,
           "n_test_good": int(tst_good.sum()), "levels": {}}
for alpha in (0.05, 0.10):
    # CRC: tau-hat = sup{tau: n/(n+1)*Rhat(tau) + 1/(n+1) <= alpha}, B=1
    cand = np.unique(np.quantile(scores[cal_good], np.linspace(0, 1, 200)))
    tau_hat, rhat = None, None
    for t in cand:
        r = float((scores[cal_good] < t).mean())
        if (n_cal / (n_cal + 1)) * r + 1.0 / (n_cal + 1) <= alpha:
            tau_hat, rhat = float(t), r
    if tau_hat is None:
        tau_hat, rhat = float(cand[0]), float((scores[cal_good] < cand[0]).mean())
    k_cal = int((scores[cal_good] < tau_hat).sum())
    k_tst = int((scores[tst_good] < tau_hat).sum())
    n_tst = int(tst_good.sum())
    test_risk = k_tst / n_tst
    retain_tst = float((scores[tst_all] >= tau_hat).mean())
    # LTT fixed-sequence style p-value at tau_hat (HB, i.i.d. assumption stated)
    crc_out["levels"][str(alpha)] = {
        "tau_hat": tau_hat, "cal_emp_risk": rhat, "k_cal": k_cal, "n_cal": n_cal,
        "cal_cp_upper_95": cp_upper(k_cal, n_cal),
        "hb_pvalue_H0_risk_ge_alpha": hb_pvalue(k_cal, n_cal, alpha),
        "test_emp_risk": test_risk, "k_test": k_tst, "n_test": n_tst,
        "test_cp_upper_95": cp_upper(k_tst, n_tst),
        "test_retention_all": retain_tst,
    }
    print(f"alpha={alpha}: tau^={tau_hat:.4f} cal_risk={rhat:.3f} "
          f"CP_UP_cal={crc_out['levels'][str(alpha)]['cal_cp_upper_95']:.3f} "
          f"test_risk={test_risk:.3f} CP_UP_tst={crc_out['levels'][str(alpha)]['test_cp_upper_95']:.3f} "
          f"retain={retain_tst:.3f}")
json.dump(crc_out, open(f"{RES}/03_crc.json", "w"), indent=2)

# LOCO-as-calibration: CRC threshold recomputed leaving each clip out (stability)
loco_taus = {}
for alpha in (0.05, 0.10):
    taus = []
    for c in uniq_clips:
        m = (clips != c) & (y == 1)
        nn = int(m.sum())
        cand = np.unique(np.quantile(scores[m], np.linspace(0, 1, 200)))
        th = cand[0]
        for t in cand:
            if (nn / (nn + 1)) * float((scores[m] < t).mean()) + 1.0 / (nn + 1) <= alpha:
                th = t
        taus.append(float(th))
    loco_taus[str(alpha)] = {"taus": taus, "mean": float(np.mean(taus)), "sd": float(np.std(taus))}
json.dump(loco_taus, open(f"{RES}/04_crc_loco_stability.json", "w"), indent=2)

# ---------------- (c) operating point + CP bounds + clustering ----------------
# Operating point: Youden-J on pooled LOCO scores ("current best", stated as in-sample-selected)
ths = np.unique(np.quantile(scores, np.linspace(0, 1, 400)))
J = np.array([(y[scores >= t].sum() / y.sum()) + ((1 - y)[scores < t].sum() / (1 - y).sum()) - 1 for t in ths])
tau_star = float(ths[int(np.argmax(J))])
pred_good = scores >= tau_star
fr = int(((y == 1) & (~pred_good)).sum())   # false rejects (good called bad)
fa = int(((y == 0) & pred_good).sum())      # false accepts (bad called good)
n1, n0 = int(y.sum()), int((1 - y).sum())
op = {"tau_star": tau_star, "J": float(J.max()),
      "false_reject": {"k": fr, "n": n1, "rate": fr / n1, "cp95": cp_two(fr, n1)},
      "false_accept": {"k": fa, "n": n0, "rate": fa / n0, "cp95": cp_two(fa, n0)}}
print("OP tau*=", tau_star, "FR:", op["false_reject"], "FA:", op["false_accept"])

# per-clip error table + ICC (one-way random effects on false-reject indicator @ tau_star)
loss = ((y == 1) & (~pred_good)).astype(float)
good = y == 1
per_clip, grp, gm = [], [], float(loss[good].mean())
for c in uniq_clips:
    m = (clips == c) & good
    k, n = int(loss[m].sum()), int(m.sum())
    per_clip.append({"clip": c, "k": k, "n": n, "rate": (k / n if n else float("nan")),
                     "cp95": cp_two(k, n)})
    grp.append(loss[m])
a_groups = [g for g in grp if len(g) > 0]
G = len(a_groups)
N = sum(map(len, a_groups))
msb = sum(len(g) * (g.mean() - gm) ** 2 for g in a_groups) / (G - 1)
msw = sum(((g - g.mean()) ** 2).sum() for g in a_groups) / (N - G) if N - G > 0 else 0.0
n0bar = (N - sum(len(g) ** 2 for g in a_groups) / N) / (G - 1)
icc = max(0.0, float((msb - msw) / (msb + (n0bar - 1) * msw))) if (msb + (n0bar - 1) * msw) > 0 else 0.0
mbar = N / G
deff = 1 + (mbar - 1) * icc
n_eff = N / deff
op["per_clip_false_reject"] = per_clip
op["icc_windows_within_clip"] = icc
op["design_effect_good"] = deff
op["n_eff_good"] = n_eff
op["kish_adjusted_FR_CP95"] = cp_two(int(round(fr / deff)), int(round(n_eff)))
print(f"ICC={icc:.3f} DEFF={deff:.2f} n_eff={n_eff:.1f} kish-CP={op['kish_adjusted_FR_CP95']}") if False else print(
    f"ICC={icc:.3f} DEFF={deff:.2f} n_eff={n_eff:.1f} kish-CP={[round(v,4) for v in op['kish_adjusted_FR_CP95']]}")

# cluster bootstrap (resample clips w/ replacement, B=5000): FR rate + tau_hat stability
B = 5000
boot_fr, boot_tau05 = [], []
cidx = {c: np.where((clips == c) & good)[0] for c in uniq_clips}
for b in range(B):
    draw = rng.choice(uniq_clips, size=len(uniq_clips), replace=True)
    idx = np.concatenate([cidx[c] for c in draw])
    boot_fr.append(float(loss[idx].mean()))
    sb = scores[idx]
    cand = np.unique(np.quantile(sb, np.linspace(0, 1, 200)))
    nn = len(sb)
    th = cand[0]
    for t in cand:
        if (nn / (nn + 1)) * float((sb < t).mean()) + 1.0 / (nn + 1) <= 0.05:
            th = t
    boot_tau05.append(float(th))
op["cluster_bootstrap_FR95"] = [float(np.quantile(boot_fr, 0.025)), float(np.quantile(boot_fr, 0.975))]
op["cluster_bootstrap_tau05_95"] = [float(np.quantile(boot_tau05, 0.025)), float(np.quantile(boot_tau05, 0.975))]
print("cluster-boot FR 95%:", op["cluster_bootstrap_FR95"], " tau05 95%:", op["cluster_bootstrap_tau05_95"])
json.dump(op, open(f"{RES}/05_operating_point.json", "w"), indent=2)

# ---------------- power: labels needed ----------------
z = 1.96
p_obs = fr / n1
powr = {"p_assumed_FR": p_obs, "targets": []}
for w in (0.05, 0.03, 0.02):  # target half-widths
    n_iid = p_obs * (1 - p_obs) * (z / w) ** 2
    n_clu = n_iid * deff
    powr["targets"].append({
        "half_width": w,
        "n_good_windows_iid": math.ceil(n_iid),
        "n_good_windows_clustered": math.ceil(n_clu),
        "extra_good_windows_vs_today": max(0, math.ceil(n_clu - n1)),
        "extra_clips_like_today": max(0, math.ceil((n_clu - n1) / mbar)),
    })
powr["zero_fail_cert"] = {
    "n84_cp_ub": arith["cp_zero_fail_ub"]["84"],
    "note": "zero-failure certification of FR<=5%@95% needs n>=59 good windows with 0 rejects; "
            "any observed reject forces larger n (see arithmetic grid)."}
powr["deff_used"] = deff
json.dump(powr, open(f"{RES}/06_power.json", "w"), indent=2)

# ---------------- manifest of this run ----------------
import subprocess
def git(*a):
    try: return subprocess.check_output(["git", *a], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
json.dump({"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "git_sha": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain")),
           "python": __import__("sys").version.split()[0], "seed": 0,
           "note": "CPU-only, numpy/sklearn. Scores are LOCO cross-fitted baseline, NOT a product model."},
          open(f"{RES}/00_manifest.json", "w"), indent=2)
print("done.")
