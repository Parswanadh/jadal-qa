"""Direction 6: INFORMATION-THEORETIC cues (CPU-only, no GPU).

(a) H.264 bits-per-frame WITHOUT pixel decode: per-window ffprobe
    -show_frames (pkt_size + pict_type), parsed from one full-clip pass per
    clip (7 passes, no per-window seeking). Per-clip normalised features
    (rate control confounds raw bytes across files): P-mean z, I-mean z,
    bytes/frame z vs clip-global median/IQR (label-free), plus raw
    P-byte-fraction / P-count-fraction.
(b) Permutation entropy (Bandt-Pompe, m=3 and m=4, tau=1) of the pixel-domain
    frame-difference (MAD) series per window. Hypothesis: looped/periodic
    content -> low PE.
(c) Head-to-head compressed-domain vs pixel-domain (same LOCO protocol):
    univariate + grouped L2-logistic LOCO AUROC, clip-bootstrap 95% CI,
    bootstrap CI on the (combined - pixel) AUROC gap.
(d) MDL-flavoured 'useful bits' is a one-paragraph analysis in REPORT.md,
    not a feature.

Read-only on shared data. Outputs results/*.json (this script).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import json, math, subprocess, time
import numpy as np

rng = np.random.default_rng(6)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

MANIFEST = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"
FFPROBE = "ffprobe"
PY = "/home/parshu/envs/jadal/bin/python"

man = json.load(open(MANIFEST))
items = man["items"]
N = len(items)
y = np.array([it["label"] for it in items])
clips = np.array([it["clip"] for it in items])
uniq_clips = sorted(set(clips.tolist()))
print(f"n={N} n_good={int(y.sum())} n_bad={int((1-y).sum())} clips={uniq_clips}")

t_all = time.time()

# ---------------- (a) compressed-domain: one ffprobe pass per clip ----------------
def ffprobe_frames(path):
    out = subprocess.check_output(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=best_effort_timestamp_time,pkt_size,pict_type",
         "-of", "csv=p=0", path], text=True)
    rows = []
    for ln in out.splitlines():
        p = ln.strip().split(",")
        if len(p) < 3:
            continue
        try:
            rows.append((float(p[0]), int(p[1]), p[2].strip()))
        except ValueError:
            continue
    return rows

clip_frames, ffprobe_sec = {}, {}
paths = {}
for it in items:
    paths.setdefault(it["clip"], it["path"])
for c, p in paths.items():
    t0 = time.time()
    clip_frames[c] = ffprobe_frames(p)
    ffprobe_sec[c] = time.time() - t0
    print(f"ffprobe {c}: {len(clip_frames[c])} frames in {ffprobe_sec[c]:.1f}s", flush=True)

# clip-global (label-free) P/I size norms for per-stream normalisation
clip_norm = {}
for c, fr in clip_frames.items():
    ps = np.array([s for _, s, t in fr if t == "P"], float)
    ii = np.array([s for _, s, t in fr if t == "I"], float)
    al = np.array([s for _, s, _ in fr], float)
    def med_iqr(a):
        if len(a) == 0:
            return (0.0, 1.0)
        m = float(np.median(a))
        q = float(np.subtract(*np.percentile(a, [75, 25])))
        return (m, q if q > 0 else 1.0)
    clip_norm[c] = {"p": med_iqr(ps), "i": med_iqr(ii), "all": med_iqr(al)}

def znorm(v, med_iqr):
    m, q = med_iqr
    return (v - m) / q

comp_rows = []
for it in items:
    fr = [(t, s, ty) for (t, s, ty) in clip_frames[it["clip"]]
          if it["start_s"] <= t < it["end_s"]]
    sz = np.array([s for _, s, _ in fr], float)
    ps = np.array([s for _, s, t in fr if t == "P"], float)
    ii = np.array([s for _, s, t in fr if t == "I"], float)
    bs = np.array([s for _, s, t in fr if t == "B"], float)
    n = len(fr)
    tot = float(sz.sum()) if n else 0.0
    p_sum = float(ps.sum())
    p_mean = float(ps.mean()) if len(ps) else 0.0
    p_med = float(np.median(ps)) if len(ps) else 0.0
    p_p10 = float(np.percentile(ps, 10)) if len(ps) else 0.0
    i_mean = float(ii.mean()) if len(ii) else clip_norm[it["clip"]]["i"][0]
    bpf = tot / n if n else 0.0
    cn = clip_norm[it["clip"]]
    comp_rows.append({
        "id": it["id"],
        "n_frames": n,
        "n_I": int(len(ii)), "n_P": int(len(ps)), "n_B": int(len(bs)),
        "bytes_total": tot,
        "p_mean": p_mean, "p_med": p_med, "p_p10": p_p10,
        "i_mean": i_mean,
        "p_byte_frac": (p_sum / tot) if tot > 0 else 0.0,
        "p_count_frac": (len(ps) / n) if n else 0.0,
        "p_mean_z": znorm(p_mean, cn["p"]),
        "i_mean_z": znorm(i_mean, cn["i"]),
        "bytes_pf_z": znorm(bpf, cn["all"]),
    })
json.dump({"clip_norm": clip_norm,
           "features": {r["id"]: r for r in comp_rows}},
          open(f"{RES}/01_bitframes.json", "w"), indent=2)
t_comp = time.time() - t_all
print(f"compressed-domain pass: {t_comp:.1f}s total = {1000*t_comp/N:.0f} ms/window "
      f"(ffprobe {sum(ffprobe_sec.values()):.1f}s + parse)")

# ---------------- (b) pixel-domain: MAD series + sharpness + PE ----------------
import cv2
print("cv2:", cv2.__version__)

def perm_entropy(x, m):
    x = np.asarray(x, float)
    n = len(x) - m + 1
    if n < 2:
        return float("nan")
    fac = math.factorial(m)
    counts = {}
    for i in range(n):
        w = x[i:i + m]
        order = tuple(np.argsort(w, kind="stable"))  # ties -> position order, deterministic
        counts[order] = counts.get(order, 0) + 1
    probs = np.array(list(counts.values()), float) / n
    return float(-(probs * np.log(probs)).sum() / math.log(fac))  # normalised to [0,1]

t0pix = time.time()
pix_rows = []
cap_cache = {}
TARGET = 50          # keep 50 frames/window (~5 fps over 10 s)
STRIDE = 2
W, H = 64, 64
for it in items:
    key = it["path"]
    cap = cap_cache.get(key)
    if cap is None:
        cap = cv2.VideoCapture(key)
        cap_cache[key] = cap
    cap.set(cv2.CAP_PROP_POS_MSEC, it["start_s"] * 1000.0)
    kept = []
    reads = 0
    while len(kept) < TARGET and reads < TARGET * STRIDE + 20:
        ok, fr = cap.read()
        reads += 1
        if not ok:
            break
        if (reads - 1) % STRIDE == 0:
            g = cv2.cvtColor(cv2.resize(fr, (W, H)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            kept.append(g)
    A = np.stack(kept) if len(kept) >= 4 else None
    if A is None:
        pix_rows.append({"id": it["id"], "n_kept": len(kept),
                         "mad_mean": float("nan"), "mad_med": float("nan"),
                         "mad_p10": float("nan"), "sharp_mean": float("nan"),
                         "pe_m3": float("nan"), "pe_m4": float("nan")})
        continue
    mad = np.abs(np.diff(A, axis=0)).mean(axis=(1, 2))
    lap = [float(cv2.Laplacian(A[k], cv2.CV_32F).var()) for k in range(0, len(A), 10)]
    pix_rows.append({
        "id": it["id"], "n_kept": int(len(A)),
        "mad_mean": float(mad.mean()), "mad_med": float(np.median(mad)),
        "mad_p10": float(np.percentile(mad, 10)),
        "sharp_mean": float(np.mean(lap)),
        "pe_m3": perm_entropy(mad, 3), "pe_m4": perm_entropy(mad, 4),
    })
for c in cap_cache.values():
    c.release()
t_pix = time.time() - t0pix
print(f"pixel pass: {t_pix:.1f}s total = {1000*t_pix/N:.0f} ms/window (decode+resize+MAD+PE)")
json.dump({"features": {r["id"]: r for r in pix_rows},
           "decode": {"target_frames": TARGET, "stride": STRIDE, "size": [W, H],
                      "note": "grab-free sequential read after POS_MSEC seek; "
                              "kept-frame count stored per window"}},
          open(f"{RES}/02_pixel.json", "w"), indent=2)

# ---------------- (c) LOCO evaluation ----------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

C = {r["id"]: r for r in comp_rows}
P = {r["id"]: r for r in pix_rows}
FEATS = {
    "p_mean_z":      (["p_mean_z"], "compressed"),
    "i_mean_z":      (["i_mean_z"], "compressed"),
    "p_byte_frac":   (["p_byte_frac"], "compressed"),
    "bytes_pf_z":    (["bytes_pf_z"], "compressed"),
    "pe_m3":         (["pe_m3"], "pe"),
    "pe_m4":         (["pe_m4"], "pe"),
    "mad_mean":      (["mad_mean"], "pixel"),
    "sharp_mean":    (["sharp_mean"], "pixel"),
    "compressed_all": (["p_mean_z", "i_mean_z", "p_byte_frac", "bytes_pf_z"], "compressed"),
    "pixel_all":     (["mad_mean", "sharp_mean"], "pixel"),
    "pe_all":        (["pe_m3", "pe_m4"], "pe"),
    "compressed+pixel": (["p_mean_z", "i_mean_z", "p_byte_frac", "bytes_pf_z",
                          "mad_mean", "sharp_mean"], "combined"),
    "all":           (["p_mean_z", "i_mean_z", "p_byte_frac", "bytes_pf_z",
                       "mad_mean", "sharp_mean", "pe_m3", "pe_m4"], "combined"),
}
ids = [it["id"] for it in items]
F = {}
for k, (cols, _) in FEATS.items():
    M = []
    for i in ids:
        v = [(C[i][c] if c in C[i] else P[i][c]) for c in cols]
        M.append(v)
    F[k] = np.array(M, float)

def loco_scores(X):
    s = np.zeros(N)
    for c in uniq_clips:
        te, tr = clips == c, clips != c
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        clf = LogisticRegression(C=1.0, max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
        s[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    return s

def boot_auc_ci(s, B=2000):
    aucs = []
    idx = {c: np.where(clips == c)[0] for c in uniq_clips}
    for _ in range(B):
        draw = rng.choice(uniq_clips, size=len(uniq_clips), replace=True)
        j = np.concatenate([idx[c] for c in draw])
        if len(np.unique(y[j])) < 2:
            continue
        aucs.append(roc_auc_score(y[j], s[j]))
    aucs = np.array(aucs)
    return [float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))], int(len(aucs))

loco_out, diffs = {}, {}
names = list(FEATS)
for k in names:
    s = loco_scores(np.nan_to_num(F[k], nan=0.0))
    auc = float(roc_auc_score(y, s))
    ci, nb = boot_auc_ci(s)
    loco_out[k] = {"family": FEATS[k][1], "cols": FEATS[k][0],
                   "loco_auroc": auc, "clip_bootstrap_95": ci, "n_boot": nb,
                   "scores": {i: float(v) for i, v in zip(ids, s)}}
    print(f"{k:16s} LOCO AUROC={auc:.4f} 95%CI=[{ci[0]:.3f},{ci[1]:.3f}]", flush=True)

# bootstrap CI on AUROC gaps (same resamples): combined-vs-pixel, compressed-vs-pixel
s_pix = np.array([loco_out["pixel_all"]["scores"][i] for i in ids])
s_cmp = np.array([loco_out["compressed_all"]["scores"][i] for i in ids])
s_cmb = np.array([loco_out["all"]["scores"][i] for i in ids])
idx = {c: np.where(clips == c)[0] for c in uniq_clips}
g1, g2 = [], []
B = 2000
nb = 0
for _ in range(B):
    draw = rng.choice(uniq_clips, size=len(uniq_clips), replace=True)
    j = np.concatenate([idx[c] for c in draw])
    if len(np.unique(y[j])) < 2:
        continue
    nb += 1
    g1.append(roc_auc_score(y[j], s_cmb[j]) - roc_auc_score(y[j], s_pix[j]))
    g2.append(roc_auc_score(y[j], s_cmp[j]) - roc_auc_score(y[j], s_pix[j]))
diffs = {
    "all_minus_pixel_all": {"mean": float(np.mean(g1)),
                            "ci95": [float(np.quantile(g1, 0.025)), float(np.quantile(g1, 0.975))]},
    "compressed_all_minus_pixel_all": {"mean": float(np.mean(g2)),
                            "ci95": [float(np.quantile(g2, 0.025)), float(np.quantile(g2, 0.975))]},
    "n_boot": nb,
}
print("gap all-pixel:", diffs["all_minus_pixel_all"])
print("gap compressed-pixel:", diffs["compressed_all_minus_pixel_all"])
json.dump({"cues": {k: {kk: vv for kk, vv in v.items() if kk != "scores"} for k, v in loco_out.items()},
           "gaps": diffs,
           "model": "L2-logistic(C=1), LOCO cross-fitted, clip-bootstrap B=2000, seed=6"},
          open(f"{RES}/03_loco.json", "w"), indent=2)
json.dump({k: v["scores"] for k, v in loco_out.items()},
          open(f"{RES}/04_scores.json", "w"), indent=2)

# ---------------- descriptives per reason ----------------
reasons = sorted(set(it.get("reason") or "GOOD" for it in items))
desc = {}
keys = ["p_mean_z", "i_mean_z", "p_byte_frac", "bytes_pf_z",
        "mad_mean", "sharp_mean", "pe_m3", "pe_m4"]
allf = {i: {**C[i], **{k: P[i][k] for k in
            ["mad_mean", "sharp_mean", "pe_m3", "pe_m4"]}} for i in ids}
for r in reasons:
    j = [it["id"] for it in items if (it.get("reason") or "GOOD") == r]
    lab = [it["label"] for it in items if (it.get("reason") or "GOOD") == r]
    d = {"n": len(j), "label_mean": float(np.mean(lab))}
    for k in keys:
        v = np.array([allf[i][k] for i in j], float)
        v = v[~np.isnan(v)]
        d[k] = {"med": float(np.median(v)) if len(v) else float("nan"),
                "q25": float(np.percentile(v, 25)) if len(v) else float("nan"),
                "q75": float(np.percentile(v, 75)) if len(v) else float("nan")} if len(v) else {}
    desc[r] = d
json.dump(desc, open(f"{RES}/05_descriptives.json", "w"), indent=2)
for r, d in desc.items():
    print(r, "n=%d" % d["n"],
          "p_mean_z=%.2f" % d.get("p_mean_z", {}).get("med", float("nan")),
          "i_mean_z=%.2f" % d.get("i_mean_z", {}).get("med", float("nan")),
          "mad=%.2f" % d.get("mad_mean", {}).get("med", float("nan")),
          "pe3=%.2f" % d.get("pe_m3", {}).get("med", float("nan")))

# ---------------- cost + verdicts ----------------
cost = {
    "compressed_no_decode_s_total": float(sum(ffprobe_sec.values())),
    "compressed_ms_per_window": float(1000 * t_comp / N),
    "ffprobe_per_clip_s": ffprobe_sec,
    "pixel_decode_s_total": float(t_pix),
    "pixel_ms_per_window": float(1000 * t_pix / N),
    "note": "compressed path parses ffprobe frame headers only (no pixel "
            "output); pixel path = cv2 decode+resize(64x64)+MAD+PE. "
            "Timings on this box, OMP_NUM_THREADS=2.",
}
json.dump(cost, open(f"{RES}/06_cost.json", "w"), indent=2)

def git(*a):
    try:
        return subprocess.check_output(["git", *a], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
json.dump({"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "git_sha": git("rev-parse", "HEAD"),
           "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
           "python": __import__("sys").version.split()[0],
           "cv2": cv2.__version__,
           "numpy": np.__version__,
           "seed": 6, "runtime_s": time.time() - t_all,
           "note": "CPU-only. Deterministic given identical ffprobe/cv2 builds "
                   "(ties in PE broken by position order; no RNG in features; "
                   "bootstrap seed fixed)."},
          open(f"{RES}/00_manifest.json", "w"), indent=2)
print(f"TOTAL {time.time()-t_all:.0f}s — done.")
