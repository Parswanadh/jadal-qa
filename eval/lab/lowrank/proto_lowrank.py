#!/usr/bin/env python3
"""Direction 1: LOW-RANK + SPARSE video decomposition prototype (Robust-PCA-flavoured).

Temporal matrix M (pixels x frames) from downscaled luma per 10-s window,
truncated SVD rank profile + cheap sparse proxy + subspace duplicate cue.

House rules: CPU only, OMP_NUM_THREADS=2 (set by caller), read-only decode,
deterministic sampling. No GMC (global-motion compensation) -- documented in REPORT.md.

Usage:
  OMP_NUM_THREADS=2 /home/parshu/envs/jadal/bin/python proto_lowrank.py [--out DIR] [--frames N]
"""
import argparse, json, os, sys, time, hashlib
import numpy as np
import cv2

# ---------------------------------------------------------------- config
W = H = 64            # downscaled luma resolution
N_FRAMES = 50         # deterministic uniform sample per window (~10fps*10s=100 -> stride 2)
K_SPARSE = 2          # truncation rank for sparse-energy fraction
K_SUB = 5             # subspace rank for duplicate cue
GOLD = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"
DUPS = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_dups.json"

def decode_uniform(path, start_s=None, end_s=None, n=N_FRAMES, size=(W, H)):
    """Read-only deterministic decode. Returns (frames float32 [0,1] (n,h,w), t_decode_ms).
    Window mode: seek to start_s, read through end_s, uniform linspace sample of n frames.
    Full-clip mode (start_s=None): uniform sample across whole file."""
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, (time.perf_counter() - t0) * 1000
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    if start_s is None:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = np.linspace(0, max(total - 1, 0), n).round().astype(int)
        want = set(idx.tolist())
        got = {}
        i = 0
        ok = True
        while cap.isOpened() and len(got) < len(want) and ok:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            if i in want:
                g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
                got[i] = g
            i += 1
        cap.release()
        if len(got) < 2:
            return None, (time.perf_counter() - t0) * 1000
        frames = np.stack([got[k] for k in sorted(got)], axis=0).astype(np.float32) / 255.0
        # deterministic top-up/pad by repeating last frame if short
        if len(frames) < n:
            pad = np.repeat(frames[-1:], n - len(frames), axis=0)
            frames = np.concatenate([frames, pad], axis=0)
        frames = frames[:n]
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(start_s * fps) - 2, 0))
        buf = []  # (t, gray_small)
        while True:
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t < start_s - 1e-6:
                continue
            if t > end_s + 1.0 / max(fps, 1):
                break
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
            buf.append(g)
            if len(buf) > 220:
                break
        cap.release()
        if len(buf) < 4:
            return None, (time.perf_counter() - t0) * 1000
        buf = np.stack(buf, axis=0).astype(np.float32) / 255.0
        take = np.linspace(0, len(buf) - 1, n).round().astype(int)
        frames = buf[take]
    dt = (time.perf_counter() - t0) * 1000
    return frames, dt

def decompose(frames):
    """M: (p x n) raw luma columns; M0: per-pixel temporal-mean-centred.
    Returns dict of features + right subspace Vk (n x K_SUB)."""
    t0 = time.perf_counter()
    n, h, w = frames.shape
    p = h * w
    M = frames.reshape(n, p).T.astype(np.float64)  # p x n
    mu = M.mean(axis=1, keepdims=True)
    M0 = M - mu
    # economy SVD of centred matrix
    U, s, Vt = np.linalg.svd(M0, full_matrices=False)
    frob2 = float((s ** 2).sum())
    s1 = float(s[0]) if s[0] > 0 else 1e-12
    stable_rank = frob2 / (s1 ** 2)
    # sparse proxy: rank-K_SPARSE residual L1 fraction
    k = min(K_SPARSE, len(s))
    L = (U[:, :k] * s[:k]) @ Vt[:k, :]
    num = float(np.abs(M0 - L).sum())
    den = float(np.abs(M0).sum())
    sparse_frac = num / den if den > 0 else 0.0
    # Marchenko-Pastur signal rank on globally-standardised entries
    # Z = M0/std; noise edge for singular values: sqrt(n)*(1+sqrt(p/n))
    std = float(M0.std())
    if std <= 0:
        mp_rank = 0
    else:
        c = p / n
        T = np.sqrt(n) * (1.0 + np.sqrt(c))
        mp_rank = int((s / std > T).sum())
    kk = min(K_SUB, Vt.shape[0])
    Vk = Vt[:kk, :].T.copy()  # n x kk
    dt = (time.perf_counter() - t0) * 1000
    top = (s[:8] / s1).tolist() if len(s) else []
    return {"stable_rank": float(stable_rank), "sparse_frac_k2": float(sparse_frac),
            "mp_rank": mp_rank, "sigma1_share": float(s1**2 / frob2) if frob2 > 0 else 0.0,
            "top_profile": top, "Vk": Vk, "t_ms": dt, "p": p, "n": n}

def subspace_dist(A, B):
    """Chordal-type distance between orthonormal column spaces: sum(sin^2 theta)."""
    qa, _ = np.linalg.qr(A)
    qb, _ = np.linalg.qr(B)
    cos = np.linalg.svd(qa.T @ qb, compute_uv=False)
    cos = np.clip(cos, -1, 1)
    return float(((1 - cos ** 2).sum()))

def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    a = ap.parse_args()
    n_frames = a.frames
    os.makedirs(a.out, exist_ok=True)

    gold = json.load(open(GOLD))["items"]
    feats, vks, tdec, tsvd, ok = [], [], [], [], []
    for it in gold:
        fr, td = decode_uniform(it["path"], it["start_s"], it["end_s"], n=n_frames)
        tdec.append(td)
        if fr is None:
            feats.append({"stable_rank": np.nan, "sparse_frac_k2": np.nan, "mp_rank": 0,
                          "sigma1_share": np.nan, "top_profile": []})
            vks.append(None); tsvd.append(0.0); ok.append(False)
            continue
        d = decompose(fr)
        vks.append(d.pop("Vk"))
        tsvd.append(d.pop("t_ms")); d.pop("p"); d.pop("n")
        feats.append(d); ok.append(True)
    F = np.array([[f["stable_rank"], f["sparse_frac_k2"], f["mp_rank"]] for f in feats], float)
    y = np.array([it["label"] for it in gold], int)
    clips = np.array([it["clip"] for it in gold])
    ids = [it["id"] for it in gold]
    reasons = [it.get("reason", "") or "" for it in gold]

    # ---- LOCO: logistic on train clips, test held-out clip
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    uniq = sorted(set(clips.tolist()))
    fold_pred = np.full(len(y), np.nan)
    fold_auc = {}
    for c in uniq:
        te = clips == c; tr = ~te
        # nan-safe: rows failing decode get score 0 (predict bad); exclude from fit
        tr_ok = tr & ~np.isnan(F).any(axis=1)
        if tr_ok.sum() < 5 or len(np.unique(y[tr_ok])) < 2:
            continue
        mu = np.nanmean(F[tr_ok], axis=0); sd = np.nanstd(F[tr_ok], axis=0) + 1e-9
        Ztr = (F[tr_ok] - mu) / sd
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Ztr, y[tr_ok])
        Zte = (F[te] - mu) / sd
        Zte[~np.isfinite(Zte)] = -3.0  # failed decode -> strongly "bad"
        fold_pred[te] = clf.predict_proba(Zte)[:, 1]
    fold_pred[~np.isfinite(fold_pred)] = 0.0
    pooled_auc = auroc(y, fold_pred)
    for c in uniq:
        m = clips == c
        fold_auc[c] = auroc(y[m], fold_pred[m])
    # threshold at pooled Youden J for per-reason sensitivity
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, fold_pred)
    J = tpr - fpr
    cut = float(thr[int(np.argmax(J))])
    pred_bad = fold_pred < cut
    per_reason = {}
    for r in sorted(set(reasons)):
        m = np.array([x == r for x in reasons])
        per_reason[r] = {"n": int(m.sum()),
                         "flagged_bad_frac": float(pred_bad[m & (y == 0)].mean()) if (m & (y == 0)).sum() else None,
                         "flagged_good_frac": float((~pred_bad[m & (y == 1)]).mean()) if (m & (y == 1)).sum() else None}
    # ---- clip-bootstrap 95% CI for pooled LOCO AUROC (1000 reps)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(1000):
        cs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(clips == c)[0] for c in cs])
        v = auroc(y[idx], fold_pred[idx])
        if np.isfinite(v):
            boots.append(v)
    boots = np.array(boots)
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if len(boots) > 100 else [None, None]
    boot_which = f"clip-bootstrap over {len(uniq)} clips, 1000 reps, {len(boots)} valid"

    # ---- gold duplicate cue: pairwise subspace distances (11175 pairs)
    idx_ok = [i for i, v in enumerate(vks) if v is not None]
    is_golddup = np.array([reasons[i] == "DUPLICATE" and y[i] == 0 for i in range(len(gold))])
    dd, pl = [], []
    for a_i in range(len(idx_ok)):
        for b_i in range(a_i + 1, len(idx_ok)):
            i, j = idx_ok[a_i], idx_ok[b_i]
            dd.append(subspace_dist(vks[i], vks[j]))
            pl.append(1 if (is_golddup[i] and is_golddup[j]) else 0)
    dd = np.array(dd); pl = np.array(pl)
    gold_dup_auc = auroc(pl, -dd)
    gold_dup_stat = {"n_dupdup_pairs": int(pl.sum()), "n_pairs": int(len(pl)),
                     "median_dist_dupdup": float(np.median(dd[pl == 1])) if pl.sum() else None,
                     "median_dist_rest": float(np.median(dd[pl == 0])),
                     "auc_dupdup_vs_rest": gold_dup_auc}

    # ---- constructed dups manifest
    D = json.load(open(DUPS))
    files = D["files"]
    fcache = {}
    for k, p in files.items():
        fr, _ = decode_uniform(p, n=n_frames)
        fcache[k] = decompose(fr)["Vk"] if fr is not None else None
    # keep Vk before pop issue: decompose returns Vk popped? No -- we popped only in gold loop.
    # Here decompose returns Vk inside dict; extract:
    for k in list(fcache.keys()):
        pass
    kind_s, kind_y, kind_k = [], [], []
    for pr in D["pairs"]:
        A, B = fcache.get(pr["a"]), fcache.get(pr["b"])
        if A is None or B is None:
            continue
        kind_s.append(subspace_dist(A, B))
        kind_y.append(1 if pr["is_duplicate"] else 0)
        kind_k.append(pr["kind"])
    kind_s = np.array(kind_s); kind_y = np.array(kind_y); kind_k = np.array(kind_k)
    dup_auc = auroc(kind_y, -kind_s)
    by_kind = {}
    for k in sorted(set(kind_k.tolist())):
        m = kind_k == k
        neg = kind_k == "easy_negative"
        # one-vs-negatives (easy negatives as reference background)
        yy = np.concatenate([np.ones(m.sum()), np.zeros(neg.sum())])
        ss = np.concatenate([-kind_s[m], -kind_s[neg]])
        by_kind[k] = {"n": int(m.sum()), "auc_vs_easyneg": auroc(yy, ss),
                      "median_dist": float(np.median(kind_s[m]))}

    json.dump([{"id": ids[i], "clip": clips[i], "label": int(y[i]), "reason": reasons[i],
                **{kk2: feats[i][kk2] for kk2 in ("stable_rank", "sparse_frac_k2", "mp_rank", "sigma1_share")},
                "decode_ok": bool(ok[i]), "t_decode_ms": tdec[i], "t_svd_ms": tsvd[i],
                "loco_score": float(fold_pred[i])} for i in range(len(gold))],
              open(os.path.join(a.out, "features.json"), "w"), indent=1)
    json.dump({"pooled_loco_auroc": pooled_auc, "per_clip_auroc": fold_auc,
               "ci95": ci, "bootstrap": boot_which, "youden_cut": cut,
               "per_reason": per_reason, "features": ["stable_rank", "sparse_frac_k2", "mp_rank"],
               "model": "StandardScaler+LogisticRegression(C=1.0), train clips -> test clip"},
              open(os.path.join(a.out, "loco.json"), "w"), indent=1)
    json.dump({"gold_dup": gold_dup_stat, "constructed_auc_dup_vs_allneg": dup_auc,
               "constructed_by_kind": by_kind, "K_SUB": K_SUB,
               "metric": "sum(sin^2 principal angles) between top-K right singular subspaces (lower=more similar)"},
              open(os.path.join(a.out, "dups.json"), "w"), indent=1)
    cost = {"ms_decode_mean": float(np.mean(tdec)), "ms_svd_mean": float(np.mean(tsvd)),
            "ms_total_mean": float(np.mean(tdec) + np.mean(tsvd)),
            "n_windows": len(gold), "n_frames": n_frames, "res": [W, H],
            "decode_failures": int(np.sum(~np.array(ok)))}
    json.dump(cost, open(os.path.join(a.out, "cost.json"), "w"), indent=1)
    json.dump({"pooled_loco_auroc": pooled_auc, "ci95": ci, "per_clip_auroc": fold_auc,
               "constructed_dup_auc": dup_auc, "cost": cost},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps({"pooled_loco_auroc": pooled_auc, "ci95": ci, "per_clip": fold_auc,
                       "gold_dup": gold_dup_stat, "constructed_dup_auc": dup_auc,
                       "by_kind": by_kind, "cost": cost}, indent=1))

if __name__ == "__main__":
    main()
