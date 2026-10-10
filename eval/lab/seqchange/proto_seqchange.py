#!/usr/bin/env python3
"""Sequential early-reject tests on noise-normalised per-frame statistics.

Direction 3: SEQUENTIAL CHANGE DETECTION with exact guarantees.
Streaming by design: per-frame update() API, fed in 2-s chunks in order.

Statistics per frame t (64x64 luma):
  MAD_t    = mean|Y_t - Y_{t-1}|            (motion / frozen-dark collapse)
  BRI_t    = mean(Y_t)                       (exposure)
  DARK_t   = frac(Y_t < 16)                  (clip fraction)
  SHARP_t  = mean(|Sx|+|Sy|) Sobel energy    (blur/focus)

Normalisation: fixed robust scale (documented choice, NOT a Poisson-Gaussian
flat-block fit): mu0 = median of pooled TRAIN-good MAD_t; sigma0 = 1.4826*MAD.
z_t = (mu0 - x_t)/sigma0, positive = degradation direction (collapse).

Methods (one-sided, reject when statistic collapses):
  M1 CUSUM:  S_n = max(0, S_{n-1} + z_t - k), fire when S_n > h.
             k=0.5, h=5.0 (ARL reasoning in REPORT).
  M2 SPRT:   Gaussian mean-shift H0: N(mu0,sig^2) vs H1: N(mu0-d*sig,sig^2),
             d=1.0, alpha=0.05, beta=0.10; LLR_n = d*z_t - d^2/2;
             fire when cumsum >= b=log((1-beta)/alpha)=2.8907;
             accept-H0 bound a=log(beta/(1-alpha))=-2.2513 (never accept early:
             truncated at N; no-fire = accept).
  M3 EB-CS:  mixture / empirical-Bernstein-style time-uniform confidence
             sequence (Howard et al. 2021, sigma-known Gaussian-mixture
             boundary, rho=1, alpha=0.05):
             width_n = sig*sqrt(2(n*rho^2+1)/(n^2*rho^2) * log(sqrt(n*rho^2+1)/alpha))
             fire when mu0 - xbar_n > width_n (lower CS excludes mu0), n>=5.

Deterministic: fixed stride sampling, no RNG. CPU only, OMP_NUM_THREADS=2.
"""
import argparse, hashlib, json, os, sys, time
from collections import defaultdict

import cv2
import numpy as np

GOLD = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"
SYNTH = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_synth.json"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

FPS_NATIVE = 10.0
STRIDE = 2            # decode every 2nd frame -> 5 fps
FPS_EFF = FPS_NATIVE / STRIDE
SMALL = (64, 64)
CHUNK_S = 2.0         # streaming chunk size (s)
CHUNK_FRAMES = int(CHUNK_S * FPS_EFF)  # 10 frames

# ---- fixed a-priori detector parameters (NOT tuned on test) ----
K_CUSUM, H_CUSUM = 0.5, 5.0
DELTA_SPRT, ALPHA_SPRT, BETA_SPRT = 1.0, 0.05, 0.10
B_SPRT = float(np.log((1 - BETA_SPRT) / ALPHA_SPRT))   # 2.8907
A_SPRT = float(np.log(BETA_SPRT / (1 - ALPHA_SPRT)))   # -2.2513
ALPHA_CS, RHO_CS, N_MIN_CS = 0.05, 1.0, 5


# ---------------- feature extraction ----------------
def decode_small_series(path, start_s=0.0, end_s=None, max_frames=60):
    """Deterministic decode: seek to start, take every STRIDE-th frame, 64x64 gray."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, {"error": "undecodable"}
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_NATIVE
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        t = start_s + idx / (fps or FPS_NATIVE)
        if end_s is not None and t >= end_s:
            break
        if idx % STRIDE == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, SMALL, interpolation=cv2.INTER_AREA)
            frames.append(g)
        idx += 1
    cap.release()
    if len(frames) < 6:
        return None, {"error": f"too_few_frames={len(frames)}"}
    return frames, {"n_frames": len(frames)}


def frame_stats(frames):
    Y = [f.astype(np.float32) for f in frames]
    bri = np.array([y.mean() for y in Y])
    dark = np.array([float((y < 16).mean()) for y in Y])
    mad = np.array([0.0] + [float(np.abs(Y[i] - Y[i - 1]).mean()) for i in range(1, len(Y))])
    shp = np.array([float((np.abs(cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)) +
                            np.abs(cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3))).mean()) for y in Y])
    return {"mad": mad[1:], "bri": bri[1:], "dark": dark[1:], "sharp": shp[1:],
            "n": len(Y) - 1}


# ---------------- streaming detectors ----------------
class CUSUM:
    def __init__(self, k=K_CUSUM, h=H_CUSUM):
        self.k, self.h, self.s, self.n, self.fire_at = k, h, 0.0, 0, None
    def update(self, z):
        self.n += 1
        self.s = max(0.0, self.s + z - self.k)
        if self.fire_at is None and self.s > self.h:
            self.fire_at = self.n
        return self.fire_at is not None


class SPRT:
    def __init__(self, delta=DELTA_SPRT, a=A_SPRT, b=B_SPRT):
        self.d, self.a, self.b, self.llr, self.n, self.fire_at = delta, a, b, 0.0, 0, None
    def update(self, z):
        self.n += 1
        self.llr += self.d * z - self.d ** 2 / 2.0
        if self.fire_at is None and self.llr >= self.b:
            self.fire_at = self.n
        return self.fire_at is not None


class EBCS:
    def __init__(self, mu0=0.0, sig=1.0, alpha=ALPHA_CS, rho=RHO_CS, n_min=N_MIN_CS):
        self.mu0, self.sig, self.alpha, self.rho, self.n_min = mu0, sig, alpha, rho, n_min
        self.sum, self.n, self.fire_at = 0.0, 0, None
    def update(self, x):
        self.n += 1
        self.sum += x
        if self.fire_at is None and self.n >= self.n_min:
            n, r = self.n, self.rho
            w = self.sig * float(np.sqrt(2 * (n * r * r + 1) / (n * n * r * r)
                                          * np.log(np.sqrt(n * r * r + 1) / self.alpha)))
            if self.mu0 - self.sum / n > w:
                self.fire_at = self.n
        return self.fire_at is not None


def run_streaming(z_series, x_series, mu0, sig0, chunk=CHUNK_FRAMES):
    """Feed z (standardised) / x (raw MAD) in 2-s chunks; return fire frame idx (1-based) or None."""
    c, s, e = CUSUM(), SPRT(), EBCS(mu0=mu0, sig=sig0)
    out = {}
    for det, ser, name in ((c, z_series, "cusum"), (s, z_series, "sprt"), (e, x_series, "ebcs")):
        fire = None
        for lo in range(0, len(ser), chunk):
            for v in ser[lo:lo + chunk]:
                if det.update(float(v)):
                    fire = det.fire_at
                    break
            if fire is not None:
                break
        out[name] = fire
    return out


def verdict_hash(wid, method, fire_at, mu0, sig0):
    h = hashlib.sha256(f"{wid}|{method}|{fire_at}|{mu0:.6f}|{sig0:.6f}|k={K_CUSUM},h={H_CUSUM},d={DELTA_SPRT},a={ALPHA_SPRT},b={BETA_SPRT}".encode()).hexdigest()[:16]
    return h


def robust_loc_scale(vals):
    vals = np.asarray(vals, dtype=float)
    mu = float(np.median(vals))
    sig = float(1.4826 * np.median(np.abs(vals - mu)))
    return mu, max(sig, 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-windows", type=int, default=150)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    gold = json.load(open(GOLD))["items"][:args.max_windows]
    synth = json.load(open(SYNTH))["items"]
    clips = sorted(set(g["clip"] for g in gold))

    # ---- 1. extract features once ----
    t0 = time.time()
    feats, per_window_ms = {}, {}
    for g in gold:
        s = time.time()
        frames, info = decode_small_series(g["path"], g["start_s"], g["end_s"], max_frames=60)
        if frames is None:
            feats[g["id"]] = {"error": info["error"], "mad": np.zeros(5), "bri": np.zeros(5),
                              "dark": np.zeros(5), "sharp": np.zeros(5), "n": 5, "decode_fail": True}
        else:
            feats[g["id"]] = frame_stats(frames)
            feats[g["id"]]["decode_fail"] = False
        per_window_ms[g["id"]] = (time.time() - s) * 1000
    feat_time = time.time() - t0

    # ---- 2. leave-one-clip-out ----
    loco_preds = []   # (id, clip, label, reason, {method: fire_idx|None}, mu0, sig0)
    fold_params = {}
    det_time_ms = []
    for held in clips:
        train_mad = np.concatenate([feats[g["id"]]["mad"] for g in gold
                                    if g["clip"] != held and not feats[g["id"]].get("decode_fail")])
        mu0, sig0 = robust_loc_scale(train_mad)
        fold_params[held] = {"mu0": mu0, "sig0": sig0}
        for g in gold:
            if g["clip"] != held:
                continue
            f = feats[g["id"]]
            s = time.time()
            if f.get("decode_fail"):
                # UNDECODABLE stream: fail-open reject at frame 1 for all methods
                fires = {"cusum": 1, "sprt": 1, "ebcs": 1}
            else:
                z = (mu0 - f["mad"]) / sig0
                fires = run_streaming(z, f["mad"], mu0, sig0)
            det_time_ms.append((time.time() - s) * 1000)
            loco_preds.append({"id": g["id"], "clip": g["clip"], "label": g["label"],
                               "reason": g["reason"], "fires": {k: (None if v is None else int(v))
                                                                for k, v in fires.items()},
                               "mu0": mu0, "sig0": sig0, "n_frames": int(f["n"]),
                               "hash": {m: verdict_hash(g["id"], m, fires[m], mu0, sig0)
                                        for m in fires}})

    def rates(preds, method):
        bad = [p for p in preds if p["label"] == 0]
        good = [p for p in preds if p["label"] == 1]
        det = float(np.mean([p["fires"][method] is not None for p in bad])) if bad else float("nan")
        fa = float(np.mean([p["fires"][method] is not None for p in good])) if good else float("nan")
        return det, fa

    # clip-bootstrap 95% CI (resample clips w/ replacement, pooled rates)
    by_clip = defaultdict(list)
    for p in loco_preds:
        by_clip[p["clip"]].append(p)
    summary = {}
    for m in ("cusum", "sprt", "ebcs"):
        det, fa = rates(loco_preds, m)
        bd, bf = [], []
        for _ in range(args.n_boot):
            samp = [by_clip[c] for c in rng.choice(clips, size=len(clips), replace=True)]
            flat = [p for grp in samp for p in grp]
            d, f_ = rates(flat, m)
            bd.append(d)
            bf.append(f_)
        summary[m] = {"detection_rate_bad": det, "false_alarm_rate_good": fa,
                      "det_95ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                      "fa_95ci": [float(np.percentile(bf, 2.5)), float(np.percentile(bf, 97.5))]}

    # ---- 3. time-to-detection: real bad windows (LOCO fires) ----
    ttd_real = []
    for p in loco_preds:
        if p["label"] == 0:
            ttd_real.append({"id": p["id"], "reason": p["reason"],
                             **{m + "_fire_s": (None if p["fires"][m] is None
                                                else round(p["fires"][m] / FPS_EFF, 2))
                                for m in ("cusum", "sprt", "ebcs")}})
    # fraction firing within 2s/4s among bad
    early = {}
    for m in ("cusum", "sprt", "ebcs"):
        fs = [p["fires"][m] for p in loco_preds if p["label"] == 0]
        early[m] = {"frac_le_2s": float(np.mean([(f or 99) * 1 <= 2 * FPS_EFF for f in fs])),
                    "frac_le_4s": float(np.mean([(f or 99) <= 4 * FPS_EFF for f in fs]))}

    # ---- 4. synthetic severity sweep (global calibration from all good) ----
    all_good_mad = np.concatenate([feats[g["id"]]["mad"] for g in gold
                                   if g["label"] == 1 and not feats[g["id"]].get("decode_fail")])
    gmu, gsig = robust_loc_scale(all_good_mad)
    synth_rows, synth_ms = [], []
    for it in synth:
        s = time.time()
        frames, info = decode_small_series(it["path"], 0.0, None, max_frames=60)
        if frames is None:
            fires = {"cusum": 1, "sprt": 1, "ebcs": 1}
            n = 0
        else:
            f = frame_stats(frames)
            z = (gmu - f["mad"]) / gsig
            fires = run_streaming(z, f["mad"], gmu, gsig)
            n = int(f["n"])
        synth_ms.append((time.time() - s) * 1000)
        synth_rows.append({"id": it["id"], "degradation": it["degradation"], "label": it["label"],
                           "n_frames": n,
                           **{m + "_fire_s": (None if fires[m] is None
                                              else round(fires[m] / FPS_EFF, 2))
                              for m in ("cusum", "sprt", "ebcs")}})

    # ---- 5. cost ----
    cost = {"feature_extract_ms_per_window_mean": float(np.mean(list(per_window_ms.values()))),
            "feature_extract_ms_p95": float(np.percentile(list(per_window_ms.values()), 95)),
            "detect_update_ms_per_window_mean": float(np.mean(det_time_ms)),
            "total_feature_s": round(feat_time, 1),
            "synth_decode_ms_mean": float(np.mean(synth_ms)) if synth_ms else None,
            "note": "single core, 64x64, OMP_NUM_THREADS=2"}

    json.dump({"loco_per_window": loco_preds,
               "fold_calibration": fold_params,
               "global_calibration": {"mu0": gmu, "sig0": gsig}},
              open(os.path.join(OUTDIR, "loco_predictions.json"), "w"), indent=1)
    json.dump({"summary": summary, "early_fire_bad": early,
               "n_windows": len(loco_preds), "n_clips": len(clips),
               "n_boot": args.n_boot, "seed": args.seed,
               "params": {"k": K_CUSUM, "h": H_CUSUM, "delta": DELTA_SPRT,
                          "alpha_sprt": ALPHA_SPRT, "beta_sprt": BETA_SPRT,
                          "A": A_SPRT, "B": B_SPRT, "alpha_cs": ALPHA_CS,
                          "rho_cs": RHO_CS, "fps_eff": FPS_EFF, "chunk_s": CHUNK_S}},
              open(os.path.join(OUTDIR, "loco_summary.json"), "w"), indent=1)
    json.dump({"real_bad_ttd": ttd_real, "synth_ttd": synth_rows},
              open(os.path.join(OUTDIR, "ttd.json"), "w"), indent=1)
    json.dump(cost, open(os.path.join(OUTDIR, "cost.json"), "w"), indent=1)
    print(json.dumps({"summary": summary, "early": early, "cost": cost}, indent=1))
    print("WROTE", OUTDIR)


if __name__ == "__main__":
    main()
