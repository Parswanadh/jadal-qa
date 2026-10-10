"""E1 shared lib: deterministic decode-once + index-subsample + LOCO eval."""
import json
import os
import time

import cv2
import numpy as np

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
MANIFEST = os.path.join(BUNDLE, "labels", "manifest_gold.json")
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GPU_SRC = os.path.join(_root, "jadal", "gpu")

W, H = 160, 90          # decode size (matches perpixel series_gold: every frame at 160x90)
NATIVE_FPS = 10.0
WIN_S = 10.0

# fps target -> (step, effective fps). Rule: idx = arange(0, T_native, step).
FPS_STEPS = {10: 1, 5: 2, 3.33: 3, 2: 5, 1: 10}


def load_manifest():
    m = json.load(open(MANIFEST))
    return m["items"]


def decode_window(path, start_s, n_frames=100):
    """Decode n_frames consecutive frames from start_s (native ~10fps).

    Method: CAP_PROP_POS_MSEC seek then read. Deterministic for fixed file.
    Returns rgb uint8 (T,H,W,3), luma float64 (T,H,W) in [0,1].
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    greys, rgbs = [], []
    t0 = time.perf_counter()
    while len(greys) < n_frames:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
        if fr.ndim == 2:
            g = fr
            fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
        elif fr.shape[2] == 1:
            g = fr[:, :, 0]
            fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
        else:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        greys.append(g)
        rgbs.append(fr[..., ::-1].copy())
    dt = (time.perf_counter() - t0) * 1000.0
    cap.release()
    if not greys:
        raise RuntimeError(f"no frames: {path} @ {start_s}")
    rgb = np.stack(rgbs)
    grey = np.stack(greys).astype(np.float64) / 255.0
    return rgb, grey, dt


def subsample_idx(t_native, step):
    return np.arange(0, t_native, step)


def gpu_features():
    import sys
    if GPU_SRC not in sys.path:
        sys.path.insert(0, GPU_SRC)
    import features as gf
    return gf


def loco_proba(X, y, clips, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    p = np.zeros(len(y))
    for c in sorted(set(clips)):
        tr = clips != c
        te = clips == c
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
        m.fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def clip_bootstrap_ci(y, p, clips, B=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    names = sorted(set(clips))
    out = []
    for _ in range(B):
        cs = rng.choice(names, len(names))
        idx = np.concatenate([np.where(clips == c)[0] for c in cs])
        if len(set(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def bal_acc(y, p, t=0.5):
    pr = (p > t).astype(int)
    return float(((pr[y == 1] == 1).mean() + (pr[y == 0] == 0).mean()) / 2)
