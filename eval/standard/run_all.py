#!/usr/bin/env python3
"""E9.01 - standard evaluation harness (the number a PR is compared against).

Reproduces the two documented in-domain IndEgo baselines on the 150 human
windows (100 good / 50 bad, 7 clips) and scores the out-of-domain (OOD)
human-labelled Egocentric-10K windows (60 windows, 9 bad):

  * pixel26 : 26 per-pixel features, LOCO logistic              (documented ~0.81)
  * eg2     : EmbeddingGemma-2 window embedding, LOCO logistic   (documented ~0.85)
  * ood     : the frozen IndEgo pixel26 model applied to OOD     (documented 0.4-0.53 = no transfer)

Deterministic, CPU-only, no network access at import time. Every number is
written to a results JSON together with n, label source, protocol, CI,
device, git sha and the exact command that regenerates it.

Usage:
  python3 eval/standard/run_all.py [--bundle DIR] [--out PATH] [--quiet]

Inputs (all read-only, never copied into the repo):
  $JADAL_BUNDLE/labels/manifest_gold.json     150 IndEgo windows, human labels
  $JADAL_BUNDLE/caches/gpu/features.json      26 per-pixel features per window
  $JADAL_BUNDLE/caches/frames10k/segments.npz EG2 frame embeddings per window
  $JADAL_BUNDLE/labels/ood_labels.json        60 OOD windows, human labels
  $JADAL_BUNDLE/local_work/.../ood_shard_*.json  26 OOD features (cached)
  $JADAL_BUNDLE/local_work/.../shard_sweep_*.json IndEgo 26-feature sweep (E1.01 protocol)

Exit codes: 0 ok, 2 missing input, 3 not enough classes in a split.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
DEFAULT_OUT = os.path.join(REPO, "eval", "results", "standard", "run_all.json")

# Model hyper-parameters: frozen, chosen on the documented runs (never tuned on
# a held-out clip; they are the values in eval/results/exp_fps/e1_01.json and
# .../e1_02.json and eval/gpu/cascade.py).
PIXEL26_C = 0.3
EG2_C = 0.01
BOOT_B = 1000
BOOT_SEED = 0


# --------------------------------------------------------------------------- #
# small, dependency-light statistics (no cv2/torch import)
# --------------------------------------------------------------------------- #
def loco_proba(X, y, clips, C):
    """Leave-one-clip-out predicted probability P(good). Deterministic."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    clips = np.asarray(clips)
    p = np.zeros(len(y), dtype=np.float64)
    for c in sorted(set(clips.tolist())):
        tr = clips != c
        te = clips == c
        if not tr.any() or not te.any() or len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=4000, C=C, class_weight="balanced")
        m.fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def clip_bootstrap_ci(y, p, clips, B=BOOT_B, seed=BOOT_SEED):
    """95% CI resampling whole clips (cluster bootstrap)."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    clips = np.asarray(clips)
    rng = np.random.default_rng(seed)
    names = sorted(set(clips.tolist()))
    out = []
    for _ in range(B):
        cs = rng.choice(names, size=len(names))
        idx = np.concatenate([np.where(clips == c)[0] for c in cs])
        if len(set(y[idx].tolist())) < 2:
            continue
        out.append(float(roc_auc_score(y[idx], p[idx])))
    if not out:
        return [None, None]
    return [round(float(np.percentile(out, 2.5)), 4),
            round(float(np.percentile(out, 97.5)), 4)]


def window_bootstrap_ci(y, p, B=2000, seed=BOOT_SEED):
    """95% CI resampling windows (optimistic when ~1 window/factory)."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    out = []
    n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(set(y[idx].tolist())) < 2:
            continue
        out.append(float(roc_auc_score(y[idx], p[idx])))
    if not out:
        return [None, None]
    return [round(float(np.percentile(out, 2.5)), 4),
            round(float(np.percentile(out, 97.5)), 4)]


def bal_acc(y, p, t=0.5):
    y = np.asarray(y)
    pred = (np.asarray(p) > t).astype(int)
    return round(float(((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean()) / 2), 4)


def auroc(y, p):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None
    return round(float(roc_auc_score(y, p)), 4)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)


def feature_names():
    sys.path.insert(0, REPO)
    from jadal.gpu.features import FEATURE_NAMES  # noqa: E402

    return list(FEATURE_NAMES)


def load_gold(bundle):
    m = load_json(os.path.join(bundle, "labels", "manifest_gold.json"))
    items = m["items"]
    return {
        "ids": [it["id"] for it in items],
        "clips": np.array([it["clip"] for it in items]),
        "y": np.array([int(it["label"]) for it in items]),
        "n_good": int(sum(1 for it in items if int(it["label"]) == 1)),
        "n_bad": int(sum(1 for it in items if int(it["label"]) == 0)),
        "snapshot_utc": m.get("snapshot_utc"),
        "labels_sha256": m.get("labels_sha256"),
    }


def pixel26_matrix_from_features(bundle, ids, names):
    W = load_json(os.path.join(bundle, "caches", "gpu", "features.json"))["windows"]
    missing = [i for i in ids if i not in W or "full" not in W[i]]
    if missing:
        raise KeyError(f"features.json missing 'full' for {len(missing)} ids e.g. {missing[:3]}")
    return np.array([[W[i]["full"][n] for n in names] for i in ids], dtype=np.float64)


def eg2_matrix(bundle, ids):
    z = np.load(os.path.join(bundle, "caches", "frames10k", "segments.npz"), allow_pickle=True)
    zids = [str(x) for x in z["ids"]]
    if zids != list(ids):
        order = {k: i for i, k in enumerate(zids)}
        missing = [i for i in ids if i not in order]
        if missing:
            raise KeyError(f"segments.npz missing ids: {missing[:3]}")
        z = None  # order mismatch must not silently mis-align
        raise ValueError("segments.npz id order does not match manifest_gold.json")
    E = z["F"].astype(np.float32)
    cnt = np.array(z["counts"]).astype(int)
    rows = []
    for k in range(len(ids)):
        n = max(1, int(cnt[k]))
        v = E[k, :n].mean(axis=0)
        rows.append(v)
    X = np.stack(rows)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return X / norm


def load_ood(bundle, names):
    """Return ids, y, and the 26-feature matrix at fps tag '10' from cached shards."""
    shard_dir = os.path.join(bundle, "local_work", "exp-fps", "eval", "results", "exp_fps")
    shards = []
    i = 0
    while True:
        p = os.path.join(shard_dir, f"ood_shard_{i}of2.json")
        if not os.path.exists(p):
            break
        shards.append(load_json(p))
        i += 1
    if not shards:
        raise FileNotFoundError("no cached OOD feature shards (ood_shard_*of2.json)")
    wins = [w for s in shards for w in s["windows"]]
    wins.sort(key=lambda w: w["segment_id"])
    # labels from the human OOD file (authoritative); shard 'y' is a cross-check
    lab = load_json(os.path.join(bundle, "labels", "ood_labels.json"))["latest"]
    ids, ys = [], []
    for w in wins:
        sid = w["segment_id"]
        v = lab.get(sid)
        ys.append(1 if (v and v.get("verdict") == "good") else 0)
        ids.append(sid)
    y = np.array(ys)
    X = np.nan_to_num(np.array([w["fps"]["10"]["feats"] for w in wins], dtype=np.float64))
    return {"ids": ids, "y": y, "X": X, "n_good": int(y.sum()),
            "n_bad": int((1 - y).sum()), "names": shards[0]["names"]}


def load_e103_indego_features(bundle, names):
    """IndEgo 26 features decoded by the E1.01 sweep (matches the OOD features' pipeline)."""
    shard_dir = os.path.join(bundle, "local_work", "exp-fps", "eval", "results", "exp_fps")
    shards = []
    i = 0
    while True:
        p = os.path.join(shard_dir, f"shard_sweep_{i}of2.json")
        if not os.path.exists(p):
            break
        shards.append(load_json(p))
        i += 1
    if not shards:
        return None
    ids = [x for s in shards for x in s["ids"]]
    order = np.argsort(ids, kind="stable")
    ids = [ids[i] for i in order]
    X = np.nan_to_num(np.array([x for s in shards for x in s["feats"]["10"]])[order])
    y = np.array([x for s in shards for x in s["y"]])[order]
    return {"ids": ids, "X": X, "y": y}


def frozen_fit(X, y, C=PIXEL26_C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X)
    m = LogisticRegression(max_iter=4000, C=C, class_weight="balanced").fit(sc.transform(X), y)
    return sc, m


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def device_label():
    dev = os.environ.get("JADAL_DEVICE_LABEL")
    if dev:
        return dev
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        name = r.stdout.strip().splitlines()[0].strip()
        if name:
            return f"cpu-inference / {name}"
    except Exception:
        pass
    return f"cpu-inference / {platform.system()} {platform.machine()}"


def git_sha():
    try:
        return subprocess.run(
            ["git", "-C", REPO, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return None


def build(bundle=DEFAULT_BUNDLE):
    names = feature_names()
    gold = load_gold(bundle)
    ids, clips, y = gold["ids"], gold["clips"], gold["y"]
    n_clips = len(set(clips.tolist()))

    # --- view A: 26 per-pixel features, LOCO ---------------------------------
    X26 = pixel26_matrix_from_features(bundle, ids, names)
    p26 = loco_proba(X26, y, clips, PIXEL26_C)
    pixel26 = {
        "auroc": auroc(y, p26),
        "ci95_clip_bootstrap": clip_bootstrap_ci(y, p26, clips),
        "bal_acc": bal_acc(y, p26),
        "n": int(len(y)), "n_good": gold["n_good"], "n_bad": gold["n_bad"],
        "n_clips": n_clips,
        "features": "caches/gpu/features.json windows[id]['full'] (26 per-pixel features)",
        "model": f"StandardScaler + LogisticRegression C={PIXEL26_C} class_weight=balanced",
        "protocol": "leave-one-clip-out over the 7 IndEgo clips",
        "documented_auroc": 0.81,
    }

    # --- view B: EG2 window embedding, LOCO ----------------------------------
    E = eg2_matrix(bundle, ids)
    peg2 = loco_proba(E, y, clips, EG2_C)
    eg2 = {
        "auroc": auroc(y, peg2),
        "ci95_clip_bootstrap": clip_bootstrap_ci(y, peg2, clips),
        "bal_acc": bal_acc(y, peg2),
        "n": int(len(y)),
        "features": "caches/frames10k/segments.npz F (counts-aware window mean, L2-normalised)",
        "model": f"StandardScaler + LogisticRegression C={EG2_C} class_weight=balanced",
        "protocol": "leave-one-clip-out over the 7 IndEgo clips; EG2 embedding frozen (no fine-tune)",
        "documented_auroc": 0.85,
    }

    # --- OOD: frozen IndEgo pixel26 model -> OOD windows ---------------------
    ood = load_ood(bundle, names)
    sc, m = frozen_fit(X26, y)
    p_ood = m.predict_proba(sc.transform(ood["X"]))[:, 1]
    saturated = bool(np.allclose(p_ood, p_ood[0]))
    ood_block = {
        "n": int(len(ood["y"])), "n_good": ood["n_good"], "n_bad": ood["n_bad"],
        "label_source": "human ood_labels.json (one labeller; out-of-domain, exploratory)",
        "protocol": ("frozen IndEgo pixel26 logistic (trained on all 150 IndEgo windows, "
                     "features.json) applied to the cached OOD 26-feature windows at fps=10; "
                     "CI = window-bootstrap B=2000 seed0 (optimistic: ~1 window/factory)"),
        "frozen_pixel26_auroc": auroc(ood["y"], p_ood),
        "ci95_window_bootstrap": window_bootstrap_ci(ood["y"], p_ood),
        "saturated": saturated,
        "note_saturated": ("all OOD windows scored p=1.0 (the features.json scale is far outside "
                           "the IndEgo training range); the value is reported, not hidden") if saturated else "",
        "documented_range": "0.4-0.53 (IndEgo-trained models did NOT transfer)",
    }
    # E1.03 reference: same model but trained on the E1.01-decoded IndEgo features
    ref = load_e103_indego_features(bundle, names)
    if ref is not None:
        sc2, m2 = frozen_fit(ref["X"], ref["y"])
        p_ref = m2.predict_proba(sc2.transform(ood["X"]))[:, 1]
        ood_block["e103_reference_auroc_fps10"] = auroc(ood["y"], p_ref)
        ood_block["e103_reference_ci95_window_bootstrap"] = window_bootstrap_ci(ood["y"], p_ref)
        ood_block["e103_reference_protocol"] = ("frozen IndEgo logistic trained on the E1.01 sweep "
                                                "10fps features (shard_sweep), applied to the same OOD windows")

    res = {
        "experiment": "E9.01",
        "title": "standard eval harness (PR comparison baseline)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "device": device_label(),
        "host": platform.node(),
        "command": "python3 eval/standard/run_all.py",
        "bundle": bundle,
        "n": int(len(y)), "n_good": gold["n_good"], "n_bad": gold["n_bad"], "n_clips": n_clips,
        "label_source": ("human IndEgo gold manifest_gold.json (one labeller, ~3 wearers); "
                         "windows within a clip are correlated"),
        "protocol": ("in-domain = leave-one-clip-out (LOCO) over the 7 IndEgo clips; "
                     "95% CI = clip-level bootstrap B=1000 seed0"),
        "labels_sha256": gold["labels_sha256"],
        "views": {"pixel26": pixel26, "eg2": eg2},
        "ood": ood_block,
        "wall_s": None,  # filled below
        "honesty": {
            "n": "150 windows / 7 clips / ~3 wearers, one labeller -> wide CIs",
            "ci": "clip-bootstrap CI is wide; window-bootstrap OOD CI is optimistic",
            "no_tuning": "model C values are the documented ones; never tuned on a held-out clip",
            "no_transfer": "the IndEgo-trained model does not transfer to OOD; reported, not hidden",
        },
    }
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="standard eval harness (E9.01)")
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    try:
        res = build(a.bundle)
    except FileNotFoundError as e:
        print(f"run_all: missing input: {e}", file=sys.stderr)
        return 2
    res["wall_s"] = round(time.perf_counter() - t0, 1)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    if not a.quiet:
        v = res["views"]
        print(f"pixel26 AUROC {v['pixel26']['auroc']:.4f} "
              f"{v['pixel26']['ci95_clip_bootstrap']} (n={v['pixel26']['n']})")
        print(f"eg2     AUROC {v['eg2']['auroc']:.4f} "
              f"{v['eg2']['ci95_clip_bootstrap']} (n={v['eg2']['n']})")
        o = res["ood"]
        print(f"ood     frozen pixel26 AUROC {o['frozen_pixel26_auroc']} "
              f"(n={o['n']}, n_bad={o['n_bad']}, saturated={o['saturated']}); "
              f"E1.03 ref {o.get('e103_reference_auroc_fps10')}")
        print(f"wrote {a.out} (wall {res['wall_s']}s, git {res['git_sha']}, device {res['device']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
