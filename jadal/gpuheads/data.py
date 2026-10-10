"""Data loading: frame shards (EG2 fp16 768-d + AI labels), spectral 53-d, gold manifest.

Alignment: by frame_id (never by position without checking). Standardization
stats come from train folds only.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

FRAMES_DIR = Path("/home/parshu/projects/jadal-qa-wt/frames10k/eval/results/frames10k/egocentric_10k")
SPEC_PATH = Path("/home/parshu/projects/jadal-qa-wt/frames-spectral/eval/results/frames_spectral/egocentric_10k.npz")
GOLD_PATH = Path("/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json")


def load_frame_eg2():
    ids, embs, hand, work = [], [], [], []
    for f in sorted(glob.glob(str(FRAMES_DIR / "shard_*.npz"))):
        d = np.load(f)
        ids.append(d["ids"].astype(str))
        embs.append(d["emb"].astype(np.float32))
        hand.append(d["hand"].astype(np.int64))
        work.append(d["work"].astype(np.int64))
    if not ids:
        return None
    return {"ids": np.concatenate(ids), "emb": np.concatenate(embs),
            "hand": np.concatenate(hand), "work": np.concatenate(work)}


def load_spectral():
    d = np.load(SPEC_PATH)
    return {"ids": d["ids"].astype(str), "X": d["X"].astype(np.float32),
            "hand": d["hand"].astype(np.int64), "work": d["work"].astype(np.int64),
            "features": [str(x) for x in d["features"]]}


def load_gold():
    m = json.load(open(GOLD_PATH))
    return m


def align(ids_a, ids_b):
    """Return index arrays ia, ib with ids_a[ia] == ids_b[ib] (inner join)."""
    bpos = {v: i for i, v in enumerate(ids_b)}
    ia, ib = [], []
    for i, v in enumerate(ids_a):
        j = bpos.get(v)
        if j is not None:
            ia.append(i)
            ib.append(j)
    return np.array(ia), np.array(ib)


def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


def pool_window(scores: np.ndarray):
    """scores: (T,) frame scores -> pooled feature vector (8-d)."""
    s = np.asarray(scores, dtype=np.float64)
    qs = np.quantile(s, [0.1, 0.25, 0.5, 0.75, 0.9])
    return np.array([s.mean(), s.max(), *qs, (s >= 0.5).mean(),
                     s.std()], dtype=np.float64)
