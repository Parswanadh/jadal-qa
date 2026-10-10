"""Train + calibrate the L1 linear-probe heads (issue #7).

Full mode (once the FIXED eval manifests land on origin/main)::

    PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/train_heads.py \
        --splits eval/results/splits.json --out eval/results/decider_baseline.json

Dev mode (default; tiny subset, DEV_ONLY outputs)::

    PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/train_heads.py \
        --dev-n 1500 --out eval/results/decider_baseline_DEV_ONLY.json

Pipeline: read frames from ``/home/parshu/projects/awing/data/ego10k_eval/*.parquet``
in small row-group batches (never print image bytes), thumbnail to 512 px,
embed with EmbeddingGemma 2 (bfloat16, batch size 1, budget 140, fixed seed),
train scikit-learn logistic regression on frozen embeddings for
``hand_count >= 1`` (hands_ok) and ``active_labor == yes`` (active_work),
fit one temperature per head on the CALIBRATION split
(``jadal.certify.fit_temperature``), choose the accept threshold on
CALIBRATION via ``pick_threshold`` (target_err 0.05, delta 0.1) and estimate
the certified error bound on the held-out TEST split. Heads are saved with
``jadal.decision.save_head`` to ``models/decider_{hands,work}.npz``
(git-ignored); their sha256 goes into the results JSON.

Caveat (recorded in the output): labels by gemini-2.5-flash, not human-checked.

NOT DONE:
- Dev mode uses the first --dev-n frames round-robined across the three
  parquet files (NOT the fixed deduped manifests); its numbers are
  model-vs-model agreement on a convenience subset only.
- One frame per segment here (parquet rows are frames); the Decider embeds
  each of its 4 sampled frames separately and averages per-frame logits.
- STEP head is separate (IndEgo stretch goal).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from jadal import certify as certify_mod
from jadal import decision as decision_mod

DATA_DIR = "/home/parshu/projects/awing/data/ego10k_eval"
FILES = ("ego4d.parquet", "egocentric_10k.parquet", "epic_kitchens.parquet")
SEED = 1234
BUDGET = 140
LABEL_CAVEAT = "labels by gemini-2.5-flash, not human-checked"
LABEL_CAVEAT_LONG = ("hand_count / active_labor labels were produced by gemini-2.5-flash "
                     "auto-labelling and were NOT human-checked: every accuracy / calibration "
                     "number here is model-vs-model agreement, not ground-truth performance.")

INDEGO_DIR = "/home/parshu/projects/awing/data/indego"
INDEGO_CLIPS = {
    "User_15_410_0411_2_480.mp4": (
        f"{INDEGO_DIR}/1_Assembly/mp4_480/User_15_410_0411_2_480.mp4",
        f"{INDEGO_DIR}/1_Assembly/annotated_keysteps_assembly_disassembly/AS_User_15_410_0411_2.json",
    ),
    "User_1_412_2010_3_480.mp4": (
        f"{INDEGO_DIR}/1_Assembly/mp4_480/User_1_412_2010_3_480.mp4",
        f"{INDEGO_DIR}/1_Assembly/annotated_keysteps_assembly_disassembly/AS_User_1_412_2010_3.json",
    ),
}
STEP_TRAIN_CLIP = "User_15_410_0411_2_480.mp4"  # 15 segments / 14 steps
STEP_TEST_CLIP = "User_1_412_2010_3_480.mp4"    # 5 segments (late steps)


def load_model():
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    if not torch.cuda.is_available():
        raise RuntimeError("train_heads requires CUDA.")
    model = SentenceTransformer(
        decision_mod.EG2_DIR,
        model_kwargs={"torch_dtype": dtype},
        trust_remote_code=True,
    )
    model.eval()
    return model


def read_dev_rows(dev_n: int) -> list[dict]:
    """Round-robin row-group scan across the three files; deterministic order."""
    import pyarrow.parquet as pq

    per_file: dict[str, list[dict]] = {}
    for f in FILES:
        rows: list[dict] = []
        pf = pq.ParquetFile(str(Path(DATA_DIR) / f))
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_groups([rg], columns=["frame_id", "image", "hand_count", "active_labor"])
            d = tbl.to_pydict()
            for i in range(tbl.num_rows):
                rows.append({
                    "frame_id": d["frame_id"][i],
                    "bytes": bytes(d["image"][i]["bytes"]),
                    "hand_count": int(d["hand_count"][i]),
                    "active_labor": str(d["active_labor"][i]),
                })
        rows.sort(key=lambda r: r["frame_id"])
        per_file[f] = rows
    out: list[dict] = []
    i = 0
    while len(out) < dev_n and any(len(v) > i for v in per_file.values()):
        for f in FILES:
            if i < len(per_file[f]) and len(out) < dev_n:
                out.append(per_file[f][i])
        i += 1
    return out


def read_manifest_rows(manifest: Path) -> dict[str, list[dict]]:
    """Read rows for the MERGED splits manifest (frame_ids per split).

    Manifest shape: {"splits": {"train": {"frame_ids": [...]}, ...}} with
    near-duplicate components kept whole (see eval harness). Labels + source
    come from a metadata-only parquet scan (no images); image bytes are
    fetched separately per split by fetch_split_rows (one at a time).
    Returns {"rows": {split: [{frame_id, hand_count, active_labor, source}]},
    "missing": {split: n_ids_not_found}}.
    """
    import pyarrow.parquet as pq

    spec = json.loads(manifest.read_text())
    groups = spec.get("splits", spec)
    index: dict[str, dict] = {}
    for f in FILES:
        source = Path(f).stem
        pf = pq.ParquetFile(str(Path(DATA_DIR) / f))
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_groups([rg], columns=["frame_id", "hand_count", "active_labor"])
            d = tbl.to_pydict()
            for i in range(tbl.num_rows):
                index[d["frame_id"][i]] = {
                    "frame_id": d["frame_id"][i],
                    "hand_count": int(d["hand_count"][i]),
                    "active_labor": str(d["active_labor"][i]),
                    "source": source,
                }
    rows: dict[str, list[dict]] = {}
    missing: dict[str, int] = {}
    for split in ("train", "calibration", "test"):
        g = groups.get(split, {})
        ids = g.get("frame_ids", g if isinstance(g, list) else [])
        rows[split], miss = [], 0
        for fid in ids:
            if fid in index:
                rows[split].append(index[fid])
            else:
                miss += 1
        missing[split] = miss
    return {"rows": rows, "missing": missing}


SUBSAMPLE_SALT = "decider-subsample-v1"
SUBSAMPLE_CAPS = {"train": 2000, "calibration": 500, "test": 500}  # per source


def subsample_split(rows: list[dict], split: str, cap_per_source: int) -> list[dict]:
    """Deterministic per-source cap: sha-order within source, take first cap."""
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    out = []
    for source in sorted(by_source):
        ordered = sorted(by_source[source],
                         key=lambda r: hashlib.sha256(
                             f"{SUBSAMPLE_SALT}|{split}|{r['frame_id']}".encode()).hexdigest())
        out.extend(ordered[:cap_per_source])
    out.sort(key=lambda r: r["frame_id"])
    return out


def fetch_split_rows(wanted_ids: set[str]) -> list[dict]:
    """Fetch image bytes for wanted frame_ids, one row at a time (small batches).

    Never prints image bytes. Returns rows sorted by frame_id.
    """
    import pyarrow.parquet as pq

    out: list[dict] = []
    for f in FILES:
        source = Path(f).stem
        pf = pq.ParquetFile(str(Path(DATA_DIR) / f))
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_groups([rg], columns=["frame_id", "image", "hand_count", "active_labor"])
            d = tbl.to_pydict()
            for i in range(tbl.num_rows):
                if d["frame_id"][i] in wanted_ids:
                    out.append({
                        "frame_id": d["frame_id"][i],
                        "bytes": bytes(d["image"][i]["bytes"]),
                        "hand_count": int(d["hand_count"][i]),
                        "active_labor": str(d["active_labor"][i]),
                        "source": source,
                    })
    out.sort(key=lambda r: r["frame_id"])
    return out


def codec_shift_check(model, rows: list[dict], cache_dir: Path, n: int = 100) -> dict:
    """Codec-shift measurement (NOT an accuracy claim): embed the SAME n test
    frames once JPEG-decoded (normal path) and once after an mp4 round-trip
    (cv2.VideoWriter mp4v at original size, then re-read), report the cosine
    similarity distribution between the two embeddings per frame."""
    import cv2

    sel = rows[:n]
    sims: list[float] = []
    for r in sel:
        img = decision_mod.preprocess_pil(Image.open(io.BytesIO(r["bytes"])))
        e_direct = _encode_frame(model, img)
        raw = np.asarray(Image.open(io.BytesIO(r["bytes"])).convert("RGB"))
        h, w = raw.shape[:2]
        h2, w2 = h - (h % 2), w - (w % 2)  # mp4 needs even dims
        raw = raw[:h2, :w2]
        tmp = str(cache_dir / "_codec_tmp.mp4")
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w2, h2))
        try:
            vw.write(cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
        finally:
            vw.release()
        cap = cv2.VideoCapture(tmp)
        try:
            ok, back = cap.read()
        finally:
            cap.release()
        try:
            os.remove(tmp)
        except OSError:
            pass
        if not ok or back is None:
            continue
        img_rt = decision_mod.preprocess_bgr(back)
        e_rt = _encode_frame(model, img_rt)
        sims.append(float(e_direct @ e_rt / (np.linalg.norm(e_direct) * np.linalg.norm(e_rt))))
    sims = np.asarray(sims, dtype=float)
    return {
        "n": int(len(sims)),
        "note": ("codec-shift measurement only: cosine sim between JPEG-decoded and "
                 "mp4-round-tripped embeddings of the same frames; not accuracy"),
        "mean": round(float(sims.mean()), 4),
        "min": round(float(sims.min()), 4),
        "p5": round(float(np.percentile(sims, 5)), 4),
        "p50": round(float(np.percentile(sims, 50)), 4),
    }


def _encode_frame(model, img) -> np.ndarray:
    with torch.inference_mode():
        e = model.encode(
            img, batch_size=1, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
            processing_kwargs={"image": {"max_soft_tokens": BUDGET}},
        )
    return np.asarray(e, dtype=np.float64).reshape(-1)


def _cached_key(png_bytes: bytes) -> str:
    return decision_mod.frame_cache_key(
        png_bytes, backbone=decision_mod.backbone_id(decision_mod.EG2_DIR),
        budget=BUDGET)


def embed_rows(model, rows: list[dict], cache_dir: Path) -> np.ndarray:
    """Embed rows with the SHARED serving preprocessing (no train/serve skew).

    Preprocessing (thumbnail) is byte-identical to serving: the cache key is
    computed over the preprocessed PNG bytes plus backbone id and budget, so a
    cache hit returns exactly what serving would compute.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    # preprocess everything first (CPU-only); keys known before any GPU use
    items: list[tuple[str, Image.Image]] = []
    for r in rows:
        img = decision_mod.preprocess_pil(Image.open(io.BytesIO(r["bytes"])))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        items.append((_cached_key(buf.getvalue()), img))
    embs: list[np.ndarray] = []
    for k, (key, img) in enumerate(items):
        hit = cache_dir / f"{key}.npy"
        if hit.exists():
            embs.append(np.load(str(hit)))
            continue
        if model is None:
            raise RuntimeError("embedding cache miss with model=None (GPU needed)")
        e = _encode_frame(model, img)
        np.save(str(hit), e)
        embs.append(e)
        if (k + 1) % 200 == 0:
            print(f"  embedded {k + 1}/{len(items)}", flush=True)
    return np.stack(embs, axis=0)


def cache_complete(rows: list[dict], cache_dir: Path) -> bool:
    for r in rows:
        img = decision_mod.preprocess_pil(Image.open(io.BytesIO(r["bytes"])))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        if not (cache_dir / f"{_cached_key(buf.getvalue())}.npy").exists():
            return False
    return True


def train_head(Xtr: np.ndarray, ytr: np.ndarray) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    clf.fit(Xtr, ytr)
    return clf.coef_.reshape(-1), float(clf.intercept_[0])


def fit_temperature_binary(logits: np.ndarray, y: np.ndarray) -> float:
    """Single temperature on 1-D logits (2-class wrapper, deterministic)."""
    logits2 = np.stack([-np.asarray(logits, dtype=float) / 2.0,
                        np.asarray(logits, dtype=float) / 2.0], axis=1)
    return certify_mod.fit_temperature(logits2, np.asarray(y))


def probs_with_T(coef, intercept, T: float, X: np.ndarray) -> np.ndarray:
    logits = np.asarray(X, dtype=np.float64) @ np.asarray(coef, dtype=np.float64) + float(intercept)
    return 1.0 / (1.0 + np.exp(-logits / float(T)))


def metrics_with_T(coef, intercept, T: float, X: np.ndarray, y: np.ndarray,
                   target_err=0.05, delta=0.1, pick=True) -> dict:
    """Accuracy/ECE/confusion at a GIVEN temperature (never fits on this split)."""
    y = np.asarray(y)
    probs = probs_with_T(coef, intercept, T, X)
    pred = (probs >= 0.5).astype(int)
    conf = np.maximum(probs, 1.0 - probs)
    out = {
        "n": int(len(y)),
        "accuracy": round(float((pred == y).mean()), 4),
        "ece": round(float(certify_mod.ece(probs, y)), 4),
        "temperature": round(float(T), 4),
        "probs_mean": round(float(probs.mean()), 4),
    }
    if pick:
        threshold = certify_mod.pick_threshold(conf, (pred == y), target_err=target_err, delta=delta)
        out["threshold_pick"] = {k: (round(float(v), 4) if isinstance(v, float) else v)
                                 for k, v in threshold.items()}
    else:
        out["conf"] = conf
        out["pred"] = pred
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results/decider_baseline_DEV_ONLY.json")
    ap.add_argument("--dev-n", type=int, default=1500)
    ap.add_argument("--splits", default=None, help="fixed splits manifest for full mode")
    ap.add_argument("--step", action="store_true",
                    help="IndEgo ECU STEP prototype head instead of hands/work heads")
    ap.add_argument("--heads-dir", default="models")
    ap.add_argument("--cache-dir", default="models/.embed_cache_ego10k")
    args = ap.parse_args()

    if args.step:
        return cmd_step(args)
    return cmd_heads(args)


def parse_via_segments(ann_path: str) -> list[dict]:
    """Parse a VIA temporal-segment JSON -> [{start_s, end_s, step}]."""
    spec = json.loads(Path(ann_path).read_text())
    options = spec["attribute"]["1"]["options"]
    segs = []
    for v in spec["metadata"].values():
        z = v.get("z") or []
        if len(z) != 2:
            continue
        opt = (v.get("av") or {}).get("1")
        if opt not in options:
            continue
        segs.append({"start_s": float(z[0]), "end_s": float(z[1]),
                     "step": str(options[opt])})
    segs.sort(key=lambda s: (s["start_s"], s["end_s"]))
    return segs


def sample_segment_frames(clip_path: str, start_s: float, end_s: float,
                          n: int = decision_mod.N_FRAMES,
                          fractions: tuple = decision_mod.FRAME_FRACTIONS) -> list[np.ndarray]:
    """Same fixed-fraction sampling as serving: fractions of [start_s, end_s].

    Serving samples fractions of the whole clip; training samples the same
    fractions of each annotated step segment, with the same per-frame
    preprocessing (decision_mod.preprocess_bgr) and the same segment pooling
    (mean of frame embeddings, renormalised) — no sampling skew (I6).
    """
    import cv2

    cap = cv2.VideoCapture(clip_path)
    try:
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out = []
        for f in fractions[:n]:
            t = start_s + (end_s - start_s) * f
            idx = min(max(int(t * fps), 0), max(total - 1, 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                out.append(frame)
        return out
    finally:
        cap.release()


def embed_pil_frames(model, imgs: list[Image.Image], cache_dir: Path) -> np.ndarray:
    """Embed preprocessed PIL frames (callers apply decision_mod.preprocess_* first)."""
    embs = []
    for img in imgs:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        key = _cached_key(buf.getvalue())
        hit = cache_dir / f"{key}.npy"
        if hit.exists():
            embs.append(np.load(str(hit)))
            continue
        if model is None:
            raise RuntimeError("embedding cache miss with model=None (GPU needed)")
        e = _encode_frame(model, img)
        np.save(str(hit), e)
        embs.append(e)
    return np.stack(embs, axis=0)


def cmd_step(args) -> int:
    """IndEgo ECU STEP head: prototypes on train clip, segment accuracy held-out.

    Sampling/preprocessing/pooling are byte-identical to serving (I6); the
    temperature grid is wide and falls back to T=1 with a note on edge
    solutions. N is stated; no step-accuracy claims are made.
    """
    repo_root = Path(__file__).resolve().parent.parent
    heads_dir = Path(args.heads_dir)
    if not heads_dir.is_absolute():
        heads_dir = repo_root / heads_dir
    cache_dir = repo_root / "models" / ".embed_cache_indego"
    cache_dir.mkdir(parents=True, exist_ok=True)

    clips: dict[str, list[dict]] = {}
    for clip_name, (clip_path, ann_path) in INDEGO_CLIPS.items():
        segs = parse_via_segments(ann_path)
        for s in segs:
            s["clip"] = clip_name
        clips[clip_name] = segs
        print(f"{clip_name}: {len(segs)} segments", flush=True)
    train_segs = clips[STEP_TRAIN_CLIP]
    test_segs = clips[STEP_TEST_CLIP]

    model = load_model()

    def seg_embedding(seg: dict) -> np.ndarray:
        # Same recipe as Decider.decide: per-frame shared preprocessing,
        # mean-pool, renormalise.
        clip_path = INDEGO_CLIPS[seg["clip"]][0]
        frames = sample_segment_frames(clip_path, seg["start_s"], seg["end_s"])
        imgs = [decision_mod.preprocess_bgr(f) for f in frames]
        E = embed_pil_frames(model, imgs, cache_dir)
        m = E.mean(axis=0)
        return m / (np.linalg.norm(m) or 1.0)

    Xtr = np.stack([seg_embedding(s) for s in train_segs])
    ytr = [s["step"] for s in train_segs]
    Xte = np.stack([seg_embedding(s) for s in test_segs])
    yte = [s["step"] for s in test_segs]
    steps = sorted(set(ytr))
    proto = np.stack([Xtr[np.array(ytr) == s].mean(axis=0) for s in steps])
    proto = proto / np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-12)

    def accuracy(X, y, P, classes) -> tuple[float, list]:
        sims = (X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)) @ P.T
        pred = [classes[int(i)] for i in sims.argmax(axis=1)]
        return float(np.mean([p == t for p, t in zip(pred, y)])), pred

    # temperature: deterministic grid fit on TRAIN segments (NLL of softmax sims/T).
    # Wide grid; an edge solution falls back to T=1 with a note (no edge solutions kept).
    T_GRID = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    best_T, best_nll = T_GRID[0], float("inf")
    str_sims = (Xtr @ proto.T)
    for T in T_GRID:
        sc = str_sims / T
        sc = sc - sc.max(axis=1, keepdims=True)
        logp = sc - np.log(np.exp(sc).sum(axis=1, keepdims=True))
        idx = np.array([steps.index(t) for t in ytr])
        nll = float(-logp[np.arange(len(ytr)), idx].mean())
        if nll < best_nll:
            best_T, best_nll = T, nll
    t_note = "grid_best"
    if best_T in (T_GRID[0], T_GRID[-1]):
        best_T, t_note = 1.0, f"edge_fallback_to_1(grid_best={best_T})"

    cross_acc, cross_pred = accuracy(Xte, yte, proto, steps)
    # leave-one-segment-out within the train clip. Only segments whose step
    # has >= 2 segments are evaluable (13/14 steps are singletons, whose
    # class prototype would be empty when held out) -> tiny N, stated as-is.
    from collections import Counter as _Counter
    _counts = _Counter(ytr)
    loo_preds, loo_true = [], []
    for j in range(len(train_segs)):
        if _counts[ytr[j]] < 2:
            continue
        mask = np.ones(len(train_segs), bool)
        mask[j] = False
        Pj = np.stack([Xtr[mask][np.array(ytr)[mask] == s].mean(axis=0) for s in steps])
        Pj = Pj / np.maximum(np.linalg.norm(Pj, axis=1, keepdims=True), 1e-12)
        sims = Xtr[j][None, :] @ Pj.T
        loo_preds.append(steps[int(np.nanargmax(sims))])
        loo_true.append(ytr[j])
    loo_acc = float(np.mean([p == t for p, t in zip(loo_preds, loo_true)])) if loo_true else float("nan")
    loo_n = len(loo_true)

    sha = decision_mod.save_step_prototypes(
        heads_dir / decision_mod.STEP_PROTO, steps=steps,
        prototypes=proto, temperature=best_T,
        meta={"provenance": "indego_ecu", "temperature_note": t_note,
              "sampling": f"n={decision_mod.N_FRAMES}:fractions={list(decision_mod.FRAME_FRACTIONS)}"})
    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "step_prototypes",
        "train_clip": STEP_TRAIN_CLIP,
        "train_segments": len(train_segs),
        "train_steps": steps,
        "test_clip": STEP_TEST_CLIP,
        "cross_clip_segment_accuracy": round(cross_acc, 4),
        "cross_clip_n": len(test_segs),
        "cross_clip_detail": [
            {"step": t, "pred": p} for t, p in zip(yte, cross_pred)],
        "loo_train_clip_segment_accuracy": round(loo_acc, 4),
        "loo_n": loo_n,
        "loo_note": "only multi-segment steps are LOO-evaluable (13/14 train steps are singletons)",
        "temperature": best_T,
        "temperature_note": t_note,
        "temperature_grid": list(T_GRID),
        "proto_file": decision_mod.STEP_PROTO,
        "proto_sha256": sha,
        "backbone": f"embeddinggemma2:budget={BUDGET}:bfloat16",
        "seed": SEED,
    }
    out = Path(args.out if args.out != "eval/results/decider_baseline_DEV_ONLY.json"
               else "eval/results/decider_step.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"cross-clip acc={cross_acc:.4f} (N={len(test_segs)}) "
          f"loo acc={loo_acc:.4f} (N={loo_n}) T={best_T}; wrote {out}", flush=True)
    return 0


def _count_by_source(rows: list[dict]) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        out[r.get("source", "?")] = out.get(r.get("source", "?"), 0) + 1
    return dict(sorted(out.items()))


def _git_sha() -> str:
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def cmd_heads(args) -> int:
    """Hands/work heads (dev-subset or fixed-manifest mode).

    Temperature is fitted on CALIBRATION only; TEST is untouched until the
    final report below (I3). Threshold picked on calibration, bound on test.
    """
    repo_root = Path(__file__).resolve().parent.parent
    heads_dir = Path(args.heads_dir)
    if not heads_dir.is_absolute():
        heads_dir = repo_root / heads_dir
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = repo_root / cache_dir
    dev_mode = args.splits is None
    if dev_mode:
        rows = read_dev_rows(args.dev_n)
        n = len(rows)
        ntr, ncal = int(0.6 * n), int(0.2 * n)
        splits = {"train": rows[:ntr], "calibration": rows[ntr:ntr + ncal], "test": rows[ntr + ncal:]}
        split_source = f"DEV_ONLY: first {args.dev_n} frames round-robined across 3 parquet files, sorted by frame_id, 60/20/20"
        provenance = decision_mod.DEV_PROVENANCE
        n_per_split_per_source = {k: _count_by_source(v) for k, v in splits.items()}
    else:
        manifest = Path(args.splits)
        man = read_manifest_rows(manifest)
        msha = decision_mod.file_sha256(manifest)[:16]
        split_source = f"fixed manifest: {args.splits} (sha256:{msha})"
        provenance = f"full_splits:{msha}"
        if man["missing"] != {"train": 0, "calibration": 0, "test": 0}:
            print(f"WARNING manifest ids missing from parquet: {man['missing']}", flush=True)
        splits = {}
        n_per_split_per_source = {}
        for split in ("train", "calibration", "test"):
            full = man["rows"][split]
            n_per_split_per_source[split] = {"manifest": _count_by_source(full)}
            sub = subsample_split(full, split, SUBSAMPLE_CAPS[split])
            n_per_split_per_source[split]["subsampled"] = _count_by_source(sub)
            splits[split] = fetch_split_rows({r["frame_id"] for r in sub})
    print(f"train/cal/test: {[len(splits[k]) for k in ('train', 'calibration', 'test')]} ({split_source})", flush=True)

    # GPU only if the embedding cache misses (retrains are CPU-only).
    all_rows = splits["train"] + splits["calibration"] + splits["test"]
    model = None if cache_complete(all_rows, cache_dir) else load_model()
    print("embedding cache complete; no GPU needed" if model is None else "embedding cache miss; loading backbone", flush=True)
    X: dict[str, np.ndarray] = {}
    for split in ("train", "calibration", "test"):
        print(f"embedding {split} ({len(splits[split])} frames)...", flush=True)
        X[split] = embed_rows(model, splits[split], cache_dir)

    y_hands = {k: np.array([1 if r["hand_count"] >= 1 else 0 for r in splits[k]]) for k in X}
    y_work = {k: np.array([1 if r["active_labor"] == "yes" else 0 for r in splits[k]]) for k in X}

    results: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "DEV_ONLY" if dev_mode else "full",
        "split_source": split_source,
        "provenance": provenance,
        "n_per_split_per_source": n_per_split_per_source,
        "subsample": None if dev_mode else {
            "salt": SUBSAMPLE_SALT, "caps_per_source": SUBSAMPLE_CAPS,
            "method": "sha-order within source, take first cap (deterministic)"},
        "label_source": "gemini-2.5-flash (not human-checked)",
        "label_caveat": LABEL_CAVEAT,
        "label_caveat_long": LABEL_CAVEAT_LONG,
        "granularity": "per single frame",
        "granularity_note": ("heads trained/calibrated/evaluated on SINGLE frames; "
                             "serving averages the per-frame logits of 4 frames, so these "
                             "numbers describe per-frame performance, not Decider output; "
                             "a labelled multi-frame check is not possible (no labels)"),
        "backbone": f"embeddinggemma2:budget={BUDGET}:bfloat16",
        "backbone_id": decision_mod.backbone_id(decision_mod.EG2_DIR),
        "seed": SEED,
        "git_sha": _git_sha(),
        "command": " ".join(["python", "eval/train_heads.py"] + sys.argv[1:]),
        "heads": {},
    }
    for name, y, fname in (("hands_ok", y_hands, decision_mod.HANDS_HEAD),
                           ("active_work", y_work, decision_mod.WORK_HEAD)):
        coef, b = train_head(X["train"], y["train"])
        # Temperature fitted on CALIBRATION ONLY; test split untouched.
        T_cal = fit_temperature_binary(X["calibration"] @ coef + b, y["calibration"])
        cal = metrics_with_T(coef, b, T_cal, X["calibration"], y["calibration"])
        tst = metrics_with_T(coef, b, T_cal, X["test"], y["test"], pick=False)
        conf_t, pred_t = tst.pop("conf"), tst.pop("pred")
        # threshold chosen on CALIBRATION, bound estimated on held-out TEST.
        thr = cal["threshold_pick"]["threshold"]
        if thr is None:
            held = {"threshold": None, "accepted": 0, "errors": 0, "upper_bound": None}
        else:
            mask = conf_t >= thr
            errs = int((pred_t[mask] != y["test"][mask]).sum())
            acc_n = int(mask.sum())
            ub = certify_mod.exact_binomial_upper(errs, acc_n, delta=0.1) if acc_n > 0 else None
            held = {"threshold": round(float(thr), 4), "accepted": acc_n, "errors": errs,
                    "upper_bound": round(float(ub), 4) if ub is not None else None}
        sha = decision_mod.save_head(heads_dir / fname, coef=coef, intercept=b,
                                     temperature=T_cal,
                                     meta={"backbone": results["backbone"],
                                           "provenance": provenance})
        results["heads"][name] = {
            "calibration": cal, "test": tst,
            "held_out_bound_at_cal_threshold": held,
            "head_file": fname, "head_sha256": sha,
            "provenance": provenance,
        }
        print(f"{name}: T={T_cal:.4f} cal_acc={cal['accuracy']} cal_ece={cal['ece']} "
              f"test_acc={tst['accuracy']} test_ece={tst['ece']} bound={held}", flush=True)

    if not dev_mode:
        # Codec-shift measurement on the first 100 TEST frames (label-free).
        results["codec_shift"] = codec_shift_check(model, splits["test"], cache_dir, n=100)
        print(f"codec_shift: {results['codec_shift']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    raise SystemExit(main())
