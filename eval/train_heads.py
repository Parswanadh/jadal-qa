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
- One frame per segment here (parquet rows are frames); the Decider
  mean-pools 4 frames per video segment at inference.
- STEP head is separate (IndEgo stretch goal).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
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
THUMB = 512
LABEL_CAVEAT = "labels by gemini-2.5-flash, not human-checked"

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
STEP_FRAMES_PER_SEG = 5
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
    """Read full train/calibration/test rows from a fixed splits manifest.

    Supports manifests shaped as {"train": [...], "calibration": [...],
    "test": [...]} where entries carry frame_id + file, or {"splits": {...}}.
    Row pixels are re-read from the parquet files by frame_id.
    """
    import pyarrow.parquet as pq

    spec = json.loads(manifest.read_text())
    splits = spec.get("splits", spec)
    by_id: dict[str, dict] = {}
    for f in FILES:
        pf = pq.ParquetFile(str(Path(DATA_DIR) / f))
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_groups([rg], columns=["frame_id", "image", "hand_count", "active_labor"])
            d = tbl.to_pydict()
            for i in range(tbl.num_rows):
                by_id[d["frame_id"][i]] = {
                    "frame_id": d["frame_id"][i],
                    "bytes": bytes(d["image"][i]["bytes"]),
                    "hand_count": int(d["hand_count"][i]),
                    "active_labor": str(d["active_labor"][i]),
                }
    out: dict[str, list[dict]] = {}
    for split in ("train", "calibration", "test"):
        entries = splits.get(split, [])
        rows = []
        for e in entries:
            fid = e.get("frame_id") if isinstance(e, dict) else e
            if fid in by_id:
                rows.append(by_id[fid])
        rows.sort(key=lambda r: r["frame_id"])
        out[split] = rows
    return out


def embed_rows(model, rows: list[dict], cache_dir: Path) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    embs: list[np.ndarray] = []
    for k, r in enumerate(rows):
        key = hashlib.sha256(r["bytes"]).hexdigest()
        hit = cache_dir / f"{key}.npy"
        if hit.exists():
            embs.append(np.load(str(hit)))
            continue
        img = Image.open(io.BytesIO(r["bytes"])).convert("RGB")
        img.thumbnail((THUMB, THUMB), Image.BILINEAR)
        with torch.inference_mode():
            e = model.encode(
                img, batch_size=1, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
                processing_kwargs={"image": {"max_soft_tokens": BUDGET}},
            )
        e = np.asarray(e, dtype=np.float64).reshape(-1)
        np.save(str(hit), e)
        embs.append(e)
        if (k + 1) % 200 == 0:
            print(f"  embedded {k + 1}/{len(rows)}", flush=True)
    return np.stack(embs, axis=0)


def train_head(Xtr: np.ndarray, ytr: np.ndarray) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    clf.fit(Xtr, ytr)
    return clf.coef_.reshape(-1), float(clf.intercept_[0])


def eval_head(coef, intercept, X, y, target_err=0.05, delta=0.1) -> dict:
    logits = X @ coef + intercept
    logits2 = np.stack([-logits / 2.0, logits / 2.0], axis=1)
    T = certify_mod.fit_temperature(logits2, y)
    probs = 1.0 / (1.0 + np.exp(-logits / T))
    pred = (probs >= 0.5).astype(int)
    acc = float((pred == y).mean())
    ece = certify_mod.ece(probs, y)
    conf = np.maximum(probs, 1.0 - probs)
    correct = (pred == y)
    pick = certify_mod.pick_threshold(conf, correct, target_err=target_err, delta=delta)
    return {
        "n": int(len(y)), "accuracy": round(acc, 4), "ece": round(float(ece), 4),
        "temperature": round(float(T), 4),
        "threshold_pick": {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in pick.items()},
        "probs_mean": round(float(probs.mean()), 4),
    }


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
                          k: int = STEP_FRAMES_PER_SEG) -> list[np.ndarray]:
    """Fixed-fraction BGR frames from [start_s, end_s] (OpenCV)."""
    import cv2

    cap = cv2.VideoCapture(clip_path)
    try:
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out = []
        for i in range(k):
            t = start_s + (end_s - start_s) * (i + 0.5) / k
            idx = min(max(int(t * fps), 0), max(total - 1, 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                out.append(frame)
        return out
    finally:
        cap.release()


def embed_pil_frames(model, imgs: list[Image.Image], cache_dir: Path) -> np.ndarray:
    embs = []
    for img in imgs:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        key = hashlib.sha256(buf.getvalue()).hexdigest()
        hit = cache_dir / f"{key}.npy"
        if hit.exists():
            embs.append(np.load(str(hit)))
            continue
        with torch.inference_mode():
            e = model.encode(
                img, batch_size=1, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
                processing_kwargs={"image": {"max_soft_tokens": BUDGET}},
            )
        e = np.asarray(e, dtype=np.float64).reshape(-1)
        np.save(str(hit), e)
        embs.append(e)
    return np.stack(embs, axis=0)


def cmd_step(args) -> int:
    """IndEgo ECU STEP head: prototypes on train clip, segment accuracy held-out."""
    import cv2  # noqa: F401  (ensures the dependency is present)

    heads_dir = Path(args.heads_dir)
    cache_dir = Path("models/.embed_cache_indego")
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
    from PIL import Image as PILImage

    def seg_embedding(seg: dict) -> np.ndarray:
        clip_path = INDEGO_CLIPS[seg["clip"]][0]
        frames = sample_segment_frames(clip_path, seg["start_s"], seg["end_s"])
        imgs = [PILImage.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        for im in imgs:
            im.thumbnail((THUMB, THUMB), PILImage.BILINEAR)
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

    # temperature: deterministic grid fit on TRAIN segments (NLL of softmax sims/T)
    best_T, best_nll = 0.05, float("inf")
    str_sims = (Xtr @ proto.T)
    for T in (0.02, 0.05, 0.1, 0.2):
        sc = str_sims / T
        sc = sc - sc.max(axis=1, keepdims=True)
        logp = sc - np.log(np.exp(sc).sum(axis=1, keepdims=True))
        idx = np.array([steps.index(t) for t in ytr])
        nll = float(-logp[np.arange(len(ytr)), idx].mean())
        if nll < best_nll:
            best_T, best_nll = T, nll

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
        prototypes=proto, temperature=best_T)
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


def cmd_heads(args) -> int:
    """Hands/work heads (dev-subset or fixed-manifest mode)."""
    heads_dir = Path(args.heads_dir)
    dev_mode = args.splits is None
    if dev_mode:
        rows = read_dev_rows(args.dev_n)
        n = len(rows)
        ntr, ncal = int(0.6 * n), int(0.2 * n)
        splits = {"train": rows[:ntr], "calibration": rows[ntr:ntr + ncal], "test": rows[ntr + ncal:]}
        split_source = f"DEV_ONLY: first {args.dev_n} frames round-robined across 3 parquet files, sorted by frame_id, 60/20/20"
    else:
        splits = read_manifest_rows(Path(args.splits))
        split_source = f"fixed manifest: {args.splits}"
    print(f"train/cal/test: {[len(splits[k]) for k in ('train', 'calibration', 'test')]} ({split_source})", flush=True)

    model = load_model()
    cache_dir = Path(args.cache_dir)
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
        "label_caveat": LABEL_CAVEAT,
        "backbone": f"embeddinggemma2:budget={BUDGET}:bfloat16",
        "seed": SEED,
        "heads": {},
    }
    for name, y, fname in (("hands_ok", y_hands, decision_mod.HANDS_HEAD),
                           ("active_work", y_work, decision_mod.WORK_HEAD)):
        coef, b = train_head(X["train"], y["train"])
        # temperature fit on CALIBRATION, metrics on CALIBRATION + held-out TEST
        cal = eval_head(coef, b, X["calibration"], y["calibration"])
        test = eval_head(coef, b, X["test"], y["test"])
        # certified bound on TEST at the CALIBRATION-chosen threshold
        thr = cal["threshold_pick"]["threshold"]
        logits_t = X["test"] @ coef + b
        probs_t = 1.0 / (1.0 + np.exp(-logits_t / cal["temperature"]))
        pred_t = (probs_t >= 0.5).astype(int)
        if thr is None:
            held = {"threshold": None, "accepted": 0, "errors": 0, "upper_bound": None}
        else:
            conf_t = np.maximum(probs_t, 1.0 - probs_t)
            mask = conf_t >= thr
            errs = int((pred_t[mask] != y["test"][mask]).sum())
            acc_n = int(mask.sum())
            ub = certify_mod.exact_binomial_upper(errs, acc_n, delta=0.1) if acc_n > 0 else None
            held = {"threshold": round(float(thr), 4), "accepted": acc_n, "errors": errs,
                    "upper_bound": round(float(ub), 4) if ub is not None else None}
        sha = decision_mod.save_head(heads_dir / fname, coef=coef, intercept=b,
                                     temperature=cal["temperature"],
                                     meta={"backbone": [results["backbone"]]})
        results["heads"][name] = {
            "calibration": cal, "test": test,
            "held_out_bound_at_cal_threshold": held,
            "head_file": fname, "head_sha256": sha,
        }
        print(f"{name}: cal_acc={cal['accuracy']} cal_ece={cal['ece']} "
              f"test_acc={test['accuracy']} test_ece={test['ece']} bound={held}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    raise SystemExit(main())
