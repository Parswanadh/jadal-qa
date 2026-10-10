"""Hub measurements on real data (GitHub issue #27).

Subcommands (all write JSON under eval/results/):
  throughput   (GPU) embedding segments/hour for 5/10/20 s x N=4/8 frames,
               GPU s/segment, peak VRAM. Wrap in flock (see docs/reports/hub.md).
  sizelat      (CPU) exact index bytes per 1000 h at dim 256/128 fp16 +
               exact-search median/p95 latency for N=1e4/1e5/1e6 SYNTHETIC
               random unit vectors.
  recall       (GPU) text->segment recall@1/5/10 per dim (768/256/128, truncated
               from the same measured 768-d vectors, renormalised) over IndEgo
               ECU-assembly clip segments; ground truth = VIA step intervals;
               hit = returned segment overlaps the step interval. N stated.
  determinism  (CPU/GPU) same query twice => identical results_hash; rebuild
               twice => identical index_hash.
  cascade      (CPU) triage-cascade table: hours of video/day a single laptop
               can process for L0-only vs L0+hub on survivors; survivor fraction
               MEASURED with a documented cheap proxy (jadal.signals is not on
               main yet). Parametric, no invented Awign numbers.
  all          sizelat + determinism + cascade (no GPU), for CI.

Never copies clips into the repo. Data: /home/parshu/projects/awing/data/indego.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
INDIGO = Path("/home/parshu/projects/awing/data/indego")
RESULTS = ROOT / "eval" / "results"

ECU_CLIPS = [
    "1_Assembly/mp4_480/User_15_410_0411_2_480.mp4",
    "1_Assembly/mp4_480/User_1_412_2010_3_480.mp4",
]
ANNOT_DIR = "1_Assembly/annotated_keysteps_assembly_disassembly"


# -- VIA annotations ----------------------------------------------------
def load_step_intervals(annot_path: str | os.PathLike) -> list[dict]:
    """Parse one VIA JSON -> [{step, start_s, end_s}]."""
    data = json.loads(Path(annot_path).read_text())
    attrs = data.get("attribute", {})
    optmap: dict[str, str] = {}
    for _aid, a in attrs.items():
        if isinstance(a, dict) and a.get("type") == 4 and "options" in a:
            optmap = {str(k): str(v) for k, v in a["options"].items()}
    out = []
    for _mid, m in data.get("metadata", {}).items():
        z = m.get("z")
        av = (m.get("av") or {})
        label_id = str(av.get("1", ""))
        step = optmap.get(label_id, label_id)
        if isinstance(z, list) and len(z) == 2:
            out.append({"step": step, "start_s": float(z[0]), "end_s": float(z[1])})
    out.sort(key=lambda r: (r["start_s"], r["end_s"]))
    return out


def step_to_query(step: str) -> str:
    s = step.strip()
    parts = s.split("_", 1)
    if parts[0].isdigit():
        s = parts[1] if len(parts) > 1 else s
    return s.replace("_", " ")


def fixed_segments(duration_s: float, seg_s: float, clip: str) -> list[dict]:
    segs, t, i = [], 0.0, 0
    while t < duration_s:
        segs.append({"segment_id": f"{Path(clip).stem}#{i:04d}",
                     "clip": clip, "start_s": round(t, 3),
                     "end_s": round(min(t + seg_s, duration_s), 3)})
        t += seg_s
        i += 1
    return segs


def clip_duration(path: str) -> float:
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return total / fps if fps > 0 else 0.0
    finally:
        cap.release()


def _write(name: str, obj: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2, sort_keys=True))
    print(f"wrote {p}")
    return p


# -- sizelat (CPU) ---------------------------------------------------------
def cmd_sizelat(args) -> dict:
    seg_s = args.seg_seconds
    segs_per_1000h = int(1000 * 3600 / seg_s)
    sizes = {str(d): segs_per_1000h * d * 2 for d in (256, 128)}
    lat: dict[str, dict] = {}
    rng = np.random.default_rng(27)
    for n in (10_000, 100_000, 1_000_000):
        mat = rng.standard_normal((n, 256)).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        q = rng.standard_normal(256).astype(np.float32)
        q /= np.linalg.norm(q)
        # warmup
        _ = mat[:1024] @ q
        ts = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            s = mat @ q
            s.partition(n - 10)
            ts.append((time.perf_counter() - t0) * 1000.0)
        lat[str(n)] = {"median_ms": round(float(statistics.median(ts)), 2),
                       "p95_ms": round(float(np.percentile(ts, 95)), 2),
                       "repeats": args.repeats, "data": "SYNTHETIC random unit vectors"}
    out = {"n": "segments counted exactly; latency vectors SYNTHETIC",
           "seg_seconds": seg_s, "segments_per_1000h": segs_per_1000h,
           "bytes_per_1000h_fp16": sizes,
           "bytes_formula": "segments * dim * 2",
           "latency_ms_dim256_fp32matmul_cpu": lat}
    _write("hub_sizelat.json", out)
    return out


# -- throughput (GPU) --------------------------------------------------------
def cmd_throughput(args) -> dict:
    import torch
    from jadal.hub_embed import HubEmbedder
    torch.use_deterministic_algorithms(True, warn_only=True)
    clip = str(INDIGO / ECU_CLIPS[1])
    dur = clip_duration(clip)
    emb = HubEmbedder(dim=256, n_frames=8,
                      cache_dir=None)  # no cache: measure raw model time
    emb._ensure_model()
    dev = next(emb._model.parameters()).device.type
    rows = []
    for seg_len in (5, 10, 20):
        for nframes in (4, 8):
            per: list[float] = []
            peak = 0.0
            for rep in range(args.reps):
                start = min(10.0 + rep * (seg_len + 5), max(dur - seg_len - 1, 0))
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    emb.embed_segment(clip, start, start + seg_len, n_frames=nframes)
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    peak = max(peak, float(torch.cuda.max_memory_allocated()) / 1e9)
                else:
                    t0 = time.perf_counter()
                    emb.embed_segment(clip, start, start + seg_len, n_frames=nframes)
                    dt = time.perf_counter() - t0
                per.append(dt)
            med = float(statistics.median(per))
            rows.append({"seg_len_s": seg_len, "n_frames": nframes,
                         "gpu_s_per_segment_median": round(med, 3),
                         "segments_per_hour": round(3600 / med, 1),
                         "peak_vram_GB": round(peak, 3),
                         "reps": args.reps, "device": dev,
                         "clip": ECU_CLIPS[1], "data": "real IndEgo clip"})
    out = {"rows": rows, "note": "GPU job must run under flock gpu.lock"}
    _write("hub_throughput.json", out)
    return out


# -- recall (GPU) ------------------------------------------------------------
def cmd_recall(args) -> dict:
    import torch
    from jadal.hub import Index
    from jadal.hub_embed import HubEmbedder
    torch.use_deterministic_algorithms(True, warn_only=True)
    seg_s = args.seg_seconds
    # Build segments over ECU clips.
    segments: list[dict] = []
    for rel in ECU_CLIPS:
        p = str(INDIGO / rel)
        if not Path(p).is_file():
            continue
        segments += fixed_segments(clip_duration(p), seg_s, rel)
    # Ground truth: VIA intervals for the ECU clips.
    gt: list[dict] = []
    for rel in ECU_CLIPS:
        stem = Path(rel).stem
        if stem.endswith("_480"):
            stem = stem[:-4]  # annotation files use the pre-480 stem
        matches = glob.glob(str(INDIGO / ANNOT_DIR / f"AS_{stem}.json"))
        for m in matches:
            for r in load_step_intervals(m):
                gt.append({"clip": rel, "step": r["step"],
                           "start_s": r["start_s"], "end_s": r["end_s"],
                           "query": step_to_query(r["step"])})
    emb = HubEmbedder(dim=768, n_frames=args.n_frames, cache_dir=None)
    # Segment embeddings (video route, batch size 1).
    seg_vecs = []
    for s in segments:
        v = emb.embed_segment(str(INDIGO / s["clip"]), s["start_s"], s["end_s"])
        seg_vecs.append(np.asarray(v, dtype=np.float32))
    seg_mat = np.stack(seg_vecs) if seg_vecs else np.zeros((0, 768), np.float32)
    out_dims: dict[str, dict] = {}
    queries = sorted({g["query"] for g in gt})
    qvecs = {q: np.asarray(emb.embed_text(q), dtype=np.float32) for q in queries}
    for dim in (768, 256, 128):
        def trunc(v):
            t = np.asarray(v)[:dim]
            return t / np.linalg.norm(t)
        sm = np.stack([trunc(v) for v in seg_vecs]) if seg_vecs else seg_mat
        hits_at = {1: 0, 5: 0, 10: 0}
        for g in gt:
            q = trunc(qvecs[g["query"]])
            scores = sm @ q
            order = np.argsort(-scores, kind="stable")
            ranked = [segments[i] for i in order[:10]]
            hit_ranks = [r + 1 for r, s in enumerate(ranked)
                         if s["clip"] == g["clip"] and
                         s["start_s"] < g["end_s"] and s["end_s"] > g["start_s"]]
            best = min(hit_ranks) if hit_ranks else None
            for kk in (1, 5, 10):
                if best is not None and best <= kk:
                    hits_at[kk] += 1
        n = len(gt)
        out_dims[str(dim)] = {
            f"recall@{kk}": round(hits_at[kk] / n, 4) if n else None
            for kk in (1, 5, 10)} | {"n_queries": n,
                                     "n_segments": len(segments),
                                     "seg_seconds": seg_s,
                                     "note": ("N is small (%d step intervals over %d "
                                              "segments); do not generalise." % (n, len(segments)))}
    out = {"dims": out_dims, "data": "real IndEgo ECU clips + VIA annotations",
           "truncation": "leading-dims of the same measured 768-d vectors, renormalised"}
    _write("hub_recall.json", out)
    # Also persist the measured 768-d segment matrix for determinism checks.
    np.save(RESULTS / "hub_recall_segvecs.npy", seg_mat)
    return out


# -- determinism ---------------------------------------------------------------
def cmd_determinism(args) -> dict:
    from jadal.hub import Index
    vecs_path = RESULTS / "hub_recall_segvecs.npy"
    if vecs_path.is_file():
        mat = np.load(vecs_path)
        dim = mat.shape[1]
        ids = [f"seg_{i:04d}" for i in range(mat.shape[0])]
        data_note = "measured 768-d IndEgo segment embeddings"
    else:
        rng = np.random.default_rng(27)
        dim, n = 256, 64
        mat = rng.standard_normal((n, dim)).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        ids = [f"seg_{i:04d}" for i in range(n)]
        data_note = "SYNTHETIC fixed-seed vectors (no measured matrix yet)"
    spin = lambda seed: mat  # build is pure function of (ids, matrix)
    hashes = []
    for _ in range(2):
        idx = Index(dim=dim)
        for i, sid in enumerate(ids):
            idx.add(sid, mat[i], {"clip": "c", "start_s": 0.0, "end_s": 1.0,
                                  "worker_id": "w", "site_id": "s",
                                  "verdict": "review", "reasons": [],
                                  "step": None, "path": None})
        hashes.append(idx.index_hash())
    idx = Index(dim=dim)
    for i, sid in enumerate(ids):
        idx.add(sid, mat[i], {"clip": "c", "start_s": 0.0, "end_s": 1.0,
                              "worker_id": "w", "site_id": "s",
                              "verdict": "review", "reasons": [],
                              "step": None, "path": None})
    q = mat[0]
    r1 = idx.search_by_vector(q, k=5)["results_hash"]
    r2 = idx.search_by_vector(q, k=5)["results_hash"]
    out = {"rebuild_index_hash_1": hashes[0], "rebuild_index_hash_2": hashes[1],
           "rebuild_identical": hashes[0] == hashes[1],
           "query_results_hash_1": r1, "query_results_hash_2": r2,
           "query_identical": r1 == r2, "data": data_note,
           "n": len(ids), "dim": dim}
    _write("hub_determinism.json", out)
    return out


# -- cascade (CPU proxy) ---------------------------------------------------------
def cmd_cascade(args) -> dict:
    """Cheap L0 proxy (jadal.signals not on main): Laplacian blur variance +
    brightness gate. Survivor fraction MEASURED on real IndEgo segments."""
    import cv2
    from jadal.hub_embed import sample_frames
    seg_s = 10.0
    segments: list[dict] = []
    for rel in ECU_CLIPS:
        p = str(INDIGO / rel)
        if Path(p).is_file():
            segments += fixed_segments(clip_duration(p), seg_s, rel)
    t0 = time.perf_counter()
    surv = 0
    for s in segments:
        try:
            frames = sample_frames(str(INDIGO / s["clip"]), s["start_s"],
                                   s["end_s"], 4)
            vars_ = []
            for f in frames:
                g = cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2GRAY)
                vars_.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
                if float(g.mean()) < 15:  # too dark proxy
                    vars_ = [-1.0]
                    break
            if float(np.median(vars_)) >= 30.0:  # in-focus proxy
                surv += 1
        except Exception:
            pass
    l0_s = (time.perf_counter() - t0) / max(len(segments), 1)
    frac = surv / max(len(segments), 1)
    # Embedding cost: from measured throughput file if present, else param.
    tp = {}
    tp_path = RESULTS / "hub_throughput.json"
    if tp_path.is_file():
        tp = json.loads(tp_path.read_text())
    emb_s = None
    for r in tp.get("rows", []):
        if r["seg_len_s"] == 10 and r["n_frames"] == 8:
            emb_s = r["gpu_s_per_segment_median"]
    if emb_s is None:
        emb_s = args.assumed_embed_s
    l0_day = 86400 / l0_s
    l0_hours = l0_day * seg_s / 3600
    both_day = 86400 / (l0_s + frac * emb_s)
    both_hours = both_day * seg_s / 3600
    out = {"proxy": "Laplacian-var>=30 + mean>=15 on 4 frames (NOT jadal.signals; signals not on main)",
           "n_segments_measured": len(segments), "seg_seconds": seg_s,
           "survivor_fraction": round(frac, 4),
           "l0_s_per_segment_cpu": round(l0_s, 4),
           "embed_s_per_segment_gpu": emb_s,
           "embed_source": "measured" if tp else f"assumed param {args.assumed_embed_s}",
           "table_hours_video_per_day": {
               "L0_only": round(l0_hours, 2),
               "L0_plus_hub_on_survivors": round(both_hours, 2)},
           "note": "parametric; no invented Awign numbers"}
    _write("hub_cascade.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("throughput"); p.add_argument("--reps", type=int, default=2)
    p = sub.add_parser("sizelat"); p.add_argument("--seg-seconds", type=float, default=10.0)
    p.add_argument("--repeats", type=int, default=20)
    p = sub.add_parser("recall"); p.add_argument("--seg-seconds", type=float, default=10.0)
    p.add_argument("--n-frames", type=int, default=8)
    p = sub.add_parser("determinism")
    p = sub.add_parser("cascade"); p.add_argument("--assumed-embed-s", type=float, default=2.0)
    p = sub.add_parser("all")
    args = ap.parse_args()
    if args.cmd == "all":
        cmd_sizelat(argparse.Namespace(seg_seconds=10.0, repeats=20))
        cmd_determinism(args)
        cmd_cascade(argparse.Namespace(assumed_embed_s=2.0))
    else:
        {"throughput": cmd_throughput, "sizelat": cmd_sizelat,
         "recall": cmd_recall, "determinism": cmd_determinism,
         "cascade": cmd_cascade}[args.cmd](args)


if __name__ == "__main__":
    main()
