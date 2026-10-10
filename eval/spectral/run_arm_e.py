"""Arm E: EmbeddingGemma-2 frozen 4-frame embedding (harness worker).

Mirrors the decider's embedding path READ-ONLY (imports preprocess/decider
helpers, never edits them). Backbone: /home/parshu/projects/awing/data/eg2.
Runs on GPU only, caller must hold: flock /home/parshu/projects/jadal-qa-wt/tools/gpu.lock

Outputs under /home/parshu/projects/jadal-qa-wt/spectral_data/arms/E/:
  gold.json   {score:null, features:{emb:[768] mean-renormed}, latency_s}
  synth.json  ditto for synth items
  dups.json   pair cosine similarities of mean embeddings
  chunks.json per-chunk single-frame embeddings + running-mean prefixes
  cost.json   per-segment latency, GPU frames/s, VRAM, cpu benchmark note

Usage: flock -w 3600 .../tools/gpu.lock python -m eval.spectral.run_arm_e
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, "/home/parshu/projects/jadal-qa-wt/decider")

DATA = Path("/home/parshu/projects/jadal-qa-wt/spectral_data")
OUTDIR = DATA / "arms" / "E"
SCRATCH = HARNESS / ".scratch" / "armE"

CHUNK_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)  # within each 2-s chunk


def version() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, cwd=str(HARNESS))
    return f"harness:{r.stdout.strip()}/eg2:frozen"


def cut_segment(src: str, start: float, dur: float, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", str(start), "-t", str(dur), "-i", src,
         "-an", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", str(dst)], check=True)


def main() -> None:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from jadal.decision import (preprocess_bgr, sample_frames_bgr,
                                FRAME_FRACTIONS, DEFAULT_BUDGET)

    assert torch.cuda.is_available(), "arm E requires CUDA"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    ver = version()

    torch.manual_seed(1234)
    model = SentenceTransformer("/home/parshu/projects/awing/data/eg2",
                                model_kwargs={"torch_dtype": torch.bfloat16},
                                trust_remote_code=True)
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    def embed(frames):
        t0 = time.perf_counter()
        with torch.inference_mode():
            embs = []
            for f in frames:
                e = model.encode(
                    preprocess_bgr(f), batch_size=1, show_progress_bar=False,
                    convert_to_numpy=True, normalize_embeddings=True,
                    processing_kwargs={"image": {"max_soft_tokens": DEFAULT_BUDGET}})
                embs.append(np.asarray(e, dtype=np.float64).reshape(-1))
        return embs, time.perf_counter() - t0

    def seg_mean(embs):
        m = np.mean(np.stack(embs), axis=0)
        n = float(np.linalg.norm(m))
        return (m / n) if n > 0 else m

    gold = json.loads((DATA / "manifest_gold.json").read_text())["items"]
    synth = json.loads((DATA / "manifest_synth.json").read_text())["items"]

    def process(path, is_gold, it):
        if is_gold:
            with tempfile.TemporaryDirectory(dir=str(SCRATCH)) as td:
                seg = Path(td) / "seg.mp4"
                cut_segment(it["path"], it["start_s"],
                            it["end_s"] - it["start_s"], seg)
                return process_file(str(seg), it["id"])
        return process_file(path, it["id"])

    def process_file(segpath, iid):
        import cv2
        frames = sample_frames_bgr(segpath, n=4, fractions=FRAME_FRACTIONS)
        if not frames:
            return None
        embs, dt = embed(frames)
        mean = seg_mean(embs)
        # per-chunk single-frame embeddings (middle of each 2-s chunk)
        cap = cv2.VideoCapture(segpath)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        chunk_embs, cdt = [], 0.0
        for c in range(5):
            idx = min(int(total * (c * 2 + 1) / 10.0), max(total - 1, 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            cap.release() if False else None
            if ok and fr is not None:
                e, d = embed([fr])
                chunk_embs.append(e[0])
                cdt += d
        cap.release()
        running = []
        for k in range(1, 6):
            m = seg_mean(chunk_embs[:k])
            running.append(m)
        return {"emb": mean, "lat": dt,
                "chunk_embs": chunk_embs, "running": running, "cdt": cdt}

    gold_items, chunk_items = {}, {}
    n_frames = 0
    t_all = time.perf_counter()
    for it in gold:
        r = process(it["path"], True, it)
        if r is None:
            continue
        gold_items[it["id"]] = {"score": None,
                                "features": {"emb": [round(float(x), 6) for x in r["emb"]]},
                                "latency_s": round(r["lat"], 3)}
        chunk_items[it["id"]] = {
            "chunk_embs": [[round(float(x), 5) for x in e] for e in r["chunk_embs"]],
            "running_mean": [[round(float(x), 5) for x in e] for e in r["running"]],
            "latency_s": round(r["cdt"], 3)}
        n_frames += 4 + len(r["chunk_embs"])
    synth_items = {}
    for it in synth:
        r = process(it["path"], False, it)
        if r is None:
            continue
        synth_items[it["id"]] = {"score": None,
                                 "features": {"emb": [round(float(x), 6) for x in r["emb"]]},
                                 "latency_s": round(r["lat"], 3)}
        chunk_items[it["id"]] = {
            "chunk_embs": [[round(float(x), 5) for x in e] for e in r["chunk_embs"]],
            "running_mean": [[round(float(x), 5) for x in e] for e in r["running"]],
            "latency_s": round(r["cdt"], 3)}
        n_frames += 4 + len(r["chunk_embs"])
    wall = time.perf_counter() - t_all

    (OUTDIR / "gold.json").write_text(json.dumps(
        {"arm": "E", "version": ver, "backbone": "embeddinggemma-2 frozen, 4 frames",
         "items": gold_items}, indent=1))
    (OUTDIR / "synth.json").write_text(json.dumps(
        {"arm": "E", "version": ver, "items": synth_items}, indent=1))
    (OUTDIR / "chunks.json").write_text(json.dumps(
        {"arm": "E", "version": ver, "prefix_until_s": [2, 4, 6, 8, 10],
         "items": chunk_items}))

    # dups cosine
    dups = json.loads((DATA / "manifest_dups.json").read_text())
    means = {}
    for fid, fpath in dups["files"].items():
        frames = sample_frames_bgr(fpath, n=4, fractions=FRAME_FRACTIONS)
        embs, _ = embed(frames)
        means[fid] = seg_mean(embs)
        n_frames += 4
    pairs = {}
    for p in dups["pairs"]:
        a, b = means[p["a"]], means[p["b"]]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        pairs[f"{p['a']}|{p['b']}"] = {"similarity": round((cos + 1) / 2, 4),
                                       "cosine": round(cos, 4),
                                       "latency_s": None}
    (OUTDIR / "dups.json").write_text(json.dumps(
        {"arm": "E", "method": "eg2_cosine", "version": ver,
         "pairs": pairs}, indent=1))

    vram_mb = torch.cuda.max_memory_allocated() / 1e6
    cost = {"gpu": torch.cuda.get_device_name(0),
            "segments": len(gold_items) + len(synth_items),
            "frames_embedded": n_frames,
            "wall_s": round(wall, 1),
            "frames_per_s": round(n_frames / wall, 2),
            "segments_per_hour": int(3600 / (wall / max(len(gold_items) + len(synth_items), 1))),
            "peak_vram_mb": round(vram_mb, 1),
            "note": "GPU-only; CPU fallback does not exist for EG2"}
    (OUTDIR / "cost.json").write_text(json.dumps(
        {"arm": "E", "version": ver, "cost": cost}, indent=1))
    print(json.dumps(cost, indent=1), flush=True)
    print(f"gold={len(gold_items)} synth={len(synth_items)} pairs={len(pairs)}", flush=True)


if __name__ == "__main__":
    main()
