"""Embed ego10k_eval parquet frames with EmbeddingGemma-2 (batched, sharded ckpts).

Same backbone + preprocessing as the decider (read-only ref:
../decider worktree jadal/decision.py): SentenceTransformer on
/home/parshu/projects/awing/data/eg2, bfloat16, thumbnail longest side 512
BILINEAR, normalize_embeddings=True, max_soft_tokens=140.
DIFFERS from serving on purpose: batched encode (batch-of-1 receipt
semantics do not matter here; noted everywhere). GPU only under flock
(caller wraps), CPU decode outside the lock is fine.
Checkpoints: <out_dir>/shard_{i:03d}.npz (float16) + done.json; reruns skip
done shards, so reruns are free. Final: embeddings.fp16.npy (N,768),
ids.npy, meta.json (throughput, vram peak, sha).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

EG2_DIR = "/home/parshu/projects/awing/data/eg2"
PREPROCESS_MAX_SIDE = 512
EMBED_DIM = 768


def preprocess_bytes(raw: bytes, gpu: bool = False):
    if gpu:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from gpu_pre import gpu_thumb_jpeg
        return gpu_thumb_jpeg(raw)
    from PIL import Image

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((PREPROCESS_MAX_SIDE, PREPROCESS_MAX_SIDE), Image.BILINEAR)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--shard-rows", type=int, default=1000)
    ap.add_argument("--max-new-shards", type=int, default=0,
                    help="0 = all todo shards; else encode at most this many new shards per invocation (keeps lock holds short)")
    ap.add_argument("--gpu-decode", action="store_true",
                    help="nvjpeg GPU decode + cuda resize (else CPU PIL); see gpu_pre.py")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / ".scratch_note").write_text("checkpoints; safe to delete shards after concat\n")

    f = pq.ParquetFile(args.parquet)
    total = f.metadata.num_rows if not args.limit else min(f.metadata.num_rows, args.limit)
    n_shards = (total + args.shard_rows - 1) // args.shard_rows
    todo = [i for i in range(n_shards) if not (out / f"shard_{i:03d}.npz").exists()]
    if args.max_new_shards:
        todo = todo[:args.max_new_shards]
        print(f"capped to {len(todo)} new shards this invocation", flush=True)
    todo_set = set(todo)
    print(f"rows={total} shards={n_shards} todo={len(todo)} batch={args.batch_size}", flush=True)

    model = None
    if todo:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.manual_seed(1234)
        np.random.seed(1234)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        assert torch.cuda.is_available(), "EG2 backbone requires CUDA"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        torch.cuda.reset_peak_memory_stats()
        t_load0 = time.time()
        model = SentenceTransformer(str(EG2_DIR), model_kwargs={"torch_dtype": dtype},
                                    trust_remote_code=True)
        model.eval()
        print(f"model loaded in {time.time()-t_load0:.1f}s dtype={dtype}", flush=True)

    # stream row-group batches, decode, accumulate per shard
    import pyarrow as pa

    n_done = 0
    frames_s = 0.0
    t0 = time.time()
    row_buf: list = []   # (frame_id, pil, y_hand, y_work)
    shard_idx_for_row = -1

    def encode_pils(pils):
        import torch

        with torch.inference_mode():
            emb = model.encode(pils, batch_size=args.batch_size, show_progress_bar=False,
                               convert_to_numpy=True, normalize_embeddings=True,
                               processing_kwargs={"image": {"max_soft_tokens": 140}})
        return np.asarray(emb, dtype=np.float16)

    # gather all (id, raw bytes, labels) row-group by row-group (never all images at once)
    pending: dict[int, dict] = {}  # shard -> {ids, pils, hands, work}
    global_row = 0
    for rg in range(f.metadata.num_row_groups):
        if global_row >= total:
            break
        tbl = f.read_row_group(rg, columns=["frame_id", "image", "hand_count", "active_labor"])
        d = tbl.to_pandas()
        for _, r in d.iterrows():
            if global_row >= total:
                break
            s = global_row // args.shard_rows
            global_row += 1
            if (out / f"shard_{s:03d}.npz").exists() or s not in todo_set:
                continue  # already checkpointed or out of this invocation's scope
            entry = pending.setdefault(s, {"ids": [], "pils": [], "hands": [], "work": []})
            raw = r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"]
            entry["ids"].append(str(r["frame_id"]))
            entry["pils"].append(preprocess_bytes(bytes(raw), gpu=args.gpu_decode))
            entry["hands"].append(int(r["hand_count"]))
            entry["work"].append(1 if str(r["active_labor"]) == "yes" else 0)
            raw = None
        # encode any full pending shards opportunistically to bound RAM
        for s in sorted(pending):
            e = pending[s]
            if len(e["pils"]) >= args.shard_rows or global_row >= total:
                t1 = time.time()
                emb = encode_pils(e["pils"])
                frames_s += time.time() - t1
                np.savez_compressed(out / f"shard_{s:03d}.npz", emb=emb,
                                    ids=np.asarray(e["ids"]),
                                    hand=np.asarray(e["hands"], dtype=np.int8),
                                    work=np.asarray(e["work"], dtype=np.int8))
                n_done += len(e["ids"])
                print(f"shard {s} done ({len(e['ids'])} rows) [{n_done}/{total}]", flush=True)
                del pending[s]
        d = None
    # leftover partial shards
    for s in sorted(pending):
        e = pending[s]
        t1 = time.time()
        emb = encode_pils(e["pils"])
        frames_s += time.time() - t1
        np.savez_compressed(out / f"shard_{s:03d}.npz", emb=emb,
                            ids=np.asarray(e["ids"]),
                            hand=np.asarray(e["hands"], dtype=np.int8),
                            work=np.asarray(e["work"], dtype=np.int8))
        n_done += len(e["ids"])
        print(f"shard {s} done ({len(e['ids'])} rows) [{n_done}/{total}]", flush=True)

    # concat (only when every shard is checkpointed; else stop — rerun continues)
    import torch

    missing = [i for i in range(n_shards) if not (out / f"shard_{i:03d}.npz").exists()]
    if missing:
        print(f"partial: {n_shards - len(missing)}/{n_shards} shards done; rerun to continue", flush=True)
        return

    embs, ids, hands, works = [], [], [], []
    for i in range(n_shards):
        z = np.load(out / f"shard_{i:03d}.npz")
        embs.append(z["emb"].astype(np.float16))
        ids.append([str(x) for x in z["ids"]])
        hands.append(z["hand"])
        works.append(z["work"])
    E = np.concatenate(embs, axis=0)
    np.save(out / "embeddings.fp16.npy", E)
    np.save(out / "ids.npy", np.asarray(sum(ids, [])))
    np.save(out / "hand.npy", np.concatenate(hands))
    np.save(out / "work.npy", np.concatenate(works))
    wall = time.time() - t0
    vram_peak = float(torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    sha = hashlib.sha256(E.tobytes()).hexdigest()[:16]
    meta = {"n": int(E.shape[0]), "dim": int(E.shape[1]), "dtype": "float16",
            "batch_encode_s": frames_s, "wall_s": wall,
            "frames_per_s_encode": float(E.shape[0] / frames_s) if frames_s else 0.0,
            "frames_per_s_wall": float(E.shape[0] / wall) if wall else 0.0,
            "vram_peak_gb": vram_peak, "embed_sha16": sha,
            "backbone": "embeddinggemma2(batched; receipt batch-size-1 semantics N/A)",
            "preprocess": f"thumbnail longest side {PREPROCESS_MAX_SIDE} BILINEAR",
            "parquet": str(args.parquet), "batch_size": args.batch_size,
            "gpu_decode": bool(args.gpu_decode)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
