"""Arm D runner: generic VLM (SmolVLM2) zero-shot usability judgement.

Modes:
  frames8   - fixed 8 frames at fixed fractions (contract baseline)
  video_fps1/video_fps2/video_max - native VIDEO input over the whole 10-s
    segment (10 / 20 / MAX_FRAMES frames uniformly sampled)

Temperature 0 (greedy). Score = P(usable) from first-answer-token logprobs
over yes/no variants; parsed JSON recorded alongside.

GPU must be acquired by the caller via:
  flock -w 3600 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock <cmd>

Writes: spectral_data/arms/D_<model>/<manifest>.json
  {"arm","version","mode","items":{id:{"score","features","latency_s"}}}
"""
import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "_vendor"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
ARM_NAME = "D_SmolVLM2-2.2B-Instruct"
MAX_FRAMES = 32  # whole-video ceiling probed for fit on 8 GB

YES_TOKENS = ["yes", "Yes", "YES", " yes", " Yes", "yes,", "Yes,"]
NO_TOKENS = ["no", "No", "NO", " no", " No", "no,", "No,"]


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def sample_frames(video_path, start_s, end_s, n_frames):
    """Uniformly sample n_frames across [start_s, end_s). Returns list of RGB arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if end_s is None:  # full-file item (e.g. synth mp4s): use whole duration
        end_s = (total / fps) if fps else 10.0
        start_s = start_s or 0.0
    frames = []
    for i in range(n_frames):
        frac = (i + 0.5) / n_frames
        t = start_s + frac * (end_s - start_s)
        idx = min(int(t * fps), max(total - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return frames


def load_model():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.to("cuda")
    model.eval()
    return proc, model


def variant_ids(tokenizer, variants):
    ids = []
    for v in variants:
        enc = tokenizer.encode(v, add_special_tokens=False)
        if enc and enc[-1] not in ids:
            ids.append(enc[-1])
    return ids


def judge(proc, model, frames, prompt, duration_s=10.0):
    import torch
    from transformers.video_utils import VideoMetadata

    metadata = VideoMetadata(
        total_num_frames=len(frames),
        fps=len(frames) / duration_s,
        duration=duration_s,
        frames_indices=list(range(len(frames))),
    )
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames},
        {"type": "text", "text": prompt},
    ]}]
    inputs = proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
        processor_kwargs={"do_sample_frames": False,
                          "video_metadata": metadata},
    ).to("cuda")
    n_in = int(inputs["input_ids"].numel())
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=64, do_sample=False,
            return_dict_in_generate=True, output_scores=True,
        )
    latency = time.time() - t0
    peak_vram_gb = float(torch.cuda.max_memory_allocated() / 1e9)
    gen_ids = out.sequences[0][n_in:]
    text = proc.decode(gen_ids, skip_special_tokens=True)
    n_out = int(gen_ids.numel())

    # P(usable) from first generated token logprobs over yes/no variants
    import torch.nn.functional as F
    logp = F.log_softmax(out.scores[0][0].float(), dim=-1)
    tok = proc.tokenizer
    y_ids = variant_ids(tok, YES_TOKENS)
    n_ids = variant_ids(tok, NO_TOKENS)
    import math
    num = torch.logsumexp(logp[y_ids], dim=0).item()
    den = torch.logsumexp(logp[y_ids + n_ids], dim=0).item()
    p_usable = float(math.exp(num - den))

    parsed = parse_answer(text)
    return text, parsed, p_usable, latency, peak_vram_gb, n_in, n_out


def parse_answer(text):
    """Extract hands_on_work / footage_usable yes-no from free text/JSON."""
    import re
    low = text.lower()
    out = {"hands_on_work": None, "footage_usable": None}
    try:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            obj = json.loads(m.group(0))
            for k in out:
                v = str(obj.get(k, "")).lower()
                if "yes" in v:
                    out[k] = "yes"
                elif "no" in v:
                    out[k] = "no"
    except Exception:
        pass
    if out["footage_usable"] is None:
        # fallback: usable<->yes association near keywords
        mu = re.search(r"footage_usable[\"']?\s*:\s*[\"']?(yes|no)", low)
        if mu:
            out["footage_usable"] = mu.group(1)
    if out["hands_on_work"] is None:
        mh = re.search(r"hands_on_work[\"']?\s*:\s*[\"']?(yes|no)", low)
        if mh:
            out["hands_on_work"] = mh.group(1)
    return out


def git_sha(workdir):
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=workdir
                              ).stdout.strip()
    except Exception:
        return "unknown"


def cmd_run(args):
    manifest = load_manifest(args.manifest)
    with open(args.prompt_file) as f:
        prompt = f.read().strip()
    with open(os.path.join(_HERE, "prompts", "config.json")) as f:
        cfg = json.load(f)

    if args.mode == "frames8":
        n_frames = 8
    elif args.mode == "video_fps1":
        n_frames = 10
    elif args.mode == "video_fps2":
        n_frames = 20
    elif args.mode == "video_max":
        n_frames = MAX_FRAMES
    else:
        raise SystemExit(f"unknown mode {args.mode}")

    items = manifest["items"]
    if args.ids:
        want = set(args.ids.split(","))
        items = [it for it in items if it["id"] in want]
    if args.limit:
        items = items[:args.limit]

    import torch
    proc, model = load_model()
    out_items = {}
    if args.resume_from and os.path.exists(args.resume_from):
        with open(args.resume_from) as f:
            out_items = json.load(f).get("items", {})
        print(f"resumed {len(out_items)} items", flush=True)

    for rep in range(args.repeat):
        for it in items:
            key = it["id"] if args.repeat == 1 else f"{it['id']}#r{rep}"
            if key in out_items:
                continue
            t_dec0 = time.time()
            try:
                frames = sample_frames(it["path"], it.get("start_s", 0.0),
                                       it.get("end_s"), n_frames)
                dur = ((it.get("end_s") or 10.0) - (it.get("start_s") or 0.0))
                text, parsed, p, lat, vram, n_in, n_out = judge(
                    proc, model, frames, prompt, duration_s=dur)
                err = None
            except Exception as e:  # record failures, don't crash the run
                text, parsed, p, lat, vram, n_in, n_out = "", {"hands_on_work": None, "footage_usable": None}, float("nan"), float("nan"), float("nan"), 0, 0
                err = f"{type(e).__name__}: {str(e)[:200]}"
            decode_s = time.time() - t_dec0 - (lat if lat == lat else 0)
            score = {"yes": 1.0, "no": 0.0}.get((parsed.get("footage_usable") or ""), float("nan"))
            out_items[key] = {
                "score": p,  # primary: logprob P(usable)
                "features": {
                    "mode": args.mode, "n_frames": n_frames,
                    "raw_text": text[:500], "parsed": parsed,
                    "parsed_score": score, "p_usable_logprob": p,
                    "input_tokens": n_in, "output_tokens": n_out,
                    "peak_vram_gb": vram, "decode_s": decode_s,
                    "error": err,
                },
                "latency_s": lat,
            }
            print(f"[{key}] p={p:.3f} parsed={parsed} lat={lat:.1f}s vram={vram:.2f}GB {err or ''}", flush=True)
            if args.save_every and len(out_items) % args.save_every == 0:
                write_out(args, out_items, cfg, partial=True)

    write_out(args, out_items, cfg)
    # free VRAM report
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)


def write_out(args, out_items, cfg, partial=False):
    doc = {"arm": ARM_NAME, "version": git_sha(os.path.dirname(_HERE) + "/.."),
           "mode": args.mode, "model": MODEL_ID,
           "prompt_file": os.path.basename(args.prompt_file),
           "sampling": cfg.get("sampling", ""),
           "items": out_items}
    tmp = args.out + (".partial" if partial else "")
    with open(tmp, "w") as f:
        json.dump(doc, f)
    if not partial:
        print(f"wrote {args.out} ({len(out_items)} items)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mode", default="frames8",
                    choices=["frames8", "video_fps1", "video_fps2", "video_max"])
    ap.add_argument("--prompt-file", default=os.path.join(_HERE, "prompts", "v1_main.txt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--resume-from", default="")
    args = ap.parse_args()
    cmd_run(args)


if __name__ == "__main__":
    main()
