"""Build eval/results/scale/label_queue_random.json + 480p window transcodes.

From eval/scale/manifest_scale.json: seeded (seed 60) stratified draw of 60
10-s windows across factories (round-robin over per-factory shuffled lists):
50 stratum 'random', 10 stratum 'uncertainty_placeholder' (random fill now;
replaced by model-uncertainty picks later — same schema, human labels the
final 60 either way).
Each window is transcoded to 480p H.264 10 s mp4 under
/home/parshu/projects/awing/data/ego10k_scale/label_windows/ (fast input
seek to start-5 then accurate -ss 5, so cuts are frame-accurate without
decoding from 0). Queue items use the segments.json schema:
{segment_id, clip, path, start_s, end_s} with path = the 480p transcode.
The human's labelling tool can load this file directly. No labelling here.
"""
import json
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor

MAN = "eval/scale/manifest_scale.json"
OUT = "eval/results/scale/label_queue_random.json"
WDIR = "/home/parshu/projects/awing/data/ego10k_scale/label_windows"
SEED = 60
N = 60
N_UNCERTAINTY_PLACEHOLDER = 10


def transcode(src, start_s, seg_id):
    os.makedirs(WDIR, exist_ok=True)
    out = os.path.join(WDIR, f"{seg_id}.mp4")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    pre = max(0.0, start_s - 5.0)
    skip = start_s - pre
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-ss", str(pre),
         "-i", src, "-ss", str(skip), "-t", "10",
         "-vf", "scale=854:480", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "23", "-an", out],
        capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-300:])
    return out


def main():
    man = json.load(open(MAN))
    byfac = {}
    for it in man["items"]:
        byfac.setdefault(it["factory"], []).append(it)
    rng = random.Random(SEED)
    for fac in byfac:
        rng.shuffle(byfac[fac])
    facs = sorted(byfac)
    order = []
    i = 0
    while len(order) < N:
        progressed = False
        for fac in facs:
            if i < len(byfac[fac]) and len(order) < N:
                order.append(byfac[fac][i])
                progressed = True
        i += 1
        if not progressed:
            break
    assert len(order) == N, f"only {len(order)} windows available, need {N}"
    seg_ids = [f"q{k:03d}_{w['id']}" for k, w in enumerate(order)]
    with ThreadPoolExecutor(max_workers=4) as ex:  # <=4 cores per house rules
        mp4s = list(ex.map(lambda a: transcode(*a),
                           [(w["path"], w["start_s"], s) for w, s in zip(order, seg_ids)]))
    items = []
    for k, (w, seg_id, mp4) in enumerate(zip(order, seg_ids, mp4s)):
        items.append({
            "segment_id": seg_id, "clip": w["clip"], "path": mp4,
            "start_s": 0.0, "end_s": 10.0,
            "source_path": w["path"], "source_start_s": w["start_s"],
            "factory": w["factory"], "worker": w["worker"],
            "stratum": ("uncertainty_placeholder" if k >= N - N_UNCERTAINTY_PLACEHOLDER
                        else "random"),
        })
    json.dump({"seed": SEED, "n": len(items),
               "note": "human labels these for the out-of-domain test; "
                       "uncertainty_placeholder rows get replaced by model-uncertainty picks",
               "segments": items}, open(OUT, "w"), indent=1)
    print(f"{len(items)} windows, {len(facs)} factories -> {OUT}")


if __name__ == "__main__":
    main()
