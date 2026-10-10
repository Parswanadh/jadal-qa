"""Capacity facts on REAL Egocentric-10K files (1080p H.265): decode RTFs.

Picks up to 3 mp4s from different factories under clips/, measures per file:
  ffprobe stats; full-decode single-core CPU RTF; scaled 854x480 CPU RTF;
  scaled 160x90 CPU RTF; NVDEC full-decode RTF (GPU lock held ONCE for all
  NVDEC runs; timed region excludes lock wait); disk read rate; transcode
  one file to 480p H.264 (1 core) cost.
Output: eval/results/scale/capacity_real1080p.json with machine info and
'NOT a Pi' labels. RTF = video_s / wall_s.
"""
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

CLIPDIR = "/home/parshu/projects/awing/data/ego10k_scale/clips"
OUT = "eval/results/scale/capacity_real1080p.json"
LOCK = "/home/parshu/projects/jadal-qa-wt/tools/gpu.lock"
SAMPLE_S = 120.0  # decode a 120 s sample per file (decode cost is uniform; full-file
# decode of 1200 s clips would take ~10x longer with the same RTF)


def run(cmd, timeout=1200):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r, time.time() - t0


def ffprobe(path):
    r, _ = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,codec_name,avg_frame_rate,duration",
                "-show_entries", "format=duration,size,bit_rate", "-of", "json", path])
    return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[-300:]}


def dur_of(probe):
    return float(probe["format"].get("duration") or probe["streams"][0].get("duration"))


def cpu_decode(path, extra=(), label=""):
    cmd = ["taskset", "-c", "0", "ffmpeg", "-hide_banner", "-v", "error",
           "-threads", "1", "-i", path, "-t", str(int(SAMPLE_S)),
           *extra, "-f", "null", "-"]
    r, wall = run(cmd)
    if r.returncode != 0:
        return {"label": label, "error": r.stderr[-300:]}
    d = min(dur_of(ffprobe(path)), SAMPLE_S)
    return {"label": label, "wall_s": round(wall, 2), "video_s": round(d, 2),
            "rtf": round(d / wall, 2), "sample_s": SAMPLE_S}


def main():
    mp4s = []
    seen = set()
    for root, _, files in os.walk(CLIPDIR):
        for fn in sorted(files):
            if fn.endswith(".mp4"):
                fac = os.path.relpath(root, CLIPDIR).split(os.sep)[0]
                if fac not in seen:
                    seen.add(fac)
                    mp4s.append(os.path.join(root, fn))
                if len(mp4s) >= 3:
                    break
        if len(mp4s) >= 3:
            break
    mp4s.sort()
    assert mp4s, "no mp4s on disk yet"
    print(f"benchmarking {len(mp4s)} files", flush=True)

    per_file = {}
    for path in mp4s:
        probe = ffprobe(path)
        d = dur_of(probe)
        per_file[path] = {
            "ffprobe": probe, "video_s": round(d, 2),
            "full_1080p_cpu1": cpu_decode(path, label="full_1080p_cpu1"),
            "scaled_854x480_cpu1": cpu_decode(
                path, extra=("-vf", "scale=854:480"), label="scaled_854x480_cpu1"),
            "scaled_160x90_cpu1": cpu_decode(
                path, extra=("-vf", "scale=160:90"), label="scaled_160x90_cpu1"),
        }
        print(os.path.basename(path),
              {k: v.get("rtf", "ERR") for k, v in per_file[path].items()
               if k.endswith("cpu1")}, flush=True)

    # NVDEC: hold the lock once, time decodes inside (excludes lock wait)
    nv_lines = ["set -e"]
    for path in mp4s:
        nv_lines.append(f"s=$(date +%s.%N); ffmpeg -hide_banner -v error "
                        f"-hwaccel cuda -hwaccel_output_format cuda -c:v hevc_cuvid "
                        f"-i '{path}' -t {int(SAMPLE_S)} -f null -; e=$(date +%s.%N); "
                        f"echo \"$(echo \"$e - $s\" | bc) {path}\"")
    r = subprocess.run(["flock", "-w", "3600", LOCK, "bash", "-c", "\n".join(nv_lines)],
                       capture_output=True, text=True, timeout=1800)
    for line in r.stdout.strip().splitlines():
        wall_s, path = line.split(" ", 1)
        wall_s = float(wall_s)
        d = min(per_file[path]["video_s"], SAMPLE_S)
        per_file[path]["full_1080p_nvdec"] = {
            "label": "full_1080p_nvdec", "wall_s": round(wall_s, 2),
            "video_s": d, "rtf": round(d / wall_s, 2)}
        print(os.path.basename(path), "nvdec rtf",
              per_file[path]["full_1080p_nvdec"]["rtf"], flush=True)

    t0 = time.time()
    total = 0
    for path in mp4s:
        with open(path, "rb") as f:
            while True:
                b = f.read(8 * 1024 * 1024)
                if not b:
                    break
                total += len(b)
    disk_s = time.time() - t0

    src = mp4s[0]
    t480 = "eval/results/scale/_proxy/t480_sample.mp4"
    _, w1 = run(["taskset", "-c", "0", "ffmpeg", "-hide_banner", "-v", "error", "-y",
                 "-threads", "1", "-i", src, "-t", str(int(SAMPLE_S)),
                 "-vf", "scale=854:480",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-an", t480])
    d0 = min(per_file[src]["video_s"], SAMPLE_S)

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True)
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device_class": "laptop workstation (x86-64, RTX 4070) — NOT a Pi; "
                        "Pi-class numbers are NOT claimed here",
        "machine": {
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "mem": open("/proc/meminfo").readline().strip(),
            "cpu_model": [l for l in open("/proc/cpuinfo").read().splitlines()
                          if l.startswith("model name")][:1],
            "gpu": gpu.stdout.strip()},
        "codec_note": "real Egocentric-10K files per ffprobe below (publisher declares H.265/1080p30)",
        "per_file": per_file,
        "disk_read": {"MB": round(total / 1e6, 1), "wall_s": round(disk_s, 2),
                      "MB_s": round(total / 1e6 / disk_s, 1)},
        "transcode_to_480p_h264_1core": {
            "wall_s": round(w1, 1), "video_s": d0, "rtf": round(d0 / w1, 2)},
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
