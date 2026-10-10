"""Local capacity facts at Egocentric-10K resolution (1080p30 H.265).

Real Ego10K content downloads are blocked (token 403), so this builds three
60 s 1080p30 H.265 PROXY files at the publisher's bitrate (~3.6 Mb/s =
16.4 TB / 10 000 h) — one synthetic (testsrc2) + two from real IndEgo work
footage upscaled to 1080p30 — and measures:
  ffprobe stats; full-decode RTF (single-core CPU); scaled-decode RTF
  (480p-ish 854x480 and cheap-tier 160x90); NVDEC RTF (hevc_cuvid, GPU lock);
  disk read rate; transcode-to-480p cost (1 core and 4 parallel).

Every proxy is labelled "proxy":true. Decode cost depends on resolution,
codec and bitrate, not scene content, so CPU/NVDEC RTF transfers to real
1080p H.265; byte sizes/durations of the real corpus do NOT.

Output: eval/results/scale/capacity_1080p.json
"""
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

SCRATCH = os.path.join("eval", "results", "scale", "_proxy")
OUT = "eval/results/scale/capacity_1080p.json"
IND_A = "/home/parshu/projects/awing/data/indego/1_Assembly/mp4_480/User_15_410_0411_2_480.mp4"
IND_C = "/home/parshu/projects/awing/data/indego/3_Logistics_Organization/mp4_480/User_1_H_pal_transport_Nar_480.mp4"
BITRATE = "3.5M"  # matches 16.4 TB / 10 000 h
DUR = 60


def run(cmd, timeout=900, **kw):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    return r, time.time() - t0


def ffprobe(path):
    r, _ = run(["ffprobe", "-v", "error", "-show_entries",
                "stream=width,height,codec_name,avg_frame_rate,r_frame_rate,duration,bit_rate",
                "-show_entries", "format=duration,size,bit_rate", "-of", "json", path])
    return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[-300:]}


def decode_rtf(path, extra=(), threads=1, hwaccel=None, label=""):
    cmd = ["taskset", "-c", "0", "ffmpeg", "-hide_banner", "-v", "error"]
    if hwaccel:
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", "hevc_cuvid"]
    cmd += ["-threads", str(threads), "-i", path, *extra, "-f", "null", "-"]
    if hwaccel:  # don't pin GPU decode to one CPU core; drop taskset
        cmd = cmd[3:]
        cmd = ["flock", "-w", "3600", "/home/parshu/projects/jadal-qa-wt/tools/gpu.lock",
               *cmd]
    r, wall = run(cmd, timeout=1200)
    if r.returncode != 0:
        return {"label": label, "error": r.stderr[-300:]}
    dur = float(ffprobe(path)["format"]["duration"])
    return {"label": label, "wall_s": round(wall, 2),
            "video_s": dur, "rtf": round(dur / wall, 2)}


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    proxies = {
        "proxy_synth_motion": ["-f", "lavfi", "-i", f"testsrc2=size=1920x1080:rate=30:duration={DUR}"],
        "proxy_indego_assembly": ["-ss", "100", "-t", str(DUR), "-i", IND_A,
                                  "-vf", "scale=1920:1080,fps=30"],
        "proxy_indego_pallet": ["-ss", "10", "-t", str(DUR), "-i", IND_C,
                                "-vf", "scale=1920:1080,fps=30"],
    }
    files = {}
    for name, src in proxies.items():
        out = os.path.join(SCRATCH, name + ".mp4")
        if not os.path.exists(out):
            cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y", *src,
                   "-c:v", "libx265", "-preset", "veryfast", "-b:v", BITRATE,
                   "-pix_fmt", "yuv420p", out]
            r, wall = run(cmd, timeout=1800)
            assert r.returncode == 0, r.stderr[-500:]
            print(f"encoded {name} in {wall:.0f}s", flush=True)
        files[name] = out

    per_file = {}
    for name, path in files.items():
        probe = ffprobe(path)
        full = decode_rtf(path, label="full_1080p_cpu1")
        s480 = decode_rtf(path, extra=("-vf", "scale=854:480"), label="scaled_854x480_cpu1")
        s160 = decode_rtf(path, extra=("-vf", "scale=160:90"), label="scaled_160x90_cpu1")
        nv = decode_rtf(path, hwaccel=True, label="full_1080p_nvdec")
        per_file[name] = {"path": path, "proxy": True, "ffprobe": probe,
                          "full_1080p_cpu1": full, "scaled_854x480_cpu1": s480,
                          "scaled_160x90_cpu1": s160, "full_1080p_nvdec": nv}
        print(name, {k: v.get("rtf", v.get("error", "?")[:60])
                     for k, v in per_file[name].items() if k.endswith(("cpu1", "nvdec"))},
              flush=True)

    # disk read rate: buffered python read of the three proxies
    t0 = time.time()
    total = 0
    for path in files.values():
        with open(path, "rb") as f:
            while f.read(8 * 1024 * 1024):
                total += 8 * 1024 * 1024
    disk_s = time.time() - t0

    # transcode to 480p (funnel format): single core, then 4-way parallel
    src = files["proxy_indego_assembly"]
    t480_1 = os.path.join(SCRATCH, "t480_1core.mp4")
    _, w1 = run(["taskset", "-c", "0", "ffmpeg", "-hide_banner", "-v", "error", "-y",
                 "-threads", "1", "-i", src, "-vf", "scale=854:480",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", t480_1],
                timeout=1200)
    t0 = time.time()
    procs = []
    for i in range(4):
        o = os.path.join(SCRATCH, f"t480_p{i}.mp4")
        procs.append(subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", src,
             "-vf", "scale=854:480", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "23", o]))
    for p in procs:
        p.wait()
    w4 = time.time() - t0

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
            "gpu": gpu.stdout.strip(),
            "ffmpeg": subprocess.run(["ffmpeg", "-version"],
                                     capture_output=True, text=True).stdout.splitlines()[0],
        },
        "proxy_note": ("1080p30 H.265 @3.5 Mb/s proxies (publisher bitrate); "
                       "CPU/NVDEC RTF transfers by res/codec/bitrate; real-corpus "
                       "sizes/durations NOT measured (download blocked)."),
        "per_file": per_file,
        "disk_read": {"MB": round(total / 1e6, 1), "wall_s": round(disk_s, 2),
                      "MB_s": round(total / 1e6 / disk_s, 1)},
        "transcode_to_480p_h264": {
            "single_core_wall_s": round(w1, 1),
            "single_core_rtf": round(DUR / w1, 2),
            "4way_parallel_wall_s": round(w4, 1),
            "4way_parallel_rtf": round(4 * DUR / w4, 2)},
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
