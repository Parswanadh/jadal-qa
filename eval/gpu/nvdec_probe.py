"""NVDEC filter-chain diagnosis: try candidate chains, record bytes/rc/stderr.

Run under lock (fast, ~2 min). Writes eval/results/gpu/nvdec_probe.json
"""
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "gpu", "nvdec_probe.json")
CLIP = "/workspace/jadal_pod_bundle/data/indego/1_Assembly/mp4_480/User_15_410_0411_2_480.mp4"

CHAINS = {
    "A_scale_cuda_hwdownload_gray": ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-ss", "60", "-t", "5", "-i", CLIP,
        "-vf", "scale_cuda=160:90,hwdownload,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
    "B_scale_cuda_nv12": ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-ss", "60", "-t", "5", "-i", CLIP,
        "-vf", "scale_cuda=160:90:format=nv12,hwdownload,format=nv12",
        "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"],
    "C_cuvid_hwdownload_cpu_scale": ["-c:v", "h264_cuvid",
        "-ss", "60", "-t", "5", "-i", CLIP,
        "-vf", "hwdownload,scale=160:90,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
    "D_hwaccel_only": ["-hwaccel", "cuda",
        "-ss", "60", "-t", "5", "-i", CLIP,
        "-vf", "scale=160:90,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
    "E_hwdownload_fullres": ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-ss", "60", "-t", "5", "-i", CLIP,
        "-vf", "hwdownload,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
    "F_scale_cuda_rgb24": ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-ss", "60", "-t", "5", "-i", CLIP,
        "-vf", "scale_cuda=160:90:format=rgb24,hwdownload",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
}


def main():
    rows = {}
    for name, args in CHAINS.items():
        cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin"] + args
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               stdin=subprocess.DEVNULL, timeout=60)
            rows[name] = {"rc": p.returncode, "bytes": len(p.stdout),
                          "stderr": p.stderr[-400:].decode("utf-8", "replace")}
        except subprocess.TimeoutExpired:
            rows[name] = {"rc": "timeout", "bytes": 0, "stderr": ""}
        print(name, rows[name]["rc"], rows[name]["bytes"],
              rows[name]["stderr"][-120:].replace("\n", " | "))
    json.dump({"device": "RTX 4070 Laptop", "chains": rows,
               "regenerate": "flock ... python3 eval/gpu/nvdec_probe.py"},
              open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
