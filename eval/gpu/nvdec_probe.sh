#!/bin/bash
# NVDEC filter-chain probe: find a working hwaccel decode->scale->pipe path.
# Usage: flock -w 3600 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock bash eval/gpu/nvdec_probe.sh
CLIP="/home/parshu/projects/awing/data/indego/1_Assembly/mp4_480/User_15_410_0411_2_480.mp4"
T=3; W=160; H=90; EXPECT=$((T * 10 * W * H))
probe() {
  local name="$1"; shift
  local out
  out=$(timeout 30 ffmpeg -hide_banner -v error "$@" -f rawvideo -pix_fmt gray pipe:1 2>/tmp/nvdec_err.txt | wc -c)
  local rc=$?
  echo "$name rc=$rc bytes=$out (expect ~$EXPECT) err: $(tail -c 200 /tmp/nvdec_err.txt | tr '\n' '|')"
}
probe V1_hwaccel_cuda_ss_before -hwaccel cuda -hwaccel_output_format cuda -ss 60 -t $T -i "$CLIP" -vf "scale_cuda=${W}:${H},hwdownload,format=gray"
probe V2_cuvid_fmt_nv12 -hwaccel cuda -hwaccel_output_format cuda -c:v h264_cuvid -ss 60 -t $T -i "$CLIP" -vf "scale_cuda=${W}:${H}:format=nv12,hwdownload,format=gray"
probe V3_ss_after -hwaccel cuda -hwaccel_output_format cuda -c:v h264_cuvid -i "$CLIP" -ss 60 -t $T -vf "scale_cuda=${W}:${H}:format=nv12,hwdownload,format=gray"
probe V4_hwaccel_nvdec -hwaccel nvdec -hwaccel_output_format cuda -i "$CLIP" -ss 60 -t $T -vf "scale_cuda=${W}:${H}:format=nv12,hwdownload,format=gray"
probe V5_cpu_ref -ss 60 -t $T -i "$CLIP" -vf "scale=${W}:${H},format=gray"
