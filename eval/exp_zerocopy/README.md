# eval/exp_zerocopy — Z1 mosaic delta + zero-copy GPU path + decode verdict

Three measurements on one deterministic **30-window sample spread over 30
factories** of `queues/manifest_scale.json` (1080p30 HEVC, label source none):

| id | what | code |
|---|---|---|
| Z1.01 | mosaic stage ON vs OFF — wall delta + verdict identity | `run_zerocopy.py --exp mosaic` |
| Z1.02 | zero-copy GPU path attempt vs the hwdownload path | `run_zerocopy.py --exp zerocopy` |
| Z1.03 | NVDEC vs CPU decode @1080p30 HEVC and @480p H.264, with/without hwdownload | `run_zerocopy.py --exp decode` |

`run_zerocopy.py` **imports** the repo pipeline (`eval/exp_rescore/run_e8.py`,
`jadal/gpu/features.py`, `jadal/ensemble/scorer.py`, `jadal/regimes/router.py`,
`jadal/mosaic/build.py`) and only inserts timing / swaps the GPU path in.

## The zero-copy question

The repo's `decode` stage is an **out-of-process ffmpeg** whose only output is a
host-memory pipe (`-f rawvideo ... pipe:1`). ffmpeg has no muxer that writes CUDA
device memory, so frames cannot stay VRAM-resident across that boundary. A
genuinely VRAM-resident path needs an **in-process NVDEC binding**.

`zerocopy.py` therefore tries `torchcodec` (`VideoDecoder(..., device="cuda")`,
in-process NVDEC → VRAM). If it is unavailable/unsupported the runner falls back
to a **partial** path (ffmpeg hwdownload → host → one small upload of the
pix+eg2 frames) and records the blocker; it never fakes a zero-copy number.

On the GPU path the two compute stages are genuinely device-resident:

* `pixel_features` — `jadal.gpu.features.gpu_window_features_batch` on CUDA
  tensors (GPU resize via `F.interpolate`, no CPU pixel pass);
* `eg2_embed` — the frozen EmbeddingGemma-2 image processor (`Gemma4ImageProcessor._preprocess`,
  all torch ops) and the model forward run **on the GPU tensors** with mean
  pooling + L2 normalise, i.e. no `PIL` / no `model.encode` re-upload.

The `descriptors` stage (E4.01) uses cv2/numpy and therefore must download its 8
native frames to the CPU — that round trip is kept and reported.

## Run

```bash
source /workspace/env.sh
export OMP_NUM_THREADS=2 JADAL_GPU_SLOTS=4
bash tools/pod/gpu_run.sh python3 eval/exp_zerocopy/run_zerocopy.py --sample-n 30
python3 eval/exp_zerocopy/analyze_zerocopy.py
```

Outputs: `eval/results/exp_zerocopy/{sample,mosaic,zerocopy,decode_verdict,Z1.01,Z1.02,Z1.03}.json`,
`docs/reports/exp_zerocopy.md`, `docs/experiments/exp_zerocopy/images/`.
