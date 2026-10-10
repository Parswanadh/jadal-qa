# NVDEC vs CPU decode on the pod A40 — correctness parity + throughput

**Claim: LB1-decode** (decode numbers behind `FACTS.md`, which were measured on a
laptop RTX 4070). Question asked here: *do they hold on this pod's A40, and is
NVDEC decode correct?*

**Verdict in one line:** NVDEC decode is **bit-exact** against the CPU decoder on
both codecs (so `jadal/poc_stream/replay.py`'s NVDEC path is safe), NVDEC is
**11-13x faster than one CPU core** for 1080p HEVC on this device, but the
"flat across 1-4 concurrent sessions" claim does **not** reproduce here, and the
*other* in-repo NVDEC path (`eval/gpu/decode.py`, `scale_cuda` + hand-rolled
nv12→rgb) is **not** pixel-equivalent to the CPU decode the ensemble was frozen on.

Artifacts: `eval/results/pod_verify/decode_verification.json` (raw JSON, all
commands + per-case numbers), drivers in `eval/scripts/decode_*.py`.

## Environment and inputs

| item | value |
|---|---|
| GPU | NVIDIA A40, 46068 MiB, driver 580.173.02 (shared pod; slot 2 lock) |
| ffmpeg | 6.1.1-3ubuntu5; `-hwaccels` lists `cuda`; `h264_cuvid`, `hevc_cuvid`, `scale_cuda` all present |
| CPU | 96 vCPU host, `OMP_NUM_THREADS=2` for the driver |
| 480p clip | `indego/1_Assembly/mp4_480/User_15_410_0411_2_480.mp4` — H.264, **480x480**, 10 fps, 6983 frames (698.3 s) |
| 1080p clip | `ego10k_scale/clips/factory_051/worker_005/factory051_worker005_00167.mp4` — HEVC, **1920x1080**, 30 fps, 5393 frames (179.8 s) |

Note the IndEgo clip is 480x480 (square), not 854x480; the FACTS 480p numbers do
not state their resolution, so the two are not ratio-comparable.

## 1. Correctness parity — bit-exact

2 s spans decoded to raw `yuv420p`; `nvdec_cuvid` = `-c:v hevc_cuvid`/`h264_cuvid`,
`nvdec_hwaccel` = `-hwaccel cuda`. Frame-count and per-pixel comparison against
the CPU decode (`-threads 1`, default software decoder):

| span | frames cpu / cuvid / hwaccel | mean abs diff | max abs diff | PSNR | ref. frame-to-frame MAE |
|---|---|---|---|---|---|
| IndEgo 480p, first 2 s | 20 / 20 / 20 | **0.0** | **0** | ∞ | 3.34 |
| IndEgo 480p, 2 s @ 30 s | 20 / 20 / 20 | **0.0** | **0** | ∞ | 2.36 |
| Ego10K 1080p, first 2 s | 60 / 60 / 60 | **0.0** | **0** | ∞ | 6.78 |
| Ego10K 1080p, 2 s @ 30 s | 60 / 60 / 60 | **0.0** | **0** | ∞ | 8.11 |

At the pipeline's working resolution (160x90, `replay.py`'s exact chain
`-hwaccel cuda -c:v hevc_cuvid -ss … -t … -i … -vf scale=160:90,fps=F,format=bgr24`):
frame counts equal and **MAE 0.0000, max 0** on both clips.

**Threshold reasoning.** Pre-registered pass criterion: max per-pixel |Δ| ≤ 2 LSB
and mean |Δ| ≤ 0.5 LSB — conformant H.264/HEVC decoders are allowed ±1-2 LSB of
inverse-transform rounding, so 0 is not required, but anything larger could move a
threshold decision. The observed 0 / 0.0 is a strict pass and makes the threshold
non-load-bearing. Reference scale: the frame-to-frame MAE of the same spans is
2.4-8.1 LSB, so the decoder difference is **0% of the signal the scorer sees**.

## 2. Does the decoder swap change a verdict?

Fed the same 10 s windows through the frozen pixel view
(`configs/ensemble_v1.json`: 160x90 rgb24 → `cpu_window_features` → logistic head)
once from a CPU decode and once from an NVDEC decode — 6 windows (3 per clip):

* `replay.py` chain (cuvid + CPU swscale): **Δp_pixel = 0.000000 on all 6 windows**,
  identical feature vectors, no verdict flips. One window sits at p = 0.4556
  (only 0.044 from the 0.5 boundary) and still does not move.
* `eval/gpu/decode.py` chain (`-hwaccel cuda -hwaccel_output_format cuda` +
  `scale_cuda` + numpy `nv12_to_rgb`): **MAE 4.9 (480p) / 10.4 (1080p) LSB** at
  160x90, and **Δp_pixel up to 0.382** (mean 0.117) on the same 6 windows.

Attribution (same 2 s span, 160x90, channel order normalised):

| contribution | IndEgo 480p | Ego10K 1080p |
|---|---|---|
| decoder alone, resampler matched (cuvid vs CPU) | **0.0000** | **0.0000** |
| resampler, bicubic vs bilinear | 1.93 | 3.40 |
| numpy nv12→rgb vs swscale bgr24 (same scale) | 2.44 | 4.16 |
| `scale_cuda` resize residual (same conversion) | 4.82 | 9.89 |
| end-to-end scale_cuda path vs CPU | 5.04 | 10.19 |

So the difference in that path is **not** the decoder: it is the GPU resizer plus
the hand-rolled nv12→rgb conversion (BT.601 limited range, nearest-neighbour
chroma upsampling). Decode-side conclusion: NVDEC is exact; a *conversion/resize*
change is what perturbs the verdict — and 5-10 LSB is comparable to or larger
than the 2.4-8.1 LSB frame-to-frame signal.

## 3. Throughput (decode-only, `-f null`, `bench:` rtime)

| case | fps | RTF |
|---|---|---|
| 480p H.264, CPU 1 thread | 503 | 50.3 |
| 480p H.264, CPU 1 process (auto threads) | 608 | 60.8 |
| **480p H.264, NVDEC `h264_cuvid`** | **1856** | **185.6** |
| 1080p HEVC, CPU 1 thread | 80.1 | 2.67 |
| 1080p HEVC, CPU 1 process (auto threads) | 81.8 | 2.73 |
| **1080p HEVC, NVDEC `hevc_cuvid`** | **903-1055** | **30.1-35.2** |
| 1080p HEVC, NVDEC `-hwaccel cuda` (auto threads) | 946 | 31.5 |
| 1080p HEVC, NVDEC `-hwaccel cuda -threads 1` | 603 | 20.1 |

CPU software-decoder thread scaling (1080p, 30 s span): 1 → 116, 2 → 187,
4 → **336**, 8 → 315, 16 → 146, auto → 146 fps. It peaks at 4 threads and
degrades after that; `-threads 0` is a poor default on this pod.

**Per-process startup cost** (0.1 s span, wall time): CPU 0.081 s (480p) /
0.126-0.149 s (1080p) vs NVDEC 0.614 s / 0.65-0.97 s. Because `replay.py` spawns
one ffmpeg per window, this dominates short windows — measured end-to-end wall
time for a real `-ss 30 -t T` window at 160x90:

| window | CPU | NVDEC cuvid | NVDEC scale_cuda |
|---|---|---|---|
| 480p, 2 s | **0.125 s** | 0.765 s | 0.640 s |
| 1080p, 2 s | 1.86 s | 1.07 s | **0.91 s** |
| 1080p, 10 s | 4.24 s | 2.26 s | **0.98 s** |

Decode-only NVDEC is 11-13x faster, but for a 2 s 480p window CPU is 5-6x faster
end-to-end, and for a 2 s 1080p window the NVDEC advantage shrinks to 1.7-2.0x.
The 10 s window keeps a 4.3x advantage for `scale_cuda` — but that path carries
the pixel/verdict difference above.

## 4. NVDEC concurrency (the "flat across 1-4 sessions" claim)

Aggregate 1080p HEVC NVDEC throughput, `hevc_cuvid`, wall-clock:

| sessions | run A (clean, co-tenant idle) | run B (co-tenant active) |
|---|---|---|
| 1 | 999-1096 fps | 441 fps |
| 2 | 1458-1675 fps | 711 fps |
| 4 | 2426-2553 fps | 1018 fps |
| 8 | 2504 fps | — |

**Not flat across 1-4 sessions.** Aggregate rises ~2.4x from 1 to 4 sessions and
plateaus only between 4 and 8. A single ffmpeg+NVDEC session reaches ~40-45% of
the achievable aggregate on this pod — one session does not saturate the decoder.
The *shape* is identical in both runs, so the finding is robust to contention;
absolute fps are not (they move ~2.3x depending on the co-tenant job: run B was
transcribed from the log of a run that crashed before writing JSON, so it is
recorded under `contention_replicate` with that provenance). Resource use of 4 concurrent
`hevc_cuvid` sessions (0.5 s sampling, 12 s window): peak **1672 MiB** total VRAM
(~0.42 GiB/session) and `nvidia-smi utilization.gpu` **79-89%** (median 79%) —
a naive monitor reads that as "GPU saturated" although the work is decoder+copy,
not SM compute. (Single-point samples taken before/after a run show 0-11% and
would be misleading; the in-run sampler is the number to use.)

## 5. Comparison with the FACTS.md numbers

| FACTS.md claim | on this A40 pod | verdict |
|---|---|---|
| "NVDEC is NOT faster than CPU decode for a single 480p stream (2,348 vs 2,820 fps)" | NVDEC 1856 vs CPU 503 (1t) / 608 (proc) | **different device: ordering reverses.** But for the pipeline's unit of work (one 2 s 480p ffmpeg process) CPU *is* 5-6x faster end-to-end, so the spirit holds for a different reason: startup, not engine saturation |
| "one CPU core decodes 43-55 fps (RTF 1.4-1.8)" (1080p HEVC) | 80 fps full clip (RTF 2.67), 116 fps on a 30 s span | same direction, **different magnitude** (different CPU) |
| "NVDEC decodes 620-730 fps (RTF 21-24)" | 903-1055 fps (RTF 30-35); 441 fps under co-tenant contention | same order, measured range brackets FACTS; **not comparable** as an exact number (different device) |
| "saturates at one engine (flat across 1-4 concurrent sessions)" | 999 → 1675 → 2553 → 2504 fps (1/2/4/8) | **does not reproduce** on this device |
| "this chip has one NVDEC engine" | nvidia-smi exposes no NVDEC engine count | **could not verify**; observed scaling is inconsistent with one session saturating the engine |

## 6. What could not be verified / caveats

* **NVDEC engine count** — not queryable from the pod (`nvidia-smi` has no
  decoder-engine field); the "one engine" claim is untested here, only its
  behavioural consequence.
* The pods' absolute fps carry contention noise from a co-tenant GPU job (both
  GPU and CPU are shared). Both concurrency runs are reported; all other numbers
  are single runs whose co-tenant state was recorded at measurement time.
* Single-clip samples: one IndEgo clip and one Ego10K clip. The 30 s-span CPU
  number (116 fps) differs from the full-clip number (80 fps) for the same
  settings — content and contention variance, not a methodology difference.
* The `eg2` view of the ensemble was not exercised (it is embedding-model based
  and its `embed_model_dir` does not resolve on the pod); only the pixel view,
  which is the ffmpeg-decode-dependent one, was tested.
* `eval/gpu/decode.py`'s chain could not be judged "wrong" on its own terms — it
  is a different (GPU-resident) pipeline; the finding is that it is **not
  interchangeable** with the CPU decode the ensemble was frozen on.

## 7. Reproduce

```bash
export OMP_NUM_THREADS=2
flock -w 3600 /workspace/state/gpu.slot2.lock bash -c '
  cd <worktree>
  python3 -u eval/scripts/decode_bench_a40.py     --out eval/results/pod_verify/decode_verification.json
  python3 -u eval/scripts/decode_bench_ext.py     --out /tmp/ext.json  --merge-into eval/results/pod_verify/decode_verification.json
  python3 -u eval/scripts/decode_pipeline_parity.py --out /tmp/pp.json --merge-into eval/results/pod_verify/decode_verification.json
  python3 -u eval/scripts/decode_scorer_delta.py  --out /tmp/sd.json  --merge-into eval/results/pod_verify/decode_verification.json
  python3 -u eval/scripts/decode_nv12_attribution.py --out /tmp/at.json --merge-into eval/results/pod_verify/decode_verification.json
  python3 -u eval/scripts/decode_scorer_delta_scalecuda.py --out /tmp/svc.json --merge-into eval/results/pod_verify/decode_verification.json
  python3 eval/scripts/decode_finalize.py eval/results/pod_verify/decode_verification.json'
```
