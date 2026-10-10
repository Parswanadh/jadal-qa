# Cost model — hours of video processed per wall-hour on the pod A40 (E7.01)

**Registry:** `E7.01` · **Results:** `eval/results/exp_cost/E7.01.json` (model),
`eval/results/exp_cost/E7.01_measure.json` (raw timings) ·
**Code:** `eval/exp_cost/run_e7.py` · **Tests:** `tests/test_exp_cost.py` (7 tests, <2 s, no network) ·
**Device:** **A40** for every number below (no cross-device claim anywhere in this page).

Regenerate:

```
bash tools/pod/gpu_run.sh python3 eval/exp_cost/run_e7.py --stage measure
python3 eval/exp_cost/run_e7.py --stage model
```

This is a **cost model**, not an accuracy result: no labels were used, and no
accuracy claim is made here. Accuracy per policy lives in E1.01/E1.02/E1.04
(fps, frame selection) and E3.01/E3.03/E3.05 (sparse probe, early exit).

## 1. What is measured and what is combined

Measured **on this pod** in this run (timed, bootstrap CI95 over repetitions;
`E7.01_measure.json`):

| input | measurement |
|---|---|
| decode | ffmpeg **CPU** vs **NVDEC** (`h264_cuvid`/`hevc_cuvid` + `scale_cuda`→160×90) on the two source families; 3 windows × {4 s, 10 s} |
| pixel26 features | `jadal.gpu.features.cpu_window_features` (CPU, `OMP_NUM_THREADS=1`), 12 real IndEgo windows, fps {10,5,2} × observed span {10 s, 6 s, 4 s} |
| EmbeddingGemma-2 | batch **1**, bf16, `max_soft_tokens=140`, on **real mosaic PNGs** (N=9, tile 224, 680×680) and spec-geometry canvases; plus a batch-8 alternative |
| mosaic build | the real builder (`jadal/mosaic/build.py`, mosaic branch sha `e8942b3`) on 6 real windows: layout+timestamps, then PNG encode |
| logistic head | 0.14 ms/predict |

Combined arithmetically per `profiles/*.json`. Source JSONs for cross-checks and
the sha of each file are recorded in `source_inputs` (E1.01, E1.02, E1.04 from
the fps branch `2cad6dd`; E2.01 from the mosaic branch `e8942b3`; E3.04 found
only in the pod bundle's `local_work` copy — its wave-1 registry line points at a
laptop path).

**Assumptions** (all listed in the results JSON):

1. serial single-stream pipeline — one window at a time; **no batching or
   pipelining across windows** (not measured);
2. decode cost ∝ native frames inside the **observed** span; a sampler cannot
   skip frames during decode (E1.04 measured selection *after* decode);
3. one processed window covers `probe_every_k_windows × 10 s` of footage (E3.01);
4. early exit = stop observing at `early_exit_s`; features/embeddings see only
   frames inside that span;
5. eg2 = per-frame EmbeddingGemma-2 image embeddings at **batch 1** (the receipt
   path hashes batch-1 embeddings); a batch-8 feature-path alternative is
   measured and reported separately;
6. mosaic evidence = one N=9 / tile-224 / 680×680 PNG per processed window,
   embedded batch 1 (E2.01 geometry);
7. `hours/wall-hour = 3600 / ms_per_window × k × 10 s / 3600`.

## 2. Decode path: NVDEC helps 1080p HEVC only

Wall time is fitted as `startup + ms_per_frame × native_frames` from the 4 s and
10 s runs; the slope is the steady-state rate (a stream kept open pays the
startup once, not per window).

| source family | CPU (marginal) | NVDEC (marginal) | startup per invocation |
|---|---|---|---|
| IndEgo 480p/10fps H.264 | **2.75 ms/frame (≈364 fps)** | fit unusable (slope ≤ 0); end-to-end 18.3 ms/frame (≈55 fps) | 138 ms (CPU) / ~1.9 s (NVDEC) |
| Egocentric-10K 1080p/30fps HEVC | 27.3 ms/frame (≈37 fps) | **7.37 ms/frame (≈136 fps)** | 1.34 s (CPU) / 1.12 s (NVDEC) |

So on this pod: **NVDEC is ~3.7× faster than CPU for the 1080p HEVC factory
footage and clearly slower for the small 480p IndEgo clips** (session/init
overhead dominates). The pod's container sees 96 CPUs but is cgroup-limited to
~4 cores; these runs shared the machine with other pod workers, so they are
**contended** numbers — the CI95 covers window-to-window variation, not machine
load.

## 3. Per-profile table (A40)

ms/window breakdown (decode + pixel26 features + eg2 embeddings + mosaic
build/encode/embed + head) and the resulting throughput. `k` = probe every k-th
window; mosaic on for every profile (`evidence.mosaic=true`).

| profile | fps | k | exit | features | decode path | decode | pixel26 | eg2 | mosaic | total | **video-h / wall-h** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `default` | 10 | 1 | – | both | CPU | 275 | 402 | 18 331 | 419 | 19 427 | **0.51** |
| `default` | 10 | 1 | – | both | NVDEC | 1 831 | 402 | 18 331 | 419 | 20 983 | 0.48 |
| `clear_stationary` | 10 | 1 | 6 s | both | CPU | 165 | 224 | 10 999 | 419 | 11 806 | **0.85** |
| `clear_stationary` | 10 | 1 | 6 s | both | NVDEC | 1 099 | 224 | 10 999 | 419 | 12 740 | 0.78 |
| `blurry_lowlight` | 5 | 1 | – | eg2 | CPU | 275 | – | 9 166 | 419 | 9 859 | **1.01** |
| `blurry_lowlight` | 5 | 1 | – | eg2 | NVDEC | 1 831 | – | 9 166 | 419 | 11 415 | 0.88 |
| `factory_1080p30` | 5 | 2 | – | both | CPU | 8 200 | 165 | 9 166 | 419 | 17 950 | **1.11** |
| `factory_1080p30` | 5 | 2 | – | both | NVDEC | 2 213 | 165 | 9 166 | 419 | 11 962 | **1.67** |
| `walking_logistics` | 10 | 1 | 4 s | pixel26 | CPU | 110 | 128 | – | 419 | 657 | **15.22** |
| `walking_logistics` | 10 | 1 | 4 s | pixel26 | NVDEC | 732 | 128 | – | 419 | 1 280 | 7.82 |

**Where the pipeline is bound.** With `features.set` containing `eg2`, the
pipeline is **embedding-bound**: batch-1 EmbeddingGemma-2 image embeddings cost
**183 ms/image ≈ 5.5 images/s** on this pod, i.e. 9–18 s per window, 47–94 % of
the total. Decode is never the bottleneck for IndEgo (0.6–2 % of total) and only
the `factory_1080p30` **CPU** row is decode-dominated (46 %); with NVDEC that
row drops to 18 %. The pixel26-only profile (`walking_logistics`) is neither:
its mosaic build+embed (419 ms) is the largest single item.

**If the eg2 features are batched** (batch 8, feature path only — the hashed
receipt stays batch 1): 79 ms/image, so `default` 1.11, `clear_stationary` 1.80,
`blurry_lowlight` 2.15, `factory_1080p30` (NVDEC) 2.96 video-h/wall-h. Measured,
but **not** what the current pipeline does.

## 4. The combination: fps × every-2nd × 4 s exit × mosaic

Like-for-like reference rows (same pixel26 feature set, mosaic included, IndEgo
480p source, CPU decode):

| policy | ms/window | video-h / wall-h | vs full sampling |
|---|---|---|---|
| full sampling, pixel26 | 1 096 | 9.13 | 1.0× |
| every-2nd + 4 s exit, pixel26 | 565 | **35.39** | **3.88×** |
| every-2nd + 4 s exit, pixel26, NVDEC decode | 1 188 | 16.84 | 1.84× (NVDEC loses on 480p) |
| every-2nd + 4 s exit, **mosaic only** (one 680×680 image per window, no per-frame embeds) | 529 | **37.81** | 4.14× |

The 4 s exit removes 60 % of the frames (100→40 at 10 fps); the every-2nd probe
doubles the footage each processed window covers; the mosaic replaces N
per-frame image embeddings with **one** image (107 ms) plus build cost
(50 ms layout + 262 ms PNG encode).

**E3.04 comparison (different metric).** E3.04's arithmetic gives ×4.86 for
every-2nd + 4 s on the 4.404 h corpus by counting *processed footage fractions*;
E7.01 counts *wall-clock throughput* of the measured pipeline and gets ×3.88 for
the same policy on the pixel26 path. The two are not expected to match and
neither replaces the other — E3.04 ignores per-window pipeline cost, E7.01
ignores that a real deployment can pipeline several streams.

## 5. Laptop reference (separate, NOT cross-device)

Quoted only as a labelled reference; **no claim that the A40 and the laptop are
interchangeable or bit-identical**:

* `eval/results/bench_eg2.json` — **RTX 4070 Laptop**, EmbeddingGemma-2 image
  embedding, batch 1, bf16, `max_soft_tokens=140`: 62.3 ms/frame (512×288),
  1 476 MB peak VRAM. A different device *and* a different image size than the
  183 ms/image (224×224) measured here.
* `eval/results/edge_bench_x86-laptop-proxy.json` — **x86-laptop-proxy**
  (Intel Core Ultra 7 155H, 1 core, *proxy* hand/face backends, placeholder
  thresholds, different pipeline): 3 489 segments/hour ≈ 9.7 video-h/wall-h.
  Throughput benchmark only; it carries no quality claim and is not this
  pipeline.

## 6. Not measured (stated, not guessed)

* end-to-end wall clock of the **whole chain in one process** (each stage was
  timed separately; the model adds them serially);
* pipelining/batching across windows and multiple concurrent streams on the A40
  (VRAM headroom exists: 1.5 GB peak for the embedder, 2.7 GB reported by E2.01);
* throughput for the **out-of-domain** 1080p path with a working scorer (the
  frozen IndEgo models did not transfer, 0.4–0.53 AUROC — E1.03/E5);
* mosaic build time excluding the frames' decode (decode is charged separately);
* accuracy of the mosaic-only row (E2.01 covers accuracy; this page is cost only).

## 7. Decision

* **Adopt** the cost table as the pod's single reference for hours/wall-hour,
  with the two decode paths kept separate (NVDEC for 1080p HEVC, CPU for 480p
  H.264) and the "embedding-bound at batch 1" caveat attached to every eg2 row.
* **Report** 0.5–15 video-h/wall-h per profile on the A40, 9.1 → 35.4 (×3.9) for
  every-2nd + 4 s exit at the same feature set, and 37.8 for the mosaic-only
  variant.
* **Do not** quote a single "speed-up" number without the profile and the
  feature set: the spread between the best and worst profile row is 30× and comes
  mostly from the feature set (eg2 per-frame embeds vs pixel26), not from
  sampling.
* Next: measure a pipelined multi-stream run and a batch-8 eg2 feature path
  end-to-end, and re-time on an idle pod (these numbers are contended).
