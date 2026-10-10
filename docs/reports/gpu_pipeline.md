# GPU pipeline PoC — RTX 4070 Laptop, NOT a Pi, NOT a datacenter GPU

Proof of concept for issue #50 (companion to #49): how much video ONE consumer
GPU can analyse with per-pixel whole-video spectral features + early-reject
cascade. Every number below comes from a JSON under `eval/results/gpu/` that
the documented command regenerates. Label source / n / CV protocol stated per
table. Two workers share this branch; `cascade.json`/`_feat_cache.json` are the
parallel worker's (cited read-only), the rest are mine.

Hardware (measured): NVIDIA GeForce RTX 4070 Laptop GPU, 8 GiB
(`nvidia-smi`: 8188 MiB, sm_89), torch 2.14.1+cu130, ffmpeg with h264_cuvid.

## 1. Optimisation ladder (centrepiece)

| rung | change | hours video / wall hour | why |
|---|---|---|---|
| R0 CPU baseline | ffmpeg CPU decode 160×90 + `cpu_window_features`, 1 proc | **22.2** (`throughput.json:R0_cpu`) | starting point |
| R1/R2 GPU kernels | same features as batched torch kernels, fp16 in | **1792–1998** (`throughput.json:kernels_only`, B1/B8/B32) | kernels-only ceiling; batching ≈ flat (already launch-bound, not compute-bound) |
| R3 NVDEC pipeline | NVDEC + `scale_cuda` at source + batched kernels, 4 decoders, sustained ≥5 min | PENDING (`nvdec_probe` found the first filter chain broken; diagnosis in `nvdec_probe.json`) | decode + compute overlap |
| R4 +fp16 +S0 skip | early-reject exits skip S1/S2 work | PENDING | effective rate = R3 / (1 − saved) |

Regenerate: `/home/parshu/envs/jadal/bin/python eval/gpu/throughput.py --r0`
(holdA/holdB GPU rungs run under `flock .../tools/gpu.lock`).

## 2. Limits table (binding resource at each operating point)

| resource | measured ceiling | % of theory | binding when |
|---|---|---|---|
| GDDR6 D2D copy | **74–108 GB/s** (`throughput.json:memcpy`, `microbench.json`) | 29–42% of 256 GB/s theory | never yet: kernels move <0.5 GB/s at 200 win/s |
| PCIe H2D pinned | **7.6–8.6 GB/s** | 47–54% of ~16 GB/s (PCIe4 x8) | never yet: 160×90×100×3 fp16 window = 8.6 MB; even 200 win/s = 1.7 GB/s |
| SM compute (kernels) | **~200 windows/s** (5 ms/window incl. PCA/phase-corr) | — | not binding: R0→kernels is 81× headroom |
| NVDEC | PENDING (chain `scale_cuda,hwdownload` fails rc=234 in this ffmpeg build; `microbench.json` NVDEC rows are marked failed, 0 bytes) | — | expected binding resource at R3; 1 NVDEC engine on AD106 |
| CPU decode (fallback, works) | **989 fps ×1 / 1512 fps ×4** @160×90 gray (`microbench.json`) | — | ×4 CPU already RTF≈150 for decode alone |
| VRAM | peak **4.3 GiB** at B32 random-tensor bench (`throughput.json`) vs 7 GiB budget | 61% | fits; B32 full windows ≈ 0.4 GiB tensors + context |
| Disk | page-cached IndEgo clips; negligible | — | — |

Per-op roofline @160×90×100 fp16 (`microbench.json:kernels`, ms/window B32):
temporal-FFT 0.28, spatial 0.002, PCA-residual 0.63, stats 0.78, motion 0.36,
skin 0.007, DCT 0.003 → **~485 windows/s op sum** (whole `gpu_window_features_batch`
≈200 win/s with slope-fit loop overhead).

What a bigger deployment changes: the ceiling moves to NVDEC engines per GPU
(+1 per card) and CPU feeding (ffmpeg procs); kernels have ~10× slack. No linear
scaling claim past the measured bottleneck.

## 3. Accuracy: per-pixel quality decision (LOCO, clip-bootstrap CIs)

Gold: FINAL 150 labels — 100 usable / 50 bad (TOO_DARK 6, HANDS_NOT_VISIBLE 14,
NO_ACTIVE_WORK 23, DUPLICATE 6, UNDECODABLE 1), 7 clips. Protocol:
leave-one-clip-out everywhere; thresholds inside train clips only.

| head | AUC | 95% CI | source |
|---|---|---|---|
| parallel `cascade.json` S1 logistic | 0.802 | 0.710–0.870 | their file, n=150 |
| independent verify (`stream.json`) | 0.793 | 0.697–0.860 | mine, n=150 |
| ablation full_27 (`ablation.json`) | 0.793 | 0.689–0.871 | mine, n=150 |

Cascade exits on gold (`cascade.json`): S0 2% (3), S1-accept 50% (75),
S1-reject 18% (27), S2-uncertain 30% (45). **False rejects: 8** (all S1, listed
in file — not hidden). Synth exits: darken/black→S0 6/6; overexpose S0 2/6;
rest decided at S1. `compute_saved_frac` in their file is −0.28 (their
accounting counts S0+S1 overhead against a zero-cost S2 stub — see §5).

Streaming (`stream.json`, 2-s chunks, my head): time-to-first-reject —
TOO_DARK/black 2 s (1 chunk), HANDS_NOT_VISIBLE 3 s, NO_ACTIVE_WORK 4 s,
DUPLICATE 3 s. Max per-chunk CPU extraction latency < 2 s (in file) →
real-time capable on 1 core for the streaming decision.

## 4. M6 bounded check: inverse-PCA + illumination rows (`ablation.json`)

| group | AUC | CI |
|---|---|---|
| full_27 | 0.793 | 0.689–0.871 |
| no_pca_resid | **0.794** | 0.677–0.873 |
| pca_only | 0.198 | 0.142–0.349 |
| illum_only | 0.231 | 0.137–0.376 |
| rppg_only | 0.288 | 0.237–0.456 |
| skin_only | 0.384 | 0.295–0.656 |
| temporal_only | 0.714 | 0.619–0.764 |
| no_illum_no_rppg | 0.804 | 0.690–0.884 |

Verdict: the inverse-PCA residual carries **no measurable signal** (removing it
changes AUC 0.793→0.794); illumination/rPPG/skin alone are at/below chance with
wide CIs. Consistent with `rgb_to_spectral.md`: no downstream gain from
reconstruction-flavoured features; metamerism is a hard limit. **No
hyperspectral claim.** Grey-world kept only as cheap normalisation. Temporal /
motion energy carries the bulk of the signal (0.714 alone).

## 5. Cost model + extrapolation

- R0 CPU single-proc: RTF 22 → 10,000 h needs 5000/22 ≈ **227 cores** (2 h) or
  3333/22 ≈ **152 cores** (3 h). Edge 50×/device: needs RTF 50 — single core
  insufficient (~3 cores); live 1× fine.
- GPU: kernels-only RTF ≈ 1800–2000 (decode-excluded). End-to-end RTF pending
  R3 sustained run; extrapolation `GPUs = 5000/RTF` lands in
  `throughput.json:extrapolation` when R3 is measured.
- Tiering: only 30% of windows reach S2; S0 kills dark/blocked/frozen after 2 s
  of tiny-res decode. The −0.28 "saved" figure in `cascade.json` is an
  accounting artefact (S2 stub cost = 0); true amortised saving = fraction of
  S2 work avoided × S2 cost — reported once EG2/MediaPipe per-window cost lands.

## 6. Determinism (`determinism.json`)

CPU: 10/10 repeats of g001 bit-identical quantised digest
(`af2c1375…`). GPU 10× pending (next lock hold). GPU numerics are NOT claimed
identical across devices or vs CPU float64; the Pi path keeps the CPU
reference.

## Pending

1. Working NVDEC chain + re-measured decode matrix (fix `microbench.json` NVDEC rows).
2. R3 sustained ≥5 min + batch sweep + dmon + extrapolation.
3. GPU 10× determinism.
4. Accuracy-vs-speed Pareto per rung (an optimisation that lowers accuracy is reported as such).
