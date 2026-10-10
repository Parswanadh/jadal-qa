# exp_shard — S1.01: sharded isolated workers, parallel scaling ceiling, multi-pod plan

**Status: MEASURED.** Branch `m/shard`. Device: **1× NVIDIA A40** (single NVDEC engine,
46 GB), cgroup CPU quota **7.65 cores** (`cpu.max = 765000/100000`, `nproc=96`). The full
corpus is **58 clips / 1,546 windows / 4.404 h** of 1080p/30 fps HEVC Egocentric-10K
(unlabelled). Scaling subset: the **six smallest clips / 102 windows / 0.283 h** (fixed,
deterministic).

## Headline

> The pipeline is now **shardable into K isolated workers** (own process, own JSONL, own
> state dir, no shared mutable state) with a **deterministic, partition-independent combined
> digest** that is byte-identical for K = 1, 2, 4, 7 (**shard-invariance proven**). On this
> pod the scaling ceiling is **≈1.1 windows/s ≈ 11 footage-h/wall-hour (K ≥ 4)** on the
> 6-clip subset and **≈19.8 footage-h/wall-hour** at full-corpus shard sizes; the ceiling is
> set by **contention on the single GPU decode/copy path**, *not* by the CPU quota, the VRAM
> budget, or average NVDEC utilisation — all three are left on the table.

| K (isolated shards) | windows/s | footage-h/wall-h | CPU cores | NVDEC dec% (mean/max) | GPU sm% (mean/max) | VRAM used (max GB) |
|---|---|---|---|---|---|---|
| 1 | 0.556 | 5.56 | 1.37 | 5.4 / 77 | 8.7 / 100 | 5.3 |
| 2 | 0.911 | 9.11 | 2.36 | 8.7 / 88 | 17.5 / 100 | 7.7 |
| 4 | **1.079** | **10.79** | 3.24 | 10.1 / 94 | 24.0 / 100 | 12.3 |
| 7 | 1.105 | 11.05 | 4.06 | 9.6 / 80 | 19.5 / 100 | 16.9 |

Ceiling ≈ **1.1 win/s / 11.0 h/h at K = 4** (K = 7 buys nothing more). CPU 4.06 / 7.65 cores,
VRAM 16.9 / 46 GB, average NVDEC ~10 % → the shared GPU is the limiter.

## 1. What "sharded into isolated workers" means here

* **Unit of sharding = clip**, never a window: a clip is decoded by exactly one worker, so no
  window is split and decode state is never shared. The plan is a deterministic
  longest-processing-time (LPT) greedy over per-clip window counts, ties broken by sorted clip
  name — byte-stable for a given (subset, K). (`common.make_plan`.)
* **Isolation**: each shard is its own OS process (`eval/exp_shard/shard.py`), single-threaded
  (`OMP_NUM_THREADS=1`, ≤1 core), with its **own output JSONL** (`shards/…/shard_<i>.jsonl`),
  its **own state/log dir**, and its **own EG2 model instance**. No queues, locks or shared
  memory between shards; the only shared resources are the OS (CPU/disk) and the GPU.
* The worker **reuses the latency runner verbatim** (`eval/exp_latency_fast/run_fast.py`,
  brought in from `m/latency-fast`) with the adopted throughput config: pixel 3.33 fps,
  EG2 1 fps, native 0.8 fps, 4 s early exit, mosaic off.

## 2. Deterministic aggregation + shard-invariance

Rows carry run-local wall timings (`latency_ms`, `stage_wall`, …). The aggregator hashes only
the **scored content**: `det_projection(row)` strips the timing fields (and the
latency-dependent `receipt_sha256`), and

```
combined_digest = sha256( canonical( sorted[ (window_id, sha256(canonical(det_projection(row)))) ] ) )
```

This is **partition-independent and order-independent** — the same whether the windows were
produced by 1 process or K isolated processes, in any arrival order. Each shard also emits its
own `shard_digest` (integrity receipt over its rows) + a per-shard meta JSON; the aggregator
recomputes and verifies it.

**Proof (shard-invariance):** the *same* 102-window subset was run at K = 1, 2, 4, 7 and the
combined digest recomputed each time:

```
K=1 9b388f92c8099a743f94a15ae12f51c89e019e1b26a46bca4d651d6811c913e8
K=2 9b388f92c8099a743f94a15ae12f51c89e019e1b26a46bca4d651d6811c913e8
K=4 9b388f92c8099a743f94a15ae12f51c89e019e1b26a46bca4d651d6811c913e8
K=7 9b388f92c8099a743f94a15ae12f51c89e019e1b26a46bca4d651d6811c913e8
```

**All equal → shard-invariant.** K = 1 is a single-process run, so this proves the K-way split
reproduces the single-process result exactly. Per-shard digests differ (they cover disjoint
clips) but combine to the same full digest.

Aggregation protocol: `shard_i.jsonl` → per-shard digest → **combined digest (sort by
window_id)**; `aggregate.py --expect-digest` fails loudly on any mismatch.

## 3. Scaling table and the ceiling

| K | windows/s | footage-h/wall-h | CPU cores | NVDEC dec% (mean/max) | GPU sm% (mean/max) | VRAM max GB | per-shard win/s |
|---|---|---|---|---|---|---|---|
| 1 | 0.556 | 5.56 | 1.37 | 5.4 / 77 | 8.7 / 100 | 5.3 | 0.566 |
| 2 | 0.911 | 9.11 | 2.36 | 8.7 / 88 | 17.5 / 100 | 7.7 | 0.477 |
| 4 | 1.079 | 10.79 | 3.24 | 10.1 / 94 | 24.0 / 100 | 12.3 | 0.378 (34-win) / 0.29 (17-win) |
| 7 | 1.105 | 11.05 | 4.06 | 9.6 / 80 | 19.5 / 100 | 16.9 | 0.197 |

**Cause: single-GPU serialization of the decode/copy path — not CPU, not VRAM, not NVDEC
saturation.** Evidence:

1. **Per-shard throughput falls ~as 1/K** (0.566 → 0.477 → 0.378 → 0.197 win/s): each extra
   worker divides a *fixed-capacity shared resource* rather than adding capacity.
2. **The CPU quota is never reached**: 4.06 / 7.65 cores at K = 7; per-shard CPU *drops*
   (1.19 → 0.66 cores) — workers spend most of their time blocked, not computing.
3. **NVDEC/SM are bursty, not saturated on average**: mean `dec` 5–10 %, mean `sm` 9–24 %
   (both peak 100 % momentarily). The 3-branch decode (NVDEC + `scale_cuda` + `hwdownload`)
   is a short GPU burst per window; across processes the bursts contend on the one GPU's
   decode + copy engines and time-slice, so aggregate throughput saturates.
4. **VRAM is not the limit**: 16.9 / 46 GB at K = 7.

**Startup caveat (important for absolute numbers).** Each shard pays a one-off EG2 model load +
torch/ffmpeg warm-up (~10–15 s). On the 6-clip subset (17–34 windows/shard) that is a large
fixed cost, which is why the subset ceiling (11 h/h) sits below the **full-corpus measured
19.79 h/h at K = 7** (`docs/reports/pod_benchmark_2026-10-10.md`, 1,546 windows, 801 s), where
shards hold ~220 windows and the warm-up amortises. The *shape* (plateau at K ≥ 4) is
startup-independent; the *level* is **~11 h/h for tiny shards, ~19.8 h/h at corpus-scale
shards**. The ceiling is GPU-contention-bound in both cases.

## 4. RAM + VRAM parallelism

{{VRAM_SECTION}}

## 5. Multi-pod plan

Containers on **one pod share the cgroup CPU quota and the single GPU**: sharding into K
isolated workers gives **reproducibility and clean failure isolation, not extra throughput**.
Horizontal scaling needs **multiple pods** (each = its own cgroup quota + its own A40/NVDEC).

From the measured per-pod ceiling **≈19.8 footage-h/wall-hour** at corpus-scale shards
(conservative small-shard floor **≈11 h/h**):

| target footage-h/wall-hour | pods @ 19.8 h/h | pods @ 11 h/h (conservative) | CPU cores total @ 19.8 | VRAM used total @ 19.8 (per-pod 17 GB of 46) |
|---|---|---|---|---|
| 50 | **3** | 5 | 21 | 51 GB |
| 100 | **6** | 10 | 42 | 102 GB |
| 1000 | **51** | 91 | 357 | 867 GB |

What to isolate per worker: the **decode + pixel/descriptor feature CPU work** (one single-core
process per shard) is what parallelises across cores; the **EG2 embed and NVDEC decode** share
the single GPU and do **not** gain from more processes beyond the GPU-contention ceiling.
Per-pod VRAM budget: **reserve ~20 GB/pod** for K = 7 workers (model + decode buffers +
activations; measured 16.9 GB peak), i.e. **≤ 40 GB usable of the 46 GB** with headroom for the
zero-copy path; across N pods that is **N × 40 GB** (e.g. 3 pods → 120 GB, 51 pods → ~2 TB).

Aggregation across pods is identical to across shards: each pod returns `shard_i.jsonl` +
digest; the controller concatenates, sorts by `window_id`, recomputes the combined digest. Same
code, same invariant, any number of pods. No shared mutable state ⇒ a slow/failed pod just
retries its shard; the digest detects any corruption.

## 6. Optimal shard size (clips per shard)

* **This pod:** K = 7 workers over 58 clips ⇒ **≈8–9 clips per shard** (≈200–230 windows,
  balanced by LPT). Use exactly K shards (or a multiple) so no worker idles at the end.
* **GPU-per-pod fleet:** each pod independently splits *its* clips into K shards, so the local
  shard size is the same ≈8–9 clips; the fleet-level split assigns whole pods.
* **Floor ≈4 clips/shard:** each shard pays a ~10–15 s warm-up (EG2 model load + torch/ffmpeg);
  a single 17-window clip is only ~30 s of work, so 1-clip shards would spend ~⅓ of their life
  warming up.

## 7. Honesty flags

* The scaling subset is the **six smallest clips / 102 windows**, not the full corpus — chosen
  to fit the timebox; the plateau shape is representative but the absolute h/h differs from the
  full-corpus run (19.79 h/h at 7 in-process workers).
* GPU/NVDEC/VRAM numbers are **GPU-wide** samples on a **shared pod**; other jobs' utilisation
  is included. Per-shard VRAM is the torch allocator peak recorded in each shard's rows.
* The 1,546 windows are **unlabelled**; verdicts are descriptive, not accuracy (see
  `contract.HONESTY_NOTE`).

## 8. Artifacts

* `eval/exp_shard/{common,shard,aggregate,run_scaling,vram_probe,make_figure}.py` +
  `tests/test_exp_shard.py`
* `eval/results/exp_shard/S1.01.json` (scaling table + invariance + VRAM), `vram_probe.json`
* `eval/results/exp_shard/shards/…` (per-K shard JSONL + metas + combined digests)
* figure: `docs/experiments/exp_shard/images/exp_shard_scaling.png`
* registry: `experiments/S1.01/`

## 9. Reproduce

```bash
source /workspace/env.sh
export OMP_NUM_THREADS=1 JADAL_GPU_SLOTS=8
bash tools/pod/gpu_run.sh python3 eval/exp_shard/vram_probe.py \
     --out eval/results/exp_shard/vram_probe.json
bash tools/pod/gpu_run.sh python3 eval/exp_shard/run_scaling.py \
     --ks 1,2,4,7 --subset smallest6
python3 eval/exp_shard/make_figure.py
```
