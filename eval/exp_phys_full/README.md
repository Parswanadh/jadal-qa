# eval/exp_phys_full — Q5.01 deterministic physics rules over the full corpus

Runs Q1's **deterministic physics rule list** over all 1,546 Egocentric-10K scale
windows at full resources, with atomic per-stage timing.

It is the L3 latency runner (`eval/exp_latency_fast/run_fast.py`) copied onto this
branch with the **decision stage swapped**:

* identical one-pass NVDEC + `scale_cuda` three-branch decode,
* identical pixel-26 features (`jadal.gpu.features.cpu_window_features`),
* identical E4.01 descriptors + deterministic router (recorded, not used),
* identical EG2 mean embedding (EmbeddingGemma-2, bf16),
* **new** atomic `phys_features` stage: `jadal.perpixel.physics.extract_physics`
  on the decoded 160x90 frames (the 16 physics features the rules consume),
* **swapped** `decision` stage: `eval/exp_phys/phys_rules.score_window` +
  `decide` with the frozen `rule_table.json` (integer score = count of crossed
  fixed quantile thresholds; no learned weights).

## Stages

| # | stage | what is timed |
|---|---|---|
| 1 | `io_probe` | open + seek + 64 KiB read at the window byte offset |
| 2 | `decode` | one NVDEC + `scale_cuda` pass, 3 branches |
| 3 | `pixel_features` | `cpu_window_features` (pixel-26) |
| 4 | `descriptors` | E4.01 descriptors on the native frames |
| 5 | `eg2_embed` | EG2 L2-normalised mean embedding |
| 6 | `phys_features` | `extract_physics` (flow/dhash/noise/bytes) |
| 7 | `decision` | deterministic rule score + verdict + receipt |
| 8 | `write` | parent JSONL append (timed in the parent) |

## Run / analyse

```bash
source /workspace/env.sh
export OMP_NUM_THREADS=1 JADAL_GPU_SLOTS=6
bash tools/pod/gpu_run.sh python3 eval/exp_phys_full/run_phys.py \
     --workers 7 --determinism-n 10 --pixel-fps 3.33 --eg2-fps 1 \
     --early-exit-s 4 --skip-mosaic --no-compare \
     --out eval/results/exp_phys_full/windows.jsonl
python3 eval/exp_phys_full/analyze_phys.py    # Q5.01.json + report + figures
```

Outputs: `eval/results/exp_phys_full/{windows.jsonl,clips.jsonl,run_meta.json,Q5.01.json}`,
`docs/reports/exp_phys_full.md`, `docs/experiments/exp_phys_full/images/`.

`receipt_sha256` hashes the contract line **excluding** `latency_ms`, so two
repeats of the same window are byte-identical (verified for 10 windows).
