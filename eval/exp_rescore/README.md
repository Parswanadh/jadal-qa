# eval/exp_rescore — E8.01: re-score the Egocentric-10K scale corpus (B2)

Scores all 1,546 windows of `$JADAL_BUNDLE/queues/manifest_scale.json` (58 clips, 4.404 h,
1920x1080 HEVC @30 fps) with the **frozen in-domain recipe `ensemble_v1`** and the
**deterministic profile router**, and writes the PoC contract file
`/workspace/artifacts/windows_rescored.jsonl` (not committed: `artifacts/` is ignored).

**These windows are unlabelled (`label_source: "none"`), so every verdict is descriptive, not
accuracy.** IndEgo-trained models did not transfer out-of-domain (0.4-0.53 AUROC on the 60
human-labelled Egocentric-10K windows, n_bad = 9), so the accept/reject share is not a quality
measure. Read `docs/reports/exp_rescore.md` before quoting any number.

## Files

| file | role |
|---|---|
| `run_e8.py` | runner: one ffmpeg/NVDEC pass per window (3 branches), pixel26 + EG2 + descriptors, scoring, JSONL + partial file for `--resume` |
| `contract.py` | canonical JSON, digests, `receipt_sha256`, `features_digest`, verdict/evidence rules, header line |
| `write_results.py` | builds `eval/results/exp_rescore/E8.01.json`, the 20-line sample and `docs/reports/exp_rescore.md`; re-verifies every receipt |
| `../../tests/test_exp_rescore.py` | 11 tests, ~23 s, no network |

## Run

```bash
source /workspace/env.sh
export JADAL_EXP_DIR=/workspace/wt/exp-rescore/experiments OMP_NUM_THREADS=1
cd /workspace/wt/exp-rescore
python3 eval/exp_rescore/run_e8.py --workers 4 --resume \
        --out /workspace/artifacts/windows_rescored.jsonl      # ~45 min on a shared A40
python3 eval/exp_rescore/run_e8.py --mode control --n 6        # frozen-vs-fast decode control
python3 eval/exp_rescore/write_results.py                      # results JSON + sample + report
python3 -m pytest tests/test_exp_rescore.py -q
```

`--resume` reuses `<out>.partial` (one JSON object per finished window: `line`, `desc`, `eg2`),
so a killed run continues where it stopped.

## Contract line (exactly these keys)

```
window_id, factory, worker, clip, start_s, end_s, verdict (accept|reject|review), stage,
reasons[{code, evidence{feature, value, op, threshold}}], score, features{}, features_digest,
receipt_sha256, versions{}, latency_ms{}, label_source
```

* `receipt_sha256` = SHA-256 of the canonical JSON of the line without `receipt_sha256`
  (sorted keys, compact separators). **Content-hashed, NOT signed.**
* `features_digest` = SHA-256 of the canonical `features{}` block (the 26 pixel features + the
  EG2 mean stored by content hash + frame counts).
* `versions{}` carries the recipe, the config digests, the chosen profile + version, the router
  margin + reasons, `unseen_regime`, the descriptor digest, the decode path, device and git sha.
* Line 1 of the file is a `type: "header"` record with n, the honesty note, the distributions and
  the SHA-256 of the window body.

`contract.py::verify_line` re-derives both digests from the line content; `write_results.py`
checks all of them and fails loudly if any mismatch.

## Measured deviations from the frozen (laptop) path — documented, not hidden

1. **Decode path**: NVDEC + `scale_cuda` (bilinear) + `hwdownload` instead of the frozen
   `ffmpeg_cpu`/swscale (bicubic) + cv2/PIL sampling. Features are not bit-identical
   (`--mode control`, n=6: |Δ p_pixel| max 0.048, mean 0.014 — exploratory, no CI).
2. **Config sidecar**: `configs/ensemble_v1.json.sha256` predates the pod path remap; the head
   numbers were verified equal to `$JADAL_BUNDLE/caches/ensemble/ensemble_v1.json` and both
   digests are recorded per line.
3. **E3.04 (every-2nd window + 4 s early exit) not applied** and `factory_1080p30`'s
   `fps=5`/`probe_every_k_windows=2` not applied to the pixel view: the contract needs one verdict
   per window and the frozen heads were fit on full-window features.
