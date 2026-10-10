# Edge feasibility (B9) — import audit + pinned-core benchmark

All numbers below are **x86 laptop cores as a proxy, NOT measured on a
Raspberry Pi**. Nothing in this report claims the pipeline "runs on a Pi".

## What ran (commands + real output tails)

`jadal.signals` / `jadal.segmenter` live on branch `origin/feat/signals`
(PR #12, not yet on main), so the real-module runs below were executed in a
THROWAWAY worktree (`/home/parshu/projects/jadal-qa-wt/tmp-sigtest`,
`origin/feat/signals` + this branch's edge files copied in, since the edge
files are new and conflict-free). The throwaway worktree was removed
afterwards. The committed scripts import-guard and SKIP with an explanation
when signals is absent.

1. `PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/edge_check.py`
   (on `feat/edge` WITHOUT signals — the skip path):
   ```
   imported: ['jadal.funnel', 'jadal.certify', 'jadal.receipts', 'jadal.config']
   missing: ['jadal.segmenter', 'jadal.signals']
   heavy_present: {'numpy': True, 'cv2': False, 'scipy': True, 'mediapipe': False, 'PIL': False, 'imagehash': False}
   forbidden_found: none
   SKIPPED: jadal.signals/jadal.segmenter absent (branch origin/feat/signals, PR #12, not yet on main): segment processing and backend probes skipped; import audit covers the modules present on this branch.
   PASS: no forbidden import on the edge path.
   exit=0
   ```
2. Same command in the throwaway worktree WITH signals (full path):
   ```
   imported: ['jadal.segmenter', 'jadal.signals', 'jadal.funnel', 'jadal.certify', 'jadal.receipts', 'jadal.config']
   missing: []
   import_time_s: 0.556
   heavy_present: {'numpy': True, 'cv2': True, 'scipy': True, 'mediapipe': False, 'PIL': True, 'imagehash': True}
   forbidden_found: none
   probe: compute_signals_s=0.748 peak_rss_MB=133.4 backend={'faces': 'proxy', 'hands': 'proxy', ...}
   mediapipe-blocked: fallback_works=True backend={'faces': 'proxy', 'hands': 'proxy', ...}
   PASS: no forbidden import on the edge path.
   exit=0
   ```
   Result JSON: `eval/results/edge_check_x86-laptop-proxy.json` (copied from
   the throwaway run; its `git_sha` is that worktree's HEAD).
3. `PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/edge_bench.py --segments 30`
   (throwaway worktree WITH signals; 30 real IndEgo 10 s segments,
   deterministic selection, 1 warm-up discarded, 3 repeats, pinned via
   taskset to 1 core (`-c 0`) and 4 cores (`-c 0-3`)):
   ```
   worker cores=0: repeat0 median/seg = 0.626s
   determinism: repeats=True cores=True
   ```
   Result JSON: `eval/results/edge_bench_x86-laptop-proxy.json`.
4. `PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_edge_check`
   → `Ran 19 tests ... OK` (0.001 s; hermetic, no signals/clips needed).

## Results (proxy hardware: Intel Core Ultra 7 155H x86 laptop)

| pinned cores | median/seg | p95/seg | segments/hour | video-hours/day | peak RSS |
|---|---|---|---|---|---|
| 1 (`-c 0`) | 0.8646 s | 1.4366 s | 4164.0 | 277.6 | 170.9 MB |
| 4 (`-c 0-3`) | 0.6110 s | 1.1885 s | 5892.0 | 392.8 | 179.4 MB |

Conditions: 30 segments x 10 s (480x480, 10 fps IndEgo, read-only),
`compute_signals` + `funnel.route` with PLACEHOLDER thresholds (throughput
only — verdict quality is NOT claimed), backends `faces=proxy, hands=proxy`
(no mediapipe model files installed; mediapipe never imported), cv2 threads
= core count, OMP/OPENBLAS/MKL threads = core count.

Import audit: the edge tier pulls in numpy, cv2, scipy (via
`jadal.certify`), pillow + ImageHash (via `jadal.signals` phash) — and NO
torch, torchvision, sentence-transformers, transformers, sklearn, pandas,
matplotlib, fastapi/uvicorn, or CUDA library. `requirements-edge.txt` is
derived from exactly this finding.

Determinism: `sha256(canonical-json(signals+route))` identical across all 3
repeats AND across the 1-core vs 4-core runs (`identical_across_repeats:
true, identical_across_cores: true`).

OpenCV fallback: with mediapipe BLOCKED (`sys.modules['mediapipe']=None`)
the edge path runs end-to-end on the proxy backends (`fallback_works:
true`). mediapipe on Pi ARM64 remains UNVERIFIED (no model files here, no Pi
hardware) — the proxy fallback is the supported edge configuration until a
real Pi run says otherwise.

## How to run on a real Raspberry Pi

```
git clone <repo> && cd <repo> && git checkout <edge-sha-or-main-after-merge>
python3 -m venv .venv-edge && . .venv-edge/bin/activate
pip install -r requirements-edge.txt
# clips: copy N IndEgo/original 10 s segments or full clips to a read-only dir
PYTHONNOUSERSITE=1 python scripts/edge_check.py --device-label pi4 --clip-dir <clips>
PYTHONNOUSERSITE=1 python eval/edge_bench.py --device-label pi4 --clip-dir <clips> --segments 30
```

The `--device-label` value names the result files
(`eval/results/edge_*_pi4.json`) and replaces the proxy disclaimer for that
run only. Copy the Pi JSONs back to replace the proxy files.

## NOT DONE

- **NOT DONE: never run on a Pi.** No Raspberry Pi hardware was available;
  every number above is from x86 laptop cores as a proxy. No Pi speed is
  extrapolated from the laptop numbers (no invented scaling factor).
- Thresholds are placeholders: throughput only, no verdict-quality claim.
- mediapipe on Pi ARM64 unverified; proxy fallback is the supported path.
- Cross-architecture determinism is not claimed (receipt re-verify stays
  same-device-class per the baseline amendment).
- The committed `eval/results/edge_*.json` files were measured in the
  throwaway signals worktree; re-run both scripts after `feat/signals`
  merges to refresh them on `main`.
