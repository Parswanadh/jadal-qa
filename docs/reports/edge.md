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
   worker cores=0: repeat0 median/seg = 0.626s   # repeat-0 only; the table
   determinism: repeats=True cores=True          # below uses the POOLED median over all 90 runs
   ```
   Result JSON: `eval/results/edge_bench_x86-laptop-proxy.json`.
4. `PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_edge_check`
   → `Ran 19 tests ... OK` (0.001 s; hermetic, no signals/clips needed).

## Results (proxy hardware: Intel Core Ultra 7 155H x86 laptop)

Measured with the CURRENT (fix-round-2) scripts in a throwaway signals
worktree (see Fix round 2); supersedes the round-1 table (old-scheme cache,
pre-fix code — round-1 values retained in Fix round 2 for audit).

| pinned cores | pooled median/seg (90 = 30 segs x 3 repeats) | p95/seg | segments/hour | video-hours/day | peak RSS |
|---|---|---|---|---|---|
| 1 (`-c 0`, requested 1, effective 1) | 1.0317 s | 1.9799 s | 3489.3 | 232.62 | 171.6 MB |
| 4 (`-c 0-3`, requested 4, effective 4) | 1.0797 s | 2.1175 s | 3334.3 | 222.29 | 180.2 MB |

This round the 4-core pooled median is slightly SLOWER than 1-core (shared
box under load at the time); reported as measured, no scaling factor
invented. Determinism (identical outputs) still holds across cores —
throughput varies with machine load, outputs do not.

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

## Fix round 1 (review of PR #48)

Every Critical/Important finding fixed in owned files; Minors fixed except
Minor 3, which is unfixable until `feat/signals` merges (proven below, not
ignored). No protected files touched (`git status --short` shows only the 7
owned paths).

- Important 1 (lazy call-time imports escaped the gate): `_PROCESS_CHILD`
  now emits `"modules": sorted(sys.modules.keys())` AFTER `compute_signals`
  runs, and `main()` runs `check_forbidden()` on it into
  `runtime_forbidden_found`, FAILing (exit 1) on any hit — same gate as
  import time. Pinned by `test_runtime_probe_modules_detect_lazy_torch`
  (clean import-time list passes, same list + `torch` fails) and
  `test_process_child_emits_modules`. Live full-path re-run still needs
  signals (absent here), so this round's evidence is the unit tests (real
  output: `Ran 28 tests ... OK`, below).
- Important 2 (cache keyed on stem+start only): new pure
  `make_seg_id()` keys the cache on stem + sha1(full source path) + start +
  seg_seconds; `extract_segments()` reuses a cached file only when its frame
  count equals the needed frames, and FAILs loudly on `got != need`
  (previously only `got == 0` failed). Pinned by `test_seg_id_keys_path_and_length`.
- Minor 4 (exit-code semantics): new `--require-probes` (exit 2 when any
  probe skipped — matches `edge_bench.py`'s SKIP exit 2). Real run on this
  branch (signals absent): `--require-probes --out
  data_runtime/edge_tmp/req_probe.json` → wrote the skip-path JSON, `exit=2`.
- Minor 5 (default run clobbered the measured JSON): new pure
  `should_overwrite()` + `--force`; a probe-skipped result never replaces a
  result containing `process_probe`. Real run: default `edge_check.py` on
  this branch → `NOT OVERWRITTEN: ... already contains measured probes ...
  --force ...`, `exit=2`, and `git status --short eval/results/` shows the
  committed file untouched. (The reviewer's restore-via-checkout incident
  cannot recur.)
- Minor 6 (contiguous-range assumption): new pure `parse_core_list()`
  (`0`→1, `0-3`→4 CPUs) replaces the first..last arithmetic; `--help`
  documents that non-contiguous sets are not expressible. Pinned by
  `test_parse_core_list` + `test_parse_core_list_garbage`.
- Minor 3 (throwaway-worktree provenance): PROVEN unfixable on this branch —
  `jadal.signals` is still absent (`missing: ['jadal.segmenter',
  'jadal.signals']` in the real run above), so no full-path re-run is
  possible here. Mitigations committed: both scripts now record `git_branch`
  in every result they write (seen live: `"branch": "feat/edge"` in the
  skip-path JSON), and the committed full-path JSONs are untouched. The
  original NOT DONE stands: re-run both scripts on `main` after
  `feat/signals` merges and recommit.
- Non-finding suggestion adopted: `onnxruntime` + `tensorflow` added to the
  forbidden list (`test_each_forbidden_exact` covers all 11 names).

Real verification outputs (this round, x86 laptop proxy hardware):

```
$ PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_edge_check
Ran 28 tests in 0.001s
OK
$ PYTHONNOUSERSITE=1 timeout 250 /home/parshu/envs/jadal/bin/python -m unittest discover -s tests 2>&1 | grep -E '^(Ran|OK|FAILED|ERROR|FAIL:)'
Ran 419 tests in 10.622s
OK
$ PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/edge_check.py   # skip path
forbidden_found: none
runtime_forbidden_found: none
SKIPPED: jadal.signals/jadal.segmenter absent (branch origin/feat/signals, PR #12, not yet on main): ...
default-exit=2   # refuses to clobber the measured JSON (Minor 5)
```

## Fix round 2 (review of PR #48, round 2)

- I1 (committed JSONs predated the fixed code): RESOLVED by re-measurement,
  not just relabelling. New throwaway worktree (`tmp-sigtest2`,
  `origin/feat/signals@a731265` + this branch, removed afterwards) with the
  CURRENT fixed scripts produced both committed JSONs fresh: new-scheme
  segment ids (`User_1_User_15_411_2910_2_1_480-25653bf9-0000.000-10s`),
  `git_branch` + `runtime_forbidden_found` present, `segment` stored relative
  (`data_runtime/edge_tmp/edge_check_seg.mp4`, M2 — no dead absolute path).
  JSON `git_sha` is `1a60b45` (the throwaway merge commit) and `git_branch`
  reads `HEAD` (detached checkout) — provenance disclosed here, not hidden.
  The results table above now shows these fixed-code numbers; the round-1
  table (1-core 4164.0 seg/h, 277.6 video-h/day, RSS 170.9 MB; 4-core 5892.0
  seg/h, 392.8 video-h/day, RSS 179.4 MB; medians 0.8646/0.6110 s, p95
  1.4366/1.1885 s) is SUPERSEDED and kept only for audit. `feat/signals`
  (PR #12) is still unmerged, so a main-branch re-run remains future work.
- M1 (mixed medians): table header now reads "pooled median/seg (90 = 30
  segs x 3 repeats)"; round-1 log annotated repeat-0 vs pooled. This round:
  1-core repeat-0 median 1.011 s vs pooled 1.0317 s.
- M3 (near-miss requirements names): `test_no_torch_family` now matches
  normalised stems (catches `pytorch`, `torchvision`, `fastapi-users`,
  `onnxruntime`, `scikit_learn`, `transformers[torch]`), pinned by
  `test_banned_stems_catch_near_misses` including legit-name negatives.
- M4 (silent core clamp): results now carry `n_cores_requested` +
  `n_cores_effective` (this round: 1/1 and 4/4 — no clamp occurred).

Real verification outputs (this round, x86 laptop proxy hardware):

```
$ PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python scripts/edge_check.py   # throwaway, WITH signals
imported: ['jadal.segmenter', 'jadal.signals', 'jadal.funnel', 'jadal.certify', 'jadal.receipts', 'jadal.config']
missing: []
heavy_present: {'numpy': True, 'cv2': True, 'scipy': True, 'mediapipe': False, 'PIL': True, 'imagehash': True}
forbidden_found: none
runtime_forbidden_found: none        # Important-1 gate verified LIVE on real compute_signals
probe: compute_signals_s=1.253 peak_rss_MB=134.5 backend={'faces': 'proxy', 'hands': 'proxy', ...}
mediapipe-blocked: fallback_works=True
PASS: no forbidden import on the edge path.
exit=0
$ PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/edge_bench.py --segments 30   # throwaway
segments: 30 x 10.0s (deterministic selection, ...)
worker cores=0: repeat0 median/seg = 1.011s
determinism: repeats=True cores=True
1-core: pooled median 1.0317s, p95 1.9799s, 3489.3 seg/h, 232.62 video-h/day, RSS 171.6 MB
4-core: pooled median 1.0797s, p95 2.1175s, 3334.3 seg/h, 222.29 video-h/day, RSS 180.2 MB
$ PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python -m unittest tests.test_edge_check
Ran 29 tests in 0.005s
OK
```
