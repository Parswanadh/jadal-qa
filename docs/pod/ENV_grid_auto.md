# ENV_grid_auto — environment, base and baseline gate (Track GRID/AUTO/TASK, DSH)

Written by DSH (lead agent) at the start of the local pass, 2026-10-18.
Everything here is measured on the builder's laptop unless marked otherwise.

## 2.1 Environment check (measured)

| Item | Value |
|---|---|
| `opencode --version` | `opencode v2.0.32` |
| GPU | `NVIDIA GeForce RTX 4070 Laptop GPU`, 8188 MiB total, ~1.0-1.3 GiB already used by other tools, driver 595.99.02 |
| `nproc` | 22 |
| `free -g` | total 15, used ~10, available ~4 (desktop baseline is heavy; swap 26 GB, 4 GB used) |
| `df -h /` | 937 G total, 865 G used, **25 G available (98 % full)** — the binding constraint |
| Python | system `python3` = 3.13.13 and `python3` = 3.14.6 — **neither has torch**. Two usable interpreters below. |
| **Project venv (use this)** | **`/home/parshu/envs/jadal/bin/python`** — Python 3.12.15, `cv2` 5.0.0, `torch` 2.14.1+cu130 (`cuda.is_available() == True`), transformers 5.19.0, sentence_transformers 6.1.0, sklearn 1.9.1, numpy 2.5.2, scipy 1.18.1, pandas 3.0.6, Pillow 12.3.0. **The brief's warning that OpenCV is unavailable is wrong for this machine** — decode works: one 480p IndEgo clip (6983 frames) decodes in 9.2 s (~760 fps). |
| Fallback interpreter | `/usr/bin/python3.12` (3.12.3) with `~/.local` packages: torch 2.7.1+cu118, numpy 2.4.4, sklearn 1.9.0 — **no cv2, no sentence_transformers**; cached-feature work only. GRID.00 reproduces 0.810 under both, so cached-feature numbers do not depend on the interpreter choice. |
| ffmpeg | 6.1.1-3ubuntu5, NVDEC/CUDA hwaccels present in the build config |
| gh | 2.93.0, authenticated as `Parswanadh` (keyring). No `GH_TOKEN`/`HF_TOKEN` in the environment. |

Consequence for this pass: **decode-dependent work (mosaics, per-frame features, S0 probes)
runs in `/home/parshu/envs/jadal`**; cached-feature and policy work runs anywhere. All commands in
the GRID reports name the interpreter they used.

## 2.2 Base branch — the brief and the repo disagree; the repo wins

The brief says main is at `53983d6` (confirmed: `git fetch --all` lands `origin/main` on
`53983d6285267669cea6198ca4d246ee98b2f9b4`) and that the tooling (`AGENTS.md`,
`tools/exp.py`, `tools/pod/secrets_guard.sh`, the `.opencode/skills/jadal-*` skills) "is not on
main; branch your work from `pod/base` or `m/findings` where they exist".

Measured facts that change the plan:

* `AGENTS.md`, `.opencode/`, and the whole `tools/` tree are **absent from `origin/main`**. They
  were dropped by the merges of the `wip-*` snapshot branches ("scope to the experiment only —
  drop stacked work (match main)"), i.e. by accident of the snapshot cleanup, not on purpose.
* `origin/m/findings` **does** have the tooling, but it is **205 commits behind main** and is
  missing `jadal/mosaic/build.py`, `jadal/poc/cascade.py` and `experiments/EXPERIMENTS.md`.
  Branching there would throw away the shipped code the tracks must measure.
* `origin/pod/base` is 38 behind / 16 ahead and **also lacks** `tools/exp.py` and `AGENTS.md`.
* `origin/wip/m/latency-fast` is the branch *closest to main* that still has `tools/exp.py`
  (99 behind / 7 ahead) — still the wrong code state.

**Chosen base: `origin/main` (`53983d6`) + one restore commit.** Branch `dsh/infra` adds
`AGENTS.md`, `.opencode/**`, `tools/**` verbatim from `origin/m/findings` (commit `a3789b8`),
so the registry protocol and the secrets guard work while the code under test is current main.
All other stream branches fork from `dsh/infra`. The builder should decide whether the tooling
restore should reach `main` or stay branch-local; it is flagged in the PR body.

## 2.3 GRID.00 baseline gate — PASS

Command: `JADAL_BUNDLE=/home/parshu/projects/jadal-qa-wt/handover/jadal_pod_bundle/pod_bundle python3 tools/dsh/grid00.py`
(`dsh/infra@55b7da1`), results `eval/results/grid/GRID.00.json`, registry `GRID.00` (verdict adopt).

| Metric | Value |
|---|---|
| Pooled LOCO AUROC (n=150, 100 good / 50 bad, 7 clips, ONE human labeller) | **0.810** |
| 95 % CI, clip-level bootstrap (2000 draws, seed 0) | [0.683, 0.892] |
| 95 % CI, window bootstrap — **optimistic** | [0.722, 0.889] |
| good vs SEMANTIC bad (NO_ACTIVE_WORK / HANDS_NOT_VISIBLE) | 0.797 |
| good vs QUALITY bad (TOO_DARK / DUPLICATE / UNDECODABLE) | 0.846 |
| per-clip AUROC | desk 0.80, ecu_full 0.95, ecu_late 0.65, logistics4 0.90, organise 0.91, pallet 0.50, pallet2 0.50 |
| Gate | expected 0.810 ± 0.01 → observed 0.810, **passed** |

The independent reproduction (`tools/port/baseline_logistic_loco.py`, unchanged pod script)
printed the same 0.810 raw / 0.797 semantic / 0.846 quality, so data paths, feature cache and
folds are the pod's. Runtime ~2 s CPU, 0 GPU. Nothing else in GRID/AUTO starts until this passed.

## 2.4 Data inventory actually present on this laptop

| Dataset | Path | n | Labels |
|---|---|---|---|
| IndEgo 480p clips (7 used) | `/home/parshu/projects/awing/data/indego/*/mp4_480/` | 7 clips | 150 windows, ONE human: 100 good / 50 bad (`NO_ACTIVE_WORK` 23, `HANDS_NOT_VISIBLE` 14, `DUPLICATE` 6, `TOO_DARK` 6, `UNDECODABLE` 1) |
| Pod bundle (labels + caches + scripts) | `/home/parshu/projects/jadal-qa-wt/handover/jadal_pod_bundle/pod_bundle/` | — | `labels/manifest_gold.json` (150), `labels/ood_labels.json` (60), `labels/ood160_labels.json` (160), `caches/gpu/features.json` (26 features, 210 windows incl. synthetic) |
| OOD-160 480p window clips | `…/ood_labels/video` (60), `…/ood_labels/video2` (100) | 160 | merged manifest: 138 good / 22 bad (19 `NO_ACTIVE_WORK`, 2 `HANDS_NOT_VISIBLE`, 1 `SHAKY`), 47 factories |
| OOD round-3 hold-out | `…/ood_labels/video3` (116 clips), labels `…/backups/ood3/labels3.20261011_000202.json` | 108 labelled of 116 | 92 good / 16 bad (10 `NO_ACTIVE_WORK`, 4 `OVEREXPOSED`, 2 `HANDS_NOT_VISIBLE`) — **already scored twice (C1 PR #162, physics v2 PR #163); report-only from here** |
| 58 original 1080p Egocentric-10K clips | `/home/parshu/projects/awing/data/ego10k_scale/clips/` | 4.404 h, 47 factories | none |
| EG2 weights | `/home/parshu/projects/awing/data/eg2/` | — | optional reference arm only |
| Frozen OOD-160 scores for the shipped head | `/home/parshu/projects/awing/interview/dsh_pack/ood160_heads_old26_vs_new22.json` | 160 | per-window 26-number (original portal) and 22-number (hardened) scores — the frozen baseline for GRID.06 / AUTO.00 |
| Demo kit | `/home/parshu/projects/awing/interview/demo_kit/` | 12 clips | kit table in `HARDEN_REPORT.md` |

## 2.5 Branch / PR state (checked after `git fetch --all`)

* 226 branches total; 67 PRs open. Every PR named in the brief (##95, #119, #162, #163, #166,
  #169) is **still open** — nothing merged since the audit.
* New since the brief: `harden/strengthen-v1` (builder's local worktree, 3 commits: H7 legacy
  threshold, H5 22-number head, H8 versions block) and the `interview/` handover pack on the
  builder's side. The brief's "main at 53983d6, 773 tests" matches the harden gate run.
* Local-only refs that the brief mentions by name and that exist: `origin/m/mosaic-res`,
  `origin/wip/m/mosaic-1080p` (the brief says the plain `m/mosaic-1080p` has code but no report
  — confirmed), `origin/pod/prwatch`, `origin/pod/base`, `origin/m/phys-ready`,
  `origin/wip/m/conformal`, `origin/wip/m/clip-sequential`, `origin/m/recipe-freeze`,
  `origin/m/domain-adapt-160`, `origin/m/new-features-160`, `origin/feat/exp-cost-model`.

## 2.6 Constraints carried into the tracks

* Disk 25 G free → scratch budget ≤5 GB, everything under `/tmp/opencode/` or
  `$JADAL_CACHE`, never git; no corpus downloads.
* ≤4 worker processes, `nice -n 10`, no GPU saturation; the GPU already carries ~1 GiB from
  other tools.
* OpenCV is available in the project venv → decode-dependent experiments (per-frame features,
  mosaic pixels) are **not** blocked; run them with `/home/parshu/envs/jadal/bin/python`.

## 2.7 Pre-existing test-suite red (not caused by any DSH stream)

`python -m unittest discover -s tests` under `/home/parshu/envs/jadal/bin/python` gives
**781 tests, 0 failures, 1 error, 11 skipped** on `dsh/grid-feats`. The single error is
`tests/test_exp_motion_cue.py:14` importing `pytest`, which is not installed in the venv; that file is
byte-identical to `origin/main`, so this is a **pre-existing red on main** that every stream hits.
The brief's note that "the `pico_status` example test fails without a Pico" is also true but is
skipped here rather than failing. 13 of the 781 tests are new (`tests/test_mosaic_features.py`).
The builder's hardened branch reported 773 tests with the same single error; 768 on main + 13 new = 781.
