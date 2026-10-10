# D1 — Egocentric-10K corpus expansion

_Generated 2026-10-10T19:18:39.241754+00:00 by `eval/scale/make_exp_corpus_report.py`._

Dataset: `builddotai/Egocentric-10K` (gated). Selection rule: SELECT-MORE: new-factories-first seeded round-robin, smallest-first, 42 GB cap
(seed 42). Media lives under
`/workspace/jadal_pod_bundle/data/ego10k_scale/` and is **not** committed.

## New footage

| metric | value |
|---|---|
| tars downloaded (planned) | 153 / 153 |
| new clips | 333 |
| new 10 s windows | 8211 |
| new footage hours | 23.258 |
| new tars bytes | 41.92 GB |
| factories in new set | 38 |
| **NEW factories gained** | **38** |
| new workers | 147 |
| ffprobe failures | 0 |

Factories gained: factory_002, factory_004, factory_005, factory_006, factory_007, factory_008, factory_011, factory_012, factory_015, factory_017, factory_018, factory_025, factory_027, factory_030, factory_031, factory_032, factory_034, factory_043, factory_045, factory_047, factory_048, factory_049, factory_050, factory_052, factory_055, factory_059, factory_063, factory_064, factory_065, factory_067, factory_071, factory_072, factory_074, factory_075, factory_078, factory_081, factory_082, factory_083

## Total corpus (existing + new)

| metric | value |
|---|---|
| clips | 391 |
| 10 s windows | 9757 |
| footage hours | 27.662 |
| factories | 85 |

## Download health

| metric | value |
|---|---|
| throughput | 93.7 MB/s |
| elapsed | 7.5 min |
| retries | 0 |
| failures | 0 |
| stopped early (disk guard) | False |
| disk free before | 73.0 GB |
| disk free after | 33.0 GB |
| free floor enforced | 15.0 GB |

## Artifacts

- provenance: `eval/results/exp_corpus/provenance.jsonl`
- extended manifest (new clips): `queues/manifest_scale_ext.json`
- full manifest (existing + new): `queues/manifest_scale_all.json`
- selection: `eval/results/exp_corpus/selection_more.json`
- verify summary: `eval/results/exp_corpus/download_verify_more.json`
- metrics: `eval/results/exp_corpus/D1.01.json`

## Resume

The download is resumable: tars already present in `provenance.jsonl` are
skipped, and partial HF downloads resume in place. To continue, re-run
`HF_HUB_DISABLE_XET=1 python eval/scale/dl_more.py` from the worktree, then
`python eval/scale/make_manifest_ext.py` and this report generator.
