# Pod experiment wave summary

Status: **skeleton** (2026-10-10). The lead fills the table from the merged registry
after wave 1; nothing in the result columns is a claim until it points to a results file.

Sources for every row: `experiments/registry.jsonl` (append-only union of
`experiments/registry_snapshot.jsonl` and the live per-worktree registries — merge with
`python3 tools/pod/union_registry.py`), the per-experiment pages written by `tools/exp.py new`
(`experiments/<id>/README.md`; only ids seeded from the snapshot exist here so far, so the
table points at `experiments/registry.jsonl` and the report files that do exist), and the
results JSON / output file named in the last column. Do not put a number in this table
until its file exists in the repo or is named as a pod path.
The rendered log is `experiments/EXPERIMENTS.md`.

## Wave 1

The table below intentionally lags `experiments/registry.jsonl` until the lead fills the
result columns from the merged wave-1 registries: rows whose registry line already has a
verdict (E0.01 `maybe`, E1.01/E1.02, E3.01-E3.05) still show `pending` here because their
result files live in the other worktrees/bundle, not in this one.

| id | verdict | headline number with CI | cost | regime | file |
|---|---|---|---|---|---|
| E0.01 | pending | pending | pending | 480p/10fps IndEgo | experiments/registry.jsonl (E0.01); prose: none in this worktree yet |
| E1.01 | pending | pending | pending | 480p 10fps H264 in-domain | experiments/registry.jsonl (E1.01); prose: none in this worktree yet |
| E1.02 | pending | pending | pending | 480p 10fps in-domain | experiments/registry.jsonl (E1.02); prose: none in this worktree yet |
| E1.03 | pending | pending | pending | 1080p 30fps HEVC out-of-domain | experiments/registry.jsonl (E1.03); prose: none in this worktree yet |
| E1.04 | pending | pending | pending | 480p 10fps in-domain | experiments/registry.jsonl (E1.04); prose: none in this worktree yet |
| E3.01 | pending | pending | pending | 480p/10fps IndEgo assembly+logistics | experiments/registry.jsonl (E3.01); docs/reports/exp_temporal.md |
| E3.02 | pending | pending | pending | 1080p/30fps Egocentric-10K factory | experiments/registry.jsonl (E3.02); docs/reports/exp_temporal.md |
| E3.03 | pending | pending | pending | 480p/10fps IndEgo | experiments/registry.jsonl (E3.03); docs/reports/exp_temporal.md |
| E3.04 | pending | pending | pending | 1080p/30fps Egocentric-10K factory | experiments/registry.jsonl (E3.04); docs/reports/exp_temporal.md |
| E3.05 | pending | pending | pending | IndEgo + Egocentric-10K | experiments/registry.jsonl (E3.05); docs/reports/exp_temporal.md |
| E4.01 | pending | pending | pending | mixed: 480p/10fps H264 vs 1080p/30fps HEVC | experiments/registry.jsonl (E4.01); docs/reports/exp_regimes.md |
| E4.02 | pending | pending | pending | terciles of sharpness/lighting/motion | experiments/registry.jsonl (E4.02); docs/reports/exp_regimes.md |
| E4.03 | pending | pending | pending | 480p/10fps transcode vs 1080p/30fps source | experiments/registry.jsonl (E4.03); docs/reports/exp_regimes.md |
| E4.04 | pending | pending | pending | all | experiments/registry.jsonl (E4.04); docs/reports/exp_regimes.md |

Tracks without a wave-1 registry entry (E2 mosaic, E5 domain adaptation, E6 motion cue,
E7 cost model, E8 re-scoring) get their own rows when the lead registers them; the mosaic
spec is `.opencode/skills/jadal-mosaic-spec/SKILL.md`.

Reproduced baselines (already runnable, not wave results — they are the reference the
wave is compared against, n = 150 human-labelled IndEgo windows, leave-one-clip-out):

| baseline | number | file |
|---|---|---|
| 26 per-pixel features + logistic (C=0.3) | AUROC 0.810 | `experiments/baselines/2026-10-10_baseline_logistic_loco.txt` |
| EmbeddingGemma-2 window embedding + logistic (C=0.01) | AUROC 0.848 [0.734, 0.909] | `experiments/baselines/2026-10-10_ensemble_views_exploratory.txt` |
| probability average of the two views | AUROC 0.853 [0.731, 0.922] | same file |

## The honesty ceiling (standing paragraph, do not drop)

The numbers above are measured on **150 ten-second windows from 7 IndEgo clips and ~3
wearers, labelled by one human** ([README.md](../../README.md), [`docs/BASELINE.md`](../BASELINE.md)
line 16: gold labels "reflect one labeller on a few IndEgo workers"), so they are
descriptive of this small set, not of head-camera video in general; window-level accuracy
is **not high** yet. The out-of-domain set is **n_bad = 9** (60 human-labelled
Egocentric-10K windows: 51 good / 9 bad, [`docs/reports/exp_temporal.md`](../reports/exp_temporal.md)),
so any out-of-domain statement is exploratory and a "no effect" result there is absence of
evidence, not evidence of absence ([AGENTS.md](../../AGENTS.md) rule 1). IndEgo-trained
scorers did not transfer out of domain (**0.4–0.53 AUROC**; the two component numbers are the
shipped PoC cascade 0.529, window-bootstrap CI [0.32, 0.74], and the "work" frame head 0.41,
recorded in the pod bundle at `/workspace/jadal_pod_bundle/docs/FACTS.md` lines 66 and 50 —
**pod path, not an in-repo results file yet**, exactly like the motion-cue correction below) —
never claim transfer that has not been tested. The motion/stability cue predicts *usable* windows; **part of that cue may
be head stability, it is not "hand work is detected"** (correction dated 2026-10-10 18:25
IST in the pod bundle's `/workspace/jadal_pod_bundle/docs/FACTS.md`; **no in-repo results
file backs this correction yet**, so it is recorded here as a correction to earlier wording,
not as a measured claim). Receipts are the **content hash** (SHA-256 of canonical JSON) of
the decision, **not signed** ([README.md](../../README.md), `jadal/receipts.py`); the
cameras are RGB, **no hyperspectral** data ([README.md](../../README.md) line 18); the
pipeline has **never been measured on a Raspberry Pi** ([`docs/reports/edge.md`](../reports/edge.md)).
