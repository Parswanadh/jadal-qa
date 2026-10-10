# Spectral study — harness-scored comparison

_Generated 2026-10-10T10:00:22Z by eval/spectral/score.py. Every number below comes from eval/results/spectral/summary.json._

## Question

Does per-pixel frequency/temporal-spectral analysis of RGB video (arm B), a bounded hyperspectral check (arm C), a generic VLM (arm D), or the EmbeddingGemma-2 head (arm E) beat the current OpenCV+MediaPipe L0 signals (arm A) at (1) near-duplicate detection and (2) deciding whether a 10-s segment is USABLE work footage — within a 10,000 h in 2–3 h throughput budget, whole-video single-pass? Sampling 4–8 frames per segment is NOT acceptable as the main method.

## Data & protocol

- HUMAN GOLD (live IndEgo labels, snapshot 2026-10-10T09:53:36.001340+00:00, sha256 0bc9e0b8f348…, n_good=65, n_bad=14). CV: leave-one-clip-out; AUROC with clip-level bootstrap CI (B=1000). A CI spanning 0.5 means 'no evidence'. Awign data differs: nothing here is evidence on Awign footage.

- SYNTHETIC (by construction, label says nothing about activity): good-labelled segments + 9 degradations; sensitivity at 5% FAR threshold set on gold-good scores.

- DUPLICATES (by construction): threshold on held-out half 1, P/R/F1 on half 2, per method and per perturbation. Horizontal-flip excluded (NOT a duplicate).

## Results per task

| arm | gold AUROC (95% CI) | bal.acc | synth sens. (median over degs) | dups F1 | status |
|---|---|---|---|---|---|
| A | — | — | — | — | NOT RUN |
| A-full | — | — | — | — | NOT RUN |
| E | — | — | — | — | NOT RUN |
| B | — | — | — | — | NOT RUN |
| D | — | — | — | — | NOT RUN |

### Gold detail (per arm, LOCO folds)

### Synthetic detail (sensitivity at 5% FAR)

**A** `{"status": "NOT RUN"}`

**E** `{"status": "NOT RUN"}`

**B** `{"status": "NOT RUN"}`

**C** `{"status": "NOT RUN"}`

**D** `{"status": "NOT RUN"}`

**A-full** `{"status": "NOT RUN"}`

### Duplicates detail (held-out P/R/F1)

**A** `{"status": "NOT RUN"}`

**E** `{"status": "NOT RUN"}`

**B** `{"status": "NOT RUN"}`

**C** `{"status": "NOT RUN"}`

**D** `{"status": "NOT RUN"}`

## Cost & Pi-class feasibility

_Single-core wall time normalised by a CPU benchmark (matmul-512 reps/s; see decode_floor.json). NOT measured on a Pi — numbers scale by benchmark ratio,/clock, and memory bandwidth._

| arm | s/segment (1 core) | RTF | cores for 10kh/2h | /3h | edge 50x | live 1x |
|---|---|---|---|---|---|---|
| A | None | None | None | None | None | None | NOT RUN |
| E | None | None | None | None | None | None | NOT RUN |
| B | None | None | None | None | None | None | NOT RUN |
| C | None | None | None | None | None | None | NOT RUN |
| D | None | None | None | None | None | None | NOT RUN |
| A-full | None | None | None | None | None | None | RUN |

## Streaming


**A** `{"status": "NOT RUN"}`

**E** `{"status": "NOT RUN"}`

**B** `{"status": "NOT RUN"}`

**C** `{"status": "NOT RUN"}`

**D** `{"status": "NOT RUN"}`

**A-full** `{"status": "NOT RUN"}`

## Hyperspectral verdict

`{"status": "NOT RUN"}`

## VLM verdict

`{"status": "NOT RUN"}`

## Recommendation for Jadal

_TBD once arms B and D report; current partial read below._

No arms scored yet.

## Limitations

- Only 3 IndEgo clips labelled so far; 14 bad labels → CIs are wide; any CI spanning 0.5 = no evidence.

- Synthetic degradations measure capture-quality sensitivity only, NOT activity judgement.

- Cost numbers are workstation single-core, not a Pi; scaled by benchmark.

- Human labels are IndEgo workers; Awign footage differs.

