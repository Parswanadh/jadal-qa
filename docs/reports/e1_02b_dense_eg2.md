# E1.02b — denser EmbeddingGemma-2 sampling (5/10 fps) vs the 2 fps view

Claim: **LB1-E1.02b** ("5/10 fps re-embed only if GPU time remains" — it remained).
Results: `eval/results/exp_fps/e1_02b.json`. Device: **A40** (idle, `flock` slot 1).

## Verdict

**drop — and the difference is NOT established.**

Denser EG2 window sampling does not beat the 2 fps view on the 150 IndEgo human
windows. Every paired clip-bootstrap CI includes 0, the point estimates
(+0.0006 / −0.0002 against the published 2 fps cache; +0.0018 / +0.0010 against a
same-pass 2 fps control) are at or below the pipeline-noise floor, and the
denser views cost 2.5x (5 fps) and 5.0x (10 fps) the embedding work. "drop" is a
cost-effectiveness decision, not a claim that denser sampling is worse: the CIs
exclude an improvement larger than ~+1.5 AUROC points, they do not exclude
smaller ones.

## Numbers (n = 150, 100 good / 50 bad, 7 clips, human gold labels, one labeller)

LOCO over the 7 clips; L2-normalised window mean of per-frame L2-normalised EG2
embeddings; StandardScaler on the train fold; logistic C=0.01, class_weight
balanced; clip-bootstrap B=2000 seed 0, **one shared resample set for every row**
(so rows and deltas are paired by construction).

| density | frames/window | AUROC | 95% CI | bal. acc |
|---|---|---|---|---|
| 2 fps — published cache (`segments.npz`) | 20 | 0.8480 | [0.733, 0.910] | 0.770 |
| 2 fps — same-pass control (rows ::5) | 20 | 0.8468 | [0.733, 0.909] | 0.765 |
| 5 fps (rows ::2) | 50 | 0.8486 | [0.743, 0.910] | 0.765 |
| 10 fps (all native rows) | 100 | 0.8478 | [0.739, 0.910] | 0.765 |

Paired deltas from the same shared bootstrap:

| comparison | Δ AUROC | 95% CI | P(Δ>0) | established |
|---|---|---|---|---|
| 5 fps − 2 fps (published) | +0.0006 | [−0.0051, +0.0140] | 0.665 | no |
| 10 fps − 2 fps (published) | −0.0002 | [−0.0082, +0.0127] | 0.583 | no |
| 2 fps same-pass − 2 fps published — **noise floor** | −0.0012 | [−0.0048, +0.0049] | 0.311 | no |
| 5 fps − 2 fps (same pass) — density only | +0.0018 | [−0.0045, +0.0151] | 0.724 | no |
| 10 fps − 2 fps (same pass) — density only | +0.0010 | [−0.0073, +0.0127] | 0.663 | no |
| 10 fps − 5 fps | −0.0008 | [−0.0069, +0.0024] | 0.276 | no |

The noise-floor row is why the control matters: re-embedding the *same frames*
in a second pass moves the AUROC by 0.0012, i.e. the same order as the whole
density effect. Together with E1.02 (1 fps 0.8596, 0.5 fps 0.8702, CIs
overlapping), the fps axis is flat from 0.5 to 10 fps within roughly ±0.02.

## Method

One GPU pass, `eval/exp_fps/embed_dense_eg2.py`:

* seek once to `start_s`, read 100 consecutive native frames (all clips are
  exactly 10.0 fps), 15 000 frames, 0 empty windows;
* same preprocessing as `eval/frames10k/embed_segments.py`: BGR→RGB, PIL RGB,
  `thumbnail(512, 512, BILINEAR)`;
* EG2 `SentenceTransformer.encode(normalize_embeddings=True, image
  max_soft_tokens=140)`, bf16 (revision `914f7f89142e33e77833254d9c9b90c3cef7303b`,
  `model.safetensors` sha256 `197a3296…3d79`), batch 16;
* 5 fps and 2 fps are **exact row subsamples** (::2, ::5) of that one pass, so
  the densities share one decode and one forward pass.

Scoring: `eval/exp_fps/run_e1_02b.py` (CPU, same `fps_lib` helpers as E1.02,
B=2000 instead of 1000).

Validation before trusting the comparison:

* the sequential decode at row index ::5 is **pixel-identical** to the published
  per-sample `CAP_PROP_POS_MSEC` seek decode (6/6 windows re-checked, including
  the five windows whose cache carries 21 frames; max abs pixel diff 0);
* per-frame cosine against the published 2 fps embeddings: mean 0.9947,
  min 0.9658 (same frames, different pass → bf16/batching numerics);
  window-mean cosine 0.9996;
* the published 2 fps cache re-scores to 0.8480 [0.733, 0.910], reproducing
  `e1_02.json` (0.8480 [0.734, 0.909] at B=1000) exactly.

## Cost (measured on this A40, batch 16, VRAM peak 2.17 GB)

26.9 frames/s encode + 5.2 ms/frame decode, 15 000 frames = 556.6 s encode +
78.5 s decode. Per 10-s window: 2 fps 0.85 s, 5 fps 2.12 s (2.5x),
10 fps 4.23 s (5.0x). The product therefore keeps the 2 fps EG2 view.

## Limits

* 7 clips → clip-bootstrap CIs are wide (~±0.01 on the deltas); the experiment
  can exclude gains above ~+0.015, not smaller ones.
* n = 150 windows, one labeller, one domain (IndEgo 480p/10 fps H.264). Nothing
  here says anything about the 1080p/30 fps out-of-domain regime.
* Only the mean-pooling recipe was tested; denser sampling under a different
  pooling (e.g. max or a temporal head) is untouched by this result.

## Provenance

* embeddings are **not** committed: `/workspace/cache_e1_02b/dense10.npz`
  (17.9 MB, regenerate with `eval/exp_fps/embed_dense_eg2.py`, ~11 min GPU);
  meta in `/workspace/cache_e1_02b/meta.json`.
* `manifest_gold.json` was restored to the pristine source-machine-paths copy at
  17:19 UTC, after the embed run started (17:16:59). The run read the earlier
  path-remapped copy and stored `labels_sha256` = `43ab44e5…`; the pristine file
  carries the **same** internal `labels_sha256` and identical ids/clips/labels
  (asserted at scoring time; file sha256 at scoring `44c7217b…`, matching
  `SHA256SUMS.txt`). Paths are remapped in memory only, 150/150 resolve. The
  scored labels are the pristine ones.
