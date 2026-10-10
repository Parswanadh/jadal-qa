# eval/results/ensemble: data provenance and licensing

Two data sources, two licence states:

- **IndEgo-derived** (may be shared under the project's IndEgo terms, private repo):
  `nested_loco.json`, `oof_probs.npz`, `determinism.json`, `PREREGISTRATION.md`,
  and `configs/ensemble_v1.json`.
- **Egocentric-10K-derived (licensed, PRIVATE repo only, excluded from public snapshots
  via `docs/PUBLIC_SNAPSHOT_EXCLUDE.txt`)**:
  - `ood_frames.npz`: despite the name, this holds EmbeddingGemma-2 per-frame embeddings
    (float16, shape 60 x 20 x 768) of 1200 Egocentric-10K frames. No images or video.
  - `ood_pixel.npz`: the 26 per-pixel features for the same 60 windows.
  - `ood_scores.json`, `ood_scores2*.json`: frozen-model scores and verdicts for those windows.
  - `transfer_probe.json`, `ood_heads.json`: analyses of the same windows.

No raw video or image is stored here. Human labels live outside the repo
(`JADAL_OOD_LABELS`, default `../ood_labels/labels.json`).
