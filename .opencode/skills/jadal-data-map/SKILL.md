---
name: jadal-data-map
description: Where the data, labels, caches and schemas live on the pod (bundle layout, label JSON schemas, queue files, manifests, environment variables). Load before touching data or writing a loader.
license: Apache-2.0
---
# Data map (pod). Env: `JADAL_BUNDLE=/workspace/jadal_pod_bundle`, `JADAL_DATA=$JADAL_BUNDLE/data`, `JADAL_EXP_DIR=<repo>/experiments`.
- `labels/manifest_gold.json`: 150 IndEgo windows `{items:[{id,clip,path,start_s,end_s,label(1=usable,0=bad),reason}]}`; clips ecu_full, ecu_late, desk, pallet, pallet2, organise, logistics4; reasons NO_ACTIVE_WORK 23, HANDS_NOT_VISIBLE 14, DUPLICATE 6, TOO_DARK 6, UNDECODABLE 1. Videos in `data/indego/...` (480p ~10 fps H.264).
- `labels/ood_labels.json` (60 random Egocentric-10K windows, 51 good/9 bad) and `labels/ood_labels2.json` (100 more, arrives later): schema `{history:[...], latest:{segment_id:{verdict:"good"|"bad", reason, note, ts}}}` — iterate `latest.items()`.
- `queues/label_queue_random{,2}.json`: `{segments:[{segment_id, path(480p 10 s transcode), source_path + source_start_s (original 1080p/30 fps HEVC clip), factory, worker}]}`; `queues/selection_small.json` (47 tars) and `queues/manifest_scale.json` (1,546 windows of 58 clips; fields id, clip, path, start_s, end_s, factory, worker, fps, width, height).
- `caches/perpixel/{series_gold.npz (150x100x8 per-frame channels: luma_mean, lap_var, diff_e, skin_frac, shift_px, under_frac, over_frac, dct_hf), gold.npz (150x121 features), physics_feats.npz}`; `caches/gpu/features.json` (26 per-pixel features per window: `windows[id]['full'][name]`); `caches/frames10k/segments.npz` (F (150,21,768) fp16 EmbeddingGemma-2 embeddings of 21 frames per window at 0.5 s spacing; ids, clips, labels, counts); `caches/frames10k/egocentric_10k_frame_embeddings/` (10k AI-labelled factory frames); `caches/ensemble/{oof_probs.npz, ood_scores.json, nested_loco.json, ensemble_v1.json}`.
- AI-labelled frames (NOT human): `Egocentric-10K-Evaluation` parquet (hand_count 0/1/2, active_labor yes/no). Original Egocentric-10K clips: fetched on the pod into `$JADAL_DATA/ego10k_scale/clips/` by `dl_small.py` (HF gated dataset).
- All JSON paths were remapped by `scripts/remap_paths.py`; if you still see `/home/parshu`, remap again.
