---
name: jadal-mosaic-spec
description: Exact specification of the mosaic ("many frames in one image") input: grid sizes, tile sizes, timestamp placement top-right of each tile, JSON sidecar, determinism tests, variants and decision rule. Load for experiment track E2.
license: Apache-2.0
---
- Frames: N in {4, 6, 9, 12, 16} at uniform times over the 10-s window (index rule documented; same rule for all datasets). Grids: 2x2, 2x3, 3x3, 3x4, 4x4, row-major time order (top-left earliest).
- Tile size 112/160/224 px (area interpolation, aspect preserved with letterbox); 2 px gutter; background mid-grey.
- **Timestamp**: text `t=3.5s` in the TOP-RIGHT corner of EVERY tile: fixed font (PIL `ImageFont.load_default()` or a bundled TTF at a fixed size), white text on a dark semi-transparent box, readable at 112 px; the position/format are part of the spec and tested.
- **Sidecar** `<window_id>.json`: `{window_id, source, grid:[rows,cols], tile_px, fps_used, tiles:[{index,t_s,src_frame_index,x,y,w,h}], mosaic_sha256, builder_version}`; the receipt hashes the mosaic bytes and the sidecar. Byte-identical PNG across repeated builds (fixed encoder params).
- Variants (each its own registry id): mean-of-frame embeddings (control), mosaic embedding, shuffled tile order, no timestamps, single middle frame, motion mosaic (|I_t - I_{t-1}| tiles), pooled per-tile statistics control. Datasets: IndEgo LOCO + OOD labels; fps {1,2,3,5,10}.
- Decision: adopt as MODEL INPUT only if within 0.02 AUROC of mean-of-frames at <= 1/4 the cost on BOTH IndEgo-LOCO and OOD; otherwise adopt only as the human-evidence artefact (dashboard 'why' view).
