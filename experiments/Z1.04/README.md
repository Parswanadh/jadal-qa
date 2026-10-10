# Z1.04 — Upload-spec ladder: codec/resolution/CRF vs decode + zero-copy throughput

**Why:** If the deployment can choose the upload encoding, the codec/resolution/quality is free and may dominate decode cost and end-to-end throughput

**Hypothesis:** Lower-res H.264 uploads cut decode cost and raise footage-h/wall-h; the NVDEC-vs-CPU advantage flips toward CPU at low resolutions

**Data / label source:** 30-window sample (1080p30 HEVC source), transcoded to 5 upload specs; label_source none

**Data regime:** 1080p30 HEVC source; upload specs 1080p HEVC, 1080p H.264 CRF23, 480p H.264 CRF23, 360p H.264 CRF28, 240p H.264 CRF28

**Method:** transcode 10-s segments per spec; measure NVDEC (with/without hwdownload) and CPU decode; compose total from measured resolution-invariant feature/EG2 costs; footage-h/wall-h per spec

**Owner:** l4-zerocopy  **Started:** 2026-10-10 19:02

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)
