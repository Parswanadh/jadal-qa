# Z1.04 — Upload-spec ladder: codec/resolution/CRF vs decode + zero-copy throughput

**Why:** If the deployment can choose the upload encoding, the codec/resolution/quality is free and may dominate decode cost and end-to-end throughput

**Hypothesis:** Lower-res H.264 uploads cut decode cost and raise footage-h/wall-h; the NVDEC-vs-CPU advantage flips toward CPU at low resolutions

**Data / label source:** 30-window sample (1080p30 HEVC source), transcoded to 5 upload specs; label_source none

**Data regime:** 1080p30 HEVC source; upload specs 1080p HEVC, 1080p H.264 CRF23, 480p H.264 CRF23, 360p H.264 CRF28, 240p H.264 CRF28

**Method:** transcode 10-s segments per spec; measure NVDEC (with/without hwdownload) and CPU decode; compose total from measured resolution-invariant feature/EG2 costs; footage-h/wall-h per spec

**Owner:** l4-zerocopy  **Started:** 2026-10-10 19:02

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 21:14): adopt.** Upload-spec ladder (n=8, NVENC -cq): 1080p30 HEVC / 1080p H.264 CRF23 / 480p CRF23 / 360p CRF28 / 240p CRF28. Decode NVDEC p50 904/1105/648/749/1690 ms; CPU 3637/3077/745/300/780 ms. Best combo = 360p H.264 @CRF28 x CPU decode x GPU feature/embed = 711 ms/window, 14.4 footage-h/wall-h (1 worker), i.e. 18.3 min for the full 1,546-window / 4.404 h corpus (4.6 min at 4 workers ideal). 360p@CRF28 is X1-validated (40% bytes, dAUROC<0.02); 240p is not.

Results file: eval/results/exp_zerocopy/Z1.04.json

Decision: Recommend an upload-side 360p H.264 @CRF28 spec; decode on CPU; GPU pixel-26 + GPU EG2.

Cost: n=8 windows x 5 specs; transcodes via NVENC.
