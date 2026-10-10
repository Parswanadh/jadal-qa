# Z1.03 — NVDEC vs CPU decode at 1080p30 HEVC and 480p H.264, with/without hwdownload

**Why:** Settle where GPU decode wins: isolate the hwdownload cost and compare NVDEC vs CPU at both resolutions

**Hypothesis:** NVDEC beats CPU at 1080p30 HEVC but not at 480p H.264; hwdownload adds measurable time

**Data / label source:** 1080p30 HEVC factory clips + 480p10 H.264 IndEgo clip; 10-s windows

**Data regime:** 1080p30 HEVC + 480p H.264

**Method:** ffmpeg CPU vs -hwaccel cuda (+ hevc_cuvid/h264_cuvid) scale_cuda, with and without hwdownload (null sink); p50 over windows

**Owner:** l4-zerocopy  **Started:** 2026-10-10 18:47

## Results
(pending — `exp.py finish` fills the registry; put figures/JSON in this folder)

**Verdict (2026-10-10 21:14): adopt.** Per-window decode (incl. process startup): NVDEC wins at 1080p30 HEVC (832 ms vs CPU 4067 ms, 4.9x); CPU wins at 480p H.264 (228 ms vs NVDEC 705 ms, 3x). hwdownload cost scales with downloaded bytes: ~0 at 160x90, 3564 ms for 300 native 1080p frames.

Results file: eval/results/exp_zerocopy/Z1.03.json

Decision: NVDEC for high-res HEVC; CPU for low-res H.264.

Cost: n=8 windows per arm.
