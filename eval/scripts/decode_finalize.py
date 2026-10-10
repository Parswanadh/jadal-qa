#!/usr/bin/env python3
"""LB1-decode, part 7: turn the raw measurements into explicit claim verdicts.

Reads the merged results JSON and adds:
  * `parity_threshold_reasoning` - why the pass criterion is what it is and
    where the observed numbers sit relative to it;
  * `facts_comparison`           - each FACTS.md decode claim vs this pod;
  * `contention_replicate`       - a second, co-tenant-affected measurement of
    the same concurrency loop (transcribed from the run log, see provenance);
  * `new_findings`               - things measured that FACTS.md does not state.

Run:  python3 eval/scripts/decode_finalize.py <results.json>
"""
from __future__ import annotations

import json
import sys
import time

FACTS_LINE_24 = ("NVDEC (GPU decode) is NOT faster than CPU decode for a "
                 "single 480p stream (2,348 vs 2,820 fps); this chip has one "
                 "NVDEC engine. [laptop RTX 4070]")
FACTS_LINE_52 = ("GPU on real 1080p/30 fps HEVC (Egocentric-10K): one CPU core "
                 "decodes 43-55 fps (RTF 1.4-1.8); NVDEC decodes 620-730 fps "
                 "(RTF 21-24) and saturates at one engine (flat across 1-4 "
                 "concurrent sessions). [laptop RTX 4070]")


def main(path):
    with open(path) as f:
        d = json.load(f)
    tp = d["throughput"]
    pa = d["parity"]
    pp = d.get("pipeline_parity", {}).get("cases", {})
    sd = d.get("scorer_delta", {})
    sds = d.get("scorer_delta_scale_cuda_path", {})
    at = d.get("nv12_path_attribution", {}).get("cases", {})
    ext = d.get("extensions", {})

    e480, e1080 = pp.get("indego_480", {}), pp.get("ego10k_1080", {})
    c480, c1080 = at.get("indego_480", {}), at.get("ego10k_1080", {})

    d["parity_threshold_reasoning"] = {
        "criterion_pre_registered": (
            "PASS if per-pixel max|delta| <= 2 LSB and mean|delta| <= 0.5 LSB, "
            "because conformant H.264/HEVC decoders may differ by +/-1-2 LSB in "
            "the inverse transform; anything larger could move a threshold "
            "decision in the downstream scorer. A bit-exact result passes "
            "strictly and makes the threshold non-load-bearing."),
        "observed_full_res": {
            "n_spans": len(pa),
            "spans": list(pa.keys()),
            "max_abs_diff_all_spans": [pa[k]["nvdec_cuvid_comparison"]["max_abs_diff"]
                                       for k in pa],
            "mae_all_spans": [pa[k]["nvdec_cuvid_comparison"]["mae_all_planes"]
                              for k in pa],
            "hwaccel_path_also_bit_exact": all(
                pa[k]["nvdec_hwaccel_comparison"]["max_abs_diff"] == 0 for k in pa),
            "reference_frame_to_frame_mae_lsb": {
                k: pa[k]["reference_frame_to_frame_mae_cpu"] for k in pa},
        },
        "verdict": ("STRICT PASS, bit-exact (max|delta| = 0, mean|delta| = 0.0) "
                    "on every span and both NVDEC invocation paths. The decoder "
                    "difference is 0% of the 2.4-8.1 LSB frame-to-frame signal "
                    "the scorer sees."),
        "verdict_path_evidence": {
            "cuvid_path_max_abs_delta_p_pixel": sd.get("summary", {}).get(
                "max_p_pixel_abs_delta"),
            "n_windows": sd.get("summary", {}).get("n_windows"),
            "n_flips": sd.get("summary", {}).get("n_verdict_flips_at_0.5"),
            "note": "6 x 10 s windows, 160x90 working resolution, frozen "
                    "configs/ensemble_v1.json pixel view; includes a window at "
                    "p=0.4556, only 0.044 from the 0.5 boundary.",
        },
        "not_negligible_other_path": {
            "chain": "eval/gpu/decode.py NVDEC_CHAIN "
                     "(-hwaccel cuda -hwaccel_output_format cuda + scale_cuda "
                     "+ numpy nv12->rgb)",
            "work_res_mae_lsb": {
                "indego_480": e480.get("nvdec_scale_cuda_nv12_vs_cpu", {}).get("mae"),
                "ego10k_1080": e1080.get("nvdec_scale_cuda_nv12_vs_cpu", {}).get("mae"),
            },
            "max_p_pixel_abs_delta": sds.get("summary", {}).get(
                "max_p_pixel_abs_delta"),
            "attribution": {
                "decoder_alone_matched_resampler": {
                    "indego_480": c480.get("c3_decoder_cuvid_vs_cpu_matched_bilinear", {}).get("mae"),
                    "ego10k_1080": c1080.get("c3_decoder_cuvid_vs_cpu_matched_bilinear", {}).get("mae"),
                },
                "resampler_bicubic_vs_bilinear": {
                    "indego_480": c480.get("c1_resampler_bicubic_vs_bilinear", {}).get("mae"),
                    "ego10k_1080": c1080.get("c1_resampler_bicubic_vs_bilinear", {}).get("mae"),
                },
                "nv12_numpy_rgb_vs_swscale_bgr24": {
                    "indego_480": c480.get("c2_conversion_nv12numpy_vs_swscale_bgr24_same_scale", {}).get("mae"),
                    "ego10k_1080": c1080.get("c2_conversion_nv12numpy_vs_swscale_bgr24_same_scale", {}).get("mae"),
                },
                "scale_cuda_resize_residual": {
                    "indego_480": c480.get("c4_scale_cuda_vs_cpu_bilinear_same_conversion", {}).get("mae"),
                    "ego10k_1080": c1080.get("c4_scale_cuda_vs_cpu_bilinear_same_conversion", {}).get("mae"),
                },
                "conclusion": "the NVDEC *decoder* is exact (c3 = 0.0000); the "
                              "difference in this path comes from the GPU "
                              "resizer + the hand-rolled nv12->rgb conversion, "
                              "not from NVDEC.",
            },
        },
    }

    c1 = tp["indego_480_cpu_1thread"]["fps"]
    c1p = tp["indego_480_cpu_process_auto"]["fps"]
    n480 = tp["indego_480_nvdec_cuvid"]["fps"]
    c10 = tp["ego10k_1080_cpu_1thread"]
    n10 = tp["ego10k_1080_nvdec_cuvid"]
    conc = tp["nvdec_1080p_concurrency"]
    agg = {k: v["aggregate_fps"] for k, v in conc.items()
           if isinstance(v, dict) and "aggregate_fps" in v}
    ext_conc = ext.get("nvdec_concurrency_1080p", {})
    agg_ext = {k: v["aggregate_fps"] for k, v in ext_conc.items()}
    win = ext.get("pipeline_window_wall_time", {})
    threads = ext.get("cpu_thread_scaling_1080p_30s_span", {})

    d["facts_comparison"] = {
        "claim_1_480p_nvdec_not_faster_than_cpu": {
            "facts_text": FACTS_LINE_24,
            "facts_device": "laptop RTX 4070",
            "this_device": "NVIDIA A40 (pod)",
            "comparable": "different device (and a different 480p: FACTS does "
                          "not say the resolution; here it is 480x480@10fps)",
            "measured_decode_only_fps": {
                "cpu_1_thread": c1, "cpu_1_process_auto_threads": c1p,
                "nvdec_hevc_cuvid": n480},
            "measured_end_to_end_2s_window_480p_s": {
                "cpu": win.get("480p_window2s_cpu", {}).get("wall_s_median"),
                "nvdec_cuvid": win.get("480p_window2s_nvdec_cuvid", {}).get("wall_s_median"),
                "nvdec_scale_cuda": win.get("480p_window2s_nvdec_scale_cuda", {}).get("wall_s_median")},
            "verdict": "NOT REPRODUCED AS STATED (ordering reverses: NVDEC is "
                       "3.1-3.7x FASTER than CPU decode-only on this pod). "
                       "However the SPIRIT holds for the pipeline's actual unit "
                       "of work: for a 2 s 480p window spawned as one ffmpeg "
                       "process, CPU is 5-6x faster end-to-end because NVDEC "
                       "pays ~0.61 s of process/CUDA init against ~0.08 s for CPU.",
        },
        "claim_2_1080p_cpu_core_43_55fps": {
            "facts_text": "one CPU core decodes 43-55 fps (RTF 1.4-1.8) on real "
                          "1080p/30 HEVC",
            "facts_device": "laptop RTX 4070 (host CPU)",
            "this_device": "pod server CPU, NVIDIA A40 pod",
            "comparable": "different CPU / different device",
            "measured": {"fps_full_clip": c10["fps"], "rtf_full_clip": c10["rtf"],
                         "fps_30s_span": threads.get("threads_1", {}).get("fps"),
                         "rtf_30s_span": threads.get("threads_1", {}).get("rtf")},
            "verdict": "DIRECTION CONFIRMED, MAGNITUDE DIFFERENT (higher here): "
                       "1 core is 1.5-3.9x real time, vs 1.4-1.8x on the laptop.",
        },
        "claim_3_1080p_nvdec_620_730fps": {
            "facts_text": "NVDEC decodes 620-730 fps (RTF 21-24) on real "
                          "1080p/30 HEVC",
            "facts_device": "laptop RTX 4070",
            "this_device": "NVIDIA A40 (pod)",
            "comparable": "different device; measured range brackets the FACTS "
                          "range, so the FACTS number is plausible on the A40 too",
            "measured_fps": {"hevc_cuvid_A": n10["fps"],
                             "hevc_cuvid_B": ext.get("nvdec_paths_1080p", {}).get("nvdec_cuvid", {}).get("fps"),
                             "hwaccel_auto": ext.get("nvdec_paths_1080p", {}).get("nvdec_hwaccel_auto", {}).get("fps"),
                             "hwaccel_threads1": ext.get("nvdec_paths_1080p", {}).get("nvdec_hwaccel_t1", {}).get("fps")},
            "measured_rtf": {"hevc_cuvid_A": n10["rtf"],
                             "hevc_cuvid_B": ext.get("nvdec_paths_1080p", {}).get("nvdec_cuvid", {}).get("rtf")},
            "verdict": "SAME ORDER OF MAGNITUDE / DIRECTION CONFIRMED on this "
                       "device (NVDEC 11-13x faster than 1 CPU core). Absolute "
                       "fps are device-specific. NOTE: `-hwaccel cuda` combined "
                       "with `-threads 1` costs ~40% of throughput; do not "
                       "benchmark that combination.",
        },
        "claim_4_nvdec_flat_across_1_to_4_sessions": {
            "facts_text": "NVDEC ... saturates at one engine (flat across 1-4 "
                          "concurrent sessions)",
            "facts_device": "laptop RTX 4070",
            "this_device": "NVIDIA A40 (pod)",
            "comparable": "different device",
            "measured_aggregate_fps_by_sessions": {
                "run_A_1080p": agg, "run_B_1080p": agg_ext},
            "verdict": "NOT REPRODUCED on this device: aggregate throughput is "
                       "NOT flat across 1-4 sessions. Run A: 1->4 sessions gives "
                       "2.4x the aggregate; run B (with 8 sessions) shows the "
                       "plateau is at 4-8 sessions. A single ffmpeg+NVDEC "
                       "session reaches only ~40-45% of the achievable NVDEC "
                       "aggregate here, so 'one session saturates the engine' "
                       "is false for this pod.",
        },
        "claim_5_this_chip_has_one_nvdec_engine": {
            "facts_text": "this chip has one NVDEC engine",
            "verdict": "COULD NOT VERIFY from the pod. nvidia-smi exposes no "
                       "NVDEC engine count (only SM utilisation, which stays "
                       "0-11% during pure NVDEC work) and no other tool is "
                       "available. The observed scaling is inconsistent with a "
                       "single session already saturating a single engine.",
        },
        "claim_6_gpu_kernels_700_hours_per_hour": {
            "facts_text": "GPU kernels alone reach ~700 hours of video per hour "
                          "at 4.6 GiB",
            "verdict": "NOT TESTED - that is the feature/kernel stage, outside "
                       "the decode claim (LB1-decode covers decode only).",
        },
    }

    d["contention_replicate"] = {
        "provenance": ("second execution of the same concurrency loop at "
                       "17:24-17:30 UTC, transcribed from the run log; that "
                       "run crashed on an unrelated KeyError before writing "
                       "its JSON, so these three numbers exist only there"),
        "nvidia_smi_co_tenant_resident_mib": 944,
        "aggregate_fps_by_sessions": {"1": 440.8, "2": 710.5, "4": 1018.4},
        "per_session_fps": {"1": 440.8, "2": 355.3, "4": 254.6},
        "reading": ("the same nominal measurement moves by 2.3x (441 vs 1096 "
                    "fps for one session) depending on whether the co-tenant "
                    "GPU job is active; the *shape* (aggregate rising with "
                    "sessions) is the same in both runs, so the 'not flat' "
                    "finding is robust to contention, the absolute fps are not"),
    }

    d["new_findings"] = {
        "per_process_startup_cost_s": tp.get("process_startup_cost"),
        "startup_reading": ("NVDEC pays 0.61-0.97 s of ffmpeg+CUDA init per "
                            "process vs 0.08-0.15 s for CPU. Any design that "
                            "spawns one ffmpeg per short window (replay.py does) "
                            "pays this on every window; at 2 s / 480p it makes "
                            "NVDEC slower than CPU end-to-end."),
        "window_wall_time_s": win,
        "cpu_thread_scaling_1080p_30s_span": threads,
        "thread_scaling_reading": ("the software HEVC decoder peaks at 4 threads "
                                   "(336 fps) and DEGRADES at 8/16/auto "
                                   "(315/146/146 fps); `-threads 0` (ffmpeg "
                                   "auto) is a bad default on this pod."),
        "scale_cuda_pixel_view_impact": sds.get("summary"),
        "scale_cuda_reading": ("eval/gpu/decode.py's NVDEC chain is not "
                               "interchangeable with the CPU decode the "
                               "ensemble was frozen on: MAE 5-10 LSB at the "
                               "160x90 working resolution and up to 0.38 change "
                               "in the frozen pixel-view probability."),
        "nvdec_resource_use_4_concurrent_sessions": {
            "vram_peak_mib_total": 1672,
            "vram_per_session_mib": 418,
            "baseline_mib": 0,
            "nvidia_smi_utilization_gpu_pct": {"median": 79, "max": 89,
                                               "n_samples": 19, "interval_s": 0.5},
            "reading": ("4 concurrent hevc_cuvid sessions hold ~1.7 GiB and "
                        "drive nvidia-smi utilization.gpu to 79-89%; a naive "
                        "monitor reads that as 'GPU saturated' although the "
                        "work is decoder+copy, not SM compute. Sampled at "
                        "0.5 s, so peaks may be slightly under-read."),
        },
        "cuvid_reading": ("jadal/poc_stream/replay.py's chain "
                          "(-hwaccel cuda -c:v hevc_cuvid + CPU swscale) is "
                          "bit-exact against CPU decode and left the pixel-view "
                          "probability unchanged to 6 decimals on all 6 windows."),
    }

    d["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print("finalized", path)


if __name__ == "__main__":
    main(sys.argv[1])
