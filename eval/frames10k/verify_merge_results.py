"""Merge the LB1-eg2-repro artifacts into the single deliverable JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

WT = Path("/workspace/wt/b-verify-eg2")
S = Path("/workspace/scratch_b_verify_eg2")
OUTD = WT / "eval" / "results" / "pod_verify"

sample = json.load(open(OUTD / "eg2_embedding_reproduction.json"))
full = json.load(open(OUTD / "eg2_embedding_reproduction_full.json"))
e2e = json.load(open(OUTD / "eg2_end_to_end_auroc.json"))
renorm = json.load(open(S / "renorm_compare.json"))
diag = json.load(open(S / "diagnosis.json"))
pix = json.load(open(S / "pixel_sensitivity.json"))

for name, src in [("eg2_diagnostics_variants.json", diag),
                  ("eg2_diagnostics_norm_invariant.json", renorm),
                  ("eg2_diagnostics_pixel_sensitivity.json", pix)]:
    (OUTD / name).write_text(json.dumps(src, indent=1))

res = {
    "task": "LB1-eg2-repro",
    "claim_tested": "the cached frames10k/human_segments/segments.npz EG2 view (source of "
                    "the 0.848 EG2-only and 0.8524 two-view ensemble AUROCs) regenerates "
                    "from the IndEgo source video at the documented recipe",
    "verdict": "REPRODUCED IN SUBSTANCE, NOT BIT-EXACTLY. All 150 windows re-embed from the "
               "source video at the documented recipe with identical frame counts (150/150, "
               "3005/3005 frames) and window cosine 0.99905-0.99979 (median 0.99962). The "
               "residual is a real but tiny per-frame directional difference (norm-invariant "
               "frame cosine median 0.9953, ||delta||~0.097) that is exactly the size of a "
               "2-3 LSB per-pixel difference - i.e. the signature of a different "
               "cv2/FFmpeg decoder build (or a re-encoded source copy), NOT of any recipe "
               "difference. Frame indices, frame counts, pooling, model revision, dtype, "
               "preprocessing size and the absence of a prompt were each verified to match. "
               "End-to-end the published numbers reproduce: eg2-only 0.848 -> 0.8468, "
               "two-view ensemble 0.8524 -> 0.8524.",
    "model": sample["model"],
    "recipe": full["recipe"],
    "recipe_source": "eval/frames10k/embed_segments.py (cache producer). Which variant ran "
                     "is recorded in configs/ensemble_v1.json: embed_command has NO "
                     "--gpu-decode flag, so the CPU cv2+PIL path is the documented producer; "
                     "the GPU-decode variant was additionally tested and is much worse "
                     "(see controls.gpu_decode_preprocessing_variant).",
    "prompt_convention": "none: no prompt/prefix/text anywhere in the cache-producing path. "
                         "EG2 is used purely as an image encoder via "
                         "SentenceTransformer.encode(list_of_PIL).",
    "n_windows": full["summary"]["n_windows"],
    "sampling_rule_primary": full["what"],
    "initial_sample": {
        "n_windows": sample["sample"]["n_windows"],
        "sampling_rule": sample["sample"]["sampling_rule"],
        "manifest_order_indices": sample["sample"]["manifest_order_indices"],
        "clips_covered": sample["sample"]["clips_covered"],
        "summary": sample["summary"],
        "note": "the 10-window sample was decoded from the pre-rewrite manifest paths "
                "(/workspace/jadal_pod_bundle/...); the full run decoded byte-identical "
                "copies (md5-checked) of the same 7 videos",
    },
    "manifest": full["manifest"],
    "video_inputs": full["video_inputs"],
    "cache": sample["cache"],
    "counts_vs_cache": full["counts_vs_cache"],
    "summary": full["summary"],
    "max_abs_elementwise_diff_normalised": full["summary"][
        "max_abs_elementwise_diff_normalised"],
    "per_window": full["per_window"],
    "five_worst_windows": full["five_worst_windows"],
    "frame_level_norm_invariant": renorm,
    "controls": {
        "batch_composition_only": {
            "what": "same 200 pixel-identical frames encoded twice with different batch "
                    "composition (sample run bs32 in window order vs full run in the cache's "
                    "own global order)",
            "frame_cosine_min": renorm["mine_vs_mine_fullbatch"]["frame_min"],
            "frame_cosine_median": renorm["mine_vs_mine_fullbatch"]["frame_median"],
            "window_cosine_min": renorm["mine_vs_mine_fullbatch"]["window_min"],
            "conclusion": "bf16 encoding on this pod is directionally deterministic "
                          "(frame cosine >= 0.999967); batching/fp16 norm jitter is "
                          "not the explanation for the cache residual",
        },
        "frame_index_shift": {
            "what": "+-1 video frame (0.1 s at 10 fps) re-decode of the 10 sample windows",
            "result": full["shift_control"],
            "conclusion": "a wrong frame index costs frame cosine ~0.77-0.80 median, two "
                          "orders of magnitude more than the observed residual; frame "
                          "indices are therefore identical, and the cache counts (20/21 per "
                          "window, 3005 total) are reproduced exactly",
        },
        "pixel_lsb_sensitivity": {
            "what": "uniform integer noise added to the same decoded thumbs before encoding",
            "result": pix,
            "conclusion": "the observed cache residual (frame cosine 0.99533 unperturbed, "
                          "i.e. ||delta||~0.097) matches the effect of ~2-3 LSB RMS "
                          "per-pixel noise (+-4 LSB -> frame 0.99006). A few-LSB decode "
                          "difference fully accounts for the residual.",
        },
        "gpu_decode_preprocessing_variant": {
            "what": "eval/frames10k/gpu_pre.py path, which UPSCALES 480x480 -> 512x512 "
                    "(the PIL thumbnail is a no-op at 480x480, the cuda path is not)",
            "result": diag["variants"]["D_gpu_preproc_512_bs32"],
            "conclusion": "clearly worse (window cosine 0.943-0.996 vs 0.99905-0.99979 for "
                          "the CPU path), so the cache was NOT produced with --gpu-decode; "
                          "this corroborates configs/ensemble_v1.json['embed_command']",
        },
        "bf16_vs_fp32": {
            "tested": False,
            "why": "fp32 weights alone are 2.98 GB, above the 3.5 GiB VRAM / 1.5 GB RAM "
                   "governors on this pod",
            "argument": "bf16 rounding of a unit vector is ||delta||~2e-3, ~50x smaller than "
                        "the observed ||delta||~0.097, and two different batchings of "
                        "identical pixels agree to 3e-5; dtype/kernel numerics therefore "
                        "cannot explain the residual",
        },
    },
    "end_to_end_auroc": e2e,
    "artifacts": {
        "deliverable": "eval/results/pod_verify/eg2_embedding_reproduction.json (this file)",
        "full_run": "eval/results/pod_verify/eg2_embedding_reproduction_full.json",
        "end_to_end": "eval/results/pod_verify/eg2_end_to_end_auroc.json",
        "diagnostics": ["eval/results/pod_verify/eg2_diagnostics_variants.json",
                        "eval/results/pod_verify/eg2_diagnostics_norm_invariant.json",
                        "eval/results/pod_verify/eg2_diagnostics_pixel_sensitivity.json"],
        "scripts": ["eval/frames10k/verify_cache_reproduction.py",
                    "eval/frames10k/verify_full_reproduction.py",
                    "eval/frames10k/verify_cache_diagnose.py",
                    "eval/frames10k/verify_pixel_sensitivity.py",
                    "eval/frames10k/verify_end_to_end_auroc.py"],
        "embeddings": "not committed (scratch: /workspace/scratch_b_verify_eg2/)",
    },
    "limitations": [
        "manifest_gold.json was rewritten at 2026-10-10T17:19:49Z to the cache producer's "
        "host paths (/home/parshu/projects/awing/...), which do not exist on this pod "
        "(0/150 resolve). Videos were resolved by basename to the bundle copies and "
        "md5-checked. Who rewrote it and why is outside this task's scope; the labels "
        "(snapshot_utc, labels_sha256, ids, clips, labels, reasons) are unchanged.",
        "the producer's own video files could not be hashed from here, so 'the bundle MP4 "
        "is byte-identical to the file the cache was built from' is NOT proven. The residual "
        "is exactly the size of a small pixel-level difference, which is what a different "
        "decoder build or a re-encode would produce. This is the one remaining unknown in "
        "the chain, and it changes the EG2 features by <=4e-4 cosine.",
        "the cache-producing environment's cv2/FFmpeg version is unknown; this pod has "
        "cv2 5.0.0 / PIL 11.0.0 / torch 2.8.0+cu128 / sentence-transformers 6.1.0.",
        "the run_repro sample comparison used raw fp16 dot products in the first pass; the "
        "frame-level numbers quoted as primary are the norm-invariant ones.",
    ],
    "environment": {"device": sample["model"]["device"], "dtype": sample["model"]["dtype"],
                    "HF_HUB_OFFLINE": "1",
                    "python": sys.version.split()[0],
                    "torch": sample["recipe"]["torch"],
                    "cv2": sample["recipe"]["cv2"], "pil": sample["recipe"]["pil"],
                    "sentence_transformers": sample["recipe"]["sentence_transformers"]},
}
OUTD.joinpath("eg2_embedding_reproduction.json").write_text(json.dumps(res, indent=1))
print("wrote", OUTD / "eg2_embedding_reproduction.json")
print(json.dumps({k: res[k] for k in ("verdict", "counts_vs_cache", "summary")}, indent=1)[:1200])
