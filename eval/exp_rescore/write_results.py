#!/usr/bin/env python3
"""E8.01 — build the committed results JSON + the committed 20-line sample.

Reads the (uncommitted) PoC contract at /workspace/artifacts/windows_rescored.jsonl
and writes:
  eval/results/exp_rescore/E8.01.json                  (n, label_source, protocol, ci95, device, cost, command, git_sha)
  eval/results/exp_rescore/windows_rescored_sample.jsonl (20 lines: header + 19 windows, seed 7)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "eval", "exp_rescore"))

from contract import HONESTY_NOTE, verify_line  # noqa: E402

ART = "/workspace/artifacts"
CONTRACT = os.path.join(ART, "windows_rescored.jsonl")
SUMMARY = os.path.join(ART, "exp_rescore", "run_summary.json")
CONTROL = os.path.join(ART, "exp_rescore", "control_frozen_path.json")
OUTDIR = os.path.join(REPO, "eval", "results", "exp_rescore")
OUT = os.path.join(OUTDIR, "E8.01.json")
SAMPLE = os.path.join(OUTDIR, "windows_rescored_sample.jsonl")

CMD = ("source /workspace/env.sh && export JADAL_EXP_DIR=/workspace/wt/exp-rescore/experiments "
       "OMP_NUM_THREADS=1 && cd /workspace/wt/exp-rescore && "
       "python3 eval/exp_rescore/run_e8.py --workers 4 --resume "
       "--out /workspace/artifacts/windows_rescored.jsonl && "
       "python3 eval/exp_rescore/write_results.py")


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]


def clip_bootstrap_share(rows, flag="accept", n_rep=2000, seed=11):
    """Cluster (clip) bootstrap CI for a verdict share; resampling clips."""
    by_clip = {}
    for r in rows:
        by_clip.setdefault(r["clip"], []).append(1.0 if r["verdict"] == flag else 0.0)
    clips = sorted(by_clip)
    vals = np.array([np.mean(by_clip[c]) for c in clips], dtype=float)
    sizes = np.array([len(by_clip[c]) for c in clips], dtype=float)
    rng = np.random.RandomState(seed)
    boots = []
    n = len(clips)
    for _ in range(n_rep):
        idx = rng.randint(0, n, n)
        w = sizes[idx]
        boots.append(float(np.sum(vals[idx] * w) / np.sum(w)))
    return [round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4)], len(clips)


def git_sha():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=REPO).stdout.strip()


def main():
    lines = [json.loads(l) for l in open(CONTRACT)]
    hdr, body = lines[0], lines[1:]
    summary = json.load(open(SUMMARY))
    os.makedirs(OUTDIR, exist_ok=True)

    n = len(body)
    ok_verify = sum(1 for r in body if verify_line(r))
    vd = Counter(r["verdict"] for r in body)
    pd = Counter(r["versions"]["profile"] for r in body)
    scores = np.array([r["score"] for r in body], dtype=float)
    lat = {k: float(np.mean([r["latency_ms"][k] for r in body]))
           for k in ("decode", "pixel_features", "descriptors", "eg2_embed", "score", "total")}
    lat_p50 = float(np.percentile([r["latency_ms"]["total"] for r in body], 50))
    lat_p95 = float(np.percentile([r["latency_ms"]["total"] for r in body], 95))
    wall = summary["wall_s"]
    fpsamp = Counter(r["versions"]["sampling"]["pixel_fps"] for r in body)

    ci_win = {v: wilson(vd.get(v, 0), n) for v in ("accept", "reject", "review")}
    ci_clip, n_clips = clip_bootstrap_share(body, "accept")
    ci_clip_rej, _ = clip_bootstrap_share(body, "reject")
    # frozen ensemble abstain band (documented alternative rule, secondary)
    frozen_band = Counter("accept" if r["score"] > 0.5 else
                          ("reject" if r["score"] < 0.4 else "review") for r in body)

    res = {
        "id": "E8.01",
        "title": "Re-score 1,546 Egocentric-10K windows with the frozen two-view recipe "
                 "plus the deterministic profile router",
        "n": n,
        "n_requested": 1546,
        "n_failed": summary["n_errors"],
        "label_source": "none",
        "protocol": "descriptive re-score of unlabelled Egocentric-10K windows "
                    "(no human labels; verdicts are descriptive, not accuracy) "
                    "+ 6-window frozen-vs-fast decode-path control",
        "honesty_note": HONESTY_NOTE,
        "device": "A40",
        "regime": "factory Egocentric-10K: 1920x1080, 30 fps, HEVC, handheld, 58 clips, 4.404 h",
        "recipe": {
            "name": "ensemble_v1 (frozen): EG2 (EmbeddingGemma-2 mean embedding, 0.5 s grid, "
                    "C=0.003) + pixel26 (160x90 rgb24, C=0.3) equal-weight probability average",
            "score_auroc_in_domain_baseline": {
                "value": 0.8534, "protocol": "full-OOF, 150 IndEgo windows / 7 clips, "
                                             "human gold, one labeller",
                "source": "configs/ensemble_v1.json:full_oof_auroc_resubstitution_free"},
            "profile_layer": "deterministic router jadal/regimes/router.py + profiles/*.json",
            "verdict_rule": "profile operating point of the selected profile "
                            "(factory_1080p30: score>=0.75 accept, <=0.25 reject, else review)",
            "secondary_rule_documented": "frozen ensemble band score>0.5 accept / <0.4 reject / "
                                         "[0.4,0.6] review (narrower abstain) -- reported, not used",
            "no_router_branch_missing": False,
        },
        "contract": {
            "path": "/workspace/artifacts/windows_rescored.jsonl (not committed: artifacts/ is ignored)",
            "n_lines": hdr["n_windows"] + 1,
            "header_line": True,
            "body_sha256": hdr["windows_body_sha256"],
            "receipts_reverified": ok_verify,
            "receipt_formula": "receipt_sha256 = sha256(canonical JSON of the line without "
                               "receipt_sha256); content-hashed, NOT signed",
            "features_digest_formula": "sha256(canonical features{})",
            "sidecars": {
                "descriptors": summary["descriptors_sidecar"],
                "eg2_mean_l2": summary["eg2_mean_sidecar"],
            },
        },
        "verdict_distribution": {k: vd.get(k, 0) for k in ("accept", "reject", "review")},
        "verdict_share": {k: round(vd.get(k, 0) / n, 4) for k in ("accept", "reject", "review")},
        "ci95": {
            "window_level_wilson": ci_win,
            "clip_bootstrap_accept_share": ci_clip,
            "clip_bootstrap_reject_share": ci_clip_rej,
            "n_clips": n_clips,
            "note": "window-level Wilson CIs are anti-conservative (windows inside a clip are "
                    "correlated); the clip bootstrap is the honest one",
        },
        "profile_distribution": dict(sorted(pd.items())),
        "router": {
            "windows_routed_factory_1080p30": pd.get("factory_1080p30", 0),
            "unseen_regime_windows": sum(1 for r in body if r["versions"]["unseen_regime"]),
            "note": "all windows are 1920x1080/30, so the hires rule fires before any "
                    "quality/motion rule; observed margins "
                    f"min={min(r['versions']['profile_margin'] for r in body):.3f} "
                    f"p50={float(np.percentile([r['versions']['profile_margin'] for r in body], 50)):.3f}",
        },
        "score_stats": {"mean": round(float(scores.mean()), 4),
                        "p05": round(float(np.percentile(scores, 5)), 4),
                        "p50": round(float(np.percentile(scores, 50)), 4),
                        "p95": round(float(np.percentile(scores, 95)), 4),
                        "frac_in_frozen_abstain_band": round(
                            float(np.mean((scores >= 0.4) & (scores <= 0.6))), 4)},
        "secondary_verdict_distribution_frozen_band": dict(sorted(frozen_band.items())),
        "cost": {
            "wall_s": wall,
            "windows_per_s": round(n / wall, 4) if wall else None,
            "ms_per_window_mean": round(lat["total"], 1),
            "ms_per_window_p50": round(lat_p50, 1),
            "ms_per_window_p95": round(lat_p95, 1),
            "stage_ms_mean": {k: round(v, 1) for k, v in lat.items()},
            "workers": summary["workers"],
            "protocol_work_per_window": {
                "pixel_frames": "300 x 160x90 rgb24 (30 fps)",
                "eg2_frames": "20 at 512x288 (0.5 s grid)",
                "descriptor_frames": "8 native 1920x1080 (fps=0.8, (i+0.5)/8 grid)",
                "ffmpeg_decodes_per_window": 1,
                "model_forwards_per_window": 1,
                "sampling_at_inference": "none (fixed frames, fixed seed, deterministic)",
                "pixel_fps_histogram": {str(k): v for k, v in sorted(fpsamp.items())},
            },
            "policy_note": "E3.04 (every-2nd-window + 4 s early exit -> x4.9 less work) was NOT "
                           "applied: the contract requires one verdict per window, and the "
                           "profile factory_1080p30 sampling (fps=5, probe_every_k=2) was not "
                           "applied to the pixel view either, because the frozen head was fit on "
                           "full-window features. Reported as a measured alternative, not used.",
        },
        "decode_path_control": (json.load(open(CONTROL)) if os.path.exists(CONTROL) else
                                {"status": "not run"}),
        "command": CMD,
        "git_sha": git_sha(),
        "files": {
            "runner": "eval/exp_rescore/run_e8.py",
            "contract_helper": "eval/exp_rescore/contract.py",
            "result_writer": "eval/exp_rescore/write_results.py",
            "tests": "tests/test_exp_rescore.py",
            "report": "docs/reports/exp_rescore.md",
            "sample": "eval/results/exp_rescore/windows_rescored_sample.jsonl",
            "poC_contract_not_committed": "/workspace/artifacts/windows_rescored.jsonl",
        },
        "not_done": [
            "no human labels for these windows -> no accuracy, no AUROC out-of-domain claim",
            "no transfer claim: IndEgo-trained heads scored 0.4-0.53 AUROC on the 60 labelled "
            "Egocentric-10K windows (n_bad=9)",
            "the fast NVDEC decode path is not bit-identical to the frozen ffmpeg_cpu path "
            "(see decode_path_control)",
            "no per-window wall-clock budget on the edge device (pod A40 numbers only)",
        ],
    }
    json.dump(res, open(OUT, "w"), indent=1, sort_keys=True)

    # 20-line committed sample: header + 19 windows spread over clips (seed 7)
    rng = np.random.RandomState(7)
    by_clip = {}
    for r in body:
        by_clip.setdefault(r["clip"], []).append(r)
    clips = sorted(by_clip)
    picks = [by_clip[clips[i]][0] for i in rng.choice(len(clips), size=19, replace=False)]
    picks.sort(key=lambda r: (r["clip"], r["start_s"]))
    with open(SAMPLE, "w") as f:
        f.write(json.dumps(hdr, sort_keys=True) + "\n")
        for r in picks:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"wrote {OUT} and {SAMPLE} ({len(picks)+1} lines, "
          f"{'all digests re-verified' if ok_verify == n else 'VERIFY FAILURES'})")
    print(json.dumps({"n": n, "verdicts": res["verdict_distribution"],
                      "share": res["verdict_share"], "ci95_clip": ci_clip,
                      "windows_per_s": res["cost"]["windows_per_s"],
                      "ms_per_window_mean": res["cost"]["ms_per_window_mean"]}, indent=1))


if __name__ == "__main__":
    main()
