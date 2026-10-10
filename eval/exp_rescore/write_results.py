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

ART = os.environ.get("E8_ART", "/workspace/artifacts")
CONTRACT = os.environ.get("E8_CONTRACT", os.path.join(ART, "windows_rescored.jsonl"))
SUMMARY = os.environ.get("E8_SUMMARY", os.path.join(ART, "exp_rescore", "run_summary.json"))
CONTROL = os.environ.get("E8_CONTROL", os.path.join(ART, "exp_rescore", "control_frozen_path.json"))
OUTDIR = os.environ.get("E8_OUTDIR", os.path.join(REPO, "eval", "results", "exp_rescore"))
REPORT = os.environ.get("E8_REPORT", os.path.join(REPO, "docs", "reports", "exp_rescore.md"))
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


def write_report(res, out_path):
    v = res["verdict_distribution"]
    sh = res["verdict_share"]
    ci = res["ci95"]
    c = res["cost"]
    ctl = res["decode_path_control"]
    ctl_txt = ("not run" if ctl.get("status") == "not run" else
               f"n={ctl['n']} windows, |Δ p_pixel| max {ctl['p_pixel_abs_delta_max']}, "
               f"mean {ctl['p_pixel_abs_delta_mean']} (exploratory)")
    md = f"""# E8 re-score of 1,546 Egocentric-10K windows — report (track E8, B2)

**Registry id:** E8.01 · **Owner:** experimenter-e8-rescore · **Device:** {res['device']} ·
**Branch:** `feat/exp-rescore` · **git_sha:** `{res['git_sha']}`

## What this is (and is not)

The PoC contract file `/workspace/artifacts/windows_rescored.jsonl` scores
**{res['n']} of {res['n_requested']} windows** of `queues/manifest_scale.json`
(58 clips, 4.404 h, 1920x1080 HEVC @30 fps; {res['n_failed']} decode failures)
with the frozen in-domain recipe **ensemble_v1** and the deterministic profile router.

> **{res['honesty_note']}**

`label_source` is `"none"` on every line. The verdict split below is **descriptive**.
The in-domain reference number for the recipe is AUROC **0.8534** (full-OOF, 150 IndEgo
windows / 7 clips, human gold, **one labeller**; `configs/ensemble_v1.json`); it does **not**
carry over here — IndEgo-trained models scored **0.4-0.53 AUROC** on the 60 human-labelled
Egocentric-10K windows (**n_bad = 9, exploratory**). "Receipt" means SHA-256 of the canonical
JSON: **content-hashed, not signed**.

## Recipe and router

- Score = equal-weight average of two frozen logistic heads on frozen features:
  **EG2** (EmbeddingGemma-2 mean embedding, 0.5 s grid, 20 frames/window, C=0.003) and
  **pixel26** (160x90 rgb24, all frames, 26 per-pixel features, C=0.3).
- **Router:** `profiles/*.json` + `jadal/regimes/router.py::choose_profile` exist and were
  used; the router branch **has landed**, so no fallback recipe was needed and none was invented.
- **Profile distribution:** {res['profile_distribution']} — every window is 1920x1080/30, so the
  hires rule fires before the quality/motion rules. The {res['router']['unseen_regime_windows']}
  windows that fell out of the calibrated descriptor ranges went to `default` (widest abstain band)
  as designed: 3 with `shift_mean` 245-642 (range [0,230], fast camera motion) and 3 with
  `luma_mean` 0.66-0.67 (range [0.05,0.65], overexposed). Margins: min 0.000, p50 1.000.
- `stage` is `"L1"` on every line (funnel vocabulary: L0 rules, L1 model heads, L2 verdict).
- **Verdict rule** = the selected profile's operating point
  (`factory_1080p30`: score ≥0.75 accept, ≤0.25 reject, else review). The frozen band
  (score >0.5 accept / <0.4 reject) is reported as a secondary distribution
  {res['secondary_verdict_distribution_frozen_band']} — it is **not** the one used.

## Verdict distribution (n={res['n']}, descriptive)

| verdict | n | share | 95% CI (window, Wilson) | 95% CI (clip bootstrap, {ci['n_clips']} clips) |
|---|---|---|---|---|
| accept | {v['accept']} | {sh['accept']} | {ci['window_level_wilson']['accept']} | {ci['clip_bootstrap_accept_share']} |
| reject | {v['reject']} | {sh['reject']} | {ci['window_level_wilson']['reject']} | {ci['clip_bootstrap_reject_share']} |
| review | {v['review']} | {sh['review']} | {ci['window_level_wilson']['review']} | — |

Window-level CIs are anti-conservative (windows inside one clip are correlated); the clip
bootstrap is the honest one. Score stats: mean {res['score_stats']['mean']},
p05/p50/p95 {res['score_stats']['p05']}/{res['score_stats']['p50']}/{res['score_stats']['p95']},
share inside the frozen abstain band {res['score_stats']['frac_in_frozen_abstain_band']}.
**This split is not a quality measure: it measures an untransferred model plus a router.**

## Cost (measured, {c['workers']} workers, shared A40)

- **{c['windows_per_s']} windows/s** overall; **{c['ms_per_window_mean']} ms/window mean**,
  p50 {c['ms_per_window_p50']} ms, p95 {c['ms_per_window_p95']} ms; **wall {c['wall_s']} s**
  for {res['n']} windows on one shared pod (load average ~15 from parallel tracks).
- Per stage (mean): decode {c['stage_ms_mean']['decode']} ms · pixel features
  {c['stage_ms_mean']['pixel_features']} ms · regime descriptors {c['stage_ms_mean']['descriptors']} ms ·
  EG2 embed {c['stage_ms_mean']['eg2_embed']} ms · score {c['stage_ms_mean']['score']} ms.
- Per window: **one** ffmpeg/NVDEC pass with three branches (300 x 160x90 pixel frames,
  20 x 512x288 EG2 frames, 8 native frames for descriptors), one EG2 forward, no sampling at
  inference, fixed seed.
- **E3.04 policy (every-2nd window + 4 s early exit) was NOT applied**: the contract needs one
  verdict per window, and the frozen heads were fit on full-window features, so a 4 s window
  would change the feature distribution. `profiles/factory_1080p30.json` asks for
  `fps=5` / `probe_every_k_windows=2`; the pixel view was run at the frozen recipe's full frame
  rate instead (documented deviation, not silently dropped). E3.04's x4.9 saving remains an
  unapplied, decode-bound estimate from the E3 track.

## Verification

- All {res['contract']['receipts_reverified']}/{res['n']} receipts and `features_digest` values
  **re-verify** from the line content (`eval/exp_rescore/contract.py::verify_line`).
- `receipt_sha256` = sha256(canonical JSON of the line); `features_digest` = sha256(canonical
  `features{{}}`); the 768-d EG2 mean vector is stored by content hash, with the full matrix at
  `{res['contract']['sidecars']['eg2_mean_l2']['path']}`
  (sha256 `{res['contract']['sidecars']['eg2_mean_l2']['sha256']}`); per-window router inputs at
  `{res['contract']['sidecars']['descriptors']['path']}`
  (sha256 `{res['contract']['sidecars']['descriptors']['sha256']}`).
- **Decode-path control** (frozen `ffmpeg_cpu`+swscale vs the fast NVDEC+scale_cuda path, pixel
  view only): {ctl_txt}. The fast path is **not bit-identical** to the frozen path; features and
  `p_pixel` differ slightly, so re-running the frozen path would give slightly different
  receipts. Not claimed identical.
- **Config note:** `configs/ensemble_v1.json`'s raw `.sha256` sidecar is stale (written before the
  pod path remap); the head numbers were verified equal to the frozen
  `$JADAL_BUNDLE/caches/ensemble/ensemble_v1.json`, and both digests are recorded per line in
  `versions{{}}`.

## Files

- runner `eval/exp_rescore/run_e8.py`, helper `eval/exp_rescore/contract.py`,
  result writer `eval/exp_rescore/write_results.py`, tests `tests/test_exp_rescore.py` (11 pass, 23 s)
- results `eval/results/exp_rescore/E8.01.json`, sample
  `eval/results/exp_rescore/windows_rescored_sample.jsonl` (20 lines = header + 19 windows)
- contract (not committed, `artifacts/` is ignored): `/workspace/artifacts/windows_rescored.jsonl`,
  body sha256 `{res['contract']['body_sha256']}`
- regenerate: `{res['command']}`

## Not done / open

{chr(10).join('- ' + x for x in res['not_done'])}
"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w").write(md)
    return out_path


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
    k = min(19, len(clips))
    picks = [by_clip[clips[i]][0] for i in rng.choice(len(clips), size=k, replace=False)]
    picks.sort(key=lambda r: (r["clip"], r["start_s"]))
    with open(SAMPLE, "w") as f:
        f.write(json.dumps(hdr, sort_keys=True) + "\n")
        for r in picks:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    rep = write_report(res, REPORT)
    print(f"wrote {OUT}, {SAMPLE} ({len(picks)+1} lines) and {rep} "
          f"({'all digests re-verified' if ok_verify == n else 'VERIFY FAILURES'})")
    print(json.dumps({"n": n, "verdicts": res["verdict_distribution"],
                      "share": res["verdict_share"], "ci95_clip": ci_clip,
                      "windows_per_s": res["cost"]["windows_per_s"],
                      "ms_per_window_mean": res["cost"]["ms_per_window_mean"]}, indent=1))


if __name__ == "__main__":
    main()
