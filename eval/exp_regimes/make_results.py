"""Emit the four deliverable results JSONs E4.01..E4.04 with the required
provenance fields (n, label_source, protocol, ci95, device, cost, command,
git_sha). Reads the analysis outputs produced by analyze_regimes.py and the
descriptor JSONL files; CPU-only."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "exp_regimes")
BUNDLE = "/workspace/jadal_pod_bundle"
DEVICE = "CPU (pod); no GPU used"


def sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.join(HERE, "..", ".."),
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def jload(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def desc_counts():
    counts = {}
    for fn in ("desc_indego.jsonl", "desc_ood.jsonl", "desc_ood_reenc.jsonl"):
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            if r.get("ok"):
                counts[r["set"]] = counts.get(r["set"], 0) + 1
    return counts


CONTENT = ["sharp_lapvar", "sharp_norm", "noise_sigma", "edge_density", "luma_mean",
           "luma_p05", "luma_p50", "luma_p95", "dark_frac", "bright_frac", "clip_frac",
           "blockiness", "chroma_spread", "shift_mean", "shift_max", "static_frac",
           "temp_energy", "dup_frac"]
CONTAINER = ["width", "height", "fps", "bitrate_kbps", "bytes_per_frame"]


def e401():
    d = jload("diagnosis.json")
    if d is None:
        return None
    cnt = desc_counts()
    n = sum(cnt.values())
    table = d["ranked_table"]
    content = [r for r in table if r["descriptor"] in CONTENT]

    def absd(r):
        v = r["cohens_d_ind_vs_oodsrc"]
        return float("inf") if v in ("+inf", "-inf") else abs(float(v))
    content.sort(key=lambda r: -absd(r))
    return {
        "id": "E4.01", "track": "E4",
        "title": "Deterministic regime descriptors: IndEgo vs Egocentric-10K domain-shift diagnosis",
        "verdict": "adopt",
        "n": n, "n_by_set": cnt,
        "label_source": ("human: 150 IndEgo gold + 60 OOD-60 (9 bad); "
                         "unlabelled: 100 OOD-2 transcodes + 300 Ego10K sample"),
        "protocol": ("deterministic per-window descriptors, 8 sampled frames/window; "
                     "ffprobe container + Laplacian sharpness, Immerkaer noise, luma "
                     "percentiles, blockiness, chroma spread, phase-correlation motion; "
                     "no model, no GPU"),
        "ci95": "n/a — deterministic descriptors",
        "device": DEVICE,
        "cost": f"{n} windows x 8 frames; CPU only; no GPU",
        "command": "python3 eval/exp_regimes/analyze_regimes.py --part diagnosis",
        "git_sha": sha(),
        "codec_mix": d["codec_mix"],
        "note_container_by_construction": ("width/height/fps/codec separate the transcode "
                                           "and source sets exactly; they are spec by "
                                           "construction, not content findings"),
        "ranked_separators_all": table,
        "ranked_separators_content_only": content,
    }


def e402():
    r = jload("regime_vs_global.json")
    if r is None:
        return None
    return {
        "id": "E4.02", "track": "E4",
        "title": "Regime-specific vs global models within IndEgo (nested LOCO)",
        "verdict": "drop",
        "n": 150, "label_source": "human (IndEgo gold, 1 labeller)",
        "protocol": "leave-one-clip-out over the 7 IndEgo clips; 2000x clip-bootstrap CI",
        "ci95": "clip bootstrap (2000 draws, 7 clips resampled)",
        "device": DEVICE,
        "cost": "150 windows; logistic on 26 feats / EG2 768-d; CPU",
        "command": "python3 eval/exp_regimes/analyze_regimes.py --part regimes",
        "git_sha": sha(),
        "results": r,
    }


def e403():
    g = jload("gap_control.json")
    if g is None:
        return None
    reenc = g.get("matching_experiment", {})
    return {
        "id": "E4.03", "track": "E4",
        "title": "Domain-gap control: re-encode OOD sources to the IndEgo spec (480x480/10)",
        "verdict": "drop",
        "n": g["n_paired"], "n_reencoded": reenc.get("n_reencoded"),
        "label_source": "human (60 OOD-60 windows, 9 bad)",
        "protocol": ("paired descriptors across 1080p/30 HEVC source, 854x480/30 H.264 "
                     "transcode, and a new 480x480/10 H.264 re-encode; frozen-ensemble "
                     "ood_scores.json baseline audit"),
        "ci95": "n/a — paired deterministic descriptors; OOD AUROC is exploratory (n_bad=9)",
        "device": DEVICE,
        "cost": "60 windows re-encoded with ffmpeg (libx264); CPU",
        "command": ("python3 eval/exp_regimes/reencode_spec.py --out "
                    "eval/results/exp_regimes/desc_ood_reenc.jsonl && "
                    "python3 eval/exp_regimes/analyze_regimes.py --part gap"),
        "git_sha": sha(),
        "frozen_ensemble_on_ood": g["frozen_ensemble_on_ood"],
        "matching_experiment": reenc,
        "transcode_vs_source": g["transcode_vs_source"],
    }


def e404():
    test_log = "/workspace/state/pids/e4_tests.log"
    passed = total = None
    if os.path.exists(test_log):
        import re
        txt = open(test_log).read()
        m = re.search(r"(\d+) passed", txt)
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+) failed", txt)
        total = (passed or 0) + (int(m.group(1)) if m else 0)
    return {
        "id": "E4.04", "track": "E4",
        "title": "Profiles product layer: schema + 5 profiles + deterministic router",
        "verdict": "adopt",
        "n": None,
        "label_source": "n/a — synthetic descriptors + E4.01 ranges",
        "protocol": ("unit tests on synthetic descriptors; determinism; UNSEEN_REGIME "
                     "fallback; profile schema conformance"),
        "ci95": "n/a — deterministic router",
        "device": DEVICE,
        "cost": "10 unit tests < 0.5 s, no video, no network",
        "command": "python3 -m pytest tests/test_exp_regimes_router.py -q",
        "git_sha": sha(),
        "tests_passed": passed, "tests_total": total,
        "profiles": sorted(os.path.basename(p)[:-5] for p in
                           __import__("glob").glob(os.path.join(HERE, "..", "..", "profiles", "*.json"))
                           if not p.endswith("schema.json")),
        "router": "jadal/regimes/router.py::choose_profile -> {profile, margin, reasons, unseen_regime}",
        "receipt_fields": ["profile", "router_margin", "router_reasons", "unseen_regime", "descriptors_sha"],
    }


def main():
    for f, name in ((e401, "E4.01.json"), (e402, "E4.02.json"),
                    (e403, "E4.03.json"), (e404, "E4.04.json")):
        out = f()
        if out is None:
            print("skip (missing input):", name)
            continue
        with open(os.path.join(RES, name), "w") as fh:
            json.dump(out, fh, indent=1)
        print("wrote", os.path.join(RES, name))


if __name__ == "__main__":
    main()
