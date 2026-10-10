#!/usr/bin/env python3
"""Q5.01 — deterministic PHYSICS RULE-LIST model over the full corpus.

This is the L3 latency runner (``eval/exp_latency_fast/run_fast.py``) with the
**decision stage swapped** from the frozen logistic ensemble
(``ensemble_prob`` + profile operating point) to Q1's deterministic physics
rule list (``eval/exp_phys/phys_rules.py`` + ``rule_table.json``).

Everything upstream of the decision is kept byte-for-byte identical:
  * same one-pass NVDEC + ``scale_cuda`` three-branch decode,
  * same pixel-26 features (``jadal.gpu.features.cpu_window_features``),
  * same E4.01 regime descriptors + deterministic router (recorded, not used),
  * same EG2 mean embedding (EmbeddingGemma-2, bf16),
  * same adopted sampling optimizations (pixel 3.33 fps, EG2 1 fps, 4 s early
    exit, mosaic off) and the same ``time.perf_counter`` atomic per-stage timing.

Added stage:
  6. ``phys_features``  ``jadal.perpixel.physics.extract_physics`` on the decoded
                        160x90 frames -> the 16 physics features the rule list uses
                        (flow_act, flow_med, dhash_loop_min, mad_mean,
                        frm_bytes_med, p_size_med, snr_db, ...).
  7. ``decision``       deterministic rule scoring: integer score = number of
                        crossed fixed quantile thresholds; quality rule -> REJECT,
                        score>=reject_threshold -> REJECT, score>=1 -> REVIEW,
                        else ACCEPT. No learned weights.

Receipt determinism: ``receipt_sha256`` is SHA-256 over the canonical JSON of the
contract line **excluding** ``latency_ms`` (timing is not deterministic), so two
repeats of the same window must produce byte-identical receipts. The full line
keeps ``latency_ms`` for reporting.

Run:
  source /workspace/env.sh
  export OMP_NUM_THREADS=1 JADAL_GPU_SLOTS=6
  bash tools/pod/gpu_run.sh python3 eval/exp_phys_full/run_phys.py \
        --workers 7 --pixel-fps 3.33 --eg2-fps 1 --early-exit-s 4 \
        --skip-mosaic --no-compare --out eval/results/exp_phys_full/windows.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)                                   # phys_rules.py
sys.path.insert(0, os.path.join(REPO, "eval", "exp_regimes"))
sys.path.insert(0, os.path.join(REPO, "eval", "exp_rescore"))

import run_e8 as e8  # noqa: E402  (the repo E8.01 pipeline: decode/features/EG2)
import phys_rules as PR  # noqa: E402  (Q1 deterministic rule list)
from contract import canonical_bytes, route_from_decision, sha256_hex  # noqa: E402
from jadal.gpu.features import cpu_window_features  # noqa: E402
from jadal.perpixel import physics as PP  # noqa: E402
from jadal.regimes.router import choose_profile  # noqa: E402

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
MANIFEST = os.path.join(BUNDLE, "queues", "manifest_scale.json")
OUTDIR = os.path.join(REPO, "eval", "results", "exp_phys_full")
RULES_PATH = os.path.join(HERE, "rule_table.json")
W, H = 160, 90
EGW, EGH = 512, 288
NATIVE_W, NATIVE_H = 1920, 1080
WINDOW_S = 10.0
DEVICE = "A40"
STAGE = "L0"                      # funnel vocabulary: L0 = deterministic rules
MEGA = 1 << 20
_OPT = None
_RULES = None
_REJECT_THR = None
_RULES_SHA = None

POD_EG2 = ("/workspace/.cache/huggingface/hub/models--google--embeddinggemma-2"
           "/snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b")

smi_log = []
smi_stop = threading.Event()


def smi_sampler():
    while not smi_stop.is_set():
        try:
            p = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            u, m = [x.strip() for x in p.stdout.strip().split(",")]
            smi_log.append({"t": time.time(), "gpu_pct": float(u),
                            "vram_mib": float(m)})
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)


# -------------------------------------------------------- adopted-opt knobs
def _opt_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def read_opts():
    return {
        "pixel_fps": _opt_float("JADAL_PIXEL_FPS", 0.0),
        "eg2_fps": _opt_float("JADAL_EG2_FPS", 2.0),
        "native_fps": _opt_float("JADAL_NATIVE_FPS", 0.8),
        "early_exit_s": _opt_float("JADAL_EARLY_EXIT_S", 0.0),
        "skip_mosaic": os.environ.get("JADAL_SKIP_MOSAIC", "0") == "1",
    }


def load_rules():
    t = json.load(open(RULES_PATH))
    sha = sha256_hex(open(RULES_PATH, "rb").read())
    return t["rules"], int(t["reject_threshold"]), sha


# ---------------------------------------------------------------- worker setup
def init_worker():
    global _OPT, _RULES, _REJECT_THR, _RULES_SHA
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    e8.init_worker()  # sets e8._CFG / e8._CFG_FRAG (head numbers checked)
    if os.path.isdir(POD_EG2):
        e8._CFG["views"]["eg2"]["embed_model_dir"] = POD_EG2
    _OPT = read_opts()
    _RULES, _REJECT_THR, _RULES_SHA = load_rules()


def ms(t0):
    return (time.perf_counter() - t0) * 1000.0


def _fin(v):
    """Finite float rounded to 4 dp, or None (JSON-safe; never guessed)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, 4) if math.isfinite(f) else None


def io_probe(path, start_s, dur_s):
    t = time.perf_counter()
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        frac = min(max(float(start_s) / max(float(dur_s), 1e-9), 0.0), 1.0)
        off = min(int(frac * size), max(size - 65536, 0))
        f.seek(off)
        f.read(65536)
    return ms(t)


def _decode_nvdec_cuvid(path, start_s, dur_s, w=W, h=H, timeout_s=180):
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
           "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
           "-c:v", "hevc_cuvid",
           "-ss", f"{float(start_s):.3f}", "-t", f"{float(dur_s):.3f}", "-i", path,
           "-vf", f"scale_cuda={w}:{h}:format=nv12,hwdownload,format=nv12,format=rgb24",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       stdin=subprocess.DEVNULL, timeout=timeout_s)
    if p.returncode != 0:
        raise RuntimeError(f"cuvid rc={p.returncode}: "
                           f"{p.stderr[-200:].decode('utf-8', 'replace')}")
    return p.stdout


def decode_compare(path, start_s, dur_s):
    out = {}
    from eval.gpu.decode import decode_window
    t = time.perf_counter()
    try:
        rgb, _ = decode_window(path, start_s, dur_s, W, H, "rgb24")
        out["cpu_ms"] = ms(t)
        out["cpu_frames"] = int(rgb.shape[0])
    except Exception as ex:  # noqa: BLE001
        out["cpu_ms"] = ms(t)
        out["cpu_error"] = str(ex)[:160]
    t = time.perf_counter()
    try:
        buf = _decode_nvdec_cuvid(path, start_s, dur_s)
        out["nvdec_cuvid_ms"] = ms(t)
        out["nvdec_cuvid_frames"] = int(len(buf) // (W * H * 3))
    except Exception as ex:  # noqa: BLE001
        out["nvdec_cuvid_ms"] = ms(t)
        out["nvdec_cuvid_error"] = str(ex)[:160]
    return out


def decode_three_branch_cfg(path, start_s, dur, pixel_fps=0.0, eg2_fps=2.0,
                            native_fps=0.8, w=W, h=H, egw=EGW, egh=EGH,
                            timeout_s=300):
    """E8.01 three-branch NVDEC pass with parameterised per-branch fps."""
    pipes = [os.pipe() for _ in range(3)]
    pfx = f"fps={pixel_fps:g}," if pixel_fps and pixel_fps > 0 else ""
    efx = f"fps={eg2_fps:g}," if eg2_fps and eg2_fps > 0 else ""
    nfx = f"fps={native_fps:g}," if native_fps and native_fps > 0 else ""
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-ss", f"{float(start_s):.3f}", "-t", f"{float(dur):.3f}", "-i", path,
        "-filter_complex",
        "[0:v]split=3[a][b][c];"
        f"[a]{pfx}scale_cuda={w}:{h}:format=nv12,hwdownload,format=nv12,format=rgb24[pix];"
        f"[b]{efx}scale_cuda={egw}:{egh}:format=nv12,hwdownload,format=nv12,format=rgb24[eg];"
        f"[c]{nfx}hwdownload,format=nv12,format=rgb24[nat]",
        "-map", "[pix]", "-f", "rawvideo", "-pix_fmt", "rgb24", f"pipe:{pipes[0][1]}",
        "-map", "[eg]", "-f", "rawvideo", "-pix_fmt", "rgb24", f"pipe:{pipes[1][1]}",
        "-map", "[nat]", "-f", "rawvideo", "-pix_fmt", "rgb24", f"pipe:{pipes[2][1]}",
    ]
    sinks = [[], [], []]
    errors = []
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, pass_fds=tuple(p[1] for p in pipes))
    for p in pipes:
        os.close(p[1])
    threads = []
    for i in range(3):
        t = threading.Thread(target=e8._drain, args=(pipes[i][0], sinks[i], errors),
                             daemon=True)
        t.start()
        threads.append(t)
    err = proc.stderr.read()
    rc = proc.wait(timeout=timeout_s)
    for t in threads:
        t.join(timeout=60)
    for p in pipes:
        try:
            os.close(p[0])
        except OSError:
            pass
    if rc != 0:
        raise RuntimeError(f"ffmpeg rc={rc}: {err[-300:].decode('utf-8', 'replace')}")
    return b"".join(sinks[0]), b"".join(sinks[1]), b"".join(sinks[2])


def _phys_features_block(feats, phys):
    block = {"full": {k: _fin(v) for k, v in sorted(feats.items())},
             "phys": {k: _fin(phys.get(k, float("nan"))) for k in PP.PHYSICS_FEATURES}}
    return block


def _build_line(item, verdict, sc, dec, profile, features_block,
                features_digest, versions, latency_ms):
    """Contract line; receipt excludes latency_ms so repeats are byte-identical."""
    reasons = []
    for e in sc["fired"]:
        reasons.append({"code": e["rule"],
                        "evidence": {"feature": e["feature"],
                                     "value": _fin(e["value"]), "op": e["op"],
                                     "threshold": _fin(e["threshold"]),
                                     "strength": e["strength"],
                                     "n_thresholds_crossed": e["n_thresholds_crossed"]}})
    line = {
        "window_id": item["window_id"], "factory": item["factory"],
        "worker": item["worker"], "clip": item["clip"],
        "start_s": round(float(item["start_s"]), 4),
        "end_s": round(float(item["end_s"]), 4),
        "verdict": verdict, "stage": STAGE,
        "decision_reason": dec["reason"],
        "score": int(sc["score"]), "reject_threshold": int(_REJECT_THR),
        "fired_rules": sc["fired_rules"], "strong_fired": sc["strong_fired"],
        "weak_fired": sc["weak_fired"],
        "profile": profile,
        "reasons": reasons,
        "features": features_block, "features_digest": features_digest,
        "versions": versions,
        "latency_ms": {k: round(float(v), 3) for k, v in latency_ms.items()},
        "label_source": "none",
    }
    det = {k: v for k, v in line.items() if k != "latency_ms"}
    line["receipt_sha256"] = sha256_hex(canonical_bytes(det))
    return line


# ----------------------------------------------------------------- per window
def score_window_timed(item, profile_map, do_compare=False):
    opts = _OPT or read_opts()
    lat = {}
    wall = {}
    t_all = time.perf_counter()
    path, s0 = item["path"], float(item["start_s"])
    s1 = float(item["end_s"])
    dur_full = s1 - s0
    dur = float(opts["early_exit_s"]) if opts["early_exit_s"] > 0 else dur_full

    # 1. io_probe ----------------------------------------------------------
    t = time.perf_counter()
    w0 = time.time()
    lat["io_probe"] = io_probe(path, s0, item.get("duration_s", dur_full))
    wall["io_probe"] = [w0, time.time()]

    # 2. decode (E8.01 three-branch NVDEC + scale_cuda) --------------------
    t = time.perf_counter()
    w0 = time.time()
    pix_raw, eg_raw, nat_raw = decode_three_branch_cfg(
        path, s0, dur, opts["pixel_fps"], opts["eg2_fps"], opts["native_fps"])
    lat["decode"] = ms(t)
    wall["decode"] = [w0, time.time()]

    # 3. pixel-26 features -------------------------------------------------
    t = time.perf_counter()
    w0 = time.time()
    pix = e8.frames_from_raw(pix_raw, H, W).astype(np.float64) / 255.0
    if pix.shape[0] == 0:
        raise RuntimeError("no pixel frames")
    luma = 0.299 * pix[..., 0] + 0.587 * pix[..., 1] + 0.114 * pix[..., 2]
    fps_pix = min(pix.shape[0] / max(dur, 1e-6), 30.0)
    feats = cpu_window_features(luma, pix, fps=fps_pix)
    lat["pixel_features"] = ms(t)
    wall["pixel_features"] = [w0, time.time()]

    # 4. descriptors (E4.01) on the native frames --------------------------
    t = time.perf_counter()
    w0 = time.time()
    meta = e8._get_meta(path)
    nat_frames = list(e8.frames_from_raw(nat_raw, NATIVE_H, NATIVE_W))
    desc = e8.describe_from_native_frames(meta, nat_frames, s0, s0 + dur)
    lat["descriptors"] = ms(t)
    wall["descriptors"] = [w0, time.time()]

    # route (deterministic; recorded for reference, NOT used by the rules)
    decision = choose_profile(desc)
    route = route_from_decision(decision, desc)

    # 5. EG2 window embedding (kept identical to the ensemble runner) ------
    t = time.perf_counter()
    w0 = time.time()
    eg2_mean, n_eg = e8.encode_eg2(e8.frames_from_raw(eg_raw, EGH, EGW), e8._CFG)
    lat["eg2_embed"] = ms(t)
    wall["eg2_embed"] = [w0, time.time()]
    try:
        import torch
        vram_peak = float(torch.cuda.max_memory_allocated() / MEGA)
    except Exception:  # noqa: BLE001
        vram_peak = None

    # 6. phys_features (16 physics features the rule list consumes) --------
    t = time.perf_counter()
    w0 = time.time()
    phys = PP.extract_physics(pix, fps_pix, path, s0, s0 + dur)
    lat["phys_features"] = ms(t)
    wall["phys_features"] = [w0, time.time()]

    # 7. decision: deterministic physics rule list -------------------------
    t = time.perf_counter()
    w0 = time.time()
    all_feats = {f"full.{k}": float(v) for k, v in feats.items()}
    all_feats.update({f"phys.{k}": float(phys.get(k, float("nan")))
                      for k in PP.PHYSICS_FEATURES})
    sc = PR.score_window(all_feats, _RULES)
    dec = PR.decide(sc, _REJECT_THR)
    verdict = dec["decision"].lower()
    fb = _phys_features_block(feats, phys)
    fb["eg2_mean_sha256"] = sha256_hex(
        canonical_bytes([round(float(x), 4) for x in eg2_mean]))
    fb["eg2_dim"] = int(len(eg2_mean))
    fb["n_frames_pixel"] = int(pix.shape[0])
    fb["n_frames_eg2"] = int(n_eg)
    fb["fps_pixel"] = round(float(fps_pix), 4)
    fdig = sha256_hex(canonical_bytes(fb))
    lat["decision"] = ms(t)
    wall["decision"] = [w0, time.time()]
    lat["total"] = ms(t_all)

    versions = {
        "recipe": "phys_rules_v1",
        "rule_table_sha256": _RULES_SHA,
        "reject_threshold": int(_REJECT_THR),
        "n_rules": len(_RULES),
        "determinism": "fixed training-clip quantiles p01/p05/p10/p25 quantised "
                       "to 4 dp; integer score = count of crossed thresholds; "
                       "no learned weights",
        "config_sha256": sha256_hex(open(e8.CONFIG, "rb").read()),
        "eg2_model": "google/embeddinggemma-2@914f7f8",
        "profile": route["profile"],
        "descriptors_sha256": sha256_hex(canonical_bytes(desc)),
        "decode_path": "nvdec(cuda)+scale_cuda(bilinear)->rgb24",
        "sampling": {
            "window_s": WINDOW_S, "pixel_fps": round(fps_pix, 4),
            "eg2_step_s": (round(1.0 / opts["eg2_fps"], 4)
                           if opts["eg2_fps"] else None),
            "descriptor_frames": 8,
            "observed_s": round(dur, 3),
            "early_exit_s": opts["early_exit_s"],
            "pixel_fps_target": opts["pixel_fps"],
            "eg2_fps_target": opts["eg2_fps"],
            "native_fps_target": opts["native_fps"],
            "mosaic_skipped": opts["skip_mosaic"],
        },
        "deterministic": True,
        "device": DEVICE,
        "git_sha": e8.git_sha(),
    }
    line = _build_line(item, verdict, sc, dec, route["profile"], fb, fdig,
                       versions, lat)

    cmp = decode_compare(path, s0, dur) if do_compare else None
    row = {
        "window_id": item["window_id"], "clip": item["clip"],
        "factory": item["factory"], "worker": item["worker"],
        "start_s": round(s0, 4), "end_s": round(s1, 4),
        "verdict": verdict, "score": int(sc["score"]),
        "reject_threshold": int(_REJECT_THR),
        "fired_rules": sc["fired_rules"],
        "strong_fired": sc["strong_fired"], "weak_fired": sc["weak_fired"],
        "decision_reason": dec["reason"],
        "profile": route["profile"],
        "latency_ms": {k: round(v, 3) for k, v in lat.items()},
        "n_frames": {"pixel": int(pix.shape[0]), "eg2": int(n_eg),
                     "native": len(nat_frames)},
        "gpu": {"eg2_vram_peak_mib": vram_peak},
        "decode_compare": cmp,
        "receipt_sha256": line["receipt_sha256"],
        "features_digest": fdig,
        "stage_wall": wall,
        "t_wall0": wall["io_probe"][0], "t_wall1": wall["decision"][1],
        "contract": line,
    }
    return row


def _worker(args):
    item, profile_map, do_compare = args
    try:
        return {"ok": True, "row": score_window_timed(item, profile_map, do_compare)}
    except Exception as ex:  # noqa: BLE001
        return {"ok": False, "window_id": item["window_id"], "clip": item["clip"],
                "factory": item["factory"], "error": f"{type(ex).__name__}: {ex}"[:300]}


def determinism_probe(items, profile_map, n=10):
    """Score the first n windows twice and compare deterministic receipts."""
    init_worker()
    out = []
    for it in items[:n]:
        try:
            r1 = score_window_timed(it, profile_map, False)
            r2 = score_window_timed(it, profile_map, False)
            out.append({
                "window_id": it["window_id"],
                "verdict_1": r1["verdict"], "verdict_2": r2["verdict"],
                "score_1": r1["score"], "score_2": r2["score"],
                "receipt_1": r1["receipt_sha256"], "receipt_2": r2["receipt_sha256"],
                "features_digest_1": r1["features_digest"],
                "features_digest_2": r2["features_digest"],
                "receipt_match": r1["receipt_sha256"] == r2["receipt_sha256"],
                "features_digest_match": r1["features_digest"] == r2["features_digest"],
                "latency_1_ms": r1["latency_ms"]["total"],
                "latency_2_ms": r2["latency_ms"]["total"],
            })
        except Exception as ex:  # noqa: BLE001
            out.append({"window_id": it["window_id"],
                        "error": f"{type(ex).__name__}: {ex}"[:200]})
    return out


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "windows.jsonl"))
    ap.add_argument("--no-compare", dest="compare", action="store_false", default=True)
    ap.add_argument("--determinism-n", type=int, default=0,
                    help="score the first N windows twice after the pool and "
                         "compare deterministic receipts")
    ap.add_argument("--pixel-fps", type=float, default=None)
    ap.add_argument("--eg2-fps", type=float, default=None)
    ap.add_argument("--native-fps", type=float, default=None)
    ap.add_argument("--early-exit-s", type=float, default=None)
    ap.add_argument("--window-stride", type=int, default=1)
    ap.add_argument("--skip-mosaic", action="store_true")
    args = ap.parse_args()

    if args.pixel_fps is not None:
        os.environ["JADAL_PIXEL_FPS"] = str(args.pixel_fps)
    if args.eg2_fps is not None:
        os.environ["JADAL_EG2_FPS"] = str(args.eg2_fps)
    if args.native_fps is not None:
        os.environ["JADAL_NATIVE_FPS"] = str(args.native_fps)
    if args.early_exit_s is not None:
        os.environ["JADAL_EARLY_EXIT_S"] = str(args.early_exit_s)
    if args.skip_mosaic:
        os.environ["JADAL_SKIP_MOSAIC"] = "1"
    opts = read_opts()

    os.makedirs(OUTDIR, exist_ok=True)
    man, items = e8.load_manifest()
    n_manifest = len(items)
    if args.limit:
        items = items[:args.limit]
    if args.window_stride > 1:
        items = items[::args.window_stride]
    n_selected = len(items)
    pmap = e8.profiles()
    order = {it["window_id"]: i for i, it in enumerate(items)}

    seen, compare_ids = set(), set()
    if args.compare:
        for it in items:
            if it["clip"] not in seen:
                seen.add(it["clip"])
                compare_ids.add(it["window_id"])

    th = threading.Thread(target=smi_sampler, daemon=True)
    th.start()
    t0 = time.perf_counter()
    wall0 = time.time()
    rows, errors = [], []

    tasks = [(it, pmap, it["window_id"] in compare_ids) for it in items]
    partial = args.out + ".partial"
    if args.workers <= 1:
        init_worker()
        with open(partial, "w") as fh:
            for k, tk in enumerate(tasks, 1):
                r = _worker(tk)
                if r["ok"]:
                    rows.append(r["row"])
                    fh.write(json.dumps({"row": r["row"]}, sort_keys=True) + "\n")
                    fh.flush()
                else:
                    errors.append(r)
                _log(k, len(tasks), len(rows), len(errors), t0)
    else:
        import multiprocessing as mp
        with mp.Pool(args.workers, initializer=init_worker) as pool:
            with open(partial, "w") as fh:
                for k, r in enumerate(pool.imap(_worker, tasks, chunksize=1), 1):
                    if r["ok"]:
                        rows.append(r["row"])
                        tw = time.perf_counter()
                        fh.write(json.dumps({"row": r["row"]}, sort_keys=True) + "\n")
                        fh.flush()
                        r["row"]["latency_ms"]["write"] = round(ms(tw), 3)
                    else:
                        errors.append(r)
                    _log(k, len(tasks), len(rows), len(errors), t0)

    # deterministic-receipt probe (after the pool: only the parent holds the GPU)
    det_rows = []
    if args.determinism_n > 0:
        print(f"determinism probe: {args.determinism_n} windows x2", flush=True)
        det_rows = determinism_probe(items, pmap, n=args.determinism_n)

    wall = time.perf_counter() - t0
    smi_stop.set()
    th.join(timeout=5)

    rows.sort(key=lambda r: order.get(r["window_id"], 1 << 30))
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    # per-clip cold probe table
    clip_rows = []
    for c in sorted({i["clip"] for i in man["items"]}):
        it = next(i for i in man["items"] if i["clip"] == c)
        row = {"clip": c, "factory": it["factory"], "path": it["path"],
               "duration_s": it.get("duration_s"), "size_bytes": it.get("size_bytes")}
        try:
            from eval.gpu.decode import clip_duration
            t = time.perf_counter()
            d = clip_duration(it["path"])
            row["ffprobe_cold_ms"] = ms(t)
            row["clip_dur_s"] = d
        except Exception as ex:  # noqa: BLE001
            row["ffprobe_error"] = str(ex)[:120]
        clip_rows.append(row)
    with open(os.path.join(OUTDIR, "clips.jsonl"), "w") as f:
        for r in clip_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    import resource
    n_windows = len(rows)
    verdicts = Counter(r["verdict"] for r in rows)
    profiles = Counter(r["profile"] for r in rows)
    meta = {
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": round(wall, 2),
        "wall_start_epoch": wall0,
        "workers": args.workers,
        "manifest": {"n_windows": man["n_windows"], "n_clips": man["n_clips"],
                     "total_hours": man["total_hours"]},
        "n_requested": len(tasks), "n_windows": n_windows,
        "n_manifest_windows": n_manifest, "n_selected": n_selected,
        "window_stride": args.window_stride,
        "n_errors": len(errors), "errors": errors[:20],
        "verdict_distribution": dict(sorted(verdicts.items())),
        "profile_distribution": dict(sorted(profiles.items())),
        "windows_per_s": round(n_windows / wall, 4) if wall else None,
        "footage_h_per_wall_h": round(man["total_hours"] / (wall / 3600.0), 3)
        if wall else None,
        "processed_footage_h_per_wall_h": round(
            (n_windows * WINDOW_S / 3600.0) / (wall / 3600.0), 3) if wall else None,
        "optimizations": opts,
        "compare_enabled": bool(args.compare),
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "device": DEVICE,
        "rule_table_sha256": _RULES_SHA,
        "reject_threshold": _REJECT_THR,
        "determinism_probe": det_rows,
        "smi_samples": len(smi_log),
        "smi": smi_log,
        "git_sha": e8.git_sha(),
    }
    json.dump(meta, open(os.path.join(OUTDIR, "run_meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items()
                      if k not in ("errors", "smi", "determinism_probe")}, indent=1))
    print(f"DONE wall={wall:.0f}s windows={n_windows} errors={len(errors)}")


def _log(k, n, nok, nerr, t0):
    if k % 25 == 0 or k == n:
        el = time.perf_counter() - t0
        print(f"  {k}/{n} ok={nok} err={nerr} {el:.0f}s {k/el:.2f} win/s "
              f"eta={(n-k)/max(k/el,1e-9)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
