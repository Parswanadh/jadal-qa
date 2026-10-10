"""E7.01 — cost model: hours of video processed per wall-hour on THIS pod (A40).

Two stages:

  --stage measure   timed measurements on this pod (no labels, deterministic inputs):
                      * decode: CPU ffmpeg vs NVDEC (cuvid) for the two source
                        families in play (480p/10fps H.264 IndEgo, 1080p/30fps
                        HEVC Egocentric-10K), decoded to the profiles' work res 160x90
                      * pixel26 features: jadal.gpu.features.cpu_window_features at
                        fps {10,5,2} on real IndEgo windows (OMP_NUM_THREADS=1)
                      * EmbeddingGemma-2 (batch 1, bf16) on REAL mosaic PNGs
                        (artifacts_mosaics, N=9 / tile 224 / 680x680) and on
                        tile-sized images of the spec geometry
                      * logistic head cost
                    -> eval/results/exp_cost/E7.01_measure.json

  --stage model     arithmetic combination of the measured numbers + the source
                    JSONs produced by other workers (E1.01, E1.02, E1.04, E2.01,
                    E3.04) into per-profile rows (profiles/*.json).
                    -> eval/results/exp_cost/E7.01.json

Every input number carries the file it came from and that file's git_sha.
Nothing is guessed: a missing input is written "not measured".

Regenerate (from the repo root):
  bash tools/pod/gpu_run.sh python3 eval/exp_cost/run_e7.py --stage measure
  python3 eval/exp_cost/run_e7.py --stage model
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

BUNDLE = os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle")
EG2_DIR = os.environ.get(
    "JADAL_EG2_DIR",
    "/workspace/.cache/huggingface/hub/models--google--embeddinggemma-2/"
    "snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b")
MOSAIC_ART = os.environ.get("JADAL_MOSAIC_ARTIFACTS", "/workspace/artifacts/mosaics")
RESULTS = os.path.join(ROOT, "eval", "results", "exp_cost")
MEASURE_JSON = os.path.join(RESULTS, "E7.01_measure.json")
OUT_JSON = os.path.join(RESULTS, "E7.01.json")

W, H = 160, 90                 # profiles/*.json features.work_res
WINDOW_S = 10.0
TILE_PX = 224                  # E2.01 tile_px
REF_N = 9                      # E2.01 reference grid (3x3)
GUTTER = 2                     # jadal/mosaic/build.py GUTTER
DEVICE = "A40"

# source families -> (native fps, codec, cuvid decoder)
FAMILIES = {
    "indego_480p10_h264": {"native_fps": 10.0, "cuvid": "h264_cuvid"},
    "ego10k_1080p30_hevc": {"native_fps": 30.0, "cuvid": "hevc_cuvid"},
}


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def git_sha(path=None):
    if path and os.path.isdir(os.path.join(path, ".git")) is False:
        pass
    cmd = ["git", "-C", path or ROOT, "rev-parse", "--short", "HEAD"]
    r = sh(cmd)
    return r.stdout.strip() or "unknown"


def boot_ci(x, B=2000, seed=0):
    """Percentile bootstrap CI95 over timed repetitions (pod-local)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(B)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


# --------------------------------------------------------------- decode ----
def timed_decode(path, start_s, dur_s, hw, cuvid, frames_expect):
    """Decode dur_s seconds to 160x90 gray rawvideo; return (wall_s, frames)."""
    if hw == "nvdec":
        pre = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
               "-c:v", cuvid]
        vf = f"scale_cuda={W}:{H}:format=nv12,hwdownload,format=nv12"
    else:
        pre, vf = [], f"scale={W}:{H}"
    cmd = (["ffmpeg", "-hide_banner", "-v", "error"] + pre +
           ["-ss", f"{start_s}", "-t", f"{dur_s}", "-i", path,
            "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"])
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    nbytes = 0
    while True:
        chunk = p.stdout.read(1 << 20)
        if not chunk:
            break
        nbytes += len(chunk)
    p.wait()
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        return None, 0, p.stderr.read().decode()[-300:]
    return wall, nbytes // (W * H), ""


def measure_decode(clips, offsets=(37.0, 91.0, 143.0), durs=(4.0, 10.0)):
    out = {}
    for fam, path in clips.items():
        cuvid = FAMILIES[fam]["cuvid"]
        for hw in ("cpu", "nvdec"):
            key = f"{fam}__{hw}"
            per_dur = {}
            for dur in durs:
                walls, frames = [], []
                for off in offsets:
                    w, f, err = timed_decode(path, off, dur, hw, cuvid, dur)
                    if w is None:
                        per_dur = {"error": err}
                        break
                    walls.append(w)
                    frames.append(f)
                if "error" in per_dur:
                    break
                fps = float(np.mean(frames) / np.mean(walls)) if walls else None
                per_dur[str(dur)] = {
                    "n_windows": len(walls),
                    "frames_mean": float(np.mean(frames)),
                    "wall_s_mean": float(np.mean(walls)),
                    "wall_s_ci95": boot_ci(walls),
                    "fps": fps,
                    "ms_per_window": float(np.mean(walls) / np.mean(frames)
                                           * dur * FAMILIES[fam]["native_fps"]
                                           * 1000.0) if walls else None,
                }
            out[key] = {"source": path, "native_fps": FAMILIES[fam]["native_fps"],
                        "by_duration_s": per_dur}
    return out


# ------------------------------------------------------------- features ----
def measure_features(n_windows=12):
    from eval.exp_fps.fps_lib import decode_window, load_manifest, subsample_idx
    from jadal.gpu import features as gf
    items = [it for it in load_manifest() if it["clip"] in
             ("ecu_full", "desk", "pallet", "organise", "logistics4")][:n_windows]
    cfgs = {"10": (1, 10.0), "5": (2, 5.0), "2": (5, 2.0)}
    ms = {k: [] for k in cfgs}
    dec_ms = []
    for it in items:
        rgb, luma, dms = decode_window(it["path"], it["start_s"])
        dec_ms.append(dms)
        T = len(luma)
        for k, (step, feff) in cfgs.items():
            idx = subsample_idx(T, step)
            t0 = time.perf_counter()
            gf.cpu_window_features(luma[idx].copy(), rgb[idx].copy(), fps=feff)
            ms[k].append((time.perf_counter() - t0) * 1000.0)
    return {
        "n_windows": len(items),
        "decode_160x90_ms_per_window": float(np.mean(dec_ms)),
        "decode_160x90_ms_ci95": boot_ci(dec_ms),
        "features_ms_per_window": {k: {"mean": float(np.mean(v)),
                                       "ci95": boot_ci(v),
                                       "n": len(v)} for k, v in ms.items()},
    }


# ------------------------------------------------------------ embedding ----
def _crop_tiles(png, n_side=3, tile=TILE_PX, gutter=GUTTER):
    from PIL import Image
    im = Image.open(png).convert("RGB")
    tiles = []
    for r in range(n_side):
        for c in range(n_side):
            x = gutter + c * (tile + gutter)
            y = gutter + r * (tile + gutter)
            tiles.append(im.crop((x, y, x + tile, y + tile)))
    return tiles


def _canvas(tiles, rows, cols, tile=TILE_PX, gutter=GUTTER):
    from PIL import Image
    w = cols * tile + (cols + 1) * gutter
    h = rows * tile + (rows + 1) * gutter
    im = Image.new("RGB", (w, h), (128, 128, 128))
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        im.paste(tiles[i % len(tiles)],
                 (gutter + c * (tile + gutter), gutter + r * (tile + gutter)))
    return im


def measure_embedding(n_timed=8, n_warm=2):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = SentenceTransformer(str(EG2_DIR), model_kwargs={"torch_dtype": dtype},
                                trust_remote_code=True)
    model.eval()

    def encode(pils):
        with torch.inference_mode():
            return model.encode(pils, batch_size=1, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=True,
                                processing_kwargs={"image": {"max_soft_tokens": 140}})

    pngs = sorted(glob.glob(os.path.join(MOSAIC_ART, "*.png")))[:4]
    if not pngs:
        return {"error": f"no mosaic PNGs under {MOSAIC_ART}"}
    tiles = _crop_tiles(pngs[0])
    from PIL import Image
    inputs = {
        "tile_224x224": [tiles[0]],
        "mosaic_2x2_454x454": [_canvas(tiles, 2, 2)],
        "mosaic_3x3_680x680_real": [Image.open(p).convert("RGB") for p in pngs],
        "mosaic_4x4_906x906": [_canvas(tiles, 4, 4)],
    }
    torch.cuda.reset_peak_memory_stats()
    res = {}
    for name, pils in inputs.items():
        for p in pils[:n_warm]:
            encode([p])
        torch.cuda.synchronize()
        # cycle the images so every group gets the same number of timed reps
        seq = [pils[i % len(pils)] for i in range(n_timed)]
        ts = []
        for p in seq:
            t0 = time.perf_counter()
            encode([p])
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        res[name] = {
            "n_timed": len(ts),
            "size": list(pils[0].size),
            "ms_per_image_mean": float(np.mean(ts) * 1000.0),
            "ms_per_image_ci95": [float(x * 1000.0) for x in boot_ci(ts)],
            "images_per_s": float(1.0 / np.mean(ts)),
        }
    return {"device": DEVICE, "batch_size": 1, "dtype": str(dtype),
            "max_soft_tokens": 140, "peak_vram_mb": float(
                torch.cuda.max_memory_allocated() / 2 ** 20),
            "mosaic_artifacts": MOSAIC_ART, "inputs": res}


def measure_head(n=1000):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 26))
    y = (rng.random(60) > 0.5).astype(int)
    m = LogisticRegression(max_iter=200).fit(X, y)
    Xt = rng.normal(size=(1, 26))
    t0 = time.perf_counter()
    for _ in range(n):
        m.predict_proba(Xt)
    return {"n_predicts": n, "ms_per_predict": (time.perf_counter() - t0)
            * 1000.0 / n}


def stage_measure(args):
    clips = {
        "indego_480p10_h264": os.path.join(
            BUNDLE, "data/indego/1_Assembly/mp4_480/User_15_410_0411_2_480.mp4"),
        "ego10k_1080p30_hevc": sorted(glob.glob(os.path.join(
            BUNDLE, "data/ego10k_scale/clips/factory_001/worker_006/*.mp4")))[0],
    }
    for k, v in clips.items():
        assert os.path.exists(v), f"missing clip for {k}: {v}"
    out = {"experiment": "E7.01", "stage": "measure", "device": DEVICE,
           "git_sha": git_sha(), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                             time.gmtime()),
           "nproc": os.cpu_count(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "work_res": [W, H]}
    out["decode"] = measure_decode(clips)
    out["features_pixel26"] = measure_features(args.feature_windows)
    out["embedding_eg2"] = measure_embedding()
    out["head_logistic"] = measure_head()
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(MEASURE_JSON, "w"), indent=1)
    print("wrote", MEASURE_JSON)


# ---------------------------------------------------------------- model ----
def video_hours_per_wall_hour(total_ms, k, window_s=WINDOW_S):
    """Hours of footage covered per wall-hour for one serial stream."""
    if not total_ms or total_ms <= 0:
        return None
    return 3.6e6 / total_ms * k * window_s / 3600.0


def decode_rate_fps(dec, fam, hw, dur_key="4.0"):
    """Measured decode rate (fps) for one source family / path.

    The rate is span-independent; the 4 s window is the reference, the 10 s
    window is a consistency check (reported in the JSON).
    """
    try:
        by = dec[f"{fam}__{hw.lower()}"]["by_duration_s"]
        return by[dur_key]["fps"], by.get("10.0", {}).get("fps")
    except (KeyError, TypeError):
        return None, None


def decode_ms_for(dec, fam, hw, obs_s):
    """ms to decode the observed span (all native frames inside it)."""
    native_fps = FAMILIES[fam]["native_fps"]
    rate, _ = decode_rate_fps(dec, fam, hw)
    if not rate:
        return None
    return obs_s * native_fps / rate * 1000.0


def find_source(relpath):
    """Locate a source results JSON: own branch first, then sibling worktrees."""
    cands = [os.path.join(ROOT, relpath)]
    cands += sorted(glob.glob(os.path.join("/workspace/wt/*", relpath)))
    cands += sorted(glob.glob(os.path.join("/workspace/repo", relpath)))
    cands += sorted(glob.glob(os.path.join(
        "/workspace/jadal_pod_bundle/local_work/*", relpath)))
    for c in cands:
        if os.path.exists(c):
            wt = c.split("/eval/")[0]
            return c, git_sha(wt if os.path.isdir(os.path.join(wt, ".git"))
                              else None)
    return None, None


def load_source(relpath):
    p, sha = find_source(relpath)
    if p is None:
        return None, {"path": relpath, "git_sha": None, "status": "missing"}
    return json.load(open(p)), {"path": p, "git_sha": sha, "status": "found"}


def stage_model(args):
    meas = json.load(open(MEASURE_JSON))
    src_meta, src = {}, {}

    def S(key, relpath):
        d, m = load_source(relpath)
        src_meta[key] = m
        src[key] = d
        return d

    e101 = S("E1.01", "eval/results/exp_fps/e1_01.json")
    e102 = S("E1.02", "eval/results/exp_fps/e1_02.json")
    e104 = S("E1.04", "eval/results/exp_fps/E1.04.json")
    e201 = S("E2.01", "eval/results/exp_mosaic/E2.01.json")
    e304 = S("E3.04", "eval/results/exp_temporal/E3.04_compute.json")
    # measured pod-local numbers
    dec = meas["decode"]
    feat = meas["features_pixel26"]
    emb = meas["embedding_eg2"]
    head_ms = meas["head_logistic"]["ms_per_predict"]

    def dec_fps(fam, hw, dur):
        return decode_rate_fps(dec, fam, hw)[0]

    def emb_ms(name):
        try:
            return emb["inputs"][name]["ms_per_image_mean"]
        except (KeyError, TypeError):
            return None

    ms_tile = emb_ms("tile_224x224")
    ms_mosaic = emb_ms("mosaic_3x3_680x680_real")

    # E1.01 fallback (measured on the same pod, OMP_NUM_THREADS=2) if a local
    # feature measurement is missing for an fps
    e101_feat = {}
    if e101:
        for r in e101.get("rows", []):
            e101_feat[str(r["fps"])] = r.get("feat_ms_mean")
    # E1.01 decode reference (CPU cv2 160x90, IndEgo 480p, 100-frame window)
    e101_decode_ms = (e101 or {}).get("rows", [{}])[0].get("decode_ms_mean")
    e104_decode_ms = (e104 or {}).get("decode_ms_mean")

    # per-profile source family: only the native-1080p profile decodes HEVC
    FAMILY_OF = {"factory_1080p30": "ego10k_1080p30_hevc"}
    DEFAULT_FAM = "indego_480p10_h264"

    rows, not_measured = [], []
    profiles = []
    for p in sorted(glob.glob(os.path.join(ROOT, "profiles", "*.json"))):
        if p.endswith("schema.json"):
            continue
        profiles.append(json.load(open(p)))

    for pr in profiles:
        s = pr["sampling"]
        fps = float(s["fps"])
        k = int(s.get("probe_every_k_windows", 1))
        exit_s = s.get("early_exit_s")
        obs_s = float(exit_s) if exit_s else WINDOW_S
        fam = FAMILY_OF.get(pr["name"], DEFAULT_FAM)
        native_fps = FAMILIES[fam]["native_fps"]
        fset = pr["features"]["set"]

        # decode: all native frames inside the observed span (samplers cannot
        # skip frames during decode; E1.04 measured selection after decode)
        dec_ms = {hw: decode_ms_for(dec, fam, hw, obs_s) for hw in ("CPU", "NVDEC")}
        rate_ref = {hw: decode_rate_fps(dec, fam, hw) for hw in ("CPU", "NVDEC")}
        # feature cost: pixel26 measured ms/window at the profile fps
        fpix = feat["features_ms_per_window"].get(str(int(fps))) or \
            feat["features_ms_per_window"].get(f"{fps:g}")
        if fpix is None and e101_feat.get(str(int(fps))):
            # fallback to the E1.01 pod measurement (OMP_NUM_THREADS=2 run)
            fpix = {"mean": e101_feat[str(int(fps))], "ci95": None,
                    "source": "E1.01"} if str(int(fps)) in e101_feat else None
        pix_ms = fpix["mean"] if (fpix and fset in ("pixel26", "pixel26_lite",
                                                    "both")) else 0.0
        if fset in ("pixel26", "pixel26_lite", "both") and fpix is None:
            not_measured.append(f"{pr['name']}: pixel26 ms at fps={fps}")
        # embedding cost: eg2 set embeds fps*obs_s frames one by one (batch 1)
        n_frames_emb = fps * obs_s
        eg2_ms = n_frames_emb * ms_tile if (fset in ("eg2", "both")
                                            and ms_tile) else 0.0
        if fset in ("eg2", "both") and ms_tile is None:
            not_measured.append(f"{pr['name']}: eg2 ms/frame")
        # mosaic evidence: one mosaic image per processed window
        mosaic_on = bool(pr.get("evidence", {}).get("mosaic"))
        mos_ms = ms_mosaic if (mosaic_on and ms_mosaic) else 0.0

        for hw in ("CPU", "NVDEC"):
            dms = dec_ms[hw]
            total = (dms or 0.0) + pix_ms + eg2_ms + mos_ms + head_ms
            wph = 3.6e6 / total if total else None
            video_hph = video_hours_per_wall_hour(total, k)
            rows.append({
                "profile": pr["name"], "source_family": fam,
                "decode_path": hw, "fps": fps, "probe_every_k_windows": k,
                "early_exit_s": exit_s, "observed_s": obs_s,
                "mosaic": mosaic_on,
                "features_set": fset,
                "decode_rate_fps_ref_4s": (round(rate_ref[hw][0], 1)
                                           if rate_ref[hw][0] else None),
                "decode_rate_fps_check_10s": (round(rate_ref[hw][1], 1)
                                              if rate_ref[hw][1] else None),
                "ms_per_window": {
                    "decode": round(dms, 1) if dms is not None else None,
                    "features_pixel26": round(pix_ms, 1),
                    "embedding_eg2": round(eg2_ms, 1),
                    "mosaic_evidence": round(mos_ms, 1),
                    "head": round(head_ms, 3),
                    "total": round(total, 1),
                },
                "decode_share_of_total": (round((dms or 0) / total, 3)
                                          if total else None),
                "windows_per_wall_hour": round(wph, 1) if wph else None,
                "video_hours_per_wall_hour": (round(video_hph, 2)
                                              if video_hph else None),
                "footage_fraction_probed": round(1.0 / k, 3),
            })

    # reference rows: full sampling (baseline) vs adopted E3 policy, IndEgo source
    def synthetic_row(name, fam, fps, k, exit_s, fset, hw):
        obs = float(exit_s) if exit_s else WINDOW_S
        dec_ms = decode_ms_for(dec, fam, hw, obs)
        fpix = feat["features_ms_per_window"].get(str(int(fps)))
        pix_ms = fpix["mean"] if (fpix and fset in ("pixel26", "both")) else 0.0
        eg2_ms = fps * obs * ms_tile if (fset == "eg2" and ms_tile) else 0.0
        total = (dec_ms or 0) + pix_ms + eg2_ms + (ms_mosaic or 0) + head_ms
        return {"profile": name, "source_family": fam, "decode_path": hw,
                "fps": fps, "probe_every_k_windows": k, "early_exit_s": exit_s,
                "observed_s": obs, "mosaic": True, "features_set": fset,
                "ms_per_window": {"decode": round(dec_ms, 1) if dec_ms else None,
                                  "features_pixel26": round(pix_ms, 1),
                                  "embedding_eg2": round(eg2_ms, 1),
                                  "mosaic_evidence": round(ms_mosaic or 0, 1),
                                  "head": round(head_ms, 3),
                                  "total": round(total, 1)},
                "decode_share_of_total": round((dec_ms or 0) / total, 3) if total else None,
                "windows_per_wall_hour": round(3.6e6 / total, 1) if total else None,
                "video_hours_per_wall_hour": (round(video_hours_per_wall_hour(total, k), 2)
                                              if total else None),
                "footage_fraction_probed": round(1.0 / k, 3)}

    ref_rows = [
        synthetic_row("E3_baseline_full_pixel26", DEFAULT_FAM, 10, 1, None, "pixel26", "CPU"),
        synthetic_row("E3_adopted_every2_exit4_pixel26", DEFAULT_FAM, 2, 2, 4, "pixel26", "CPU"),
        synthetic_row("E3_adopted_every2_exit4_pixel26_nvdec", DEFAULT_FAM, 2, 2, 4, "pixel26", "NVDEC"),
        synthetic_row("mosaic_only_every2_exit4", DEFAULT_FAM, 0.9, 2, 4, "eg2", "NVDEC"),
    ]

    out = {
        "experiment": "E7.01",
        "title": "Cost model: hours of video per wall-hour on the pod A40, per profile",
        "n": {"windows_timed_decode": sum(
                  v["by_duration_s"].get("4.0", {}).get("n_windows", 0)
                  for v in dec.values()),
              "windows_timed_features": feat["n_windows"],
              "images_timed_embedding": sum(v["n_timed"] for v in
                                            emb["inputs"].values()),
              "profiles_modelled": len(profiles)},
        "label_source": "n/a (cost model; no labels used; no accuracy claim)",
        "protocol": ("timed measurements on the pod A40 host (decode CPU vs NVDEC, "
                     "pixel26 CPU features OMP_NUM_THREADS=1, EmbeddingGemma-2 "
                     "batch-1 bf16 images) combined arithmetically per profiles/*.json; "
                     "CI = percentile bootstrap over timed repetitions (decode 6 "
                     "windows/config, features 12 windows, embedding 8 images)"),
        "ci95": {"decode_fps": {k: {d: v.get("wall_s_ci95") for d, v in
                                    cfg["by_duration_s"].items()}
                                for k, cfg in dec.items()},
                 "features_ms_per_window": {k: v["ci95"] for k, v in
                                            feat["features_ms_per_window"].items()},
                 "embedding_ms_per_image": {k: v["ms_per_image_ci95"] for k, v in
                                            emb["inputs"].items()}},
        "device": DEVICE,
        "decode_path_note": ("NVDEC = ffmpeg -hwaccel cuda -hwaccel_output_format cuda "
                             "-c:v {h264,hevc}_cuvid + scale_cuda to 160x90; "
                             "CPU = ffmpeg -vf scale=160:90; both measured on the same clips"),
        "command": ("bash tools/pod/gpu_run.sh python3 eval/exp_cost/run_e7.py --stage measure "
                    "&& python3 eval/exp_cost/run_e7.py --stage model"),
        "git_sha": git_sha(),
        "measured_this_run": {
            "decode": dec, "features_pixel26": feat,
            "embedding_eg2": {k: v for k, v in emb.items() if k != "inputs"},
            "embedding_ms_per_image": {k: v for k, v in emb["inputs"].items()},
            "head_logistic": meas["head_logistic"],
        },
        "source_inputs": src_meta,
        "assumptions": [
            "serial single-stream pipeline: one window processed at a time, no batching/pipelining (not measured)",
            "decode cost is proportional to native frames inside the OBSERVED span; samplers cannot skip frames during decode (E1.04: selection happens after decode)",
            "one processed window covers probe_every_k_windows x 10 s of footage (sparse probe; E3.01)",
            "early exit = stop observing after early_exit_s; features/embeddings use only frames inside that span",
            "pixel26 features are CPU float64 at 160x90 (jadal.gpu.features.cpu_window_features)",
            "eg2 embeddings are EmbeddingGemma-2 image embeddings, batch 1, bf16, max_soft_tokens=140",
            "mosaic evidence = one N=9/tile-224/680x680 PNG per processed window (E2.01 geometry), embedded batch 1",
            "logistic head cost measured on this pod",
            "hours/wall-hour = 3600 / ms_per_window * probe_k * 10 s / 3600",
        ],
        "not_measured": not_measured,
        "cross_checks": {},
        "rows": rows,
        "reference_rows": ref_rows,
        "headline": None,
    }
    # cross-checks against the source JSONs (independent measurements of the
    # same quantities, different runs/settings) -- reported, never averaged in
    cc = {}
    if e101:
        mine10 = feat["features_ms_per_window"].get("10", {}).get("mean")
        cc["features_pixel26_10fps"] = {
            "this_run_OMP1_ms": round(mine10, 1) if mine10 else None,
            "E1.01_OMP2_ms": e101_feat.get("10"),
            "note": "E1.01 ran two shards in parallel; this run is single-process OMP_NUM_THREADS=1",
        }
    if e101_decode_ms:
        cc["decode_indego_480p_cpu"] = {
            "this_run_ms_per_100f_window": (
                round(100 / decode_rate_fps(dec, DEFAULT_FAM, "CPU")[0] * 1000.0, 1)
                if decode_rate_fps(dec, DEFAULT_FAM, "CPU")[0] else None),
            "E1.01_cv2_ms": e101_decode_ms,
            "E1.04_ffmpeg_ms": e104_decode_ms,
            "note": "E1.01/E1.04 decoded with cv2 at 160x90; this run uses ffmpeg -vf scale=160:90 (same work res)",
        }
    if e201:
        cc["mosaic_cost_aggregate_E2.01"] = {
            "decode_ms": e201["cost"]["decode_ms"], "build_ms": e201["cost"]["build_ms"],
            "embed_ms": e201["cost"]["embed_ms"], "n_windows": e201["cost"]["n_windows"],
            "note": ("E2.01 aggregate over 28 windows x 15 mosaic/variant images "
                     "(all variants in one pass) -> NOT a per-window number; "
                     "this run times the mosaic embed alone"),
        }
    if e304:
        cc["E3.04_arithmetic"] = {
            "corpus_hours": e304.get("corpus_hours"),
            "per_policy": e304.get("per_policy"),
            "with_early_exit": e304.get("with_early_exit"),
        }
        cc["E3.04_vs_E7.01"] = ("E3.04 counts PROCESSED FOOTAGE fractions "
                                "(arithmetic, no wall clock); E7.01 counts wall-clock "
                                "throughput given the measured pipeline cost -- the two "
                                "speed-ups are different metrics and are not expected to match")
    if e102:
        cc["E1.02_eg2_fps"] = {"rows": e102.get("eg2_rows"),
                               "note": "cached 0.5s-spacing EG2 embeddings; fewer frames did not hurt"}
    out["cross_checks"] = cc

    # headline: best IndEgo profile row and speed-up vs the full-sampling profile
    def best(name_filter=None, hw="NVDEC"):
        c = [r for r in rows if r["decode_path"] == hw and
             (name_filter is None or r["profile"] == name_filter)]
        return max(c, key=lambda r: r["video_hours_per_wall_hour"] or 0) if c else None
    base = [r for r in rows if r["profile"] == "default" and r["decode_path"] == "NVDEC"]
    top = best()
    out["headline"] = {
        "best_profile_row": top,
        "baseline_default_nvdec": base[0] if base else None,
        "speedup_x": (round(top["video_hours_per_wall_hour"] /
                            base[0]["video_hours_per_wall_hour"], 2)
                      if top and base and base[0]["video_hours_per_wall_hour"] else None),
        "bound": ("decode-bound" if top and top["decode_share_of_total"] and
                  top["decode_share_of_total"] > 0.5 else "feature/embedding-bound")
        if top else "not measured",
    }
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print("wrote", OUT_JSON)
    for r in rows:
        print(f"  {r['profile']:22s} {r['decode_path']:5s} total={r['ms_per_window']['total']:8.1f}ms "
              f"video_h/wall_h={r['video_hours_per_wall_hour']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["measure", "model", "all"], default="all")
    ap.add_argument("--feature-windows", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if a.stage in ("measure", "all"):
        stage_measure(a)
    if a.stage in ("model", "all"):
        stage_model(a)


if __name__ == "__main__":
    main()
