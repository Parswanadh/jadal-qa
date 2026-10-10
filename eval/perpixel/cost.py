"""M4 cost: RTF on 1 CPU core and GPU, decode in/excluded; where time goes.

Output: eval/results/perpixel/cost.json
Regenerate: PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/perpixel/cost.py
GPU section needs tools/gpu.lock (flock -w 60); records lock-busy honestly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results", "perpixel")
SPEC_DIR = "/home/parshu/projects/jadal-qa-wt/spectral_data"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def main():
    gm = json.load(open(os.path.join(SPEC_DIR, "manifest_gold.json")))
    wins = [(it["path"], it["start_s"], it["end_s"]) for it in gm["items"][:6]]
    from jadal.perpixel import series as S
    from jadal.perpixel import theory as T
    from jadal.perpixel import physics as P
    from jadal.perpixel import features as F
    # warmup
    fr, fps = S.decode_160x90(*wins[0])
    # decode-only floor (1 core via taskset)
    t0 = time.time()
    for w in wins:
        subprocess.run(["taskset", "-c", "0", "ffmpeg", "-hide_banner", "-v",
                        "error", "-ss", str(w[1]), "-i", w[0], "-t",
                        str(w[2] - w[1]), "-map", "0:v:0", "-vf",
                        "scale=160:90", "-pix_fmt", "rgb24", "-f", "rawvideo",
                        "-"], capture_output=True, timeout=300)
    dec_s = time.time() - t0
    # full extract on 1 core
    per = {}
    for name, fn in (("series", lambda f, fp: S.describe_series(
            S.per_frame_series(f), fp)),
                     ("theory", lambda f, fp: T.extract_theory(f, fp)),
                     ("physics", lambda f, fp: P.extract_physics(
                         f, fp, wins[0][0], wins[0][1], wins[0][2]))):
        t0 = time.time()
        for w in wins:
            fr, fp = S.decode_160x90(*w)
            fn(fr, fp)
        per[name] = time.time() - t0
    # M1 row (armB + gpu-ref + series) on 1 core
    t0 = time.time()
    for w in wins:
        F.extract_row(*w)
    per["m1_row"] = time.time() - t0
    vid_s = sum(w[2] - w[1] for w in wins)
    out = {"n_windows": len(wins), "video_s": vid_s,
           "cpu_1core_s": {"decode_only": dec_s, **per}}
    full_1core = per["m1_row"] + per["theory"] + per["physics"]
    rtf = vid_s / full_1core
    out["cpu_1core"] = {"full_s": full_1core, "RTF_incl_decode": rtf,
                        "segments_per_hour": 3600 * len(wins) / full_1core,
                        "cores_for_5000x": 5000 / rtf,
                        "cores_for_3333x": 3333 / rtf}
    # GPU: frames/s of the G_* kernels under flock (short wait)
    try:
        p = subprocess.run(
            ["flock", "-w", "60",
             "/home/parshu/projects/jadal-qa-wt/tools/gpu.lock",
             "/home/parshu/envs/jadal/bin/python", "-c",
             ("import time,torch,numpy as np,sys;"
              "sys.path.insert(0,'.');"
              "from jadal.perpixel import series as S;"
              "from jadal.gpu import features as G;"
              "fr,fps=S.decode_160x90(*" + repr(wins[0]) + ");"
              "import numpy as _np;"
              "luma=(0.299*fr[...,0]+0.587*fr[...,1]+0.114*fr[...,2]);"
              "t0=time.time();n=0;"
              "dev=torch.device('cuda');"
              "lt=torch.from_numpy(luma[None]).to(dev).half();"
              "rt=torch.from_numpy(fr[None]).to(dev).half();"
              "import torch as _t;"
              "with _t.no_grad():"
              "  [G.gpu_window_features_batch(lt,rt,fps=fps) for _ in range(5)];"
              "_t.cuda.synchronize();"
              "dt=time.time()-t0;"
              "print('GPU_FRAMES_S='+str(5*fr.shape[0]/dt))")],
            capture_output=True, text=True, timeout=300)
        line = [x for x in (p.stdout or "").splitlines() if "GPU_FRAMES_S" in x]
        out["gpu"] = {"frames_s": float(line[0].split("=")[1])} if line else {
            "note": "gpu probe produced no output", "rc": p.returncode,
            "stderr": (p.stderr or "")[-300:]}
    except Exception as e:  # noqa: BLE001
        out["gpu"] = {"note": f"lock busy or failed: {e}"}
    with open(os.path.join(RES, "cost.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1)[:1500], flush=True)


if __name__ == "__main__":
    main()
