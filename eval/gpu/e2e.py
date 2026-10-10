"""End-to-end sustained run: CPU decode -> GPU batched features -> S0/S1 cascade.

>= 120 ten-second windows (>= 20 video-minutes) round-robin over the 7 clips.
Run under lock (needs GPU for the batched kernels):
  flock -w 3600 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock \
    /home/parshu/envs/jadal/bin/python eval/gpu/e2e.py [nwin] [batch]
Writes eval/results/gpu/e2e.json (incl. start/end timestamps for telemetry join).
Telemetry join needs no lock: eval/gpu/telemetry.py window_stats().
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "..", "results", "gpu", "e2e.json")
GOLD = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"
FEATS = os.path.join(HERE, "..", "results", "gpu", "features.json")

W, H = 160, 90
THR, MARGIN = 0.6, 0.25


def main(nwin=120, B=8):
    import numpy as np
    import torch
    from eval.gpu.decode import decode_window, safe_offset
    from eval.gpu.stream import S0_RULES
    from eval.gpu.telemetry import now_str
    from jadal.gpu.features import gpu_window_features_batch, FEATURE_NAMES
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()

    gold = json.load(open(GOLD))["items"]
    seen, clips = set(), []
    for it in gold:
        if it["clip"] not in seen:
            seen.add(it["clip"])
            clips.append(it)
    # train S1 head once on full-window features (CPU, seconds)
    from sklearn.linear_model import LogisticRegression
    feats = json.load(open(FEATS))["windows"]
    names = FEATURE_NAMES
    ids = [i["id"] for i in gold if i["id"] in feats and "full" in feats[i["id"]]]
    X = np.stack([[feats[i]["full"][k] for k in names] for i in ids])
    y = np.array([next(g["label"] for g in gold if g["id"] == i) for i in ids])
    mu, sd = X.mean(0), X.std(0) + 1e-12
    clf = LogisticRegression(max_iter=2000, C=1.0).fit((X - mu) / sd, y)

    t_start = now_str()
    t0 = time.perf_counter()
    dec_s = gpu_s = 0.0
    exits = {"S0": 0, "S1_accept": 0, "S1_reject": 0, "S2": 0}
    s0_evidence = []
    batch, done = [], 0
    for i in range(nwin):
        it = clips[i % len(clips)]
        t1 = time.perf_counter()
        rgb, fps = decode_window(it["path"], safe_offset(it["path"], i, 10.0),
                                 10.0, W, H, "rgb24")
        dec_s += time.perf_counter() - t1
        # S0 probe on first 2 s
        g2 = (0.299 * rgb[:20, ..., 0] + 0.587 * rgb[:20, ..., 1]
              + 0.114 * rgb[:20, ..., 2])
        probe = {"s0p95": float(np.percentile(g2, 95)),
                 "s0p50": float(np.percentile(g2, 50)),
                 "s0var": float(g2[:10].mean(axis=0).var()),
                 "s0bright": 1.0 if float(np.percentile(g2, 95)) > 0.95 else 0.0,
                 "s0mot": float(np.abs(np.diff(g2, axis=0)).mean())}
        hit = None
        for name, key, op, thr in S0_RULES:
            v = probe[key]
            if (op == "<" and v < thr) or (op == ">" and v > thr):
                hit = (name, key, op, thr, v)
                break
        if hit:
            exits["S0"] += 1
            s0_evidence.append((it["clip"], hit[0], round(hit[4], 5)))
            done += 1
            continue
        batch.append((rgb.astype(np.float32), probe))
        if len(batch) == B or i == nwin - 1:
            t2 = time.perf_counter()
            ten = torch.from_numpy(np.stack([b[0] for b in batch])).half().cuda()
            luma = (0.299 * ten[..., 0] + 0.587 * ten[..., 1]
                    + 0.114 * ten[..., 2])
            with torch.no_grad():
                T = gpu_window_features_batch(luma, ten, fps=10.0).float().cpu().numpy()
            gpu_s += time.perf_counter() - t2
            P = clf.predict_proba((T - mu) / sd)[:, 1]
            for p in P:
                if p >= THR + MARGIN:
                    exits["S1_accept"] += 1
                elif p <= THR - MARGIN:
                    exits["S1_reject"] += 1
                else:
                    exits["S2"] += 1
            done += len(batch)
            batch = []
    wall = time.perf_counter() - t0
    if batch:  # flush partial tail (e.g. last windows exited at S0)
        t2 = time.perf_counter()
        ten = torch.from_numpy(np.stack(batch)).half().cuda()
        luma = (0.299 * ten[..., 0] + 0.587 * ten[..., 1]
                + 0.114 * ten[..., 2])
        with torch.no_grad():
            T = gpu_window_features_batch(luma, ten, fps=10.0).float().cpu().numpy()
        gpu_s += time.perf_counter() - t2
        P = clf.predict_proba((T - mu) / sd)[:, 1]
        for p in P:
            if p >= THR + MARGIN:
                exits["S1_accept"] += 1
            elif p <= THR - MARGIN:
                exits["S1_reject"] += 1
            else:
                exits["S2"] += 1
        done += len(batch)
        batch = []
    torch.cuda.synchronize()
    t_end = now_str()
    wps = done / wall
    res = {
        "device": "RTX 4070 Laptop, NOT a Pi, NOT a datacenter GPU",
        "regenerate": ("flock -w 3600 /home/parshu/projects/jadal-qa-wt/tools/gpu.lock "
                       "/home/parshu/envs/jadal/bin/python eval/gpu/e2e.py"),
        "nwin": done, "video_min": round(done * 10 / 60, 1),
        "corpus": "round-robin 10-s windows over 7 IndEgo clips (content repeats; decode cost realistic)",
        "batch": B, "threshold": THR, "margin": MARGIN,
        "wall_s": round(wall, 1), "decode_s": round(dec_s, 1),
        "gpu_batch_s": round(gpu_s, 1),
        "windows_s": round(wps, 2), "hours_per_hour": round(wps * 10, 1),
        "video_s_per_wall_s": round(wps * 10, 1),
        "exits": exits,
        "exit_frac": {k: round(v / done, 3) for k, v in exits.items()},
        "s0_evidence_sample": s0_evidence[:10],
        "s2_note": "S2 = queue for expensive tier (mediapipe CPU / EG2 GPU); not run here",
        "gpu_peak_vram_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "vram_budget_mib": 7168,
        "t_start": t_start, "t_end": t_end,
    }
    ext = 5000 / (wps * 10)
    res["extrapolation"] = {"rtf": round(wps * 10, 1),
                            "gpus_for_10kh_in_2h": round(ext, 1),
                            "gpus_for_10kh_in_3h": round(3333 / (wps * 10), 1)}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(n, b)
