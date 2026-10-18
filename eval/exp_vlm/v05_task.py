#!/usr/bin/env python3
"""GRID.V05 — task level on IndEgo: 12-tile sheets vs video, then the L2 text call.

L1 (per 30 s slice: is relevant work progressing?) is answerable from video alone and
has FREE ground truth on IndEgo: the VIA keystep annotations mark the start and end of
every step, so a slice inside a step is ``advancing`` and a slice in a gap is
``no_relevant_work``.

L2 (per clip: was the task completed?) is judged with the BUILDER'S working definition
(Addendum 1: pick an object from one place, put it in another place, accurately -- right
object, right destination, no drop and no redo; judge the visible start and end state,
not how busy the hands look).  Reported as "builder's definition, not Awign's".

Phases (run in order; each is resumable and writes its own cache):
  map     -- resolve clip -> annotation file, print the mapping.  NO API calls.
  sheet   -- build and send the 24 slice sheets (slice_l1).      24 image calls
  video   -- send the same 24 slices through analyze_video.      24 video calls
  l2      -- one text call per clip over its six slice answers.   4 text calls
  report  -- metrics only, no calls

Run: JADAL_VLM_LEDGER=/tmp/opencode/dsh/vlm_ledger_v05.jsonl \
     JADAL_BUNDLE=<bundle> python3 eval/exp_vlm/v05_task.py --phase map
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "dsh"))
sys.path.insert(0, os.path.join(ROOT, "eval"))

import common as C  # noqa: E402
import vlm_mosaic as M  # noqa: E402
import vlm_sheet as S  # noqa: E402
from exp_vlm import via as VIA  # noqa: E402
from exp_vlm import vlm_common as VC  # noqa: E402

OUT = os.path.join(ROOT, "eval", "results", "exp_vlm")
CACHE = os.path.join(OUT, "v05_cache.json")
SLICE_S, N_SLICES, N_TILES = 30.0, 6, 12
CLIPS = ["ecu_full", "ecu_late", "desk", "organise"]      # the 4 annotated gold clips

PROMPT_SHA = M.PROMPT_SHA256


def clip_map():
    """{clip_key: {'path':..., 'ann':..., 'segments':[...], 'duration_s':...}}"""
    anns = VIA.find_annotations()
    g = C.load_gold()
    by_clip = {}
    for i, cl in enumerate(g["clip"]):
        by_clip.setdefault(cl, g["path"][i])
    out = {}
    for key in CLIPS:
        path = by_clip.get(key)
        if not path:
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        bare = stem.replace("_480", "")
        hit = None
        for fname, ap in anns.items():
            fs = os.path.splitext(os.path.basename(fname))[0].replace("_480", "")
            if fs == bare or fs.startswith(bare) or bare.startswith(fs):
                hit = (fname, ap)
                break
        if not hit:
            out[key] = {"path": path, "ann": None, "error": "no annotation matched"}
            continue
        v = VIA.parse_via(hit[1])
        frames, fps = VC.decode_window(path, 0.0, max_frames=30)
        import cv2
        cap = cv2.VideoCapture(path)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps_full = cap.get(cv2.CAP_PROP_FPS) or 10.0
        cap.release()
        dur = float(n) / float(fps_full)
        out[key] = {"path": path, "ann": hit[1], "fname": hit[0],
                    "n_segments": len(v["segments"]), "segments": v["segments"],
                    "duration_s": round(dur, 2),
                    "n_gaps": len(VIA.gap_list(v["segments"], dur))}
    return out


def slices_for(cm):
    out = []
    for key, info in cm.items():
        if not info.get("segments"):
            continue
        for k, t0 in enumerate(VIA.slice_starts(info["duration_s"], N_SLICES, SLICE_S)):
            out.append({"clip": key, "slice": k, "t0": t0, "t1": t0 + SLICE_S,
                        "path": info["path"], "segments": info["segments"],
                        "duration_s": info["duration_s"]})
    return out


def load_cache():
    return json.load(open(CACHE)) if os.path.exists(CACHE) else {"sheet": {}, "video": {}, "l2": {}}


def save_cache(c):
    os.makedirs(OUT, exist_ok=True)
    json.dump(c, open(CACHE, "w"), indent=1, sort_keys=True)


def key_of(s):
    return f"{s['clip']}#{s['slice']}"


def phase_map():
    cm = clip_map()
    for k, v in cm.items():
        print(f"{k:10s} dur={v.get('duration_s')} segs={v.get('n_segments')} "
              f"gaps={v.get('n_gaps')} ann={os.path.basename(v['ann']) if v.get('ann') else None}")
    print(json.dumps({k: {"duration_s": v.get("duration_s"), "n_segments": v.get("n_segments"),
                          "ann": v.get("ann")} for k, v in cm.items()}, indent=1)[:1200])


def phase_sheet():
    c = load_cache()
    for s in slices_for(clip_map()):
        k = key_of(s)
        if k in c["sheet"]:
            continue
        frames, fps = VC.decode_window(s["path"], s["t0"], max_frames=int(SLICE_S * 30) + 30)
        times = S.cell_times(N_TILES, SLICE_S, s["t0"])          # clip-absolute labels
        idx = S.frame_indices(times, fps, len(frames), s["t0"])
        use = [frames[i] for i in idx]
        img, meta = S.build_sheet(use, times, cols=4)
        png = S.png_bytes(img)
        r = M.ask(png, "slice_l1", extra={"clip": s["clip"], "slice": s["slice"],
                                          "t0": s["t0"], "phase": "V05_sheet"})
        c["sheet"][k] = {"clip": s["clip"], "slice": s["slice"], "t0": s["t0"],
                         "png_sha256": M.png_sha256(png), "seconds": r["seconds"],
                         "ok": r["ok"], "error": r.get("error", ""), "raw": r["raw"],
                         "parsed": r["parsed"], "labels": [m["label"] for m in meta],
                         "truth_l1": VIA.l1_truth(s["segments"], s["t0"], s["t1"])}
        save_cache(c)
        print(f"  {k} ok={r['ok']} {r['seconds']}s truth={c['sheet'][k]['truth_l1']} "
              f"pred={(r['parsed'] or {}).get('progressing')}", flush=True)
    print("calls:", M.count_calls())


def phase_video():
    import muse_video as MV
    c = load_cache()
    for s in slices_for(clip_map()):
        k = key_of(s)
        if k in c["video"]:
            continue
        tmp = f"/tmp/opencode/dsh/v05_slice_{s['clip']}_{s['slice']}.mp4"
        if not os.path.exists(tmp):
            import subprocess
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(s["t0"]), "-t", str(SLICE_S),
                            "-i", s["path"], "-c:v", "libx264", "-crf", "28", "-an", tmp],
                           check=True)
        res = MV.analyze_video(tmp)
        c["video"][k] = {"clip": s["clip"], "slice": s["slice"], "t0": s["t0"],
                         "bytes": os.path.getsize(tmp), "seconds": res["seconds"],
                         "status": res["status"], "usable": res["usable"],
                         "reason_code": res["reason_code"], "raw": res["raw"],
                         "truth_l1": VIA.l1_truth(s["segments"], s["t0"], s["t1"])}
        save_cache(c)
        print(f"  {k} status={res['status']} {res['seconds']}s usable={res['usable']}", flush=True)


def phase_l2():
    c = load_cache()
    for clip in CLIPS:
        rows = [v for kk, v in c["sheet"].items() if v["clip"] == clip]
        rows.sort(key=lambda r: r["slice"])
        if not rows or clip in c["l2"]:
            continue
        payload = [{"slice": r["slice"], "t0_s": r["t0"], "answer": r["parsed"],
                    "answer_raw": (r["raw"] or "")[-400:]} for r in rows]
        r = M.ask_text(json.dumps(payload, indent=1), "l2_text",
                       extra={"clip": clip, "phase": "V05_l2"})
        c["l2"][clip] = {"ok": r["ok"], "seconds": r["seconds"], "raw": r["raw"],
                         "parsed": r["parsed"], "n_slices": len(rows),
                         "slice_answers": [x["answer"] for x in payload]}
        save_cache(c)
        print(f"  {clip}: {json.dumps(r['parsed'])[:200]}", flush=True)
    print("calls:", M.count_calls())


def phase_report():
    c = load_cache()
    rows = [v for v in c["sheet"].values() if v.get("parsed")]
    pred = [1 if (v["parsed"] or {}).get("progressing") == "advancing" else 0 for v in rows]
    truth = [1 if v["truth_l1"] == "advancing" else 0 for v in rows]
    clips = [v["clip"] for v in rows]
    agree = float(np.mean([p == t for p, t in zip(pred, truth)])) if rows else float("nan")
    base = float(np.mean(truth)) if rows else float("nan")
    kappa = VIA.cohen_kappa(pred, truth)
    sel = [v for v in rows if v["truth_l1"] == "no_relevant_work"]
    rec = float(np.mean([1 if (v["parsed"] or {}).get("progressing") != "advancing" else 0
                         for v in sel])) if sel else float("nan")
    out = {
        "n_slices": len(rows), "n_gap_slices": len(sel), "n_clips": len(set(clips)),
        "l1_agreement": round(agree, 4), "l1_always_advancing_baseline": round(base, 4),
        "l1_kappa": round(kappa, 4), "l1_gap_recall": round(rec, 4) if sel else None,
        "sheet_seconds_median": round(float(np.median([v["seconds"] for v in rows])), 2) if rows else None,
        "l2": {k: (v.get("parsed") or {}) for k, v in c["l2"].items()},
        "l2_truth_note": "IndEgo L2 truth is 4/4 'completed' (TASK.00b): ZERO variance, so no L2 accuracy can be computed against it",
        "video": {},
    }
    vids = [v for v in c["video"].values() if v.get("status") == "ok"]
    if vids:
        vp = [1 if v.get("usable") else 0 for v in vids]
        vt = [1 if v["truth_l1"] == "advancing" else 0 for v in vids]
        out["video"] = {"n": len(vids),
                        "agreement_with_truth": round(float(np.mean([a == b for a, b in zip(vp, vt)])), 4),
                        "seconds_median": round(float(np.median([v["seconds"] for v in vids])), 2),
                        "bytes_median": int(np.median([v["bytes"] for v in vids]))}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["map", "sheet", "video", "l2", "report"])
    a = ap.parse_args()
    {"map": phase_map, "sheet": phase_sheet, "video": phase_video,
     "l2": phase_l2, "report": phase_report}[a.phase]()
