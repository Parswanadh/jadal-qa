"""Compute regime descriptors for the E4 window sets. CPU-only.

Sets:
  indego : 150 gold windows from manifest_gold.json (decode path @ start_s/end_s)
  ood1   : 60 queue-1 windows; descriptors on BOTH the 480p transcode
           (item `path`) and the 1080p30 source sub-window
           (item `source_path` @ source_start_s + 10 s)  -> E4.03 gap
  ood2   : 100 queue-2 windows (480p transcode only; labels pending)
  ego10k : stratified sample (seed 7, n=300) of poc_data/windows.jsonl,
           decoded from source clip @ start_s/end_s
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from descriptors import describe_window

WT = "/home/parshu/projects/jadal-qa-wt"


def load_indego():
    m = json.load(open(f"{WT}/spectral_data/manifest_gold.json"))
    out = []
    for it in m["items"]:
        out.append({"window_id": it["id"], "set": "indego", "clip": it["clip"],
                    "path": it["path"], "start_s": it["start_s"], "end_s": it["end_s"],
                    "label": it["label"]})
    return out


def load_queue(name):
    q = json.load(open(f"{WT}/scale-data/eval/results/scale/{name}.json"))
    return q["segments"]


def load_ood1():
    segs = load_queue("label_queue_random")
    lab = json.load(open(f"{WT}/ood_labels/labels.json"))["latest"]
    out = []
    for s in segs:
        sid = s["segment_id"]
        v = lab.get(sid, {}).get("verdict")
        out.append({"window_id": sid, "set": "ood1_transcode", "clip": s["clip"],
                    "path": s["path"], "start_s": s["start_s"], "end_s": s["end_s"],
                    "label": 1 if v == "good" else (0 if v == "bad" else None)})
        dur = s["end_s"] - s["start_s"]
        out.append({"window_id": sid, "set": "ood1_source", "clip": s["clip"],
                    "path": s["source_path"], "start_s": s["source_start_s"],
                    "end_s": s["source_start_s"] + dur, "label": out[-1]["label"]})
    return out


def load_ood2():
    out = []
    for s in load_queue("label_queue_random2"):
        out.append({"window_id": s["segment_id"], "set": "ood2_transcode", "clip": s["clip"],
                    "path": s["path"], "start_s": s["start_s"], "end_s": s["end_s"],
                    "label": None})
    return out


def load_ego10k(n=300, seed=7):
    import random
    rows = [json.loads(l) for l in open(f"{WT}/poc_data/windows.jsonl")]
    byf = {}
    for r in rows:
        byf.setdefault(r["factory"], []).append(r)
    rng = random.Random(seed)
    per = max(1, n // len(byf))
    out = []
    for f, rs in sorted(byf.items()):
        out += rng.sample(rs, min(per, len(rs)))
    while len(out) < n:
        extra = rng.choice(rows)
        if extra["window_id"] not in {o["window_id"] for o in out}:
            out.append(extra)
    res = []
    for r in out[:n]:
        res.append({"window_id": r["window_id"], "set": "ego10k_sample", "clip": r["clip"],
                    "path": r["clip"], "start_s": r["start_s"], "end_s": r["end_s"],
                    "label": None})
    return res


def run_one(w):
    try:
        d = describe_window(w["path"], w["start_s"], w["end_s"])
        return {"window_id": w["window_id"], "set": w["set"], "clip": w.get("clip"),
                "label": w.get("label"), "ok": True, "desc": d,
                "src": w["path"]}
    except Exception as e:  # noqa: BLE001
        return {"window_id": w["window_id"], "set": w["set"], "clip": w.get("clip"),
                "label": w.get("label"), "ok": False, "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="indego,ood1,ood2,ego10k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    want = set(args.sets.split(","))
    work = []
    if "indego" in want:
        work += load_indego()
    if "ood1" in want:
        work += load_ood1()
    if "ood2" in want:
        work += load_ood2()
    if "ego10k" in want:
        work += load_ego10k()
    if args.limit:
        work = work[:args.limit]
    print(f"windows: {len(work)}", flush=True)
    with Pool(args.jobs) as pool:
        recs = pool.map(run_one, work, chunksize=1)
    ok = sum(1 for r in recs if r["ok"])
    print(f"ok {ok}/{len(recs)}", flush=True)
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
