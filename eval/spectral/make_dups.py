"""Build the DUPS near-duplicate pair set (by construction labels).

Reads (frozen, read-only): spectral_data/manifest_gold.json
Writes (only): spectral_data/dups/*.mp4, spectral_data/manifest_dups.json,
               spectral_data/READY_dups, scratch under .scratch/dups/.

Deterministic (seed 0). Re-runnable: skips up-to-date outputs.
8 base sources x (1 orig + 5 perturbations) = 48 files.
Positives: 8 x 5 = 40 (orig vs perturbed, same content).
Negatives: >=80 (hard = same clip different source, easy = different clips).

python: /home/parshu/envs/jadal/bin/python ; no pip installs ; <=4 workers.
"""
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

SEED = 0
N_SRC = 8
PERTS = ["reencode", "crop5", "bright", "speed11", "offset1"]
N_HARD = 44
N_EASY = 44

GOLD = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_gold.json"
OUTDIR = "/home/parshu/projects/jadal-qa-wt/spectral_data/dups"
MANIFEST = "/home/parshu/projects/jadal-qa-wt/spectral_data/manifest_dups.json"
READY = "/home/parshu/projects/jadal-qa-wt/spectral_data/READY_dups"
SCRATCH = "/home/parshu/projects/jadal-qa-wt/spectral-harness/.scratch/dups"

BASE_VF = "scale=480:480,fps=10"
BASE_ENC = ["-an", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-r", "10"]


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True)
    if p.returncode != 0:
        raise RuntimeError(f"FAILED: {' '.join(cmd)}\n{p.stderr[-3000:]}")
    return p


def probe(path):
    p = run(["ffprobe", "-v", "error", "-count_frames",
             "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
             "-of", "json", path])
    s = json.loads(p.stdout)["streams"][0]
    fps_s = s.get("avg_frame_rate", "0/0")
    num, den = fps_s.split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    return {"w": int(s["width"]), "h": int(s["height"]),
            "frames": int(s["nb_read_frames"]), "fps": fps}


def build_jobs(sources):
    """Return list of (file_id, kind, build_fn_args). Encoded in run_all."""
    jobs = []
    for i, s in enumerate(sources):
        tag = f"dup_src{i:02d}"
        jobs.append((f"{tag}_orig", "orig", ("orig", s, None)))
        for pert in PERTS:
            jobs.append((f"{tag}_{pert}", pert, (pert, s, tag)))
    return jobs


def encode(job, outdir):
    fid, kind, (op, s, tag) = job
    out = os.path.join(outdir, f"{fid}.mp4")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return (fid, out, True)  # skipped
    src = s["path"]
    if op == "orig":
        cmd = (["ffmpeg", "-y", "-ss", str(s["start_s"]), "-i", src,
                "-t", "10", "-vf", BASE_VF] + BASE_ENC + [out])
    elif op == "reencode":
        inp = os.path.join(outdir, f"{tag}_orig.mp4")
        cmd = (["ffmpeg", "-y", "-i", inp,
                "-vf", "fps=10"] + ["-an", "-c:v", "libx264", "-crf", "30",
                "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", "10", out])
    elif op == "crop5":
        inp = os.path.join(outdir, f"{tag}_orig.mp4")
        cmd = (["ffmpeg", "-y", "-i", inp, "-vf",
                "crop=456:456,scale=480:480,fps=10"] + BASE_ENC + [out])
    elif op == "bright":
        inp = os.path.join(outdir, f"{tag}_orig.mp4")
        cmd = (["ffmpeg", "-y", "-i", inp, "-vf",
                "scale=480:480,eq=brightness=0.12:contrast=1.12,fps=10"]
               + BASE_ENC + [out])
    elif op == "speed11":
        cmd = (["ffmpeg", "-y", "-ss", str(s["start_s"]), "-i", src,
                "-t", "11", "-vf", "scale=480:480,setpts=PTS/1.1,fps=10",
                "-t", "10"] + BASE_ENC + [out])
    elif op == "offset1":
        cmd = (["ffmpeg", "-y", "-ss", str(s["start_s"] + 1.0), "-i", src,
                "-t", "10", "-vf", BASE_VF] + BASE_ENC + [out])
    else:
        raise ValueError(op)
    run(cmd)
    return (fid, out, False)


def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(SCRATCH, exist_ok=True)
    with open(GOLD) as f:
        gold = json.load(f)
    items = sorted(gold["items"], key=lambda d: d["id"])

    # Deterministic base selection: prefer label==1, spread across >=2 clips.
    # Fixed choice verified against durations (all need start_s+11 <= duration):
    # 5x ecu_full + 2x ecu_late + 1x desk.
    want = ["g001", "g010", "g020", "g030", "g040", "g048", "g055", "g062"]
    by_id = {d["id"]: d for d in items}
    sources = []
    for gid in want:
        assert gid in by_id, f"gold id {gid} missing"
        sources.append(by_id[gid])
    assert len({s["clip"] for s in sources}) >= 2
    assert all(s["label"] == 1 for s in sources)

    jobs = build_jobs(sources)
    assert len(jobs) == N_SRC * (1 + len(PERTS)) == 48

    # Order matters: origs first (pertiurbations read orig files), then perts in parallel.
    origs = [j for j in jobs if j[1] == "orig"]
    perts = [j for j in jobs if j[1] != "orig"]
    failures = []
    skipped = 0
    for j in origs:  # sequential, cheap
        try:
            _, _, sk = encode(j, OUTDIR)
            skipped += sk
        except Exception as e:
            failures.append(f"{j[0]}: {e}")
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(encode, j, OUTDIR): j for j in perts}
        for fut in futs:
            try:
                _, _, sk = fut.result()
                skipped += sk
            except Exception as e:
                failures.append(f"{futs[fut][0]}: {e}")
    # Retry any pert whose orig was missing on first pass (re-runnable anyway)
    if failures:
        still = []
        for j in perts:
            fid = j[0]
            if any(f.startswith(fid + ":") for f in failures) and \
                    os.path.exists(os.path.join(OUTDIR, f"{fid}.mp4")):
                continue
            if any(f.startswith(fid + ":") for f in failures):
                try:
                    encode(j, OUTDIR)
                except Exception as e:
                    still.append(f"{fid}: {e}")
        failures = still

    # Verify geometry / frame counts.
    files = {}
    bad = []
    for j in jobs:
        fid = j[0]
        path = os.path.abspath(os.path.join(OUTDIR, f"{fid}.mp4"))
        if not os.path.exists(path):
            bad.append(f"{fid}: missing file")
            continue
        try:
            pr = probe(path)
        except Exception as e:
            bad.append(f"{fid}: probe failed: {e}")
            continue
        if not (pr["w"] == 480 and pr["h"] == 480):
            bad.append(f"{fid}: geometry {pr['w']}x{pr['h']}")
        if not (99 <= pr["frames"] <= 101):
            bad.append(f"{fid}: frames={pr['frames']}")
        files[fid] = path

    # Pairs.
    pairs = []
    for i in range(N_SRC):
        tag = f"dup_src{i:02d}"
        for pert in PERTS:
            pairs.append({"a": f"{tag}_orig", "b": f"{tag}_{pert}",
                          "is_duplicate": True, "kind": pert,
                          "params": ("crf30" if pert == "reencode"
                                     else "crop456+resize" if pert == "crop5"
                                     else "eq_b0.12_c1.12" if pert == "bright"
                                     else "1.1x_from_11s_truncated" if pert == "speed11"
                                     else "shift+1s_window")})
    # Negative candidates: same-clip/different-source (hard), different-clip (easy).
    clip_of = {}
    src_of = {}
    for i, s in enumerate(sources):
        tag = f"dup_src{i:02d}"
        for fid in [f"{tag}_orig"] + [f"{tag}_{p}" for p in PERTS]:
            clip_of[fid] = s["clip"]
            src_of[fid] = i
    fids = sorted(files.keys())
    hard_cand = sorted((a, b) for x, a in enumerate(fids) for b in fids[x + 1:]
                       if clip_of[a] == clip_of[b] and src_of[a] != src_of[b])
    easy_cand = sorted((a, b) for x, a in enumerate(fids) for b in fids[x + 1:]
                       if clip_of[a] != clip_of[b])
    rng = random.Random(SEED)
    rng.shuffle(hard_cand)
    rng.shuffle(easy_cand)
    hard = hard_cand[:N_HARD]
    easy = easy_cand[:N_EASY]
    # Require at least one orig-orig hard pair per same-clip group (info, not fatal).
    for a, b in hard:
        pairs.append({"a": a, "b": b, "is_duplicate": False, "kind": "hard_negative"})
    for a, b in easy:
        pairs.append({"a": a, "b": b, "is_duplicate": False, "kind": "easy_negative"})

    n_pos = sum(1 for p in pairs if p["is_duplicate"])
    n_neg = sum(1 for p in pairs if not p["is_duplicate"])
    assert n_pos >= 40, n_pos
    assert n_neg >= 80, n_neg
    assert all(p["a"] in files and p["b"] in files for p in pairs)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "sources": {f"dup_src{i:02d}": {"gold_id": s["id"], "clip": s["clip"],
                                        "path": s["path"], "start_s": s["start_s"],
                                        "end_s": s["end_s"],
                                        "gold_label": s["label"]} for i, s in enumerate(sources)},
        "files": files,
        "pairs": pairs,
        "failures": failures + bad,
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)

    # Cleanup scratch intermediates (none expected besides stray tmp files).
    for name in os.listdir(SCRATCH):
        p = os.path.join(SCRATCH, name)
        try:
            os.remove(p) if os.path.isfile(p) else shutil.rmtree(p)
        except OSError:
            pass

    ok = not failures and not bad
    if ok:
        with open(READY, "w") as f:
            f.write("")
    wall = time.time() - t0
    from collections import Counter
    c = Counter((p["kind"], p["is_duplicate"]) for p in pairs)
    print(f"files={len(files)} pairs={len(pairs)} pos={n_pos} neg={n_neg}")
    print("per-kind:", dict(c))
    print(f"failures={failures + bad}")
    print(f"skipped_up_to_date={skipped} wall_s={wall:.1f} READY={ok}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
