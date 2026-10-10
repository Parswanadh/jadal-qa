"""Background download of the SELECT-8G subset (robust to timeouts).

Reads eval/results/scale/selection_small.json. For each tar:
  hf_hub_download(resume_download=True), up to 6 attempts, exp backoff;
  size check vs listing; tar member scan; extract mp4+json sidecars to
  $DATA/clips/<factory>/<worker>/; ffprobe every mp4.
Progress appended to eval/results/scale/download.log (stdout) and
eval/results/scale/download_status.json is rewritten after every shard so
foreground work can poll. Final: download_verify_small.json.
Only complete verified files land in clips/ (partials stay in HF cache).

Run (background): HF_HUB_DISABLE_XET=1 nohup .../python eval/scale/dl_small.py ...
"""
import json
import os
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone

DATA = "/home/parshu/projects/awing/data/ego10k_scale"
TARDIR = os.path.join(DATA, "tars")
CLIPDIR = os.path.join(DATA, "clips")
SEL = "eval/results/scale/selection_small.json"
STATUS = "eval/results/scale/download_status.json"
OUT = "eval/results/scale/download_verify_small.json"
REPO = "builddotai/Egocentric-10K"
MAX_ATTEMPTS = 6


def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def ffprobe(path):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=width,height,codec_name,avg_frame_rate,duration",
             "-show_entries", "format=duration,size", "-of", "json", path],
            capture_output=True, text=True, timeout=300)
        return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[-300:]}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def download_one(repo_path, expected_size):
    from huggingface_hub import hf_hub_download
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            p = hf_hub_download(REPO, repo_type="dataset", filename=repo_path,
                                local_dir=TARDIR, resume_download=True)
            got = os.path.getsize(p)
            if got != expected_size:
                last = f"size mismatch {got} != {expected_size}"
                raise IOError(last)
            with tarfile.open(p, "r") as tf:
                tf.getmembers()
            return {"path": p, "bytes": got, "attempts": attempt}
        except Exception as e:  # noqa: BLE001 - retry anything transient
            last = f"{type(e).__name__}: {str(e)[-200:]}"
            wait = min(30 * 2 ** (attempt - 1), 900)
            log(f"attempt {attempt} failed for {repo_path}: {last}; sleep {wait}s")
            time.sleep(wait)
    return {"error": last, "attempts": MAX_ATTEMPTS}


def main():
    os.makedirs(TARDIR, exist_ok=True)
    os.makedirs(CLIPDIR, exist_ok=True)
    sel = json.load(open(SEL))
    results = []
    for i, item in enumerate(sel["files"]):
        rp, exp = item["path"], item["size_bytes"]
        log(f"[{i + 1}/{len(sel['files'])}] GET {rp} ({exp / 1e6:.0f} MB)")
        dl = download_one(rp, exp)
        entry = {"repo_path": rp, **dl, "clips": []}
        if "error" not in dl:
            fac, worker = rp.split("/")[0], rp.split("/")[2]
            dest = os.path.join(CLIPDIR, fac, worker)
            os.makedirs(dest, exist_ok=True)
            with tarfile.open(dl["path"], "r") as tf:
                tf.extractall(dest, filter="data")
            for root, _, files in os.walk(dest):
                for fn in sorted(files):
                    if fn.endswith(".mp4"):
                        mp = os.path.join(root, fn)
                        entry["clips"].append({"mp4": mp, "ffprobe": ffprobe(mp)})
            bad = [c for c in entry["clips"] if "error" in c["ffprobe"]]
            log(f"OK {rp}: {len(entry['clips'])} mp4s, {len(bad)} ffprobe failures")
        else:
            log(f"FAILED {rp}: {dl['error']}")
        results.append(entry)
        ok = sum(1 for r in results if "error" not in r)
        json.dump({"done": i + 1, "total": len(sel["files"]), "ok": ok,
                   "results": results}, open(STATUS, "w"), indent=1)
    ok = sum(1 for r in results if "error" not in r)
    nclips = sum(len(r.get("clips", [])) for r in results)
    json.dump({"n_tars_ok": ok, "n_tars_total": len(results),
               "n_clips": nclips, "results": results}, open(OUT, "w"), indent=1)
    log(f"DONE: {ok}/{len(results)} tars, {nclips} clips")


if __name__ == "__main__":
    sys.exit(main())
