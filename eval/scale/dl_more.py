"""EXTENDED Ego10K corpus download: resumable, verified, provenance-tracked.

Reads eval/results/exp_corpus/selection_more.json. For each tar:
  hf_hub_download(resume_download=True), up to 6 attempts + exp backoff;
  byte-size check vs the selection metadata; tar member scan; extract mp4+json
  sidecars to $DATA/clips_more/<factory>/<worker>/; ffprobe every mp4;
  sha256 the tar; append a provenance row; then DELETE the tar to save disk.
A hard disk guard stops the loop (recording partial state) before /workspace
drops below MIN_FREE_BYTES. Resumable: tars already in provenance are skipped.
Progress is rewritten to download_status_more.json after every tar.

Run (background): HF_HUB_DISABLE_XET=1 nohup python eval/scale/dl_more.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

DATA = "/workspace/jadal_pod_bundle/data/ego10k_scale"
TARDIR = os.path.join(DATA, "tars_more")
CLIPDIR = os.path.join(DATA, "clips_more")
SEL = "eval/results/exp_corpus/selection_more.json"
PROV = "eval/results/exp_corpus/provenance.jsonl"
STATUS = "eval/results/exp_corpus/download_status_more.json"
OUT = "eval/results/exp_corpus/download_verify_more.json"
REPO = "builddotai/Egocentric-10K"
MAX_ATTEMPTS = 6
MIN_FREE_BYTES = 15_000_000_000
PREFETCH = 3


def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def ffprobe(path):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,codec_name,avg_frame_rate,duration",
             "-show_entries", "format=duration,size", "-of", "json", path],
            capture_output=True, text=True, timeout=300)
        return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[-300:]}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def cleanup_tar(p, repo_path):
    """Delete the tar and its local_dir cache metadata to reclaim disk."""
    try:
        os.remove(p)
    except FileNotFoundError:
        pass
    meta = os.path.join(TARDIR, ".cache", "huggingface", "download", repo_path)
    for suffix in (".metadata", ".lock"):
        try:
            os.remove(meta + suffix)
        except FileNotFoundError:
            pass


def load_done():
    done = {}
    if os.path.exists(PROV):
        for line in open(PROV):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done[row["tar"]] = row
            except json.JSONDecodeError:
                pass
    return done


def main():
    os.makedirs(TARDIR, exist_ok=True)
    os.makedirs(CLIPDIR, exist_ok=True)
    sel = json.load(open(SEL))
    files = sel["files"]
    done = load_done()
    results = []
    t0 = time.time()
    bytes_new = 0
    n_fail = 0
    n_retries = 0
    stopped_early = False

    def process(i, item, dl):
        nonlocal bytes_new, n_fail, n_retries
        rp, exp = item["path"], item["size_bytes"]
        if "error" in dl:
            n_fail += 1
            log(f"[{i + 1}/{len(files)}] FAILED {rp}: {dl['error']}")
            return {"repo_path": rp, **dl, "clips": []}
        n_retries += max(0, dl.get("attempts", 1) - 1)
        digest = sha256_of(dl["path"])
        fac, worker = rp.split("/")[0], rp.split("/")[2]
        dest = os.path.join(CLIPDIR, fac, worker)
        os.makedirs(dest, exist_ok=True)
        with tarfile.open(dl["path"], "r") as tf:
            tf.extractall(dest, filter="data")
        clips = []
        for root, _, fs in os.walk(dest):
            for fn in sorted(fs):
                if fn.endswith(".mp4"):
                    mp = os.path.join(root, fn)
                    clips.append({"mp4": mp, "ffprobe": ffprobe(mp)})
        bad = [c for c in clips if "error" in c["ffprobe"]]
        bytes_new += dl["bytes"]
        row = {"tar": rp, "bytes": dl["bytes"], "sha256": digest,
               "clips": clips, "n_clips": len(clips),
               "n_bad": len(bad), "ts": datetime.now(timezone.utc).isoformat()}
        with open(PROV, "a") as f:
            f.write(json.dumps(row) + "\n")
        cleanup_tar(dl["path"], rp)
        log(f"[{i + 1}/{len(files)}] OK {rp}: {len(clips)} mp4s, {len(bad)} bad, "
            f"{dl['bytes'] / 1e6:.0f} MB, attempt {dl.get('attempts')}, deleted tar")
        return {"repo_path": rp, **dl, "clips": clips}

    with ThreadPoolExecutor(max_workers=PREFETCH) as ex:
        pending = {}
        idx = 0

        def top_up(i):
            nonlocal idx, stopped_early
            while idx < len(files) and idx - i < PREFETCH:
                nxt = files[idx]
                if nxt["path"] in done:
                    idx += 1
                    continue
                free = shutil.disk_usage("/workspace").free
                if free - nxt["size_bytes"] < MIN_FREE_BYTES:
                    log(f"STOP: disk guard {free / 1e9:.1f} GB free; cannot queue {nxt['path']}")
                    stopped_early = True
                    break
                pending[idx] = ex.submit(download_one, nxt["path"], nxt["size_bytes"])
                idx += 1

        i = 0
        top_up(i)
        while i < len(files) and i < idx:
            item = files[i]
            if item["path"] in done:
                log(f"skip (already in provenance): {item['path']}")
                results.append({"repo_path": item["path"], "skipped": True, "clips": []})
                i += 1
                top_up(i)
                continue
            dl = pending.pop(i).result()
            results.append(process(i, item, dl))
            json.dump({"done": i + 1, "total": len(files),
                       "n_ok": sum(1 for r in results if "error" not in r),
                       "n_fail": n_fail, "bytes_new": bytes_new,
                       "stopped_early": stopped_early,
                       "results": results}, open(STATUS, "w"), indent=1)
            i += 1
            top_up(i)

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if "error" not in r and not r.get("skipped"))
    n_clips = sum(len(r.get("clips", [])) for r in results)
    summary = {"n_tars_ok": n_ok, "n_tars_total": len(files),
               "n_fail": n_fail, "n_retries": n_retries,
               "bytes_new": bytes_new, "MB_per_s": round(bytes_new / 1e6 / elapsed, 1)
               if elapsed else 0, "elapsed_s": round(elapsed, 1),
               "stopped_early": stopped_early, "n_clips": n_clips,
               "results": results}
    json.dump(summary, open(OUT, "w"), indent=1)
    log(f"DONE: {n_ok}/{len(files)} tars, {n_clips} clips, {bytes_new / 1e9:.2f} GB, "
        f"{elapsed / 60:.1f} min, stopped_early={stopped_early}")


if __name__ == "__main__":
    sys.exit(main())
