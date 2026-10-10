"""Download the SELECT-53 subset with resume + retry + verification.

Reads eval/results/scale/selection.json, downloads each tar to
/home/parshu/projects/awing/data/ego10k_scale/tars/, extracts mp4+json
sidecars to .../clips/<factory>/<worker>/, ffprobe-verifies every mp4.

Robustness (network has timed out before):
  - huggingface_hub resume_download=True (no .incomplete leftovers: partial
    blobs stay in HF cache, final tree only gets complete verified files)
  - up to 6 attempts per file, exponential backoff (30, 60, 120, ...)
  - per-file size check against repo_listing.json, `tar -tf` integrity check
  - ffprobe every extracted mp4 (resolution/fps/codec/duration recorded)
Writes eval/results/scale/download_verify.json.

Run: HF_HUB_DISABLE_XET=1 .../python eval/scale/download_subset.py
Requires a token with 'public gated repositories' permission (see report).
"""
import json
import os
import subprocess
import tarfile
import time

DATA = "/home/parshu/projects/awing/data/ego10k_scale"
TARDIR = os.path.join(DATA, "tars")
CLIPDIR = os.path.join(DATA, "clips")
SEL = "eval/results/scale/selection.json"
LISTING = "eval/results/scale/repo_listing.json"
OUT = "eval/results/scale/download_verify.json"
REPO = "builddotai/Egocentric-10K"
MAX_ATTEMPTS = 6


def ffprobe(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=width,height,codec_name,avg_frame_rate,duration",
         "-show_entries", "format=duration,size", "-of", "json", path],
        capture_output=True, text=True, timeout=300)
    return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[-500:]}


def download_one(repo_path, expected_size):
    from huggingface_hub import hf_hub_download
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            p = hf_hub_download(
                REPO, repo_type="dataset", filename=repo_path,
                local_dir=TARDIR, resume_download=True)
            got = os.path.getsize(p)
            if got != expected_size:
                last = f"size mismatch {got} != {expected_size}"
                raise IOError(last)
            with tarfile.open(p, "r") as tf:  # integrity: full member scan
                tf.getmembers()
            return {"path": p, "bytes": got, "attempts": attempt}
        except Exception as e:  # noqa: BLE001 - retry any transient failure
            last = f"{type(e).__name__}: {str(e)[-200:]}"
            time.sleep(min(30 * 2 ** (attempt - 1), 900))
    return {"error": last, "attempts": MAX_ATTEMPTS}


def main():
    os.makedirs(TARDIR, exist_ok=True)
    os.makedirs(CLIPDIR, exist_ok=True)
    sel = json.load(open(SEL))
    sizes = {t["path"]: t["size_bytes"] for t in json.load(open(LISTING))["tars"]}
    results = []
    for rp in sel["files"]:
        fac, worker = rp.split("/")[0], rp.split("/")[1]
        print(f"--- {rp}", flush=True)
        dl = download_one(rp, sizes[rp])
        entry = {"repo_path": rp, **dl, "clips": []}
        if "error" not in dl:
            dest = os.path.join(CLIPDIR, fac, worker)
            os.makedirs(dest, exist_ok=True)
            with tarfile.open(dl["path"], "r") as tf:
                tf.extractall(dest, filter="data")
            for root, _, files in os.walk(dest):
                for fn in sorted(files):
                    if fn.endswith(".mp4"):
                        mp = os.path.join(root, fn)
                        entry["clips"].append({"mp4": mp, "ffprobe": ffprobe(mp)})
        results.append(entry)
    ok = sum(1 for r in results if "error" not in r)
    nclips = sum(len(r.get("clips", [])) for r in results)
    with open(OUT, "w") as f:
        json.dump({"n_tars_ok": ok, "n_tars_total": len(results),
                   "n_clips": nclips, "results": results}, f, indent=1)
    print(f"done: {ok}/{len(results)} tars, {nclips} clips -> {OUT}")


if __name__ == "__main__":
    main()
