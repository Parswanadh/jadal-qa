"""Dump Egocentric-10K repo file listing (public metadata) to repo_listing.json.

Uses huggingface_hub repo_info(files_metadata=True), which needs NO gated
content permission (unlike actual file downloads). Run:
  HF_HUB_DISABLE_XET=1 /home/parshu/envs/jadal/bin/python eval/scale/repo_listing.py
"""
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone

from huggingface_hub import repo_info

REPO = "builddotai/Egocentric-10K"
OUT = "eval/results/scale/repo_listing.json"

info = repo_info(REPO, repo_type="dataset", files_metadata=True)

tars = []          # (path, size)
intrinsics = []
other = []
per_worker = defaultdict(list)
factories = set()
workers = set()
for s in info.siblings:
    fn = s.rfilename
    sz = getattr(s, "size", None) or 0
    if fn.endswith(".tar"):
        tars.append({"path": fn, "size_bytes": sz})
        p = fn.split("/")
        factories.add(p[0])
        workers.add(p[0] + "/" + p[1])
        per_worker[p[0] + "/" + p[1]].append(sz)
    elif "intrinsics" in fn:
        intrinsics.append(fn)
    else:
        other.append(fn)

total = sum(t["size_bytes"] for t in tars)
sizes = sorted(t["size_bytes"] for t in tars)
ppw = sorted(len(v) for v in per_worker.values())
per_factory_workers = Counter(w.split("/")[0] for w in workers)
per_factory_tars = Counter(t["path"].split("/")[0] for t in tars)
per_factory_bytes = Counter()
for t in tars:
    per_factory_bytes[t["path"].split("/")[0]] += t["size_bytes"]

summary = {
    "repo": REPO,
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "commit_sha": getattr(info, "sha", None),
    "n_files_total": len(info.siblings),
    "n_tars": len(tars),
    "n_intrinsics": len(intrinsics),
    "n_other": len(other),
    "other_files": other,
    "total_tar_bytes": total,
    "total_tar_GB": round(total / 1e9, 2),
    "n_factories": len(factories),
    "factories": sorted(factories),
    "n_workers": len(workers),
    "tar_size_bytes": {
        "min": min(sizes), "median": int(statistics.median(sizes)), "max": max(sizes)
    },
    "parts_per_worker": {"min": min(ppw), "median": statistics.median(ppw), "max": max(ppw)},
    "workers_per_factory": dict(sorted(per_factory_workers.items())),
    "tars_per_factory": dict(sorted(per_factory_tars.items())),
    "GB_per_factory": {k: round(v / 1e9, 2) for k, v in sorted(per_factory_bytes.items())},
    # Publisher-declared (README.md, NOT ffprobe-verified: content downloads are
    # currently blocked by token permission; see docs/reports/scale_data.md).
    "publisher_declared": {
        "total_hours": 10000,
        "total_clips": 192900,
        "median_clip_s": 180.0,
        "mean_hours_per_worker": 4.68,
        "storage_TB": 16.4,
        "format": "H.265/MP4",
        "resolution": "1920x1080",
        "fps": 30,
        "codec": "h265",
    },
    "per_file_durations_s": None,
    "per_file_durations_note": (
        "Per-tar/per-clip durations live inside the tar JSON sidecars; content "
        "download returned 403 (token lacks 'public gated repositories' "
        "permission), so durations are NOT verified. Median 180 s per README."
    ),
}

with open(OUT, "w") as f:
    json.dump({"summary": summary, "tars": sorted(tars, key=lambda t: t["path"])}, f, indent=1)
print(f"wrote {OUT}: {len(tars)} tars, {summary['total_tar_GB']} GB, "
      f"{summary['n_factories']} factories, {summary['n_workers']} workers")
