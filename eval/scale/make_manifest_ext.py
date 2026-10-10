"""Build the EXTENDED Ego10K window manifest from newly downloaded clips.

Walks $DATA/clips_more/<fac>/<worker>/*.mp4, ffprobes each, emits
non-overlapping [start, start+10) windows (partial tail < 10 s dropped),
ids s_<sha1(clip_relpath,start)[:12]> -- same schema as make_manifest.py.

Outputs:
  queues/manifest_scale_ext.json : NEW clips only (the extension)
  queues/manifest_scale_all.json : existing (manifest_scale.json) + NEW clips

Both carry {built_utc, window_s, n_clips, n_windows, total_hours, items:[...]}.
"""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

CLIPDIR = "/workspace/jadal_pod_bundle/data/ego10k_scale/clips_more"
EXISTING = "/workspace/jadal_pod_bundle/queues/manifest_scale.json"
OUT_EXT = "queues/manifest_scale_ext.json"
OUT_ALL = "queues/manifest_scale_all.json"
WIN = 10.0


def ffprobe(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,codec_name,avg_frame_rate,duration",
         "-show_entries", "format=duration,size", "-of", "json", path],
        capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-300:])
    return json.loads(r.stdout)


def fps_of(avg):
    try:
        n, d = avg.split("/")
        return float(n) / float(d)
    except Exception:  # noqa: BLE001
        return None


def build_items(clipdir):
    clips = []
    for root, _, files in os.walk(clipdir):
        for fn in sorted(files):
            if fn.endswith(".mp4"):
                clips.append(os.path.join(root, fn))
    clips.sort()
    items, total_dur = [], 0.0
    for mp in clips:
        rel = os.path.relpath(mp, clipdir)
        parts = rel.split(os.sep)
        fac = parts[0]
        worker = parts[1] if len(parts) > 2 else "?"
        pr = ffprobe(mp)
        st = pr["streams"][0]
        dur = float(pr["format"].get("duration") or st.get("duration") or 0)
        total_dur += dur
        for i in range(int(dur // WIN)):
            s = round(i * WIN, 3)
            hid = hashlib.sha1(f"{rel}@{s}".encode()).hexdigest()[:12]
            items.append({
                "id": f"s_{hid}", "clip": rel, "path": mp,
                "start_s": s, "end_s": round(s + WIN, 3),
                "factory": fac, "worker": worker, "duration_s": round(dur, 2),
                "width": st["width"], "height": st["height"],
                "fps": fps_of(st.get("avg_frame_rate", "0/0")),
                "codec": st.get("codec_name"),
                "size_bytes": int(pr["format"].get("size", 0)),
            })
    return clips, items, total_dur


def main():
    clips, items, total_dur = build_items(CLIPDIR)
    ext = {"built_utc": datetime.now(timezone.utc).isoformat(), "window_s": WIN,
           "n_clips": len(clips), "n_windows": len(items),
           "total_hours": round(total_dur / 3600, 3), "items": items}
    json.dump(ext, open(OUT_EXT, "w"), indent=1)

    all_items = list(items)
    n_exist_clips = 0
    if os.path.exists(EXISTING):
        ex = json.load(open(EXISTING))
        all_items = ex["items"] + items
        n_exist_clips = ex.get("n_clips", 0)
    allman = {"built_utc": datetime.now(timezone.utc).isoformat(), "window_s": WIN,
              "n_clips": n_exist_clips + len(clips), "n_new_clips": len(clips),
              "n_existing_clips": n_exist_clips, "n_windows": len(all_items),
              "total_hours": round((sum(it["duration_s"] for it in all_items)
                                    if False else 0) / 3600, 3),
              "items": all_items}
    # total_hours for the union = per-clip durations counted once
    seen, tot = set(), 0.0
    for it in all_items:
        key = it["path"]
        if key not in seen:
            seen.add(key)
            tot += it["duration_s"]
    allman["total_hours"] = round(tot / 3600, 3)
    json.dump(allman, open(OUT_ALL, "w"), indent=1)

    facs = sorted({it["factory"] for it in items})
    print(f"NEW: {len(clips)} clips, {len(items)} windows, "
          f"{ext['total_hours']} h, {len(facs)} factories -> {OUT_EXT}")
    print(f"ALL: {allman['n_clips']} clips, {allman['n_windows']} windows, "
          f"{allman['total_hours']} h -> {OUT_ALL}")


if __name__ == "__main__":
    main()
