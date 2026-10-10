"""Build eval/scale/manifest_scale.json: every 10-s window of every mp4 on disk.

Walks /home/parshu/projects/awing/data/ego10k_scale/clips/<fac>/<worker>/*.mp4,
ffprobes each file, emits non-overlapping [start, start+10) windows
(partial tail < 10 s dropped), ids s_<sha1(clip_relpath,start)[:12]>.
Output: {built_utc, n_clips, n_windows, total_hours, items:[{id, clip, path,
start_s, end_s, factory, worker, duration_s, width, height, fps, size_bytes}]}.
"""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

CLIPDIR = "/home/parshu/projects/awing/data/ego10k_scale/clips"
OUT = "eval/scale/manifest_scale.json"
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


def main():
    items, total_dur = [], 0.0
    clips = []
    for root, _, files in os.walk(CLIPDIR):
        for fn in sorted(files):
            if fn.endswith(".mp4"):
                clips.append(os.path.join(root, fn))
    clips.sort()
    for mp in clips:
        rel = os.path.relpath(mp, CLIPDIR)
        fac = rel.split(os.sep)[0]
        worker = rel.split(os.sep)[1] if len(rel.split(os.sep)) > 2 else "?"
        pr = ffprobe(mp)
        st = pr["streams"][0]
        dur = float(pr["format"].get("duration") or st.get("duration") or 0)
        total_dur += dur
        n = int(dur // WIN)
        for i in range(n):
            s = round(i * WIN, 3)
            hid = hashlib.sha1(f"{rel}@{s}".encode()).hexdigest()[:12]
            items.append({
                "id": f"s_{hid}", "clip": os.path.splitext(fn)[0] if False else rel,
                "path": mp, "start_s": s, "end_s": round(s + WIN, 3),
                "factory": fac, "worker": worker, "duration_s": round(dur, 2),
                "width": st["width"], "height": st["height"],
                "fps": fps_of(st.get("avg_frame_rate", "0/0")),
                "codec": st.get("codec_name"),
                "size_bytes": int(pr["format"].get("size", 0)),
            })
    man = {"built_utc": datetime.now(timezone.utc).isoformat(), "window_s": WIN,
           "n_clips": len(clips), "n_windows": len(items),
           "total_hours": round(total_dur / 3600, 3), "items": items}
    json.dump(man, open(OUT, "w"), indent=1)
    facs = sorted({it["factory"] for it in items})
    print(f"{len(clips)} clips, {len(items)} windows, {man['total_hours']} h, "
          f"{len(facs)} factories -> {OUT}")


if __name__ == "__main__":
    main()
